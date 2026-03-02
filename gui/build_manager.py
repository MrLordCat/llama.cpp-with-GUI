"""
Модуль для управления сборкой llama.cpp с различными backend
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any
from PyQt6.QtCore import QThread, pyqtSignal


class ConfigureThread(QThread):
    """Поток для CMake конфигурирования в фоне"""
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)  # success
    
    def __init__(self, command: List[str], working_dir: Path, env: Optional[Dict[str, str]] = None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.should_stop = False
        self.env = env
        
    def run(self):
        try:
            # Use provided environment or inherit from current
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
                universal_newlines=True,
                env=process_env
            )
            
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    self.finished_signal.emit(False)
                    return
                self.output.emit(line)
                
            process.wait()
            
            if process.returncode == 0:
                self.output.emit("\n✅ Configuration completed successfully!\n")
                self.output.emit("You can now click 'Build' to compile the project.\n")
                self.finished_signal.emit(True)
            else:
                self.output.emit(f"\n❌ Configuration failed with code {process.returncode}\n")
                self.finished_signal.emit(False)
                
        except Exception as e:
            self.output.emit(f"\n❌ Error: {str(e)}\n")
            self.finished_signal.emit(False)
            
    def stop(self):
        self.should_stop = True


class BuildThread(QThread):
    """Поток для сборки в фоне"""
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)  # success
    progress = pyqtSignal(int)  # процент 0-100
    build_progress = pyqtSignal(int, int, str)  # current, total, filename
    
    def __init__(self, commands: List[List[str]], working_dir: Path, env: Optional[Dict[str, str]] = None):
        super().__init__()
        self.commands = commands
        self.working_dir = working_dir
        self.should_stop = False
        self.env = env
        
    def _parse_ninja_progress(self, line: str) -> tuple:
        """Parse ninja/cmake progress from output like [17/413] Building..."""
        import re
        # Match patterns like [17/413] or [17/413]
        match = re.match(r'\[(\d+)/(\d+)\]\s*(.*)', line)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            action = match.group(3).strip()
            # Extract filename from action like "Building CXX object path/to/file.cpp.obj"
            filename = ""
            if "Building" in action or "Compiling" in action or "Linking" in action:
                parts = action.split()
                if parts:
                    filename = parts[-1].split('/')[-1].split('\\')[-1]
                    # Remove .obj suffix for cleaner display
                    if filename.endswith('.obj'):
                        filename = filename[:-4]
            return current, total, filename
        return None, None, None
        
    def run(self):
        try:
            # Use provided environment or inherit from current
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            
            for i, command in enumerate(self.commands):
                if self.should_stop:
                    self.finished_signal.emit(False)
                    return
                    
                self.output.emit(f"\n▶️ Выполнение: {' '.join(command)}\n")
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.working_dir,
                    bufsize=1,
                    universal_newlines=True,
                    env=process_env
                )
                
                for line in process.stdout:
                    if self.should_stop:
                        process.terminate()
                        self.finished_signal.emit(False)
                        return
                    self.output.emit(line)
                    
                    # Parse ninja/cmake progress
                    current, total, filename = self._parse_ninja_progress(line)
                    if current is not None and total is not None:
                        percent = int(current / total * 100)
                        self.progress.emit(percent)
                        self.build_progress.emit(current, total, filename)
                    
                process.wait()
                
                if process.returncode != 0:
                    self.output.emit(f"\n❌ Ошибка при выполнении команды (код: {process.returncode})\n")
                    self.finished_signal.emit(False)
                    return
                
            self.output.emit("\n✅ Сборка завершена успешно!\n")
            self.finished_signal.emit(True)
            
        except Exception as e:
            self.output.emit(f"\n❌ Исключение: {str(e)}\n")
            self.finished_signal.emit(False)
            
    def stop(self):
        self.should_stop = True


class BuildManager:
    """Класс для управления сборкой llama.cpp"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        self.os_type = platform.system()
    
    def detect_rocm_version(self) -> Optional[str]:
        """Detect installed ROCm/HIP SDK version
        
        Returns:
            Version string like '6.2' or '6.4.2' or None if not found
        """
        if self.os_type != "Windows":
            # On Linux, check /opt/rocm/.info/version
            version_file = Path("/opt/rocm/.info/version")
            if version_file.exists():
                try:
                    return version_file.read_text().strip()
                except:
                    pass
            return None
        
        # On Windows, check ROCm installation folder name
        rocm_base = Path("C:/Program Files/AMD/ROCm")
        if rocm_base.exists():
            versions = sorted(rocm_base.iterdir(), reverse=True)
            if versions:
                # Folder name is the version (e.g., "6.2", "6.4.2")
                return versions[0].name
        return None
    
    def detect_amd_gpu_targets(self) -> Optional[str]:
        """Auto-detect AMD GPU architecture using rocminfo or hipInfo
        
        Returns:
            GPU target string like 'gfx1201' or None if detection fails
        """
        if self.os_type != "Windows":
            # On Linux, try rocminfo
            try:
                result = subprocess.run(
                    ["rocminfo"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    import re
                    # Look for "Name: gfx..." pattern
                    matches = re.findall(r'Name:\s+(gfx\d+)', result.stdout)
                    if matches:
                        # Return unique GPU targets
                        unique_targets = list(dict.fromkeys(matches))
                        return ";".join(unique_targets)
            except:
                pass
            return None
        
        # On Windows, try hipInfo.exe from ROCm
        rocm_paths = [
            Path("C:/Program Files/AMD/ROCm"),
        ]
        
        hipinfo_exe = None
        for path in rocm_paths:
            if path.exists():
                versions = sorted(path.iterdir(), reverse=True)
                for ver in versions:
                    candidate = ver / "bin" / "hipInfo.exe"
                    if candidate.exists():
                        hipinfo_exe = candidate
                        break
                if hipinfo_exe:
                    break
        
        if not hipinfo_exe:
            return None
        
        try:
            # Need ROCm in PATH to run hipInfo
            env = os.environ.copy()
            rocm_bin = hipinfo_exe.parent
            env["PATH"] = f"{rocm_bin};{env.get('PATH', '')}"
            
            result = subprocess.run(
                [str(hipinfo_exe)],
                capture_output=True,
                text=True,
                timeout=15,
                env=env
            )
            
            if result.returncode == 0:
                import re
                # hipInfo output contains "gcnArchName: gfx1201" or similar
                matches = re.findall(r'gcnArchName:\s*(gfx\d+)', result.stdout)
                if matches:
                    unique_targets = list(dict.fromkeys(matches))
                    return ";".join(unique_targets)
                
                # Alternative pattern: look for gfxXXXX anywhere
                matches = re.findall(r'\bgfx(\d{4})\b', result.stdout)
                if matches:
                    unique_targets = list(dict.fromkeys([f"gfx{m}" for m in matches]))
                    return ";".join(unique_targets)
        except Exception as e:
            pass
        
        return None
    
    def get_msvc_env(self) -> Optional[Dict[str, str]]:
        """Get MSVC environment variables by running vcvarsall.bat"""
        if self.os_type != "Windows":
            return None
        
        # Find vcvarsall.bat
        vs_paths = [
            Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Auxiliary/Build"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build"),
        ]
        
        vcvarsall = None
        for vs_path in vs_paths:
            candidate = vs_path / "vcvarsall.bat"
            if candidate.exists():
                vcvarsall = candidate
                break
        
        if not vcvarsall:
            return None
        
        # Run vcvarsall.bat and capture environment
        try:
            # Run vcvarsall x64 and print all environment variables
            cmd = f'"{vcvarsall}" x64 && set'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return None
            
            # Parse environment variables from output
            env = {}
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, _, value = line.partition('=')
                    env[key.strip()] = value.strip()
            
            return env if env else None
            
        except Exception:
            return None
        
    def get_rocm_env(self) -> Optional[Dict[str, str]]:
        """Get environment variables for ROCm on Windows"""
        if self.os_type != "Windows":
            return None
        
        # Start with current environment (ROCm uses clang, not MSVC)
        env = os.environ.copy()
        
        # Find ROCm installation
        rocm_paths = [
            Path("C:/Program Files/AMD/ROCm"),
        ]
        
        rocm_root = None
        for path in rocm_paths:
            if path.exists():
                # Find latest version
                versions = list(path.iterdir())
                if versions:
                    rocm_root = sorted(versions, reverse=True)[0]
                    break
        
        if not rocm_root:
            return env if env else None
        
        # Add ROCm paths to environment
        hip_path = rocm_root / "bin"
        cmake_prefix = rocm_root / "lib" / "cmake"
        
        if hip_path.exists():
            current_path = env.get("PATH", "")
            
            # Add Strawberry Perl to PATH if installed (required for HIP)
            perl_paths = [
                "C:\\Strawberry\\perl\\bin",
                "C:\\Strawberry\\c\\bin",
                "C:\\Perl64\\bin",
                "C:\\Perl\\bin",
            ]
            extra_paths = [p for p in perl_paths if Path(p).exists()]
            
            # Build complete PATH: ROCm first, then Perl, then existing (which includes MSVC)
            path_parts = [str(hip_path), str(rocm_root)] + extra_paths + [current_path]
            env["PATH"] = ";".join(path_parts)
            
            env["HIP_PATH"] = str(rocm_root)
            env["ROCM_PATH"] = str(rocm_root)
            # Critical: Set HIP_PLATFORM for AMD GPUs
            env["HIP_PLATFORM"] = "amd"
            env["HIP_COMPILER"] = "clang"
            
            # RDNA4 (gfx1201) workaround: ROCm/HIP SDK < 6.4 doesn't have rocBLAS kernels for gfx1201
            # Force using gfx1100 (RDNA3) kernels which are compatible
            # ROCm 6.4+ has native gfx1201 support, so workaround is not needed
            rocm_version = self.detect_rocm_version()
            needs_rdna4_workaround = True
            if rocm_version:
                try:
                    # Parse version like "6.4" or "6.4.2"
                    major_minor = float(rocm_version.split('.')[0] + '.' + rocm_version.split('.')[1])
                    if major_minor >= 6.4:
                        needs_rdna4_workaround = False
                except:
                    pass
            
            if needs_rdna4_workaround:
                detected_gpu = self.detect_amd_gpu_targets()
                if detected_gpu and ("gfx1201" in detected_gpu or "gfx1200" in detected_gpu):
                    env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"  # Maps to gfx1100
            
            if cmake_prefix.exists():
                env["CMAKE_PREFIX_PATH"] = str(cmake_prefix)
        
        return env if env else None
        
    def get_cmake_generator(self, backend: Optional[str] = None) -> Optional[str]:
        """Определение генератора CMake для платформы
        
        Args:
            backend: Backend type - ROCm requires Ninja generator
        """
        if self.os_type == "Windows":
            # ROCm/HIP requires Ninja generator (Visual Studio doesn't work with HIP)
            if backend and backend.upper() == "ROCM":
                if self._check_generator("Ninja"):
                    return "Ninja"
                else:
                    return None  # Will fail, but user needs to install Ninja
            
            # For other backends, prefer Visual Studio
            generators = [
                "Visual Studio 17 2022",
                "Visual Studio 16 2019",
                "Visual Studio 18 2022",
                "Ninja",
            ]
            
            for gen in generators:
                if self._check_generator(gen):
                    return gen
                    
        return None  # Использовать генератор по умолчанию
        
    def _check_generator(self, generator: str) -> bool:
        """Проверка доступности генератора CMake"""
        try:
            result = subprocess.run(
                ["cmake", "-G", generator, "--help"],
                capture_output=True,
                timeout=3
            )
            return result.returncode == 0
        except:
            return False
            
    def get_configure_command(
        self,
        backend: Optional[str] = None,
        additional_options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Построение команды конфигурирования CMake
        
        Args:
            backend: Выбранный backend (CUDA, Metal, Vulkan, и т.д.)
            additional_options: Дополнительные опции CMake
        """
        command = ["cmake", "-B", str(self.build_dir)]
        
        # Выбор генератора (backend-aware - ROCm needs Ninja)
        generator = self.get_cmake_generator(backend)
        if generator:
            command.extend(["-G", generator])
        
        # Отключаем CURL (требует libcurl, не критична для функциональности)
        command.append("-DLLAMA_CURL=OFF")
            
        # Настройка backend
        if backend:
            backend_upper = backend.upper()
            
            if backend_upper == "CUDA":
                command.append("-DGGML_CUDA=ON")
            elif backend_upper == "METAL":
                command.append("-DGGML_METAL=ON")
            elif backend_upper == "VULKAN":
                command.append("-DGGML_VULKAN=ON")
            elif backend_upper == "SYCL":
                command.append("-DGGML_SYCL=ON")
            elif backend_upper == "ROCM":
                # Modern llama.cpp uses GGML_HIP instead of GGML_HIPBLAS
                command.append("-DGGML_HIP=ON")
                
                # Auto-detect GPU architecture
                detected_targets = self.detect_amd_gpu_targets()
                if detected_targets:
                    command.append(f"-DGPU_TARGETS={detected_targets}")
                else:
                    # Fallback: build for common AMD GPU architectures
                    # gfx1100 = RDNA3 (RX 7900), gfx1101 = RDNA3 (RX 7800/7700)
                    # gfx1200/gfx1201 = RDNA4 (RX 9070 series)
                    command.append("-DGPU_TARGETS=gfx1100;gfx1101;gfx1200;gfx1201")
                
                # Set CMAKE_BUILD_TYPE for Ninja (single-config generator)
                command.append("-DCMAKE_BUILD_TYPE=Release")
                
                # For Windows ROCm, must use clang/clang++ from ROCm (not MSVC)
                # See docs/build.md for official instructions
                rocm_env = self.get_rocm_env()
                if rocm_env and "HIP_PATH" in rocm_env:
                    hip_path = rocm_env["HIP_PATH"]
                    clang_c = Path(hip_path) / "bin" / "clang.exe"
                    clang_cxx = Path(hip_path) / "bin" / "clang++.exe"
                    
                    if clang_c.exists() and clang_cxx.exists():
                        command.append(f"-DCMAKE_C_COMPILER={clang_c}")
                        command.append(f"-DCMAKE_CXX_COMPILER={clang_cxx}")
                
        # Дополнительные опции
        if additional_options:
            for key, value in additional_options.items():
                if isinstance(value, bool):
                    command.append(f"-D{key}={'ON' if value else 'OFF'}")
                else:
                    command.append(f"-D{key}={value}")
                    
        return command
        
    def get_build_command(
        self,
        config: str = "Release",
        jobs: Optional[int] = None,
        backend: Optional[str] = None
    ) -> List[str]:
        """
        Построение команды сборки
        
        Args:
            config: Конфигурация сборки (Release, Debug)
            jobs: Количество параллельных задач
            backend: Backend type (ROCm uses Ninja which ignores --config)
        """
        command = ["cmake", "--build", str(self.build_dir)]
        
        # Ninja (used for ROCm) ignores --config flag, build type is set during configure
        # Visual Studio uses --config flag
        if not (backend and backend.upper() == "ROCM"):
            command.extend(["--config", config])
        
        if jobs:
            command.extend(["-j", str(jobs)])
        else:
            # Используем количество ядер процессора
            import os
            cpu_count = os.cpu_count() or 4
            
            # ROCm/HIP compilation is very memory-intensive
            # Limit parallel jobs to prevent system freeze
            if backend and backend.upper() == "ROCM":
                # Use max 4 parallel jobs for ROCm to avoid memory issues
                # Each clang++ process can use 2-4GB RAM for HIP compilation
                cpu_count = min(cpu_count, 4)
            
            command.extend(["-j", str(cpu_count)])
            
        return command
        
    def install_dependencies_windows(self) -> List[str]:
        """Получение команд для установки зависимостей на Windows"""
        commands = []
        
        # Проверка наличия winget
        try:
            subprocess.run(["winget", "--version"], capture_output=True, timeout=3)
            has_winget = True
        except:
            has_winget = False
            
        if has_winget:
            # Установка CMake через winget
            commands.append(["winget", "install", "-e", "--id", "Kitware.CMake"])
            
            # Установка Git
            commands.append(["winget", "install", "-e", "--id", "Git.Git"])
            
        return commands
        
    def install_vulkan_sdk_windows(self) -> List[str]:
        """Инструкции по установке Vulkan SDK на Windows"""
        # Vulkan SDK нужно устанавливать вручную
        # Возвращаем команду для открытия браузера
        return [
            ["start", "https://vulkan.lunarg.com/sdk/home#windows"]
        ]
        
    def install_cuda_windows(self) -> List[str]:
        """Инструкции по установке CUDA на Windows"""
        return [
            ["start", "https://developer.nvidia.com/cuda-downloads"]
        ]
        
    def check_build_prerequisites(self, backend: Optional[str] = None) -> Dict[str, bool]:
        """
        Проверка наличия необходимых инструментов для сборки
        
        Returns:
            Словарь с результатами проверки
        """
        checks = {
            "cmake": self._check_tool("cmake"),
            "git": self._check_tool("git"),
        }
        
        if backend:
            backend_upper = backend.upper()
            
            if backend_upper == "CUDA":
                checks["nvcc"] = self._check_tool("nvcc")
            elif backend_upper == "VULKAN":
                checks["vulkan_sdk"] = self._check_vulkan_sdk()
            elif backend_upper == "ROCM":
                checks["rocm"] = self._check_rocm()
                checks["ninja"] = self._check_ninja()
                checks["perl"] = self._check_perl()
                
        # Проверка компилятора
        if self.os_type == "Windows":
            checks["msvc"] = self._check_msvc()
        else:
            checks["gcc"] = self._check_tool("gcc") or self._check_tool("clang")
            
        return checks
        
    def _check_tool(self, tool: str) -> bool:
        """Проверка наличия инструмента"""
        try:
            subprocess.run(
                [tool, "--version"],
                capture_output=True,
                timeout=3
            )
            return True
        except:
            return False
    
    def _check_ninja(self) -> bool:
        """Check if Ninja build system is installed"""
        return self._check_tool("ninja")
    
    def _check_perl(self) -> bool:
        """Check if Perl is installed (required for ROCm/HIP)"""
        return self._check_tool("perl")
            
    def _check_msvc(self) -> bool:
        """Проверка наличия MSVC компилятора"""
        try:
            # Проверяем наличие cl.exe (компилятор MSVC)
            result = subprocess.run(
                ["where", "cl.exe"],
                capture_output=True,
                timeout=3
            )
            return result.returncode == 0
        except:
            return False
            
    def _check_vulkan_sdk(self) -> bool:
        """Проверка наличия Vulkan SDK"""
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            return Path(vulkan_sdk).exists()
            
        # Проверка стандартных путей
        if self.os_type == "Windows":
            program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
            vulkan_path = Path(program_files) / "VulkanSDK"
            return vulkan_path.exists()
            
        return False
        
    def _check_rocm(self) -> bool:
        """Check for ROCm installation"""
        if self.os_type == "Windows":
            # Check standard ROCm paths on Windows
            rocm_paths = [
                Path("C:/Program Files/AMD/ROCm"),
                Path("C:/Program Files (x86)/AMD/ROCm"),
                Path(os.environ.get("HIP_PATH", "")),
                Path(os.environ.get("ROCM_PATH", "")),
            ]
            for path in rocm_paths:
                if str(path) and path.exists():
                    return True
            return False
        else:
            # On Linux, check for hipcc command
            return self._check_tool("hipcc")
        
    def get_build_info(self) -> Dict[str, Any]:
        """Получение информации о текущей сборке"""
        info = {
            "build_exists": self.build_dir.exists(),
            "executables": {},
            "backend": None,
            "build_type": None,
            "features": [],
            "build_date": None,
        }
        
        if self.build_dir.exists():
            # Try to read CMakeCache.txt for build info
            cmake_cache = self.build_dir / "CMakeCache.txt"
            if cmake_cache.exists():
                try:
                    cache_content = cmake_cache.read_text(errors='ignore')
                    
                    # Detect backend from CMake cache
                    if "GGML_CUDA:BOOL=ON" in cache_content:
                        info["backend"] = "CUDA (NVIDIA GPU)"
                    elif "GGML_HIP:BOOL=ON" in cache_content or "GGML_ROCM:BOOL=ON" in cache_content:
                        info["backend"] = "ROCm (AMD GPU)"
                    elif "GGML_METAL:BOOL=ON" in cache_content:
                        info["backend"] = "Metal (macOS GPU)"
                    elif "GGML_VULKAN:BOOL=ON" in cache_content:
                        info["backend"] = "Vulkan"
                    elif "GGML_SYCL:BOOL=ON" in cache_content:
                        info["backend"] = "SYCL (Intel GPU)"
                    else:
                        info["backend"] = "CPU"
                    
                    # Detect build type
                    import re
                    build_type_match = re.search(r'CMAKE_BUILD_TYPE:STRING=(\w+)', cache_content)
                    if build_type_match:
                        info["build_type"] = build_type_match.group(1)
                    
                    # Detect features
                    features = []
                    if "LLAMA_CURL:BOOL=ON" in cache_content:
                        features.append("HTTP Downloads (curl)")
                    if "LLAMA_SERVER_SSL:BOOL=ON" in cache_content:
                        features.append("SSL Support")
                    if "GGML_OPENMP:BOOL=ON" in cache_content:
                        features.append("OpenMP Parallelism")
                    if "GGML_NATIVE:BOOL=ON" in cache_content:
                        features.append("Native CPU Optimizations")
                    if "GGML_AVX:BOOL=ON" in cache_content or "GGML_AVX2:BOOL=ON" in cache_content:
                        features.append("AVX/AVX2 SIMD")
                    if "GGML_AVX512:BOOL=ON" in cache_content:
                        features.append("AVX-512 SIMD")
                    info["features"] = features
                    
                    # Get build date from CMakeCache file modification time
                    import datetime
                    mtime = cmake_cache.stat().st_mtime
                    info["build_date"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    
                except Exception:
                    pass
            
            # Поиск исполняемых файлов
            bin_dir = self.build_dir / "bin"
            if bin_dir.exists():
                # Comprehensive list of llama.cpp executables
                all_executables = [
                    "llama-cli", "llama-server", "llama-quantize",
                    "llama-bench", "llama-perplexity", "llama-embedding",
                    "llama-speculative", "llama-batched", "llama-parallel",
                    "llama-save-load-state", "llama-simple", "llama-lookup",
                    "llama-infill", "llama-eval-callback", "llama-gguf",
                    "llama-gguf-hash", "llama-gguf-split", "llama-imatrix",
                    "llama-cvector-generator", "llama-tokenize"
                ]
                
                for exe_name in all_executables:
                    exe_path = None
                    
                    # Check various possible paths
                    possible_paths = [
                        bin_dir / f"{exe_name}.exe",
                        bin_dir / "Release" / f"{exe_name}.exe",
                        bin_dir / "Debug" / f"{exe_name}.exe",
                        bin_dir / exe_name,  # Linux/Mac
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            exe_path = path
                            break
                    
                    if exe_path:
                        # Get file size
                        try:
                            size_mb = exe_path.stat().st_size / (1024 * 1024)
                        except Exception:
                            size_mb = 0
                            
                        info["executables"][exe_name] = {
                            "exists": True,
                            "path": str(exe_path),
                            "size_mb": round(size_mb, 2)
                        }
                    
        return info
