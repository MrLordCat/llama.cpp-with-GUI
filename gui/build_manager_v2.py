"""
Build management module for llama.cpp with various backends and ROCm/Vulkan support
"""

import os
import subprocess
import platform
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PyQt6.QtCore import QThread, pyqtSignal
import urllib.request
import urllib.error


class BuildThread(QThread):
    """Thread for building in background"""
    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)  # success
    progress = pyqtSignal(int)  # percentage
    
    def __init__(self, commands: List[List[str]], working_dir: Path):
        super().__init__()
        self.commands = commands
        self.working_dir = working_dir
        self.should_stop = False
        
    def run(self):
        try:
            for i, command in enumerate(self.commands):
                if self.should_stop:
                    self.finished_signal.emit(False)
                    return
                    
                self.output.emit(f"\n▶️ Running: {' '.join(command)}\n")
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.working_dir,
                    bufsize=1,
                    universal_newlines=True
                )
                
                for line in process.stdout:
                    if self.should_stop:
                        process.terminate()
                        self.finished_signal.emit(False)
                        return
                    self.output.emit(line)
                    
                process.wait()
                
                if process.returncode != 0:
                    self.output.emit(f"\n❌ Error executing command (code: {process.returncode})\n")
                    self.finished_signal.emit(False)
                    return
                    
                # Update progress
                progress = int((i + 1) / len(self.commands) * 100)
                self.progress.emit(progress)
                
            self.output.emit("\n✅ Build completed successfully!\n")
            self.finished_signal.emit(True)
            
        except Exception as e:
            self.output.emit(f"\n❌ Exception: {str(e)}\n")
            self.finished_signal.emit(False)
            
    def stop(self):
        self.should_stop = True


class BuildManager:
    """Class for managing llama.cpp build with ROCm/Vulkan support for AMD GPU"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build"
        self.os_type = platform.system()
        
    def get_cmake_generator(self) -> Optional[str]:
        """Determine CMake generator for platform"""
        if self.os_type == "Windows":
            # Check availability of various generators
            generators = [
                "Visual Studio 17 2022",
                "Visual Studio 16 2019",
                "Ninja",
            ]
            
            for gen in generators:
                if self._check_generator(gen):
                    return gen
                    
        return None  # Use default generator
        
    def _check_generator(self, generator: str) -> bool:
        """Check CMake generator availability"""
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
        Build CMake configuration command
        
        Args:
            backend: Selected backend (CUDA, Metal, Vulkan, ROCm)
            additional_options: Additional CMake options
        """
        command = ["cmake", "-B", str(self.build_dir)]
        
        # Select generator
        generator = self.get_cmake_generator()
        if generator:
            command.extend(["-G", generator])
            
        # Configure backend
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
                # ROCm with support for AMD RDNA 4 (9070XT uses gfx1201)
                command.append("-DGGML_HIPBLAS=ON")
                command.append("-DAMD_ARCH=gfx1201")
                
        # Additional options
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
        jobs: Optional[int] = None
    ) -> List[str]:
        """
        Build CMake build command
        
        Args:
            config: Build configuration (Release, Debug)
            jobs: Number of parallel jobs
        """
        command = ["cmake", "--build", str(self.build_dir), "--config", config]
        
        if jobs:
            command.extend(["-j", str(jobs)])
        else:
            # Use number of CPU cores
            cpu_count = os.cpu_count() or 4
            command.extend(["-j", str(cpu_count)])
            
        return command
        
    def check_build_prerequisites(self, backend: Optional[str] = None) -> Dict[str, bool]:
        """
        Check for required build tools
        
        Returns:
            Dictionary with check results
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
                
        # Check compiler
        if self.os_type == "Windows":
            checks["msvc"] = self._check_msvc()
        else:
            checks["gcc_or_clang"] = self._check_tool("gcc") or self._check_tool("clang")
            
        return checks
        
    def _check_tool(self, tool: str) -> bool:
        """Check if tool is available"""
        try:
            subprocess.run(
                [tool, "--version"],
                capture_output=True,
                timeout=3
            )
            return True
        except:
            return False
            
    def _check_msvc(self) -> bool:
        """Check for MSVC compiler"""
        try:
            result = subprocess.run(
                ["where", "cl.exe"],
                capture_output=True,
                timeout=3
            )
            return result.returncode == 0
        except:
            return False
            
    def _check_vulkan_sdk(self) -> bool:
        """Check for Vulkan SDK"""
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            return Path(vulkan_sdk).exists()
            
        # Check standard paths
        if self.os_type == "Windows":
            program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
            vulkan_path = Path(program_files) / "VulkanSDK"
            return vulkan_path.exists()
            
        return False
        
    def _check_rocm(self) -> bool:
        """Check for ROCm installation (important for AMD 9070XT)"""
        try:
            if self.os_type == "Windows":
                # Check common ROCm Windows paths
                rocm_paths = [
                    Path("C:/Program Files/AMD/ROCm"),
                    Path("C:/Program Files (x86)/AMD/ROCm"),
                    Path(os.environ.get("HIP_PATH", "")),
                ]
                for path in rocm_paths:
                    if path.exists():
                        bin_dir = path / "bin"
                        if bin_dir.exists() and (bin_dir / "hipcc.exe").exists():
                            return True
            elif self.os_type == "Linux":
                # Check hipcc command on Linux
                result = subprocess.run(
                    ["which", "hipcc"],
                    capture_output=True,
                    timeout=3
                )
                return result.returncode == 0
        except:
            pass
        return False
        
    def get_rocm_download_url(self) -> str:
        """Get ROCm download URL"""
        if self.os_type == "Windows":
            return "https://rocmdocs.amd.com/en/latest/deploy/windows/quick_start.html"
        else:
            return "https://rocmdocs.amd.com/en/latest/deploy/linux/index.html"
        
    def get_vulkan_download_url(self) -> str:
        """Get Vulkan SDK download URL (fallback for AMD GPU if ROCm unavailable)"""
        return "https://vulkan.lunarg.com/sdk/home"
        
    def get_recommended_backend_for_amd(self) -> Tuple[str, str]:
        """
        Get recommended backend for AMD GPU (9070XT)
        Returns: (backend_name, reason)
        """
        if self._check_rocm():
            return "ROCm", "ROCm is installed - native AMD GPU acceleration"
        else:
            return "Vulkan", "ROCm not found - using Vulkan (cross-platform GPU acceleration)"
        
    def get_build_info(self) -> Dict[str, Any]:
        """Get information about current build"""
        info = {
            "build_exists": self.build_dir.exists(),
            "executables": {},
        }
        
        if self.build_dir.exists():
            # Search for executables
            bin_dir = self.build_dir / "bin"
            if bin_dir.exists():
                for exe_name in ["llama-cli", "llama-server", "llama-quantize"]:
                    exe_path = None
                    
                    # Check various paths
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
                            
                    info["executables"][exe_name] = {
                        "exists": exe_path is not None,
                        "path": str(exe_path) if exe_path else None
                    }
                    
        return info
        
    def download_dependency(self, url: str, dest_file: Path, description: str = "Downloading") -> bool:
        """
        Download a dependency file
        
        Args:
            url: URL to download from
            dest_file: Destination file path
            description: Description of what's being downloaded
            
        Returns:
            True if successful
        """
        try:
            print(f"{description}...")
            urllib.request.urlretrieve(url, dest_file)
            return dest_file.exists()
        except Exception as e:
            print(f"Failed to download: {e}")
            return False
