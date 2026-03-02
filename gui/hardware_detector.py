"""
Модуль для определения характеристик железа и рекомендации оптимальных настроек
"""

import os
import platform
import subprocess
import psutil
from typing import Dict, List, Any, Optional
from pathlib import Path


class HardwareDetector:
    """Класс для определения характеристик железа"""
    
    def __init__(self):
        self.os_type = platform.system()
        
    def get_cpu_info(self) -> Dict[str, Any]:
        """Получение информации о процессоре"""
        info = {
            "name": platform.processor() or "Unknown CPU",
            "cores": psutil.cpu_count(logical=False) or 1,
            "threads": psutil.cpu_count(logical=True) or 1,
            "frequency": None,
        }
        
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["frequency"] = f"{cpu_freq.max:.0f} MHz"
        except:
            pass
            
        # Более детальная информация для Windows
        if self.os_type == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info["name"] = lines[1].strip()
            except:
                pass
                
        return info
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Получение информации об оперативной памяти"""
        mem = psutil.virtual_memory()
        
        return {
            "total": mem.total,
            "total_gb": mem.total / (1024 ** 3),
            "available": mem.available,
            "available_gb": mem.available / (1024 ** 3),
            "percent_used": mem.percent,
        }
        
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Получение информации о GPU"""
        gpus = []
        
        # Проверка NVIDIA GPU через nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            gpus.append({
                                "name": parts[0].strip(),
                                "memory": parts[1].strip(),
                                "type": "NVIDIA",
                                "backend": "CUDA"
                            })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Проверка GPU через PowerShell (более надежно чем WMIC)
        if self.os_type == "Windows":
            try:
                # Method 1: PowerShell Get-CimInstance (preferred)
                result = subprocess.run(
                    ["powershell", "-Command", 
                     "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        line = line.strip()
                        if line and not any(g['name'] == line for g in gpus):  # Avoid duplicates
                            gpu_entry = {
                                "name": line,
                                "type": "Unknown",
                                "backend": "Unknown"
                            }
                            
                            line_upper = line.upper()
                            if "AMD" in line_upper or "RADEON" in line_upper:
                                gpu_entry["type"] = "AMD"
                                gpu_entry["backend"] = "ROCm or Vulkan"
                                # Check for RDNA4 (9000 series)
                                if "9070" in line or "9080" in line or "9050" in line:
                                    gpu_entry["is_rdna4"] = True
                                    gpu_entry["is_9070xt"] = "9070" in line
                            elif "NVIDIA" in line_upper or "GEFORCE" in line_upper or "RTX" in line_upper or "GTX" in line_upper:
                                gpu_entry["type"] = "NVIDIA"
                                gpu_entry["backend"] = "CUDA"
                            elif "INTEL" in line_upper:
                                gpu_entry["type"] = "Intel"
                                gpu_entry["backend"] = "Vulkan or SYCL"
                            
                            gpus.append(gpu_entry)
            except Exception:
                pass
            
            # Method 2: Fallback to WMIC if PowerShell failed
            if not gpus:
                try:
                    result = subprocess.run(
                        ["wmic", "path", "win32_VideoController", "get", "name"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    for line in result.stdout.strip().split('\n')[1:]:
                        line = line.strip()
                        if line:
                            gpu_entry = {
                                "name": line,
                                "type": "Unknown",
                                "backend": "Unknown"
                            }
                            
                            line_upper = line.upper()
                            if "AMD" in line_upper or "RADEON" in line_upper:
                                gpu_entry["type"] = "AMD"
                                gpu_entry["backend"] = "ROCm or Vulkan"
                            elif "NVIDIA" in line_upper:
                                gpu_entry["type"] = "NVIDIA"
                                gpu_entry["backend"] = "CUDA"
                            elif "INTEL" in line_upper:
                                gpu_entry["type"] = "Intel"
                                gpu_entry["backend"] = "Vulkan or SYCL"
                            
                            gpus.append(gpu_entry)
                except:
                    pass
                    
            # Method 3: Try DirectX diagnostics as last resort
            if not gpus:
                try:
                    result = subprocess.run(
                        ["powershell", "-Command",
                         "(Get-WmiObject Win32_VideoController).Name"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            line = line.strip()
                            if line:
                                gpu_entry = {"name": line, "type": "Unknown", "backend": "Unknown"}
                                if "AMD" in line.upper() or "RADEON" in line.upper():
                                    gpu_entry["type"] = "AMD"
                                    gpu_entry["backend"] = "ROCm or Vulkan"
                                elif "NVIDIA" in line.upper():
                                    gpu_entry["type"] = "NVIDIA"
                                    gpu_entry["backend"] = "CUDA"
                                gpus.append(gpu_entry)
                except:
                    pass
                    
        elif self.os_type == "Linux":
            # Для Linux можно использовать lspci
            try:
                result = subprocess.run(
                    ["lspci"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                for line in result.stdout.split('\n'):
                    if "VGA" in line or "3D" in line:
                        if "AMD" in line or "ATI" in line:
                            gpus.append({
                                "name": line.split(': ')[1] if ': ' in line else line,
                                "type": "AMD",
                                "backend": "Vulkan or ROCm"
                            })
                        elif "Intel" in line:
                            gpus.append({
                                "name": line.split(': ')[1] if ': ' in line else line,
                                "type": "Intel",
                                "backend": "Vulkan or SYCL"
                            })
            except:
                pass
                
        # Проверка Metal (macOS)
        if self.os_type == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Парсинг вывода system_profiler
                if result.returncode == 0:
                    gpus.append({
                        "name": "Apple GPU",
                        "type": "Metal",
                        "backend": "Metal"
                    })
            except:
                pass
                
        return gpus
        
    def recommend_backend(self, gpu_info: List[Dict[str, Any]]) -> str:
        """Рекомендация оптимального backend на основе железа"""
        if not gpu_info:
            return "CPU (GPU не обнаружен)"
            
        # Приоритет: Metal > CUDA > Vulkan > CPU
        for gpu in gpu_info:
            gpu_type = gpu.get("type", "").upper()
            
            if "METAL" in gpu_type:
                return "Metal (рекомендуется для macOS)"
            elif "NVIDIA" in gpu_type:
                return "CUDA (рекомендуется для NVIDIA GPU)"
                
        # Если есть AMD или Intel GPU
        for gpu in gpu_info:
            gpu_type = gpu.get("type", "").upper()
            
            if "AMD" in gpu_type:
                return "Vulkan или ROCm (рекомендуется для AMD GPU)"
            elif "INTEL" in gpu_type:
                return "Vulkan или SYCL (рекомендуется для Intel GPU)"
                
        return "CPU"
        
    def get_hardware_info(self) -> Dict[str, Any]:
        """Получение полной информации о железе"""
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        gpu_info = self.get_gpu_info()
        
        return {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "recommended_backend": self.recommend_backend(gpu_info),
            "rocm_available": self._check_rocm(),
            "vulkan_available": self._check_vulkan_sdk(),
        }
        
    def check_dependencies(self, backend: str) -> Dict[str, bool]:
        """Проверка наличия необходимых зависимостей для выбранного backend"""
        checks = {
            "cmake": self._check_command("cmake"),
            "git": self._check_command("git"),
        }
        
        backend_lower = backend.lower()
        
        if "cuda" in backend_lower:
            checks["nvcc"] = self._check_command("nvcc")
            checks["nvidia-smi"] = self._check_command("nvidia-smi")
        elif "vulkan" in backend_lower:
            checks["vulkan_sdk"] = self._check_vulkan_sdk()
        elif "rocm" in backend_lower:
            checks["rocm"] = self._check_rocm()
        elif "metal" in backend_lower:
            checks["xcode"] = self._check_command("xcodebuild")
            
        return checks
        
    def _check_command(self, command: str) -> bool:
        """Проверка наличия команды в системе"""
        try:
            subprocess.run(
                [command, "--version"],
                capture_output=True,
                timeout=3
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
            
    def _check_vulkan_sdk(self) -> bool:
        """Проверка наличия Vulkan SDK"""
        # Проверка переменной окружения
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            return True
            
        # Проверка стандартных путей установки
        if self.os_type == "Windows":
            program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
            vulkan_path = Path(program_files) / "VulkanSDK"
            return vulkan_path.exists()
        elif self.os_type == "Linux":
            return Path("/usr/share/vulkan").exists()
            
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
            return self._check_command("hipcc")
    
    def is_rocm_available(self) -> bool:
        """Check if ROCm is available on this system"""
        return self._check_rocm()
    
    def is_vulkan_available(self) -> bool:
        """Check if Vulkan SDK is available"""
        return self._check_vulkan_sdk()


if __name__ == "__main__":
    # Тестирование
    detector = HardwareDetector()
    info = detector.get_hardware_info()
    
    import json
    print(json.dumps(info, indent=2, ensure_ascii=False))
