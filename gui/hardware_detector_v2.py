"""
Hardware detection module with detailed AMD GPU and ROCm/Vulkan support
"""

import os
import platform
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class HardwareDetector:
    """Class for detecting hardware and recommending optimal configuration"""
    
    def __init__(self):
        self.os_type = platform.system()
        
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
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
            
        # More detailed info for Windows
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
        """Get RAM information"""
        mem = psutil.virtual_memory()
        
        return {
            "total": mem.total,
            "total_gb": mem.total / (1024 ** 3),
            "available": mem.available,
            "available_gb": mem.available / (1024 ** 3),
            "percent_used": mem.percent,
        }
        
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information including AMD detection"""
        gpus = []
        
        # Check NVIDIA GPU via nvidia-smi
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
                                "backend": "CUDA",
                                "rocm_support": False
                            })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Check AMD GPU
        if self.os_type == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name,adapterram"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if line and ("AMD" in line.upper() or "ATI" in line.upper() or "RADEON" in line.upper()):
                        # Check if it's 9070XT
                        is_9070xt = "9070" in line or "RX 9070" in line
                        is_rdna4 = is_9070xt  # 9070XT is RDNA 4
                        
                        gpu_info = {
                            "name": line,
                            "type": "AMD",
                            "backend": "ROCm (recommended) or Vulkan",
                            "rocm_support": True,
                            "rdna_generation": 4 if is_rdna4 else 3,
                            "is_9070xt": is_9070xt,
                        }
                        
                        # Try to get memory info if available
                        if "," in line:
                            parts = line.split(",")
                            if len(parts) >= 2:
                                gpu_info["memory"] = parts[1].strip()
                                
                        gpus.append(gpu_info)
                    elif line and "INTEL" in line.upper():
                        gpus.append({
                            "name": line,
                            "type": "Intel",
                            "backend": "Vulkan or SYCL",
                            "rocm_support": False
                        })
            except:
                pass
                
        elif self.os_type == "Linux":
            # For Linux, use lspci
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
                            is_9070xt = "9070" in line or "RX 9070" in line
                            gpus.append({
                                "name": line.split(': ')[1] if ': ' in line else line,
                                "type": "AMD",
                                "backend": "ROCm (recommended) or Vulkan",
                                "rocm_support": True,
                                "is_9070xt": is_9070xt,
                            })
                        elif "Intel" in line:
                            gpus.append({
                                "name": line.split(': ')[1] if ': ' in line else line,
                                "type": "Intel",
                                "backend": "Vulkan or SYCL",
                                "rocm_support": False
                            })
            except:
                pass
                
        # Check Metal (macOS)
        if self.os_type == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    gpus.append({
                        "name": "Apple GPU",
                        "type": "Metal",
                        "backend": "Metal",
                        "rocm_support": False
                    })
            except:
                pass
                
        return gpus
        
    def check_rocm_installed(self) -> bool:
        """Check if ROCm is installed (important for AMD)"""
        try:
            if self.os_type == "Windows":
                rocm_paths = [
                    Path("C:/Program Files/AMD/ROCm"),
                    Path(os.environ.get("HIP_PATH", "")),
                ]
                for path in rocm_paths:
                    if path.exists():
                        return True
            elif self.os_type == "Linux":
                result = subprocess.run(
                    ["which", "hipcc"],
                    capture_output=True,
                    timeout=3
                )
                return result.returncode == 0
        except:
            pass
        return False
        
    def check_vulkan_installed(self) -> bool:
        """Check if Vulkan SDK is installed"""
        try:
            vulkan_sdk = os.environ.get("VULKAN_SDK")
            if vulkan_sdk:
                return Path(vulkan_sdk).exists()
                
            if self.os_type == "Windows":
                program_files = os.environ.get("PROGRAMFILES", "C:\\Program Files")
                vulkan_path = Path(program_files) / "VulkanSDK"
                return vulkan_path.exists()
        except:
            pass
        return False
        
    def recommend_backend(self, gpu_info: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Recommend optimal backend based on hardware
        
        Returns:
            Tuple of (backend_name, recommendation_reason)
        """
        if not gpu_info:
            return ("CPU", "GPU not detected - will use CPU inference")
            
        # Priority: Metal > CUDA > ROCm > Vulkan > CPU
        for gpu in gpu_info:
            gpu_type = gpu.get("type", "").upper()
            
            if "METAL" in gpu_type:
                return ("Metal", f"Metal GPU detected: {gpu.get('name', 'Unknown')}")
            elif "NVIDIA" in gpu_type:
                return ("CUDA", f"NVIDIA GPU detected: {gpu.get('name', 'Unknown')}")
                
        # For AMD GPUs
        for gpu in gpu_info:
            gpu_type = gpu.get("type", "").upper()
            
            if "AMD" in gpu_type:
                is_9070xt = gpu.get("is_9070xt", False)
                if is_9070xt:
                    return ("ROCm/Vulkan", "AMD 9070XT detected - ROCm recommended (RDNA 4)")
                else:
                    return ("ROCm/Vulkan", f"AMD GPU detected: {gpu.get('name', 'Unknown')}")
            elif "INTEL" in gpu_type:
                return ("Vulkan/SYCL", f"Intel GPU detected: {gpu.get('name', 'Unknown')}")
                
        return ("CPU", "CPU-only inference")
        
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get complete hardware information"""
        cpu_info = self.get_cpu_info()
        memory_info = self.get_memory_info()
        gpu_info = self.get_gpu_info()
        
        backend_name, backend_reason = self.recommend_backend(gpu_info)
        rocm_available = self.check_rocm_installed()
        vulkan_available = self.check_vulkan_installed()
        
        return {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "recommended_backend": backend_name,
            "backend_reason": backend_reason,
            "rocm_available": rocm_available,
            "vulkan_available": vulkan_available,
        }
        
    def get_amd_gpu_details(self) -> Optional[Dict[str, Any]]:
        """Get detailed AMD GPU info if present"""
        gpus = self.get_gpu_info()
        for gpu in gpus:
            if gpu.get("type") == "AMD":
                return gpu
        return None
