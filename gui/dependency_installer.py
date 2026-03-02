"""
Dependency installer module - automatically downloads and installs required tools
"""

import os
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PyQt6.QtCore import QThread, pyqtSignal
import urllib.request
import urllib.error
import shutil


class DependencyInstallThread(QThread):
    """Thread for downloading and installing dependencies"""
    progress = pyqtSignal(str, int)  # (message, percentage)
    status_update = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, dependencies: List[str], os_type: str = "Windows"):
        super().__init__()
        self.dependencies = dependencies
        self.os_type = os_type
        self.should_stop = False
        
    def run(self):
        """Download and install dependencies"""
        try:
            total = len(self.dependencies)
            
            for i, dep in enumerate(self.dependencies):
                if self.should_stop:
                    self.finished_signal.emit(False, "Installation cancelled by user")
                    return
                
                percentage = int((i / total) * 100)
                self.status_update.emit(f"Installing {dep}...")
                self.progress.emit(f"Installing {dep}...", percentage)
                
                success = self._install_dependency(dep)
                
                if not success:
                    self.finished_signal.emit(False, f"Failed to install {dep}")
                    return
            
            self.progress.emit("Installation complete", 100)
            self.finished_signal.emit(True, "All dependencies installed successfully!")
            
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")
            
    def _install_dependency(self, dep: str) -> bool:
        """Install a single dependency"""
        if self.os_type == "Windows":
            return self._install_windows(dep)
        else:
            return self._install_linux(dep)
            
    def _install_windows(self, dep: str) -> bool:
        """Install dependency on Windows using winget or direct download"""
        dep_lower = dep.lower()
        
        if "cmake" in dep_lower:
            return self._install_cmake_windows()
        elif "git" in dep_lower:
            return self._install_git_windows()
        elif "msvc" in dep_lower or "build" in dep_lower:
            return self._install_msvc_windows()
        elif "vulkan" in dep_lower:
            return self._install_vulkan_windows()
        elif "rocm" in dep_lower:
            return self._install_rocm_windows()
        
        return False
        
    def _install_linux(self, dep: str) -> bool:
        """Install dependency on Linux using apt/dnf"""
        dep_lower = dep.lower()
        
        # Try to detect package manager
        has_apt = shutil.which("apt") is not None
        has_dnf = shutil.which("dnf") is not None
        
        if has_apt:
            return self._install_debian(dep)
        elif has_dnf:
            return self._install_fedora(dep)
        
        return False
        
    def _install_cmake_windows(self) -> bool:
        """Install CMake on Windows"""
        try:
            self.status_update.emit("Checking if CMake can be installed via winget...")
            
            if shutil.which("winget"):
                self.status_update.emit("Installing CMake via winget...")
                result = subprocess.run(
                    ["winget", "install", "-e", "--id", "Kitware.CMake", "-h"],
                    capture_output=True,
                    timeout=300
                )
                return result.returncode == 0
            else:
                self.status_update.emit("winget not found - will need manual installation")
                return False
        except Exception as e:
            self.status_update.emit(f"CMake installation failed: {str(e)}")
            return False
            
    def _install_git_windows(self) -> bool:
        """Install Git on Windows"""
        try:
            self.status_update.emit("Installing Git via winget...")
            
            if shutil.which("winget"):
                result = subprocess.run(
                    ["winget", "install", "-e", "--id", "Git.Git", "-h"],
                    capture_output=True,
                    timeout=300
                )
                return result.returncode == 0
            else:
                return False
        except Exception as e:
            self.status_update.emit(f"Git installation failed: {str(e)}")
            return False
            
    def _install_msvc_windows(self) -> bool:
        """Install Visual Studio Build Tools on Windows"""
        try:
            self.status_update.emit("Visual Studio Build Tools requires manual installation")
            self.status_update.emit("Downloading installer...")
            
            # Download Visual Studio Build Tools installer
            vs_url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
            vs_installer = Path.home() / "Downloads" / "vs_buildtools.exe"
            
            if not vs_installer.exists():
                urllib.request.urlretrieve(vs_url, vs_installer)
                self.status_update.emit(f"Downloaded: {vs_installer}")
            
            # Run installer with default C++ workload
            self.status_update.emit("Launching installer (please complete the installation)...")
            subprocess.Popen([str(vs_installer), "--includeRecommended"])
            
            return True
        except Exception as e:
            self.status_update.emit(f"MSVC installation setup failed: {str(e)}")
            return False
            
    def _install_vulkan_windows(self) -> bool:
        """Download Vulkan SDK installer for Windows"""
        try:
            self.status_update.emit("Downloading Vulkan SDK...")
            
            # Get latest Vulkan SDK version
            vulkan_url = "https://sdk.lunarg.com/sdk/download/latest/windows/vulkan-sdk.exe"
            vulkan_installer = Path.home() / "Downloads" / "VulkanSDK.exe"
            
            urllib.request.urlretrieve(vulkan_url, vulkan_installer)
            self.status_update.emit(f"Downloaded Vulkan SDK: {vulkan_installer}")
            
            # Launch installer
            self.status_update.emit("Launching Vulkan SDK installer...")
            subprocess.Popen([str(vulkan_installer)])
            
            return True
        except Exception as e:
            self.status_update.emit(f"Vulkan SDK download failed: {str(e)}")
            return False
            
    def _install_rocm_windows(self) -> bool:
        """Download ROCm installer for Windows"""
        try:
            self.status_update.emit("Downloading ROCm SDK...")
            
            # ROCm doesn't have a direct single-file URL, need to use package
            rocm_url = "https://rocmdocs.amd.com/en/latest/deploy/windows/quick_start.html"
            self.status_update.emit(f"Please download ROCm from: {rocm_url}")
            self.status_update.emit("Opening browser for ROCm download...")
            
            import webbrowser
            webbrowser.open(rocm_url)
            
            return True
        except Exception as e:
            self.status_update.emit(f"ROCm download failed: {str(e)}")
            return False
            
    def _install_debian(self, dep: str) -> bool:
        """Install package on Debian/Ubuntu"""
        try:
            dep_lower = dep.lower()
            package_map = {
                "cmake": "cmake",
                "git": "git",
                "gcc": "build-essential",
                "rocm": "rocm-hip-sdk",
                "vulkan": "vulkan-tools libvulkan-dev",
            }
            
            package = None
            for key, value in package_map.items():
                if key in dep_lower:
                    package = value
                    break
            
            if not package:
                self.status_update.emit(f"Unknown package: {dep}")
                return False
            
            self.status_update.emit(f"Installing {package} via apt...")
            
            # First update package list
            subprocess.run(["sudo", "apt", "update"], capture_output=True, timeout=60)
            
            # Install package
            result = subprocess.run(
                ["sudo", "apt", "install", "-y"] + package.split(),
                capture_output=True,
                timeout=300
            )
            
            return result.returncode == 0
        except Exception as e:
            self.status_update.emit(f"Installation failed: {str(e)}")
            return False
            
    def _install_fedora(self, dep: str) -> bool:
        """Install package on Fedora/RHEL"""
        try:
            dep_lower = dep.lower()
            package_map = {
                "cmake": "cmake",
                "git": "git",
                "gcc": "gcc gcc-c++ make",
                "rocm": "rocm-hip-devel",
                "vulkan": "vulkan-tools vulkan-devel",
            }
            
            package = None
            for key, value in package_map.items():
                if key in dep_lower:
                    package = value
                    break
            
            if not package:
                self.status_update.emit(f"Unknown package: {dep}")
                return False
            
            self.status_update.emit(f"Installing {package} via dnf...")
            
            result = subprocess.run(
                ["sudo", "dnf", "install", "-y"] + package.split(),
                capture_output=True,
                timeout=300
            )
            
            return result.returncode == 0
        except Exception as e:
            self.status_update.emit(f"Installation failed: {str(e)}")
            return False
            
    def stop(self):
        """Stop installation"""
        self.should_stop = True


class DependencyManager:
    """Manages dependency detection and installation"""
    
    def __init__(self):
        self.os_type = platform.system()
        
    def get_missing_dependencies(self, backend: Optional[str] = None) -> List[str]:
        """Get list of missing dependencies"""
        missing = []
        
        # Basic tools
        if not self._check_tool("cmake"):
            missing.append("CMake")
        if not self._check_tool("git"):
            missing.append("Git")
            
        # Compiler
        if self.os_type == "Windows":
            if not self._check_msvc():
                missing.append("MSVC Build Tools")
        else:
            if not self._check_tool("gcc") and not self._check_tool("clang"):
                missing.append("GCC/Clang")
        
        # Backend-specific
        if backend and backend.upper() == "CUDA":
            if not self._check_cuda():
                missing.append("CUDA Toolkit")
        elif backend and backend.upper() == "VULKAN":
            if not self._check_vulkan():
                missing.append("Vulkan SDK")
        elif backend and backend.upper() == "ROCM":
            if not self._check_rocm():
                missing.append("ROCm")
        
        return missing
        
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
        """Check for MSVC Build Tools"""
        # Method 1: Check if cl.exe is in PATH
        try:
            result = subprocess.run(
                ["where", "cl.exe"],
                capture_output=True,
                timeout=3,
                text=True
            )
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Method 2: Check common MSVC installation paths
        msvc_paths = [
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
            Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community"),
            Path("C:/Program Files/Microsoft Visual Studio/2019/BuildTools"),
            Path("C:/Program Files/Microsoft Visual Studio/2019/Community"),
        ]
        
        for path in msvc_paths:
            if path.exists():
                # Check for VC folder with compiler
                vc_path = path / "VC" / "Tools" / "MSVC"
                if vc_path.exists():
                    return True
        
        # Method 3: Check Windows Registry
        try:
            import winreg
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\VisualStudio\SxS\VS7"
                )
                # If key exists and has values, MSVC is installed
                value, _ = winreg.QueryValueEx(key, "15.0")
                if value:
                    return True
            except WindowsError:
                pass
            
            # Also check for Build Tools registry
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\VisualStudio\SxS\VS7"
                )
                value, _ = winreg.QueryValueEx(key, "16.0")
                if value:
                    return True
            except WindowsError:
                pass
        except ImportError:
            pass
        
        return False
    
    def _check_cuda(self) -> bool:
        """Check for CUDA Toolkit (nvcc compiler)"""
        # Method 1: Check nvcc in PATH
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Method 2: Check CUDA_PATH environment variable
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            nvcc_path = Path(cuda_path) / "bin" / "nvcc.exe"
            if nvcc_path.exists():
                return True
        
        # Method 3: Check standard CUDA installation paths
        cuda_paths = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path("C:/CUDA"),
        ]
        
        for base_path in cuda_paths:
            if base_path.exists():
                # Find latest version
                versions = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
                for ver_path in versions:
                    nvcc = ver_path / "bin" / "nvcc.exe"
                    if nvcc.exists():
                        return True
        
        return False
            
    def _check_vulkan(self) -> bool:
        """Check for Vulkan"""
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk:
            return Path(vulkan_sdk).exists()
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
    
    def initialize_msvc_env(self) -> bool:
        """Initialize MSVC environment variables and PATH"""
        try:
            # Find MSVC installation
            msvc_root = None
            msvc_paths = [
                Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
                Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community"),
                Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional"),
                Path("C:/Program Files/Microsoft Visual Studio/2022/BuildTools"),
                Path("C:/Program Files/Microsoft Visual Studio/2022/Community"),
                Path("C:/Program Files/Microsoft Visual Studio/2022/Professional"),
                Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools"),
                Path("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community"),
                Path("C:/Program Files/Microsoft Visual Studio/2019/BuildTools"),
                Path("C:/Program Files/Microsoft Visual Studio/2019/Community"),
            ]
            
            for path in msvc_paths:
                if path.exists():
                    msvc_root = path
                    break
            
            if not msvc_root:
                return False
            
            # Find the latest MSVC version folder
            vc_tools = msvc_root / "VC" / "Tools" / "MSVC"
            if vc_tools.exists():
                versions = sorted([d for d in vc_tools.iterdir() if d.is_dir()], reverse=True)
                if versions:
                    latest_msvc = versions[0]
                    
                    # Add cl.exe to PATH
                    bin_paths = [
                        latest_msvc / "bin" / "Hostx64" / "x64",
                        latest_msvc / "bin" / "Hostx86" / "x86",
                    ]
                    
                    current_path = os.environ.get("PATH", "")
                    for bin_path in bin_paths:
                        if bin_path.exists() and str(bin_path) not in current_path:
                            os.environ["PATH"] = f"{bin_path};{current_path}"
                            current_path = os.environ.get("PATH", "")
                    
                    # Also add Windows SDK path
                    kit_root = msvc_root / "Windows Kits"
                    if kit_root.exists():
                        kit_versions = sorted([d for d in kit_root.iterdir() if d.is_dir()], reverse=True)
                        if kit_versions:
                            kit_version = kit_versions[0]
                            kit_bin = kit_version / "bin"
                            if kit_bin.exists():
                                current_path = os.environ.get("PATH", "")
                                os.environ["PATH"] = f"{kit_bin};{current_path}"
                    
                    return True
            
            return False
        except Exception as e:
            print(f"Error initializing MSVC environment: {e}")
            return False
