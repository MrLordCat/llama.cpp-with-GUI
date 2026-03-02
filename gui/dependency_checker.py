"""
Dependency checker and installer for llama.cpp GUI
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


class DependencyChecker:
    """Check and install required Python packages"""
    
    # Required packages: (package_name, import_name, minimum_version_optional)
    REQUIRED_PACKAGES = [
        ("PyQt6", "PyQt6", None),
        ("PyQt6-Qt6", "PyQt6.QtCore", None),
        ("huggingface-hub", "huggingface_hub", None),
        ("psutil", "psutil", None),
        ("requests", "requests", None),
        ("tqdm", "tqdm", None),
    ]
    
    # Optional but recommended packages
    OPTIONAL_PACKAGES = [
        ("hf-xet", "xet", None),  # Better HuggingFace Hub performance (optional)
    ]
    
    # System tools required for building with specific backends
    # Format: (tool_name, check_command, install_hint, required_for)
    SYSTEM_TOOLS = {
        "cmake": {
            "check_cmd": ["cmake", "--version"],
            "install_hint_windows": "winget install Kitware.CMake",
            "install_hint_linux": "sudo apt install cmake  # or: sudo dnf install cmake",
            "description": "CMake build system",
            "required_for": ["all"],
        },
        "git": {
            "check_cmd": ["git", "--version"],
            "install_hint_windows": "winget install Git.Git",
            "install_hint_linux": "sudo apt install git  # or: sudo dnf install git",
            "description": "Git version control",
            "required_for": ["all"],
        },
        "ninja": {
            "check_cmd": ["ninja", "--version"],
            "install_hint_windows": "winget install Ninja-build.Ninja",
            "install_hint_linux": "sudo apt install ninja-build  # or: sudo dnf install ninja-build",
            "description": "Ninja build system (required for ROCm)",
            "required_for": ["rocm"],
        },
        "perl": {
            "check_cmd": ["perl", "--version"],
            "install_hint_windows": "winget install StrawberryPerl.StrawberryPerl",
            "install_hint_linux": "sudo apt install perl",
            "description": "Perl interpreter (required for ROCm/HIP on Windows)",
            "required_for": ["rocm"],
            "windows_only": True,
            "standard_paths": [
                "C:\\Strawberry\\perl\\bin\\perl.exe",
                "C:\\Perl64\\bin\\perl.exe",
                "C:\\Perl\\bin\\perl.exe",
            ],
        },
    }
    
    @staticmethod
    def check_package(package_name: str, import_name: str) -> bool:
        """Check if a package is installed"""
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False
    
    @staticmethod
    def check_system_tool(tool_name: str) -> bool:
        """Check if a system tool is installed"""
        tool_info = DependencyChecker.SYSTEM_TOOLS.get(tool_name)
        if not tool_info:
            return False
        
        # First try command in PATH
        try:
            result = subprocess.run(
                tool_info["check_cmd"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except:
            pass
        
        # Check standard installation paths
        from pathlib import Path
        standard_paths = tool_info.get("standard_paths", [])
        for path in standard_paths:
            if Path(path).exists():
                return True
        
        return False
    
    @staticmethod
    def get_tool_path(tool_name: str) -> Optional[str]:
        """Get the path to a system tool, checking standard locations"""
        tool_info = DependencyChecker.SYSTEM_TOOLS.get(tool_name)
        if not tool_info:
            return None
        
        # First try command in PATH
        try:
            import shutil
            cmd = tool_info["check_cmd"][0]
            path = shutil.which(cmd)
            if path:
                return path
        except:
            pass
        
        # Check standard installation paths
        from pathlib import Path
        standard_paths = tool_info.get("standard_paths", [])
        for path in standard_paths:
            if Path(path).exists():
                return path
        
        return None
    
    @classmethod
    def get_install_hint(cls, tool_name: str) -> str:
        """Get platform-appropriate install hint for a tool"""
        import platform
        tool_info = cls.SYSTEM_TOOLS.get(tool_name, {})
        if platform.system() == "Linux":
            return tool_info.get("install_hint_linux", tool_info.get("install_hint_windows", "N/A"))
        return tool_info.get("install_hint_windows", "N/A")
    
    @classmethod
    def get_missing_system_tools(cls, backend: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Get list of missing system tools for a backend
        
        Args:
            backend: Backend type (e.g., 'rocm', 'cuda') or None for common tools
            
        Returns:
            List of (tool_name, description, install_hint)
        """
        missing = []
        backend_lower = backend.lower() if backend else None
        import platform as _platform
        os_type = _platform.system()
        
        for tool_name, tool_info in cls.SYSTEM_TOOLS.items():
            required_for = tool_info["required_for"]
            
            # Skip Windows-only tools on non-Windows
            if tool_info.get("windows_only", False) and os_type != "Windows":
                continue
            
            # Check if this tool is needed
            needed = "all" in required_for or (backend_lower and backend_lower in required_for)
            
            if needed and not cls.check_system_tool(tool_name):
                missing.append((
                    tool_name,
                    tool_info["description"],
                    cls.get_install_hint(tool_name)
                ))
        
        return missing
    
    @classmethod
    def get_missing_packages(cls) -> List[str]:
        """Get list of missing required packages"""
        missing = []
        for pip_name, import_name, _ in cls.REQUIRED_PACKAGES:
            if not cls.check_package(pip_name, import_name):
                missing.append(pip_name)
        return missing
    
    @classmethod
    def get_missing_optional_packages(cls) -> List[Tuple[str, str]]:
        """Get list of missing optional packages (name, description)"""
        missing = []
        for pip_name, import_name, description in cls.OPTIONAL_PACKAGES:
            if not cls.check_package(pip_name, import_name):
                missing.append((pip_name, description))
        return missing
    
    @staticmethod
    def _is_externally_managed() -> bool:
        """Check if the Python environment is externally managed (PEP 668)"""
        import sysconfig
        marker = Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED"
        return marker.exists()

    @staticmethod
    def install_packages(packages: List[str], quiet: bool = False) -> bool:
        """Install packages using pip
        
        Args:
            packages: List of package names to install
            quiet: If True, suppress output
            
        Returns:
            True if installation successful, False otherwise
        """
        if not packages:
            return True
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            if quiet:
                cmd.append("-q")
            # Handle PEP 668 externally-managed environments (Ubuntu 23+, Fedora 38+, etc.)
            if DependencyChecker._is_externally_managed():
                cmd.append("--break-system-packages")
            cmd.extend(packages)
            
            result = subprocess.run(cmd, capture_output=quiet)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error installing packages: {e}")
            return False
    
    @classmethod
    def auto_install_missing(cls) -> bool:
        """Automatically install missing required packages
        
        Returns:
            True if all packages are now available, False if installation failed
        """
        missing = cls.get_missing_packages()
        
        if not missing:
            return True
        
        print("\n[INSTALL] Installing missing dependencies...")
        print(f"Packages to install: {', '.join(missing)}\n")
        
        if cls.install_packages(missing):
            # Refresh the import cache so newly installed packages are visible
            import importlib
            import site
            importlib.invalidate_caches()
            # Re-add user site-packages to sys.path (pip --user installs go there)
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.insert(0, user_site)
            print("[OK] All required packages installed successfully!\n")
            return True
        else:
            print("[ERROR] Failed to install some packages. Please install manually:\n")
            print(f"  pip install {' '.join(missing)}\n")
            return False
    
    @classmethod
    def check_and_recommend_optional(cls) -> None:
        """Check optional packages and recommend installation"""
        missing_optional = cls.get_missing_optional_packages()
        
        if missing_optional:
            print("\n[TIP] Optional packages for better performance:\n")
            for pkg_name, description in missing_optional:
                print(f"  * {pkg_name}: {description}")
                print(f"    Install with: pip install {pkg_name}\n")
    
    @staticmethod
    def install_system_tool(tool_name: str) -> bool:
        """Try to install a system tool using package manager
        
        Args:
            tool_name: Name of the tool to install
            
        Returns:
            True if installation succeeded, False otherwise
        """
        import platform
        tool_info = DependencyChecker.SYSTEM_TOOLS.get(tool_name)
        if not tool_info:
            return False
        
        os_type = platform.system()
        
        if os_type == "Windows":
            install_hint = tool_info.get("install_hint_windows", "")
            if not install_hint.startswith("winget"):
                print(f"[INFO] Cannot auto-install {tool_name}. Please run: {install_hint}")
                return False
            try:
                cmd_parts = install_hint.split()
                print(f"[INSTALL] Running: {install_hint}")
                result = subprocess.run(cmd_parts, timeout=300)
                return result.returncode == 0
            except Exception as e:
                print(f"[ERROR] Failed to install {tool_name}: {e}")
                return False
        
        elif os_type == "Linux":
            import shutil
            # Try apt first, then dnf
            package_map = {
                "cmake": {"apt": "cmake", "dnf": "cmake"},
                "git": {"apt": "git", "dnf": "git"},
                "ninja": {"apt": "ninja-build", "dnf": "ninja-build"},
                "perl": {"apt": "perl", "dnf": "perl"},
            }
            packages = package_map.get(tool_name, {})
            
            if shutil.which("apt"):
                pkg = packages.get("apt", tool_name)
                try:
                    print(f"[INSTALL] Running: sudo apt install -y {pkg}")
                    result = subprocess.run(["sudo", "apt", "install", "-y", pkg], timeout=300)
                    return result.returncode == 0
                except Exception as e:
                    print(f"[ERROR] Failed to install {tool_name}: {e}")
                    return False
            elif shutil.which("dnf"):
                pkg = packages.get("dnf", tool_name)
                try:
                    print(f"[INSTALL] Running: sudo dnf install -y {pkg}")
                    result = subprocess.run(["sudo", "dnf", "install", "-y", pkg], timeout=300)
                    return result.returncode == 0
                except Exception as e:
                    print(f"[ERROR] Failed to install {tool_name}: {e}")
                    return False
        
        print(f"[INFO] Cannot auto-install {tool_name} on this platform")
        return False
    
    @classmethod
    def check_and_install_system_tools(cls, backend: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Check and optionally install missing system tools
        
        Args:
            backend: Backend type for which to check tools
            
        Returns:
            Tuple of (all_available, list_of_missing_tools)
        """
        missing = cls.get_missing_system_tools(backend)
        
        if not missing:
            return True, []
        
        missing_names = [m[0] for m in missing]
        return False, missing_names


def is_frozen() -> bool:
    """Check if running as a frozen/packaged executable"""
    import sys
    return getattr(sys, 'frozen', False) or hasattr(sys, '_MEIPASS')


def init_dependencies() -> bool:
    """Initialize and check all dependencies at startup
    
    Returns:
        True if all dependencies are met, False otherwise
    """
    # Skip dependency check if running as frozen exe
    # All dependencies should be bundled in the exe
    if is_frozen():
        return True
    
    print("=" * 60)
    print("[CHECK] Checking dependencies...")
    print("=" * 60 + "\n")
    
    # Check required packages
    missing = DependencyChecker.get_missing_packages()
    
    if missing:
        print(f"[MISSING] Missing required packages: {', '.join(missing)}\n")
        
        # Try to auto-install
        if not DependencyChecker.auto_install_missing():
            print("[WARNING] Some dependencies are missing!")
            print("Please install them manually and try again.\n")
            return False
    else:
        print("[OK] All required packages are installed\n")
    
    # Check optional packages and recommend (if any exist)
    missing_optional = DependencyChecker.get_missing_optional_packages()
    if missing_optional:
        DependencyChecker.check_and_recommend_optional()
    
    print("=" * 60 + "\n")
    return True
