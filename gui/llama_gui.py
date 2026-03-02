#!/usr/bin/env python3
"""
LLaMA.cpp GUI - Graphical interface for llama.cpp
Allows configuring parameters, downloading models and running inference
"""

import sys
import os
import json
import subprocess
import platform
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QProgressBar, QTabWidget, QGroupBox,
    QCheckBox, QMessageBox, QListWidget, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QUrl
from PyQt6.QtGui import QFont, QTextCursor, QDesktopServices
import webbrowser

from model_downloader import ModelDownloader, DownloadThread, ListFilesThread
from hardware_detector import HardwareDetector
from build_manager import BuildManager, BuildThread, ConfigureThread
from dependency_installer import DependencyInstallThread, DependencyManager
from dependency_checker import init_dependencies


class ServerThread(QThread):
    """Thread for running llama-server in background"""
    output_ready = pyqtSignal(str)
    server_ready = pyqtSignal(str)  # Server URL
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, command: list, working_dir: str, port: int = 8080, env: dict = None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        self.port = port
        self.env = env
        
    def run(self):
        try:
            # Build environment
            process_env = os.environ.copy()
            if self.env:
                process_env.update(self.env)
            
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
                universal_newlines=True,
                env=process_env
            )
            
            server_started = False
            for line in self.process.stdout:
                self.output_ready.emit(line)
                
                # Detect when server is ready
                if not server_started and ("HTTP server listening" in line or "server is listening" in line):
                    self.server_ready.emit(f"http://localhost:{self.port}")
                    server_started = True
                
            self.process.wait()
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


class InferenceThread(QThread):
    """Thread for running inference in background"""
    output_ready = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, command: list, working_dir: str):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None
        
    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in self.process.stdout:
                self.output_ready.emit(line)
                
            self.process.wait()
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))
            
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


class LlamaCppGUI(QMainWindow):
    """Main window for llama.cpp GUI"""
    
    # Build directory names for different backends
    BUILD_DIRS = {
        "CPU": "build-cpu",
        "CUDA": "build-cuda",
        "Metal": "build-metal",
        "Vulkan": "build-vulkan",
        "SYCL": "build-sycl",
        "ROCm": "build-rocm",
    }
    
    def __init__(self, project_root: Optional[Path] = None):
        super().__init__()
        self.settings = QSettings("LlamaCpp", "GUI")
        
        # Find or set project root
        if project_root:
            self.project_root = project_root
        else:
            self.project_root = self._find_or_select_project_root()
        
        if not self.project_root:
            # User cancelled, exit
            sys.exit(0)
        
        # Save the project root for next time
        self.settings.setValue("project_root", str(self.project_root))
        
        self.build_dir = self.project_root / "build"  # Default, will be updated based on backend
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.server_thread: Optional[ServerThread] = None
        self.server_url: Optional[str] = None
        self.inference_thread: Optional[InferenceThread] = None
        self.download_thread: Optional[DownloadThread] = None
        self.list_files_thread: Optional[ListFilesThread] = None
        self.build_thread: Optional[BuildThread] = None
        self.install_thread: Optional[DependencyInstallThread] = None
        self.hardware_detector = HardwareDetector()
        self.build_manager = BuildManager(self.project_root)
        self.model_downloader = ModelDownloader(self.models_dir)
        self.dependency_manager = DependencyManager()
        self.os_type = platform.system()
        
        # Initialize MSVC environment if on Windows
        if self.os_type == "Windows":
            self.dependency_manager.initialize_msvc_env()
        
        self.init_ui()
        self.detect_hardware()
        self.load_settings()
        self.check_project_build()
    
    def _find_or_select_project_root(self) -> Optional[Path]:
        """Find llama.cpp repository or ask user to select it"""
        
        # 1. Check saved path first
        saved_path = self.settings.value("project_root", "")
        if saved_path:
            saved_path = Path(saved_path)
            if self._is_valid_llama_cpp_repo(saved_path):
                return saved_path
        
        # 2. Check if running from within the repo
        current_file = Path(__file__).resolve()
        possible_roots = [
            current_file.parent.parent,  # gui/ -> project root
            current_file.parent,  # if in root
            Path.cwd(),  # current working directory
            Path.cwd().parent,
        ]
        
        for root in possible_roots:
            if self._is_valid_llama_cpp_repo(root):
                return root
        
        # 3. Search common locations
        search_paths = self._get_common_repo_locations()
        for path in search_paths:
            if path.exists() and self._is_valid_llama_cpp_repo(path):
                return path
        
        # 4. Ask user to select the folder
        return self._ask_user_for_repo_path()
    
    def _is_valid_llama_cpp_repo(self, path: Path) -> bool:
        """Check if path is a valid llama.cpp repository"""
        if not path.exists() or not path.is_dir():
            return False
        
        # Check for key files that should exist in llama.cpp
        required_files = [
            "CMakeLists.txt",
            "include/llama.h",
        ]
        
        # At least one of these should exist
        optional_indicators = [
            "src/llama.cpp",
            "ggml",
            "examples",
            "AGENTS.md",  # Our custom file
        ]
        
        # Check required files
        for req_file in required_files:
            if not (path / req_file).exists():
                return False
        
        # Check at least one optional indicator
        for opt_file in optional_indicators:
            if (path / opt_file).exists():
                return True
        
        return False
    
    def _get_common_repo_locations(self) -> List[Path]:
        """Get common locations where llama.cpp might be cloned"""
        locations = []
        
        # User's home directory
        home = Path.home()
        
        # Common development folders
        common_folders = [
            home / "Documents" / "GitHub" / "llama.cpp",
            home / "Documents" / "GitHub" / "llama.cpp-with-GUI",
            home / "source" / "repos" / "llama.cpp",
            home / "Projects" / "llama.cpp",
            home / "dev" / "llama.cpp",
            home / "code" / "llama.cpp",
            home / "git" / "llama.cpp",
            home / "llama.cpp",
            Path("C:/llama.cpp"),
            Path("D:/llama.cpp"),
            Path("C:/GitHub/llama.cpp"),
            Path("D:/GitHub/llama.cpp"),
        ]
        
        # Add variations with -with-GUI suffix
        for folder in common_folders.copy():
            if "llama.cpp" in str(folder) and "-with-GUI" not in str(folder):
                locations.append(Path(str(folder) + "-with-GUI"))
        
        locations.extend(common_folders)
        return locations
    
    def _ask_user_for_repo_path(self) -> Optional[Path]:
        """Show dialog to ask user to select llama.cpp folder"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("llama.cpp Repository Not Found")
        msg.setText(
            "Could not find llama.cpp repository automatically.\n\n"
            "Please select the folder where you cloned llama.cpp.\n\n"
            "The folder should contain:\n"
            "• CMakeLists.txt\n"
            "• include/llama.h\n"
            "• ggml/ folder"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        
        result = msg.exec()
        
        if result == QMessageBox.StandardButton.Cancel:
            return None
        
        # Open folder selection dialog
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select llama.cpp Repository Folder",
            str(Path.home() / "Documents"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return None
        
        folder_path = Path(folder)
        
        if not self._is_valid_llama_cpp_repo(folder_path):
            QMessageBox.warning(
                None,
                "Invalid Repository",
                f"The selected folder does not appear to be a valid llama.cpp repository.\n\n"
                f"Selected: {folder_path}\n\n"
                "Please clone llama.cpp first:\n"
                "git clone https://github.com/ggml-org/llama.cpp.git"
            )
            # Try again
            return self._ask_user_for_repo_path()
        
        return folder_path
    
    def get_build_dir_for_backend(self, backend: Optional[str]) -> Path:
        """Get the build directory for a specific backend"""
        if backend is None:
            backend = "CPU"
        
        # First check if backend-specific directory exists
        backend_dir_name = self.BUILD_DIRS.get(backend, f"build-{backend.lower()}")
        backend_dir = self.project_root / backend_dir_name
        
        if backend_dir.exists():
            return backend_dir
        
        # Fallback to generic 'build' directory
        generic_build = self.project_root / "build"
        if generic_build.exists():
            return generic_build
        
        # Return the backend-specific path even if it doesn't exist yet
        return backend_dir
    
    def get_available_builds(self) -> dict:
        """Get all available builds with their backends"""
        builds = {}
        
        # Check generic build directory
        generic_build = self.project_root / "build"
        if generic_build.exists() and (generic_build / "CMakeCache.txt").exists():
            backend = self._detect_build_backend(generic_build)
            builds["build"] = {
                "path": generic_build,
                "backend": backend,
                "display_name": f"build ({backend})"
            }
        
        # Check backend-specific directories from BUILD_DIRS
        for backend_name, dir_name in self.BUILD_DIRS.items():
            dir_path = self.project_root / dir_name
            if dir_path.exists() and (dir_path / "CMakeCache.txt").exists():
                detected_backend = self._detect_build_backend(dir_path)
                builds[dir_name] = {
                    "path": dir_path,
                    "backend": detected_backend or backend_name,
                    "display_name": f"{dir_name} ({detected_backend or backend_name})"
                }
        
        # Also scan for any other build-* directories
        try:
            for item in self.project_root.iterdir():
                if item.is_dir() and item.name.startswith("build") and item.name not in builds:
                    if (item / "CMakeCache.txt").exists():
                        detected_backend = self._detect_build_backend(item)
                        builds[item.name] = {
                            "path": item,
                            "backend": detected_backend,
                            "display_name": f"{item.name} ({detected_backend})"
                        }
        except Exception:
            pass
        
        return builds
    
    def _detect_build_backend(self, build_path: Path) -> str:
        """Detect backend from CMakeCache.txt"""
        cmake_cache = build_path / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                content = cmake_cache.read_text(errors='ignore')
                if "GGML_CUDA:BOOL=ON" in content:
                    return "CUDA"
                elif "GGML_HIP:BOOL=ON" in content or "GGML_ROCM:BOOL=ON" in content:
                    return "ROCm"
                elif "GGML_METAL:BOOL=ON" in content:
                    return "Metal"
                elif "GGML_VULKAN:BOOL=ON" in content:
                    return "Vulkan"
                elif "GGML_SYCL:BOOL=ON" in content:
                    return "SYCL"
            except:
                pass
        return "CPU"
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("LLaMA.cpp GUI - Model Management")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Tab: Server Launch
        tabs.addTab(self.create_server_tab(), "🚀 Launch Server")
        
        # Tab: Inference (CLI mode)
        tabs.addTab(self.create_inference_tab(), "⚡ Inference")
        
        # Tab: Model Downloads
        tabs.addTab(self.create_download_tab(), "📥 Download Models")
        
        # Tab: Build and Setup
        tabs.addTab(self.create_build_tab(), "🔧 Build & Setup")
        
        # Tab: Installed Builds Info
        tabs.addTab(self.create_builds_info_tab(), "📋 Installed Builds")
        
        # Tab: Hardware Info
        tabs.addTab(self.create_hardware_tab(), "💻 System Info")
        
        # Now refresh models list after all widgets are created
        self.refresh_models_list()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_server_tab(self) -> QWidget:
        """Create tab for launching server"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        model_select_layout = QHBoxLayout()
        self.server_model_path_edit = QLineEdit()
        self.server_model_path_edit.setPlaceholderText("Path to .gguf model")
        model_select_layout.addWidget(QLabel("Model:"))
        model_select_layout.addWidget(self.server_model_path_edit)
        
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self.browse_server_model)
        model_select_layout.addWidget(browse_btn)
        
        model_layout.addLayout(model_select_layout)
        
        # Available models list (will be populated later)
        self.server_models_list = QComboBox()
        self.server_models_list.currentTextChanged.connect(self.on_server_model_selected)
        model_layout.addWidget(QLabel("Quick select:"))
        model_layout.addWidget(self.server_models_list)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Server mode
        mode_group = QGroupBox("Server Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        
        self.mode_web_radio = QRadioButton("🌐 Web Interface (for browser chat)")
        self.mode_web_radio.setChecked(True)
        self.mode_button_group.addButton(self.mode_web_radio)
        mode_layout.addWidget(self.mode_web_radio)
        
        self.mode_api_radio = QRadioButton("🔌 API Mode (for VSCode/agents)")
        self.mode_button_group.addButton(self.mode_api_radio)
        mode_layout.addWidget(self.mode_api_radio)
        
        mode_info = QLabel("💡 In web mode, browser will open for chat.\nIn API mode, server will be available for integration with VSCode and other tools.")
        mode_info.setWordWrap(True)
        mode_info.setStyleSheet("color: #666; font-style: italic;")
        mode_layout.addWidget(mode_info)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Server Parameters
        server_params_group = QGroupBox("Server Parameters")
        server_params_layout = QVBoxLayout()
        
        # Port and parameters in two columns
        params_grid = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.server_port_spin = QSpinBox()
        self.server_port_spin.setRange(1024, 65535)
        self.server_port_spin.setValue(8080)
        port_layout.addWidget(self.server_port_spin)
        col1.addLayout(port_layout)
        
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Size:"))
        self.server_ctx_slider = QSlider(Qt.Orientation.Horizontal)
        self.server_ctx_slider.setRange(1, 32)  # 1-32 steps (8192 * step)
        self.server_ctx_slider.setValue(1)  # 8192
        self.server_ctx_slider.setSingleStep(1)
        self.server_ctx_slider.setPageStep(1)
        self.server_ctx_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.server_ctx_slider.setTickInterval(4)  # Every 32K
        ctx_layout.addWidget(self.server_ctx_slider)
        self.server_ctx_label = QLabel("8192")
        self.server_ctx_label.setMinimumWidth(60)
        ctx_layout.addWidget(self.server_ctx_label)
        self.server_ctx_slider.valueChanged.connect(lambda v: self.server_ctx_label.setText(str(v * 8192)))
        col1.addLayout(ctx_layout)
        
        params_grid.addLayout(col1)
        
        # Column 2
        col2 = QVBoxLayout()
        
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        self.server_threads_spin = QSpinBox()
        self.server_threads_spin.setRange(1, 64)
        self.server_threads_spin.setValue(os.cpu_count() or 4)
        threads_layout.addWidget(self.server_threads_spin)
        col2.addLayout(threads_layout)
        
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.server_batch_slider = QSlider(Qt.Orientation.Horizontal)
        self.server_batch_slider.setRange(1, 256)  # 1-256 steps (32 * step)
        self.server_batch_slider.setValue(16)  # 512
        self.server_batch_slider.setSingleStep(1)
        self.server_batch_slider.setPageStep(4)
        self.server_batch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.server_batch_slider.setTickInterval(32)  # Every 1024
        batch_layout.addWidget(self.server_batch_slider)
        self.server_batch_label = QLabel("512")
        self.server_batch_label.setMinimumWidth(50)
        batch_layout.addWidget(self.server_batch_label)
        self.server_batch_slider.valueChanged.connect(lambda v: self.server_batch_label.setText(str(v * 32)))
        col2.addLayout(batch_layout)
        
        params_grid.addLayout(col2)
        
        server_params_layout.addLayout(params_grid)
        
        # GPU parameters
        self.server_gpu_checkbox = QCheckBox("Use GPU")
        self.server_gpu_checkbox.setChecked(True)
        server_params_layout.addWidget(self.server_gpu_checkbox)
        
        # GPU Layers with VRAM estimation
        gpu_layers_group = QVBoxLayout()
        
        # VRAM estimation labels
        vram_hints_layout = QHBoxLayout()
        vram_hints_layout.addWidget(QLabel("VRAM estimate (7B/13B/34B/70B):"))
        vram_hints_layout.addStretch()
        gpu_layers_group.addLayout(vram_hints_layout)
        
        vram_scale_layout = QHBoxLayout()
        vram_scale_layout.addWidget(QLabel("30"))
        vram_scale_layout.addWidget(QLabel("~4/6/10/18GB"))
        vram_scale_layout.addStretch()
        vram_scale_layout.addWidget(QLabel("~5/8/14/24GB"))
        vram_scale_layout.addStretch()
        vram_scale_layout.addWidget(QLabel("~6/10/18/32GB"))
        vram_scale_layout.addStretch()
        vram_scale_layout.addWidget(QLabel("~8/12/24/48GB"))
        vram_scale_layout.addWidget(QLabel("100"))
        gpu_layers_group.addLayout(vram_scale_layout)
        
        gpu_layers_layout = QHBoxLayout()
        gpu_layers_layout.addWidget(QLabel("GPU Layers:"))
        self.server_gpu_layers_slider = QSlider(Qt.Orientation.Horizontal)
        self.server_gpu_layers_slider.setRange(0, 100)
        self.server_gpu_layers_slider.setValue(33)
        self.server_gpu_layers_slider.setSingleStep(1)
        self.server_gpu_layers_slider.setPageStep(10)
        self.server_gpu_layers_slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.server_gpu_layers_slider.setTickInterval(10)
        gpu_layers_layout.addWidget(self.server_gpu_layers_slider)
        self.server_gpu_layers_label = QLabel("33")
        self.server_gpu_layers_label.setMinimumWidth(30)
        gpu_layers_layout.addWidget(self.server_gpu_layers_label)
        self.server_gpu_layers_slider.valueChanged.connect(lambda v: self.server_gpu_layers_label.setText(str(v)))
        gpu_layers_group.addLayout(gpu_layers_layout)
        
        server_params_layout.addLayout(gpu_layers_group)
        
        # CORS and API key
        self.server_cors_checkbox = QCheckBox("Enable CORS (for web access)")
        self.server_cors_checkbox.setChecked(True)
        server_params_layout.addWidget(self.server_cors_checkbox)
        
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key (optional):"))
        self.server_api_key_edit = QLineEdit()
        self.server_api_key_edit.setPlaceholderText("Leave empty for open access")
        api_key_layout.addWidget(self.server_api_key_edit)
        server_params_layout.addLayout(api_key_layout)
        
        # Backend selection for server
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("Inference Backend:"))
        self.server_backend_combo = QComboBox()
        self.server_backend_combo.addItems([
            "CPU only (default)",
            "CUDA",
            "Metal",
            "Vulkan",
            "ROCm"
        ])
        backend_layout.addWidget(self.server_backend_combo)
        backend_layout.addStretch()
        server_params_layout.addLayout(backend_layout)
        
        server_params_group.setLayout(server_params_layout)
        layout.addWidget(server_params_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.server_start_btn = QPushButton("▶️ Start Server")
        self.server_start_btn.clicked.connect(self.start_server)
        self.server_start_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; background-color: #4CAF50; color: white; }")
        buttons_layout.addWidget(self.server_start_btn)
        
        self.server_stop_btn = QPushButton("⏹️ Stop Server")
        self.server_stop_btn.clicked.connect(self.stop_server)
        self.server_stop_btn.setEnabled(False)
        self.server_stop_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.server_stop_btn)
        
        self.server_open_web_btn = QPushButton("🌐 Open Web Interface")
        self.server_open_web_btn.clicked.connect(self.open_web_interface)
        self.server_open_web_btn.setEnabled(False)
        self.server_open_web_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.server_open_web_btn)
        
        self.server_clear_btn = QPushButton("🗑️ Clear Log")
        self.server_clear_btn.clicked.connect(lambda: self.server_output_text.clear())
        buttons_layout.addWidget(self.server_clear_btn)
        
        layout.addLayout(buttons_layout)
        
        # Server status
        self.server_status_label = QLabel("⚪ Server stopped")
        self.server_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        layout.addWidget(self.server_status_label)
        
        # Server output
        layout.addWidget(QLabel("Server Log:"))
        self.server_output_text = QTextEdit()
        self.server_output_text.setReadOnly(True)
        self.server_output_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.server_output_text)
        
        return widget
        
    def create_inference_tab(self) -> QWidget:
        """Create tab for running inference"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("Выбор модели")
        model_layout = QVBoxLayout()
        
        model_select_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Путь к модели .gguf")
        model_select_layout.addWidget(QLabel("Model:"))
        model_select_layout.addWidget(self.model_path_edit)
        
        browse_btn = QPushButton("📁 Обзор")
        browse_btn.clicked.connect(self.browse_model)
        model_select_layout.addWidget(browse_btn)
        
        model_layout.addLayout(model_select_layout)
        
        # Список доступных models
        self.models_list = QComboBox()
        self.refresh_models_list()
        self.models_list.currentTextChanged.connect(self.on_model_selected)
        model_layout.addWidget(QLabel("Быстрый выбор:"))
        model_layout.addWidget(self.models_list)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Inference parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Prompt
        params_layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        self.prompt_edit.setMaximumHeight(100)
        params_layout.addWidget(self.prompt_edit)
        
        # Parameters in multiple columns
        params_grid = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        
        n_predict_layout = QHBoxLayout()
        n_predict_layout.addWidget(QLabel("Tokens:"))
        self.n_predict_spin = QSpinBox()
        self.n_predict_spin.setRange(-1, 8192)
        self.n_predict_spin.setValue(128)
        n_predict_layout.addWidget(self.n_predict_spin)
        col1.addLayout(n_predict_layout)
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.8)
        temp_layout.addWidget(self.temp_spin)
        col1.addLayout(temp_layout)
        
        params_grid.addLayout(col1)
        
        # Column 2
        col2 = QVBoxLayout()
        
        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(QLabel("Top-P:"))
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        top_p_layout.addWidget(self.top_p_spin)
        col2.addLayout(top_p_layout)
        
        top_k_layout = QHBoxLayout()
        top_k_layout.addWidget(QLabel("Top-K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 200)
        self.top_k_spin.setValue(40)
        top_k_layout.addWidget(self.top_k_spin)
        col2.addLayout(top_k_layout)
        
        params_grid.addLayout(col2)
        
        # Column 3
        col3 = QVBoxLayout()
        
        ctx_size_layout = QHBoxLayout()
        ctx_size_layout.addWidget(QLabel("Context Size:"))
        self.ctx_size_spin = QSpinBox()
        self.ctx_size_spin.setRange(128, 262144)  # Up to 256K
        self.ctx_size_spin.setValue(2048)
        self.ctx_size_spin.setSingleStep(1024)  # Step by 1K
        ctx_size_layout.addWidget(self.ctx_size_spin)
        col3.addLayout(ctx_size_layout)
        
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(os.cpu_count() or 4)
        threads_layout.addWidget(self.threads_spin)
        col3.addLayout(threads_layout)
        
        params_grid.addLayout(col3)
        
        params_layout.addLayout(params_grid)
        
        # Additional Options
        self.gpu_layers_checkbox = QCheckBox("Use GPU")
        self.gpu_layers_checkbox.setChecked(True)
        params_layout.addWidget(self.gpu_layers_checkbox)
        
        gpu_layers_layout = QHBoxLayout()
        gpu_layers_layout.addWidget(QLabel("GPU Layers:"))
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 100)
        self.gpu_layers_spin.setValue(33)
        gpu_layers_layout.addWidget(self.gpu_layers_spin)
        gpu_layers_layout.addStretch()
        params_layout.addLayout(gpu_layers_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶️ Run")
        self.run_btn.clicked.connect(self.run_inference)
        self.run_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.stop_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear Output")
        self.clear_btn.clicked.connect(lambda: self.output_text.clear())
        buttons_layout.addWidget(self.clear_btn)
        
        layout.addLayout(buttons_layout)
        
        # Вывод
        layout.addWidget(QLabel("Model Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.output_text)
        
        return widget
        
    def create_download_tab(self) -> QWidget:
        """Создание вкладки для загрузки models"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("📥 Search and Download Models from HuggingFace")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        # Search models
        search_group = QGroupBox("Search Models on HuggingFace")
        search_layout = QVBoxLayout()
        
        # Search field and buttons
        search_controls_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter query (e.g.: 'llama', 'mistral', 'codellama')...")
        self.search_input.returnPressed.connect(self.search_hf_models)
        search_controls_layout.addWidget(self.search_input)
        
        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.clicked.connect(self.search_hf_models)
        search_controls_layout.addWidget(self.search_btn)
        
        self.load_popular_btn = QPushButton("⭐ Popular")
        self.load_popular_btn.clicked.connect(self.load_popular_models)
        search_controls_layout.addWidget(self.load_popular_btn)
        
        search_layout.addLayout(search_controls_layout)
        
        # Date filter
        date_filter_layout = QHBoxLayout()
        date_filter_layout.addWidget(QLabel("Min Date:"))
        
        self.filter_year_combo = QComboBox()
        self.filter_year_combo.addItems(["All", "2025", "2024", "2023", "2022"])
        date_filter_layout.addWidget(self.filter_year_combo)
        
        self.filter_month_combo = QComboBox()
        self.filter_month_combo.addItems([
            "All months",
            "January (01)", "February (02)", "March (03)", "April (04)",
            "May (05)", "June (06)", "July (07)", "August (08)",
            "September (09)", "October (10)", "November (11)", "December (12)"
        ])
        date_filter_layout.addWidget(self.filter_month_combo)
        
        self.apply_date_filter_btn = QPushButton("🔄 Apply")
        self.apply_date_filter_btn.clicked.connect(self.apply_date_filter)
        date_filter_layout.addWidget(self.apply_date_filter_btn)
        date_filter_layout.addStretch()
        
        search_layout.addLayout(date_filter_layout)
        
        # Sorting
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort:"))
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["By Downloads ⬇️", "By Likes ❤️", "By Date Updated 📅", "By Name 🔤"])
        self.sort_combo.currentIndexChanged.connect(self.on_sort_changed)
        sort_layout.addWidget(self.sort_combo)
        sort_layout.addStretch()
        
        search_layout.addLayout(sort_layout)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Results table
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout()
        
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels([
            "Model Name", "Author", "Downloads", "Likes", "Updated"
        ])
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.models_table.itemDoubleClicked.connect(self.on_model_double_clicked)
        results_layout.addWidget(self.models_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Selected Model
        selected_group = QGroupBox("Selected Model")
        selected_layout = QVBoxLayout()
        
        self.selected_model_label = QLabel("Not Selected")
        self.selected_model_label.setStyleSheet("font-weight: bold;")
        selected_layout.addWidget(self.selected_model_label)
        
        # File selection from repository
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.model_file_combo = QComboBox()
        self.model_file_combo.setEnabled(False)
        file_layout.addWidget(self.model_file_combo)
        selected_layout.addLayout(file_layout)
        
        selected_group.setLayout(selected_layout)
        layout.addWidget(selected_group)
        
        # Download Progress
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()
        
        self.download_progress = QProgressBar()
        progress_layout.addWidget(self.download_progress)
        
        self.download_status_label = QLabel("Ready to Download")
        progress_layout.addWidget(self.download_status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("📥 Download Model")
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.download_btn)
        
        self.cancel_download_btn = QPushButton("❌ Cancel")
        self.cancel_download_btn.setEnabled(False)
        self.cancel_download_btn.clicked.connect(self.cancel_download)
        buttons_layout.addWidget(self.cancel_download_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Initialization: загружаем популярные модели
        QThread.msleep(100)  # Небольшая задержка
        self.load_popular_models()
        
        return widget
        
    def create_build_tab(self) -> QWidget:
        """Create tab for building project"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("🔧 Build llama.cpp with Selected Hardware Support")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        # Repository Path Section
        repo_group = QGroupBox("Repository Location")
        repo_layout = QHBoxLayout()
        
        repo_layout.addWidget(QLabel("Path:"))
        self.repo_path_label = QLabel(str(self.project_root))
        self.repo_path_label.setStyleSheet("color: #0066cc; font-family: Consolas;")
        self.repo_path_label.setWordWrap(True)
        repo_layout.addWidget(self.repo_path_label, 1)
        
        change_repo_btn = QPushButton("📁 Change...")
        change_repo_btn.clicked.connect(self.change_repository_path)
        change_repo_btn.setMaximumWidth(120)
        repo_layout.addWidget(change_repo_btn)
        
        repo_group.setLayout(repo_layout)
        layout.addWidget(repo_group)
        
        # Backend Selection
        backend_group = QGroupBox("Backend Selection")
        backend_layout = QVBoxLayout()
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "CPU (only processor)",
            "CUDA (NVIDIA GPU)",
            "Metal (macOS GPU)",
            "Vulkan (AMD/Intel/NVIDIA)",
            "SYCL (Intel GPU)",
            "ROCm (AMD GPU)",
        ])
        backend_layout.addWidget(self.backend_combo)
        
        backend_group.setLayout(backend_layout)
        layout.addWidget(backend_group)
        
        # Build options
        options_group = QGroupBox("Additional Options")
        options_layout = QVBoxLayout()
        
        self.build_server_checkbox = QCheckBox("Build llama-server")
        self.build_server_checkbox.setChecked(True)
        options_layout.addWidget(self.build_server_checkbox)
        
        self.build_tests_checkbox = QCheckBox("Build tests")
        options_layout.addWidget(self.build_tests_checkbox)
        
        self.use_ccache_checkbox = QCheckBox("Use ccache (faster build)")
        self.use_ccache_checkbox.setChecked(True)
        options_layout.addWidget(self.use_ccache_checkbox)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.configure_btn = QPushButton("⚙️ Configure")
        self.configure_btn.clicked.connect(self.configure_build)
        self.configure_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.configure_btn)
        
        self.build_btn = QPushButton("🔨 Build")
        self.build_btn.clicked.connect(self.build_project)
        self.build_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.build_btn)
        
        self.install_deps_btn = QPushButton("📦 Install Dependencies")
        self.install_deps_btn.clicked.connect(self.install_dependencies)
        buttons_layout.addWidget(self.install_deps_btn)
        
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Build progress section
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.build_progress_bar = QProgressBar()
        self.build_progress_bar.setMinimum(0)
        self.build_progress_bar.setMaximum(100)
        self.build_progress_bar.setValue(0)
        self.build_progress_bar.setTextVisible(True)
        self.build_progress_bar.setFormat("%v%")
        progress_layout.addWidget(self.build_progress_bar)
        layout.addLayout(progress_layout)
        
        # Current file being compiled
        self.build_current_file_label = QLabel("Ready to build")
        self.build_current_file_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.build_current_file_label)
        
        # Build log
        layout.addWidget(QLabel("Build Log:"))
        self.build_log = QTextEdit()
        self.build_log.setReadOnly(True)
        self.build_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.build_log)
        
        return widget
    
    def create_builds_info_tab(self) -> QWidget:
        """Create tab showing installed builds and their capabilities"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("📋 Installed llama.cpp Builds")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        # Available Builds Section
        builds_group = QGroupBox("Available Builds")
        builds_layout = QVBoxLayout()
        
        builds_info = QLabel(
            "💡 You can have multiple builds for different backends.\n"
            "Rename 'build/' folder to keep it (e.g., 'build-rocm/', 'build-cuda/').\n"
            "The server will automatically use the correct build based on selected backend."
        )
        builds_info.setWordWrap(True)
        builds_info.setStyleSheet("color: #666; margin-bottom: 10px;")
        builds_layout.addWidget(builds_info)
        
        # Builds table
        self.builds_table = QTableWidget()
        self.builds_table.setColumnCount(4)
        self.builds_table.setHorizontalHeaderLabels(["Folder", "Backend", "Status", "llama-server"])
        self.builds_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.builds_table.setMaximumHeight(150)
        self.builds_table.setAlternatingRowColors(True)
        builds_layout.addWidget(self.builds_table)
        
        # Rename build button
        rename_layout = QHBoxLayout()
        rename_layout.addWidget(QLabel("Rename current 'build/' to:"))
        self.rename_build_combo = QComboBox()
        self.rename_build_combo.addItems(["build-cpu", "build-cuda", "build-rocm", "build-vulkan", "build-metal", "build-sycl"])
        self.rename_build_combo.setEditable(True)
        rename_layout.addWidget(self.rename_build_combo)
        rename_btn = QPushButton("📁 Rename Build")
        rename_btn.clicked.connect(self.rename_build_folder)
        rename_layout.addWidget(rename_btn)
        rename_layout.addStretch()
        builds_layout.addLayout(rename_layout)
        
        builds_group.setLayout(builds_layout)
        layout.addWidget(builds_group)
        
        # Build Status Summary
        summary_group = QGroupBox("Current Build Details")
        summary_layout = QVBoxLayout()
        
        self.build_summary_text = QTextEdit()
        self.build_summary_text.setReadOnly(True)
        self.build_summary_text.setFont(QFont("Consolas", 10))
        self.build_summary_text.setMaximumHeight(120)
        summary_layout.addWidget(self.build_summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Executables Table
        exe_group = QGroupBox("Executables in Selected Build")
        exe_layout = QVBoxLayout()
        
        self.executables_table = QTableWidget()
        self.executables_table.setColumnCount(4)
        self.executables_table.setHorizontalHeaderLabels(["Executable", "Status", "Size", "Path"])
        self.executables_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.executables_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.executables_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.executables_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.executables_table.setAlternatingRowColors(True)
        
        exe_layout.addWidget(self.executables_table)
        exe_group.setLayout(exe_layout)
        layout.addWidget(exe_group)
        
        # VS Code Integration Section
        vscode_group = QGroupBox("VS Code Integration")
        vscode_layout = QVBoxLayout()
        
        vscode_info = QLabel(
            "To use llama.cpp with VS Code GitHub Copilot:\n"
            "1. Start the server from 'Launch Server' tab\n"
            "2. In VS Code, open Settings (Ctrl+,)\n"
            "3. Search for 'github.copilot.chat.models'\n"
            "4. Add a custom model pointing to http://localhost:8080"
        )
        vscode_info.setWordWrap(True)
        vscode_layout.addWidget(vscode_info)
        
        self.vscode_config_text = QTextEdit()
        self.vscode_config_text.setReadOnly(True)
        self.vscode_config_text.setFont(QFont("Consolas", 9))
        self.vscode_config_text.setMaximumHeight(150)
        self.vscode_config_text.setPlaceholderText("Start the server to see VS Code configuration...")
        vscode_layout.addWidget(self.vscode_config_text)
        
        copy_config_btn = QPushButton("📋 Copy VS Code Configuration")
        copy_config_btn.clicked.connect(self.copy_vscode_config)
        vscode_layout.addWidget(copy_config_btn)
        
        vscode_group.setLayout(vscode_layout)
        layout.addWidget(vscode_group)
        
        # Refresh Button
        buttons_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh Build Info")
        refresh_btn.clicked.connect(self.refresh_builds_info)
        buttons_layout.addWidget(refresh_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Initial load
        self.refresh_builds_info()
        
        return widget
    
    def rename_build_folder(self):
        """Rename the generic 'build' folder to a backend-specific name"""
        generic_build = self.project_root / "build"
        
        if not generic_build.exists():
            QMessageBox.warning(
                self,
                "No Build Found",
                "The 'build/' folder does not exist.\n\n"
                "Build the project first, then rename it."
            )
            return
        
        new_name = self.rename_build_combo.currentText().strip()
        if not new_name:
            return
        
        new_path = self.project_root / new_name
        
        if new_path.exists():
            reply = QMessageBox.question(
                self,
                "Folder Exists",
                f"'{new_name}' already exists.\n\n"
                "Do you want to replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Remove existing folder
            import shutil
            try:
                shutil.rmtree(new_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove existing folder:\n{e}")
                return
        
        try:
            generic_build.rename(new_path)
            QMessageBox.information(
                self,
                "Success",
                f"Build folder renamed to '{new_name}'.\n\n"
                "You can now create a new build for a different backend."
            )
            self.refresh_builds_info()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename folder:\n{e}")
    
    def refresh_builds_info(self):
        """Refresh information about installed builds"""
        # First, update builds table
        available_builds = self.get_available_builds()
        
        self.builds_table.setRowCount(len(available_builds))
        row = 0
        for dir_name, build_data in sorted(available_builds.items()):
            # Folder name
            folder_item = QTableWidgetItem(dir_name)
            self.builds_table.setItem(row, 0, folder_item)
            
            # Backend
            backend_item = QTableWidgetItem(build_data["backend"])
            if build_data["backend"] == "ROCm":
                backend_item.setForeground(Qt.GlobalColor.darkRed)
            elif build_data["backend"] == "CUDA":
                backend_item.setForeground(Qt.GlobalColor.darkGreen)
            elif build_data["backend"] == "Vulkan":
                backend_item.setForeground(Qt.GlobalColor.darkBlue)
            self.builds_table.setItem(row, 1, backend_item)
            
            # Status - check if llama-server exists
            server_exists = self._check_server_in_build(build_data["path"])
            status_item = QTableWidgetItem("✅ Ready" if server_exists else "⚠️ No server")
            self.builds_table.setItem(row, 2, status_item)
            
            # Server path
            server_path = self._find_server_in_build(build_data["path"])
            path_item = QTableWidgetItem(str(server_path) if server_path else "Not found")
            self.builds_table.setItem(row, 3, path_item)
            
            row += 1
        
        # Now update the summary for the first available build
        build_info = self.build_manager.get_build_info()
        
        summary = []
        if available_builds:
            summary.append(f"✅ {len(available_builds)} build(s) available")
            for name, data in available_builds.items():
                summary.append(f"   • {name}: {data['backend']}")
        else:
            summary.append("❌ No builds found")
            summary.append("   Go to 'Build & Setup' tab to build llama.cpp")
        
        self.build_summary_text.setPlainText("\n".join(summary))
        
        # Update executables table (from first available or generic build)
        executables = build_info.get("executables", {})
        self.executables_table.setRowCount(len(executables))
        
        row = 0
        for exe_name, exe_info in sorted(executables.items()):
            name_item = QTableWidgetItem(exe_name)
            self.executables_table.setItem(row, 0, name_item)
            
            status_item = QTableWidgetItem("✅ Built")
            status_item.setForeground(Qt.GlobalColor.darkGreen)
            self.executables_table.setItem(row, 1, status_item)
            
            size = exe_info.get("size_mb", 0)
            size_item = QTableWidgetItem(f"{size:.1f} MB")
            self.executables_table.setItem(row, 2, size_item)
            
            path_item = QTableWidgetItem(exe_info.get("path", ""))
            self.executables_table.setItem(row, 3, path_item)
            
            row += 1
        
        # Update VS Code config
        self.update_vscode_config()
    
    def _check_server_in_build(self, build_path: Path) -> bool:
        """Check if llama-server exists in a build directory"""
        return self._find_server_in_build(build_path) is not None
    
    def _find_server_in_build(self, build_path: Path) -> Optional[Path]:
        """Find llama-server in a build directory"""
        possible_paths = [
            build_path / "bin" / "llama-server.exe",
            build_path / "bin" / "Release" / "llama-server.exe",
            build_path / "bin" / "Debug" / "llama-server.exe",
            build_path / "bin" / "llama-server",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def update_vscode_config(self):
        """Update VS Code configuration example"""
        port = self.server_port_spin.value() if hasattr(self, 'server_port_spin') else 8080
        
        config = f'''// Add to VS Code settings.json:
{{
  "github.copilot.chat.models": [
    {{
      "name": "llama.cpp Local",
      "url": "http://localhost:{port}/v1",
      "vendor": "llama.cpp",
      "provider": "openai"
    }}
  ]
}}

// Or use Continue extension:
// Install "Continue" extension, then add to ~/.continue/config.json:
{{
  "models": [
    {{
      "title": "llama.cpp Local",
      "provider": "openai",
      "model": "local",
      "apiBase": "http://localhost:{port}/v1"
    }}
  ]
}}'''
        self.vscode_config_text.setPlainText(config)
    
    def copy_vscode_config(self):
        """Copy VS Code configuration to clipboard"""
        config = self.vscode_config_text.toPlainText()
        if config:
            clipboard = QApplication.clipboard()
            clipboard.setText(config)
            QMessageBox.information(self, "Copied", "VS Code configuration copied to clipboard!")
        
    def create_hardware_tab(self) -> QWidget:
        """Create tab with hardware information"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("💻 Your System Information")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        self.hardware_info_text = QTextEdit()
        self.hardware_info_text.setReadOnly(True)
        self.hardware_info_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.hardware_info_text)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh Information")
        refresh_btn.clicked.connect(self.detect_hardware)
        buttons_layout.addWidget(refresh_btn)
        
        # ROCm update button (Windows only)
        if platform.system() == "Windows":
            self.rocm_update_btn = QPushButton("📦 Update/Install HIP SDK")
            self.rocm_update_btn.clicked.connect(self.update_rocm)
            self.rocm_update_btn.setToolTip("Open AMD website to download HIP SDK for ROCm support")
            buttons_layout.addWidget(self.rocm_update_btn)
        
        layout.addLayout(buttons_layout)
        
        return widget
        
    def browse_server_model(self):
        """Select model file for server"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.server_model_path_edit.setText(file_path)
            
    def on_server_model_selected(self, model_name: str):
        """Handle model selection from list for server"""
        if model_name and model_name != "-- Выберите модель --":
            model_path = self.models_dir / model_name
            self.server_model_path_edit.setText(str(model_path))
            
    def start_server(self):
        """Start llama-server"""
        model_path = self.server_model_path_edit.text()
        
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self,
                "Error",
                "Please select an existing model file"
            )
            return
        
        # Get selected backend
        backend_idx = self.server_backend_combo.currentIndex()
        backend_names = ["CPU", "CUDA", "Metal", "Vulkan", "ROCm"]
        selected_backend = backend_names[backend_idx] if backend_idx < len(backend_names) else "CPU"
        
        # Find the right build directory for selected backend
        build_dir = self.get_build_dir_for_backend(selected_backend if backend_idx > 0 else None)
        
        # Determining executable file - check multiple possible locations
        possible_paths = [
            build_dir / "bin" / "llama-server.exe",
            build_dir / "bin" / "Release" / "llama-server.exe",
            build_dir / "bin" / "Debug" / "llama-server.exe",
            build_dir / "bin" / "llama-server",  # Linux/Mac
        ]
        
        # Also check generic 'build' folder as fallback
        if build_dir != self.project_root / "build":
            fallback_paths = [
                self.project_root / "build" / "bin" / "llama-server.exe",
                self.project_root / "build" / "bin" / "Release" / "llama-server.exe",
                self.project_root / "build" / "bin" / "llama-server",
            ]
            possible_paths.extend(fallback_paths)
        
        llama_server = None
        used_build_dir = None
        for path in possible_paths:
            if path.exists():
                llama_server = path
                used_build_dir = path.parent.parent if path.parent.name == "bin" else path.parent.parent.parent
                break
        
        if not llama_server:
            # Show helpful error with available builds
            available = self.get_available_builds()
            if available:
                builds_info = "\n".join([f"  • {b['display_name']}" for b in available.values()])
                msg = (f"llama-server not found for {selected_backend} backend!\n\n"
                       f"Available builds:\n{builds_info}\n\n"
                       f"Either:\n"
                       f"1. Select a backend that matches an existing build\n"
                       f"2. Build the project for {selected_backend}")
            else:
                msg = ("llama-server executable not found!\n\n"
                       "Please build the project first:\n"
                       "1. Go to 'Build' tab\n"
                       "2. Click 'Configure Build'\n"
                       "3. Click 'Build Project'")
            
            QMessageBox.warning(self, "Server Not Built", msg)
            return
        
        # Log which build is being used
        self.server_output_text.clear()
        detected_backend = self._detect_build_backend(used_build_dir) if used_build_dir else "Unknown"
        self.server_output_text.append(f"📁 Using build: {used_build_dir}\n")
        self.server_output_text.append(f"🎯 Build backend: {detected_backend}\n")
        self.server_output_text.append(f"🔧 Selected backend: {selected_backend}\n")
        
        # Warn if backend mismatch
        if backend_idx > 0 and detected_backend != selected_backend:
            self.server_output_text.append(f"⚠️ WARNING: Build backend ({detected_backend}) differs from selected ({selected_backend})!\n")
        
        self.server_output_text.append("-" * 50 + "\n")
            
        port = self.server_port_spin.value()
        
        backend_args = {
            0: [],  # CPU only (default)
            1: ["-ngl", str(self.server_gpu_layers_slider.value())],  # CUDA
            2: ["-ngl", str(self.server_gpu_layers_slider.value())],  # Metal
            3: ["-ngl", str(self.server_gpu_layers_slider.value())],  # Vulkan
            4: ["-ngl", str(self.server_gpu_layers_slider.value())],  # ROCm
        }
        
        # Building command
        command = [
            str(llama_server),
            "-m", model_path,
            "--port", str(port),
            "-c", str(self.server_ctx_slider.value() * 8192),
            "-t", str(self.server_threads_spin.value()),
            "--batch-size", str(self.server_batch_slider.value() * 32),
        ]
        
        # Add backend-specific arguments if GPU is enabled
        if self.server_gpu_checkbox.isChecked():
            command.extend(backend_args.get(backend_idx, []))
            
        # CORS
        if self.server_cors_checkbox.isChecked():
            command.extend(["--cors", "*"])
            
        # API ключ
        api_key = self.server_api_key_edit.text().strip()
        if api_key:
            command.extend(["--api-key", api_key])
        
        # Get backend name for logging
        backend_names = [
            "CPU only",
            "CUDA",
            "Metal",
            "Vulkan",
            "ROCm"
        ]
        backend_name = backend_names[backend_idx]
            
        self.server_output_text.clear()
        self.server_output_text.append(f"🚀 Starting server with {backend_name} backend:\n{' '.join(command)}\n\n")
        
        # Get environment for ROCm (need DLLs in PATH)
        server_env = None
        if backend_idx == 4:  # ROCm
            server_env = self.build_manager.get_rocm_env()
            if server_env:
                rocm_version = self.build_manager.detect_rocm_version()
                self.server_output_text.append(f"🎯 ROCm environment loaded (HIP SDK {rocm_version or 'unknown'})\n")
                if "HSA_OVERRIDE_GFX_VERSION" in server_env:
                    self.server_output_text.append(f"⚠️ RDNA4 compatibility mode: HSA_OVERRIDE_GFX_VERSION={server_env['HSA_OVERRIDE_GFX_VERSION']}\n")
                    self.server_output_text.append(f"   💡 Upgrade to HIP SDK 6.4+ for native RDNA4 support\n")
                self.server_output_text.append("\n")
        
        # Running in separate thread
        self.server_thread = ServerThread(command, str(self.project_root), port, env=server_env)
        self.server_thread.output_ready.connect(self.append_server_output)
        self.server_thread.server_ready.connect(self.on_server_ready)
        self.server_thread.finished_signal.connect(self.server_finished)
        self.server_thread.error_signal.connect(self.server_error)
        self.server_thread.start()
        
        self.server_start_btn.setEnabled(False)
        self.server_stop_btn.setEnabled(True)
        self.server_status_label.setText("🟡 Server starting...")
        self.server_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px; color: orange;")
        self.statusBar().showMessage("Starting server...")
        
    def stop_server(self):
        """Stop server"""
        if self.server_thread:
            self.server_thread.stop()
            self.server_output_text.append("\n\n⏹️ Stopped by user\n")
            
    def on_server_ready(self, url: str):
        """Handle server ready"""
        self.server_url = url
        self.server_status_label.setText(f"🟢 Server running: {url}")
        self.server_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px; color: green;")
        self.server_open_web_btn.setEnabled(True)
        self.statusBar().showMessage(f"Server running on {url}")
        
        # Automatically open browser if web mode selected
        if self.mode_web_radio.isChecked():
            QThread.msleep(1000)  # Подождать секунду
            self.open_web_interface()
            
    def open_web_interface(self):
        """Open web interface in browser"""
        if self.server_url:
            webbrowser.open(self.server_url)
            self.server_output_text.append(f"\n🌐 Opened web interface: {self.server_url}\n")
        
    def append_server_output(self, text: str):
        """Add text to server output"""
        self.server_output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.server_output_text.insertPlainText(text)
        self.server_output_text.moveCursor(QTextCursor.MoveOperation.End)
        
    def server_finished(self):
        """Server finished"""
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_open_web_btn.setEnabled(False)
        self.server_status_label.setText("⚪ Server stopped")
        self.server_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px;")
        self.statusBar().showMessage("Server stopped")
        self.server_output_text.append("\n\n✅ Server stopped\n")
        self.server_url = None
        
    def server_error(self, error: str):
        """Handle server error"""
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_open_web_btn.setEnabled(False)
        self.server_status_label.setText("🔴 Error сервера")
        self.server_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 5px; color: red;")
        self.statusBar().showMessage("Error при запуске сервера")
        self.server_output_text.append(f"\n\n❌ Error: {error}\n")
        self.server_url = None
        
    def search_hf_models(self):
        """Search Models on HuggingFace"""
        query = self.search_input.text().strip()
        if not query:
            query = "gguf"
            
        self.statusBar().showMessage(f"Search models: {query}...")
        self.models_table.setRowCount(0)
        
        try:
            # Getting current sort
            sort_methods = ["downloads", "likes", "updated", "id"]
            sort_method = sort_methods[self.sort_combo.currentIndex()]
            
            # Build date filter
            min_date = None
            year = self.filter_year_combo.currentText()
            month_text = self.filter_month_combo.currentText()
            
            if year != "All" and month_text != "All months":
                # Extract month number from text like "July (07)"
                month_num = month_text.split("(")[1].rstrip(")")
                min_date = f"{year}-{month_num}"
            elif year != "All":
                min_date = f"{year}-01"
            
            # Search models
            results = self.model_downloader.search_models(
                query=query,
                sort=sort_method,
                limit=50,
                min_date=min_date
            )
            
            self.display_search_results(results)
            date_info = f" (after {min_date})" if min_date else ""
            self.statusBar().showMessage(f"Found {len(results)} models{date_info}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error поиска", f"Failed to search:\n{str(e)}")
            self.statusBar().showMessage("Error поиска")
            
    def apply_date_filter(self):
        """Apply date filter and refresh search"""
        self.search_hf_models()
            
    def load_popular_models(self):
        """Downloading популярных models"""
        self.statusBar().showMessage("Downloading популярных models...")
        self.models_table.setRowCount(0)
        
        try:
            results = self.model_downloader.get_popular_gguf_models(limit=30)
            self.display_search_results(results)
            self.statusBar().showMessage(f"Loaded {len(results)} популярных models")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load popular models:\n{str(e)}")
            self.statusBar().showMessage("Error загрузки")
            
    def display_search_results(self, results: list):
        """Display search results in table"""
        self.models_table.setRowCount(len(results))
        
        for row, model in enumerate(results):
            # Model Name
            name_item = QTableWidgetItem(model['model_name'])
            self.models_table.setItem(row, 0, name_item)
            
            # Author
            author_item = QTableWidgetItem(model['author'])
            self.models_table.setItem(row, 1, author_item)
            
            # Downloads
            downloads = model.get('downloads', 0)
            downloads_text = f"{downloads:,}" if downloads else "—"
            downloads_item = QTableWidgetItem(downloads_text)
            self.models_table.setItem(row, 2, downloads_item)
            
            # Likes
            likes = model.get('likes', 0)
            likes_text = f"{likes:,}" if likes else "—"
            likes_item = QTableWidgetItem(likes_text)
            self.models_table.setItem(row, 3, likes_item)
            
            # Update Date
            updated = model.get('updated', '')
            updated_text = updated.split('T')[0] if updated else "—"
            updated_item = QTableWidgetItem(updated_text)
            self.models_table.setItem(row, 4, updated_item)
            
            # Save full ID in first cell
            name_item.setData(Qt.ItemDataRole.UserRole, model['id'])
            
    def on_model_double_clicked(self, item):
        """Handle double click on model"""
        row = item.row()
        name_item = self.models_table.item(row, 0)
        model_id = name_item.data(Qt.ItemDataRole.UserRole)
        
        self.selected_model_label.setText(f"Model: {model_id}")
        
        # Clear and disable controls while loading
        self.statusBar().showMessage(f"Loading file list for {model_id}...")
        self.model_file_combo.clear()
        self.model_file_combo.setEnabled(False)
        self.download_btn.setEnabled(False)
        
        # Store model_id for use in callbacks
        self._current_model_id = model_id
        
        # Load files in background thread to prevent UI freeze
        self.list_files_thread = ListFilesThread(model_id)
        self.list_files_thread.status_signal.connect(self._on_list_files_status)
        self.list_files_thread.finished_signal.connect(self._on_list_files_finished)
        self.list_files_thread.error_signal.connect(self._on_list_files_error)
        self.list_files_thread.start()
        
    def _on_list_files_status(self, status: str):
        """Update status during file listing"""
        self.statusBar().showMessage(status)
        
    def _on_list_files_finished(self, files_dict: dict):
        """Handle file list loaded successfully"""
        if files_dict:
            # Add files with size to combo box
            for filename, size in files_dict.items():
                # Format size in human readable format
                size_str = self._format_size(size)
                display_name = f"{filename} ({size_str})"
                self.model_file_combo.addItem(display_name, filename)  # Store actual filename in data
                
            self.model_file_combo.setEnabled(True)
            self.download_btn.setEnabled(True)
            self.statusBar().showMessage(f"Found {len(files_dict)} files")
        else:
            model_id = getattr(self, '_current_model_id', 'unknown')
            QMessageBox.warning(self, "Warning", f"No .gguf files found in repository {model_id}")
            self.statusBar().showMessage("No files found")
            
    def _on_list_files_error(self, error: str):
        """Handle file listing error"""
        QMessageBox.warning(self, "Error", f"Failed to load file list:\n{error}")
        self.statusBar().showMessage("Error loading files")
            
    def _format_size(self, bytes_size: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
            
    def on_sort_changed(self):
        """Handle sort change"""
        # If search text exists, repeat search with new sorting
        if self.search_input.text().strip():
            self.search_hf_models()
        else:
            self.load_popular_models()
        
    def browse_model(self):
        """Select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
    def refresh_models_list(self):
        """Update available models list"""
        # Update all model comboboxes (only if they exist)
        combos = []
        if hasattr(self, 'models_list'):
            combos.append(self.models_list)
        if hasattr(self, 'server_models_list'):
            combos.append(self.server_models_list)
        
        for combo in combos:
            combo.clear()
            combo.addItem("-- Select Model --")
            
            if self.models_dir.exists():
                for model_file in self.models_dir.glob("*.gguf"):
                    combo.addItem(str(model_file.name))
                
    def on_model_selected(self, model_name: str):
        """Handle model selection from list"""
        if model_name and model_name != "-- Select Model --":
            model_path = self.models_dir / model_name
            self.model_path_edit.setText(str(model_path))
            
    def run_inference(self):
        """Run inference"""
        model_path = self.model_path_edit.text()
        
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self,
                "Error",
                "Please select an existing model file"
            )
            return
            
        prompt = self.prompt_edit.toPlainText()
        if not prompt:
            QMessageBox.warning(
                self,
                "Error",
                "Please enter a prompt"
            )
            return
            
        # Determining executable file - check multiple possible locations
        possible_paths = [
            self.build_dir / "bin" / "llama-cli.exe",
            self.build_dir / "bin" / "Release" / "llama-cli.exe",
            self.build_dir / "bin" / "Debug" / "llama-cli.exe",
            self.build_dir / "bin" / "llama-cli",  # Linux/Mac
        ]
        
        llama_cli = None
        for path in possible_paths:
            if path.exists():
                llama_cli = path
                break
            
        if not llama_cli:
            QMessageBox.critical(
                self,
                "Error",
                f"llama-cli executable not found. Please build the project first."
            )
            return
            
        # Building command
        command = [
            str(llama_cli),
            "-m", model_path,
            "-p", prompt,
            "-n", str(self.n_predict_spin.value()),
            "--temp", str(self.temp_spin.value()),
            "--top-p", str(self.top_p_spin.value()),
            "--top-k", str(self.top_k_spin.value()),
            "-c", str(self.ctx_size_spin.value()),
            "-t", str(self.threads_spin.value()),
        ]
        
        if self.gpu_layers_checkbox.isChecked():
            command.extend(["-ngl", str(self.gpu_layers_spin.value())])
            
        self.output_text.clear()
        self.output_text.append(f"🚀 Running command:\n{' '.join(command)}\n\n")
        
        # Running in separate thread
        self.inference_thread = InferenceThread(command, str(self.project_root))
        self.inference_thread.output_ready.connect(self.append_output)
        self.inference_thread.finished_signal.connect(self.inference_finished)
        self.inference_thread.error_signal.connect(self.inference_error)
        self.inference_thread.start()
        
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Running inference...")
        
    def stop_inference(self):
        """Stop inference"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.output_text.append("\n\n⏹️ Stopped by user")
            
    def append_output(self, text: str):
        """Add text to output"""
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)
        
    def inference_finished(self):
        """Inference finished"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Inference completed")
        self.output_text.append("\n\n✅ Completed")
        
    def inference_error(self, error: str):
        """Handle inference error"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Error при выполнении")
        self.output_text.append(f"\n\n❌ Error: {error}")
        
    def download_model(self):
        """Downloading модели"""
        if not self.download_btn.isEnabled():
            return
            
        model_id = self.selected_model_label.text().replace("Model: ", "").strip()
        if not model_id or model_id == "Not Selected":
            QMessageBox.warning(self, "Error", "First select a model from the list")
            return
            
        # Get actual filename from combo box data (without size info)
        filename = self.model_file_combo.currentData()
        if not filename:
            filename = self.model_file_combo.currentText()
            # Extract filename from "filename (size)" format
            if "(" in filename:
                filename = filename.split("(")[0].strip()
        
        if not filename:
            QMessageBox.warning(self, "Error", "Select file to download")
            return
            
        # Check if file is already downloaded
        target_path = self.models_dir / filename
        if target_path.exists():
            reply = QMessageBox.question(
                self,
                "File exists",
                f"File {filename} already exists. Download again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
                
        self.download_status_label.setText(f"Downloading {filename}...")
        self.download_progress.setValue(0)
        self.download_btn.setEnabled(False)
        self.cancel_download_btn.setEnabled(True)
        self.search_btn.setEnabled(False)
        self.load_popular_btn.setEnabled(False)
        self.statusBar().showMessage(f"Downloading {filename}...")
        
        # Store download start time for stats
        self._download_start_time = time.time()
        
        # Start download in background thread
        self.download_thread = DownloadThread(model_id, filename, self.models_dir)
        self.download_thread.status.connect(self.on_download_status)
        self.download_thread.finished_signal.connect(self.on_download_finished)
        self.download_thread.error_signal.connect(self.on_download_error)
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.start()
        
    def cancel_download(self):
        """Cancel current download"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.stop()
            self.download_status_label.setText("Cancelling download...")
            self.statusBar().showMessage("Cancelling download...")
            # Wait a bit for thread to stop, then reset UI
            self.download_thread.wait(2000)  # Wait up to 2 seconds
            self._reset_download_ui()
            self.download_status_label.setText("Download cancelled")
            self.statusBar().showMessage("Download cancelled")
    
    def _reset_download_ui(self):
        """Reset download UI to initial state"""
        self.download_btn.setEnabled(True)
        self.cancel_download_btn.setEnabled(False)
        self.search_btn.setEnabled(True)
        self.load_popular_btn.setEnabled(True)
        self.download_progress.setValue(0)
    
    def on_download_status(self, status: str):
        """Update download status"""
        self.download_status_label.setText(status)
        self.statusBar().showMessage(status)
        
    def on_download_progress(self, downloaded, total, speed_mbps: float, eta_seconds: float):
        """Update download progress with speed and ETA"""
        # Convert to int for safety (may come as object type for large values)
        downloaded = int(downloaded) if downloaded else 0
        total = int(total) if total else 0
        
        print(f"[GUI] Progress received: {downloaded}/{total} ({speed_mbps:.1f} MB/s)")
        
        if total > 0:
            progress = int((downloaded / total) * 100)
            self.download_progress.setValue(progress)
            
            # Format sizes
            downloaded_mb = downloaded / 1024 / 1024
            total_mb = total / 1024 / 1024
            
            # Format ETA
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds / 3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds / 60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
            
            # Show detailed progress
            if total_mb >= 1024:
                # Show in GB for large files
                downloaded_gb = downloaded_mb / 1024
                total_gb = total_mb / 1024
                self.download_status_label.setText(
                    f"Downloading: {downloaded_gb:.2f} / {total_gb:.2f} GB | "
                    f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_str}"
                )
            else:
                self.download_status_label.setText(
                    f"Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB | "
                    f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_str}"
                )
        
    def on_download_finished(self, file_path: str):
        """Handle successful download"""
        # Calculate total download time
        elapsed = time.time() - getattr(self, '_download_start_time', time.time())
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
        
        self.download_progress.setValue(100)
        self.download_status_label.setText(f"[OK] Downloaded: {Path(file_path).name} in {elapsed_str}")
        self.statusBar().showMessage("Download completed")
        self.refresh_models_list()
        
        QMessageBox.information(
            self,
            "Success",
            f"Model downloaded successfully:\n{file_path}\n\nTime: {elapsed_str}"
        )
        
        # Reset UI
        self._reset_download_ui()
        
    def on_download_error(self, error: str):
        """Handle download error"""
        self.download_status_label.setText(f"[ERROR] {error}")
        self.statusBar().showMessage(f"Download error: {error}")
        
        QMessageBox.critical(
            self,
            "Download Error",
            f"Failed to download model:\n{error}"
        )
        
        # Reset UI
        self._reset_download_ui()
        
    def configure_build(self):
        """Configure build with selected backend"""
        from dependency_checker import DependencyChecker
        
        backend_idx = self.backend_combo.currentIndex()
        backend_map = {
            0: None,  # CPU only
            1: "CUDA",
            2: "Metal",
            3: "Vulkan",
            4: "SYCL",
            5: "ROCm",
        }
        
        backend = backend_map.get(backend_idx)
        
        self.build_log.clear()
        self.build_log.append(f"⚙️ Starting CMake configuration for backend: {backend or 'CPU only'}\n")
        
        # Check system tools required for this backend
        if backend:
            all_available, missing_tools = DependencyChecker.check_and_install_system_tools(backend)
            if not all_available:
                self.build_log.append("\n❌ Missing required system tools:\n")
                for tool_name in missing_tools:
                    tool_info = DependencyChecker.SYSTEM_TOOLS.get(tool_name, {})
                    self.build_log.append(f"  • {tool_name}: {tool_info.get('description', '')}\n")
                    self.build_log.append(f"    Install: {tool_info.get('install_hint', 'N/A')}\n")
                
                reply = QMessageBox.question(
                    self,
                    "Missing System Tools",
                    f"Missing required tools for {backend}:\n\n"
                    + "\n".join(f"• {t}" for t in missing_tools) +
                    "\n\nWould you like to install them now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    for tool_name in missing_tools:
                        self.build_log.append(f"\n🔧 Installing {tool_name}...\n")
                        if DependencyChecker.install_system_tool(tool_name):
                            self.build_log.append(f"✅ {tool_name} installed successfully\n")
                        else:
                            tool_info = DependencyChecker.SYSTEM_TOOLS.get(tool_name, {})
                            self.build_log.append(f"❌ Failed to install {tool_name}\n")
                            self.build_log.append(f"   Please run manually: {tool_info.get('install_hint', 'N/A')}\n")
                            return
                    self.build_log.append("\n✅ All tools installed. Please restart the application.\n")
                    QMessageBox.information(
                        self, "Restart Required",
                        "System tools installed successfully.\n\n"
                        "Please restart the application for changes to take effect."
                    )
                    return
                else:
                    self.build_log.append("\n⏹️ Configuration cancelled - missing tools\n")
                    return
        
        # Check prerequisites
        prerequisites = self.build_manager.check_build_prerequisites(backend)
        self.build_log.append("\n📋 Checking prerequisites:\n")
        
        for tool, available in prerequisites.items():
            status = "✅" if available else "❌"
            self.build_log.append(f"  {status} {tool}: {available}\n")
        
        # Check if ROCm needed for AMD
        if backend == "ROCm" and not prerequisites.get("rocm", False):
            self.build_log.append("\n⚠️ WARNING: ROCm not installed\n")
            self.build_log.append("Please install ROCm from: https://rocmdocs.amd.com/\n")
            reply = QMessageBox.warning(
                self,
                "ROCm Not Found",
                "ROCm is not installed on your system.\n\n"
                "For AMD 9070XT, ROCm is highly recommended.\n\n"
                "Would you like to continue with Vulkan instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                backend = "Vulkan"
                self.backend_combo.setCurrentIndex(3)
                self.build_log.append("\n✓ Switched to Vulkan backend\n")
            else:
                self.build_log.append("\n⏹️ Configuration cancelled\n")
                return
        
        # Build configuration command
        options = {
            "LLAMA_BUILD_TESTS": self.build_tests_checkbox.isChecked(),
            "LLAMA_INSTALL": True,
        }
        
        if self.use_ccache_checkbox.isChecked():
            options["LLAMA_CCACHE"] = True
        
        # Show which generator will be used
        generator = self.build_manager.get_cmake_generator(backend)
        self.build_log.append(f"\n🔧 CMake Generator: {generator or 'default'}\n")
        if backend and backend.upper() == "ROCM" and generator != "Ninja":
            self.build_log.append("⚠️ WARNING: ROCm requires Ninja generator!\n")
        
        # Check if we need to clean CMakeCache due to generator mismatch
        cmake_cache = self.build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                with open(cmake_cache, 'r') as f:
                    cache_content = f.read()
                    # Check for generator mismatch
                    if generator and f"CMAKE_GENERATOR:INTERNAL={generator}" not in cache_content:
                        # Generator changed - need to clean
                        self.build_log.append("⚠️ Detected generator mismatch with existing build\n")
                        reply = QMessageBox.question(
                            self,
                            "Generator Mismatch",
                            "The build directory was configured with a different CMake generator.\n\n"
                            "To use a different generator, the build directory must be cleaned.\n\n"
                            "Clean build directory and continue?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            import shutil
                            self.build_log.append("🧹 Cleaning build directory...\n")
                            # Remove CMakeCache.txt and CMakeFiles
                            cmake_cache.unlink()
                            cmake_files = self.build_dir / "CMakeFiles"
                            if cmake_files.exists():
                                shutil.rmtree(cmake_files)
                            self.build_log.append("✅ Build directory cleaned\n")
                        else:
                            self.build_log.append("⏹️ Configuration cancelled\n")
                            return
            except Exception as e:
                self.build_log.append(f"⚠️ Could not check CMakeCache: {e}\n")
        
        command = self.build_manager.get_configure_command(backend, options)
        
        self.build_log.append(f"\n▶️ CMake command:\n{' '.join(command)}\n\n")
        
        # Get ROCm environment if using ROCm backend
        rocm_env = None
        if backend and backend.upper() == "ROCM":
            rocm_env = self.build_manager.get_rocm_env()
            if rocm_env:
                self.build_log.append(f"🎯 ROCm environment configured:\n")
                self.build_log.append(f"   HIP_PATH: {rocm_env.get('HIP_PATH', 'N/A')}\n")
                
                # Show ROCm version
                rocm_version = self.build_manager.detect_rocm_version()
                if rocm_version:
                    self.build_log.append(f"   📦 HIP SDK version: {rocm_version}\n")
                
                # Show detected GPU
                gpu_target = self.build_manager.detect_amd_gpu_targets()
                if gpu_target:
                    self.build_log.append(f"   🎮 Detected GPU: {gpu_target}\n")
                    # Show RDNA4 workaround info
                    if "gfx1200" in gpu_target or "gfx1201" in gpu_target:
                        # Check if workaround is needed
                        needs_workaround = True
                        if rocm_version:
                            try:
                                major_minor = float(rocm_version.split('.')[0] + '.' + rocm_version.split('.')[1])
                                if major_minor >= 6.4:
                                    needs_workaround = False
                                    self.build_log.append(f"   ✅ Native RDNA4 support (HIP SDK 6.4+)\n")
                            except:
                                pass
                        
                        if needs_workaround:
                            self.build_log.append(f"   ⚠️ RDNA4 detected - using gfx1100 compatibility mode\n")
                            self.build_log.append(f"      💡 Upgrade to HIP SDK 6.4+ for native RDNA4 support\n")
                            self.build_log.append(f"      Download: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html\n")
                    self.build_log.append("\n")
                else:
                    self.build_log.append(f"   ⚠️ GPU auto-detection failed, using fallback targets\n\n")
            else:
                self.build_log.append("⚠️ Warning: ROCm not found in standard locations\n\n")
        
        # Execute configuration in background thread to prevent UI freezing
        self.configure_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        
        self.configure_thread = ConfigureThread(command, self.project_root, env=rocm_env)
        self.configure_thread.output.connect(self._on_configure_output)
        self.configure_thread.finished_signal.connect(self._on_configure_finished)
        self.configure_thread.start()
        
    def _on_configure_output(self, text: str):
        """Handle output from configure thread"""
        self.build_log.append(text)
        # Auto-scroll to bottom
        self.build_log.verticalScrollBar().setValue(
            self.build_log.verticalScrollBar().maximum()
        )
        
    def _on_configure_finished(self, success: bool):
        """Handle configure thread completion"""
        self.configure_btn.setEnabled(True)
        if success:
            self.build_btn.setEnabled(True)
        
        
    def build_project(self):
        """Build llama.cpp project in background"""
        # Check if build directory exists
        if not self.build_dir.exists():
            self.build_log.clear()
            self.build_log.append("❌ Build directory does not exist!\n\n")
            self.build_log.append("═" * 60 + "\n")
            self.build_log.append("⚠️ BUILD REQUIRES CONFIGURATION FIRST!\n")
            self.build_log.append("═" * 60 + "\n\n")
            self.build_log.append("The 'Configure Build' step creates the build/ folder\n")
            self.build_log.append("and generates build files using CMake.\n\n")
            self.build_log.append("CORRECT ORDER:\n")
            self.build_log.append("1️⃣  Click 'Install Dependencies' (one time)\n")
            self.build_log.append("2️⃣  Click 'Configure Build' (before first build)\n")
            self.build_log.append("3️⃣  Click 'Build Project' (actual compilation)\n\n")
            self.build_log.append("TIME ESTIMATES:\n")
            self.build_log.append("├─ Install Dependencies: 4-6 minutes\n")
            self.build_log.append("├─ Configure Build:     30 seconds to 2 minutes\n")
            self.build_log.append("└─ Build Project:       10-30 minutes (first time)\n")
            
            QMessageBox.warning(
                self,
                "⚠️ Configuration Required",
                "Build directory does not exist!\n\n"
                "You must run 'Configure Build' first.\n\n"
                "This step:\n"
                "• Creates the build/ folder\n"
                "• Generates build files with CMake\n"
                "• Detects your compiler and libraries\n\n"
                "Steps:\n"
                "1. Click 'Configure Build' button\n"
                "2. Wait for ✅ success message\n"
                "3. Then click 'Build Project'\n\n"
                "For details, see BUILD_STEPS.md"
            )
            return
        
        config = "Release"
        jobs = os.cpu_count() or 4
        
        self.build_log.clear()
        
        # Get selected backend
        backend_idx = self.backend_combo.currentIndex()
        backend = {0: None, 1: "CUDA", 2: "Metal", 3: "Vulkan", 4: "SYCL", 5: "ROCm"}.get(backend_idx)
        
        # ROCm builds are limited to 4 parallel jobs due to high memory usage
        if backend and backend.upper() == "ROCM":
            jobs = min(jobs, 4)
            self.build_log.append(f"🔨 Starting ROCm build with {jobs} parallel jobs...\n")
            self.build_log.append("⚠️ ROCm compilation is memory-intensive and may take 15-30 minutes.\n")
            self.build_log.append("   Please be patient and don't close the application.\n\n")
        else:
            self.build_log.append(f"🔨 Starting build with {jobs} parallel jobs...\n")
        
        # Pass backend to build command (ROCm uses Ninja which doesn't need --config)
        build_command = self.build_manager.get_build_command(config, jobs, backend)
        
        self.build_log.append(f"Build command: {' '.join(build_command)}\n\n")
        
        # Get ROCm environment if using ROCm backend
        rocm_env = None
        if backend and backend.upper() == "ROCM":
            rocm_env = self.build_manager.get_rocm_env()
        
        # Create build thread
        self.build_thread = BuildThread([build_command], self.project_root, env=rocm_env)
        self.build_thread.output.connect(self.on_build_output)
        self.build_thread.finished_signal.connect(self.on_build_finished)
        self.build_thread.progress.connect(self.on_build_progress)
        self.build_thread.build_progress.connect(self.on_build_progress_detail)
        
        # Reset progress UI
        self.build_progress_bar.setValue(0)
        self.build_current_file_label.setText("Starting build...")
        self.build_current_file_label.setStyleSheet("color: blue;")
        
        self.configure_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.install_deps_btn.setEnabled(False)
        
        self.build_thread.start()
        
    def on_build_output(self, text: str):
        """Update build output"""
        self.build_log.append(text)
        
    def on_build_finished(self, success: bool):
        """Handle build completion"""
        if success:
            self.build_log.append("\n✅ Build completed successfully!\n")
            build_info = self.build_manager.get_build_info()
            self.build_log.append("\n📋 Built executables:\n")
            for exe_name, info in build_info["executables"].items():
                if info["exists"]:
                    self.build_log.append(f"  ✅ {exe_name}: {info['path']}\n")
        else:
            self.build_log.append("\n❌ Build failed!\n")
        
        self.configure_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        self.install_deps_btn.setEnabled(True)
        
        # Update progress UI on completion
        if success:
            self.build_progress_bar.setValue(100)
            self.build_current_file_label.setText("✅ Build completed!")
            self.build_current_file_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.build_current_file_label.setText("❌ Build failed!")
            self.build_current_file_label.setStyleSheet("color: red; font-weight: bold;")
        
    def on_build_progress(self, percentage: int):
        """Update build progress bar"""
        self.build_progress_bar.setValue(percentage)
        self.statusBar().showMessage(f"Building... {percentage}%")
    
    def on_build_progress_detail(self, current: int, total: int, filename: str):
        """Update detailed build progress"""
        self.build_progress_bar.setValue(int(current / total * 100))
        if filename:
            self.build_current_file_label.setText(f"[{current}/{total}] Compiling: {filename}")
        else:
            self.build_current_file_label.setText(f"[{current}/{total}] Building...")
        self.build_current_file_label.setStyleSheet("color: blue;")
        
    def install_dependencies(self):
        """Automatically download and install required dependencies"""
        self.build_log.clear()
        self.build_log.append("📦 Checking dependencies...\n\n")
        
        # Check what dependencies are needed
        backend_idx = self.backend_combo.currentIndex()
        backend = {0: None, 1: "CUDA", 2: "Metal", 3: "Vulkan", 4: "SYCL", 5: "ROCm"}.get(backend_idx)
        
        missing_deps = self.dependency_manager.get_missing_dependencies(backend)
        
        self.build_log.append("📋 Dependency Status:\n")
        self.build_log.append("-" * 50 + "\n")
        
        # Check basic tools
        basic_tools = ["cmake", "git"]
        if self.os_type == "Windows":
            basic_tools.append("MSVC Build Tools")
        else:
            basic_tools.append("GCC/Clang")
        
        for tool in basic_tools:
            is_missing = any(tool.lower() in dep.lower() for dep in missing_deps)
            status = "❌" if is_missing else "✅"
            self.build_log.append(f"  {status} {tool}\n")
        
        # Check backend tools
        if backend:
            self.build_log.append(f"\n🎮 {backend} Backend:\n")
            if backend == "CUDA":
                is_missing = any("cuda" in dep.lower() for dep in missing_deps)
                status = "❌" if is_missing else "✅"
                self.build_log.append(f"  {status} CUDA Toolkit (nvcc)\n")
            elif backend == "Vulkan":
                is_missing = any("vulkan" in dep.lower() for dep in missing_deps)
                status = "❌" if is_missing else "✅"
                self.build_log.append(f"  {status} Vulkan SDK\n")
            elif backend == "ROCm":
                is_missing = any("rocm" in dep.lower() for dep in missing_deps)
                status = "❌" if is_missing else "✅"
                self.build_log.append(f"  {status} ROCm SDK\n")
        
        if not missing_deps:
            self.build_log.append("\n✅ All dependencies are installed!\n")
            self.build_log.append("Ready to configure and build the project.\n")
            QMessageBox.information(
                self,
                "Dependencies OK",
                "All required dependencies are already installed.\n\n"
                "You can now proceed to 'Configure' and 'Build'."
            )
            return
        
        self.build_log.append(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}\n")
        
        # Ask user if they want auto-install
        reply = QMessageBox.question(
            self,
            "Missing Dependencies",
            f"Missing: {', '.join(missing_deps)}\n\n"
            "Would you like to automatically download and install them?\n\n"
            "Note: This will download installers and launch them.\n"
            "Some installations may require admin privileges.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.build_log.append("\n🚀 Starting automatic installation...\n")
            self.build_log.append("-" * 50 + "\n\n")
            
            # Disable buttons during installation
            self.configure_btn.setEnabled(False)
            self.build_btn.setEnabled(False)
            self.install_deps_btn.setEnabled(False)
            
            # Start installation thread
            self.install_thread = DependencyInstallThread(missing_deps, self.os_type)
            self.install_thread.status_update.connect(self.on_install_status)
            self.install_thread.progress.connect(self.on_install_progress)
            self.install_thread.finished_signal.connect(self.on_install_finished)
            self.install_thread.start()
        else:
            self.build_log.append("\n📖 Manual Installation Instructions:\n")
            self.build_log.append("-" * 50 + "\n\n")
            self._show_manual_install_instructions(missing_deps, backend)
            
    def on_install_status(self, status: str):
        """Update installation status"""
        self.build_log.append(f"{status}\n")
        self.statusBar().showMessage(status)
        
    def on_install_progress(self, message: str, percentage: int):
        """Update installation progress"""
        self.build_log.append(f"[{percentage}%] {message}\n")
        self.statusBar().showMessage(f"Installing... {percentage}%")
        
    def on_install_finished(self, success: bool, message: str):
        """Handle installation completion"""
        self.configure_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        self.install_deps_btn.setEnabled(True)
        
        if success:
            self.build_log.append(f"\n✅ {message}\n")
            self.build_log.append("Please restart the application for changes to take effect.\n")
            
            QMessageBox.information(
                self,
                "Installation Complete",
                f"{message}\n\n"
                "Please restart the application for changes to take effect."
            )
        else:
            self.build_log.append(f"\n❌ {message}\n")
            
            QMessageBox.warning(
                self,
                "Installation Failed",
                f"Installation encountered an error:\n{message}\n\n"
                "Please check the log above for details."
            )
            
    def _show_manual_install_instructions(self, missing_deps: List[str], backend: Optional[str]):
        """Show manual installation instructions"""
        self.build_log.append("Windows Installation:\n")
        self.build_log.append("=" * 50 + "\n\n")
        
        if any("cmake" in dep.lower() for dep in missing_deps):
            self.build_log.append("1️⃣ CMake:\n")
            self.build_log.append("   Download: https://cmake.org/download/\n")
            self.build_log.append("   Or: winget install Kitware.CMake\n\n")
        
        if any("git" in dep.lower() for dep in missing_deps):
            self.build_log.append("2️⃣ Git:\n")
            self.build_log.append("   Download: https://git-scm.com/download/win\n")
            self.build_log.append("   Or: winget install Git.Git\n\n")
        
        if any("msvc" in dep.lower() for dep in missing_deps):
            self.build_log.append("3️⃣ Visual Studio Build Tools:\n")
            self.build_log.append("   Download: https://visualstudio.microsoft.com/downloads/\n")
            self.build_log.append("   Install C++ workload\n\n")
        
        if any("vulkan" in dep.lower() for dep in missing_deps):
            self.build_log.append("4️⃣ Vulkan SDK:\n")
            self.build_log.append("   Download: https://vulkan.lunarg.com/sdk/home\n\n")
        
        if any("rocm" in dep.lower() for dep in missing_deps):
            self.build_log.append("5️⃣ ROCm SDK:\n")
            self.build_log.append("   Download: https://rocmdocs.amd.com/en/latest/deploy/windows/quick_start.html\n\n")
        
        self.build_log.append("⚠️ After installing, please restart the application.\n")
        
    def _auto_select_backend(self):
        """Auto-select optimal backend based on detected hardware"""
        if hasattr(self, 'server_backend_combo') and self.hardware_detector:
            gpu_info = self.hardware_detector.get_gpu_info()
            for gpu in gpu_info:
                if gpu.get("type") == "AMD":
                    self.server_backend_combo.setCurrentIndex(4)  # ROCm
                    break
                elif gpu.get("type") == "NVIDIA":
                    self.server_backend_combo.setCurrentIndex(1)  # CUDA
                    break
        
    def detect_hardware(self):
        """Detect hardware characteristics and show ROCm/Vulkan info"""
        info = self.hardware_detector.get_hardware_info()
        
        # Auto-select backend after hardware detection
        self._auto_select_backend()
        
        text = "💻 SYSTEM INFORMATION\n"
        text += "=" * 60 + "\n\n"
        
        text += f"Operating System: {info['os']}\n"
        text += f"CPU: {info['cpu']['name']}\n"
        text += f"Cores: {info['cpu']['cores']} | Threads: {info['cpu']['threads']}\n"
        text += f"RAM: {info['memory']['total_gb']:.1f} GB ({info['memory']['percent_used']:.1f}% used)\n\n"
        
        text += "🎮 GPU INFORMATION:\n"
        text += "-" * 60 + "\n"
        
        if info['gpu']:
            for i, gpu in enumerate(info['gpu']):
                text += f"\nGPU {i+1}: {gpu['name']}\n"
                
                if gpu.get('type') == 'AMD':
                    text += f"  Type: AMD Radeon\n"
                    text += f"  Backend: {gpu.get('backend', 'ROCm or Vulkan')}\n"
                    
                    if gpu.get('is_9070xt'):
                        text += f"  🎯 AMD 9070XT DETECTED (RDNA 4)\n"
                        text += f"  Recommended: ROCm with gfx1201 support\n"
                else:
                    text += f"  Type: {gpu.get('type', 'Unknown')}\n"
                    text += f"  Backend: {gpu.get('backend', 'Unknown')}\n"
                
                if 'memory' in gpu:
                    text += f"  Memory: {gpu['memory']}\n"
        else:
            text += "  ❌ No GPU detected - will use CPU inference\n"
        
        text += "\n" + "=" * 60 + "\n"
        text += "📋 BACKEND RECOMMENDATION:\n"
        text += "-" * 60 + "\n"
        text += f"  Recommended: {info.get('recommended_backend', 'CPU')}\n"
        
        # Handle both old and new hardware detector versions
        if 'backend_reason' in info:
            text += f"  Reason: {info['backend_reason']}\n"
        
        text += "\n✅ DEPENDENCY STATUS:\n"
        text += "-" * 60 + "\n"
        
        if info.get('rocm_available', False):
            text += f"  ✅ ROCm: Installed\n"
            # Show ROCm version
            rocm_version = self.build_manager.detect_rocm_version()
            if rocm_version:
                text += f"     Version: {rocm_version}\n"
                # Check if upgrade recommended for RDNA4
                try:
                    major_minor = float(rocm_version.split('.')[0] + '.' + rocm_version.split('.')[1])
                    if major_minor < 6.4:
                        text += f"     ⚠️ Upgrade to 6.4+ recommended for RDNA4 native support\n"
                    else:
                        text += f"     ✅ Native RDNA4 support available\n"
                except:
                    pass
        else:
            text += f"  ❌ ROCm: NOT installed\n"
        
        if info.get('vulkan_available', False):
            text += f"  ✅ Vulkan: Installed\n"
        else:
            text += f"  ❌ Vulkan: NOT installed\n"
        
        text += "\n💡 For AMD 9070XT: ROCm is HIGHLY RECOMMENDED\n"
        text += "   If ROCm is not available, Vulkan can be used as fallback.\n"
        text += "\n📦 Latest HIP SDK: 6.4.2 (download button above)\n"
        
        self.hardware_info_text.setText(text)
    
    def update_rocm(self):
        """Download and install/update HIP SDK"""
        # Get current version
        current_version = self.build_manager.detect_rocm_version()
        latest_version = "6.4.2"
        
        # Build message
        if current_version:
            msg = f"Current HIP SDK version: {current_version}\n"
            msg += f"Latest available version: {latest_version}\n\n"
            try:
                current_major_minor = float(current_version.split('.')[0] + '.' + current_version.split('.')[1])
                if current_major_minor >= 6.4:
                    msg += "✅ You have a recent version with RDNA4 support.\n\n"
                else:
                    msg += "⚠️ Upgrade recommended for native RDNA4 support.\n\n"
            except:
                pass
        else:
            msg = "HIP SDK is not currently installed.\n\n"
        
        msg += "Open AMD HIP SDK download page?\n\n"
        msg += "Note: AMD requires accepting a license agreement before download."
        
        reply = QMessageBox.question(
            self,
            "HIP SDK Update",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Open AMD HIP SDK download page
            url = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
            webbrowser.open(url)
            
            # Show additional instructions
            QMessageBox.information(
                self,
                "Installation Instructions",
                "1. On the AMD page, find 'HIP SDK' for Windows 10 & 11\n\n"
                "2. Click 'HIP SDK' link next to version 6.4.2\n\n"
                "3. Accept the license agreement\n\n"
                "4. Download and run the installer (~1.5 GB)\n\n"
                "5. Follow the installation wizard\n\n"
                "6. Restart this application after installation\n\n"
                "7. Click 'Refresh Information' to verify\n\n"
                "Note: You may need to reconfigure and rebuild llama.cpp."
            )
        
    def load_settings(self):
        """Load saved settings"""
        # Load server settings first - they are always available
        if hasattr(self, 'server_port_spin'):
            self.server_port_spin.setValue(self.settings.value("server_port", 8080, type=int))
        if hasattr(self, 'server_gpu_checkbox'):
            self.server_gpu_checkbox.setChecked(self.settings.value("server_gpu", True, type=bool))
        if hasattr(self, 'server_cors_checkbox'):
            self.server_cors_checkbox.setChecked(self.settings.value("server_cors", True, type=bool))
        if hasattr(self, 'server_api_key_edit'):
            self.server_api_key_edit.setText(self.settings.value("server_api_key", ""))
        if hasattr(self, 'server_backend_combo'):
            self.server_backend_combo.setCurrentIndex(self.settings.value("server_backend", 0, type=int))
        if hasattr(self, 'server_model_path_edit'):
            self.server_model_path_edit.setText(self.settings.value("server_model_path", ""))
        
        # Load server parameters (sliders)
        if hasattr(self, 'server_ctx_slider'):
            ctx_val = self.settings.value("server_ctx", 8192, type=int)
            self.server_ctx_slider.setValue(max(1, ctx_val // 8192))  # Convert to step
        if hasattr(self, 'server_batch_slider'):
            batch_val = self.settings.value("server_batch", 512, type=int)
            self.server_batch_slider.setValue(max(1, batch_val // 32))  # Convert to step
        if hasattr(self, 'server_threads_spin'):
            self.server_threads_spin.setValue(self.settings.value("server_threads", os.cpu_count() or 4, type=int))
        if hasattr(self, 'server_gpu_layers_slider'):
            self.server_gpu_layers_slider.setValue(self.settings.value("server_gpu_layers", 33, type=int))
        
        # Load Quick Select model selection
        saved_model_name = self.settings.value("server_quick_select_model", "")
        if saved_model_name and hasattr(self, 'server_models_list'):
            index = self.server_models_list.findText(saved_model_name)
            if index >= 0:
                self.server_models_list.setCurrentIndex(index)
        
        # Load inference settings
        if hasattr(self, 'n_predict_spin'):
            self.n_predict_spin.setValue(self.settings.value("n_predict", 128, type=int))
        if hasattr(self, 'temp_spin'):
            self.temp_spin.setValue(self.settings.value("temperature", 0.8, type=float))
        if hasattr(self, 'top_p_spin'):
            self.top_p_spin.setValue(self.settings.value("top_p", 0.9, type=float))
        if hasattr(self, 'top_k_spin'):
            self.top_k_spin.setValue(self.settings.value("top_k", 40, type=int))
        if hasattr(self, 'ctx_size_spin'):
            self.ctx_size_spin.setValue(self.settings.value("ctx_size", 2048, type=int))
        if hasattr(self, 'threads_spin'):
            self.threads_spin.setValue(self.settings.value("threads", os.cpu_count() or 4, type=int))
        if hasattr(self, 'gpu_layers_spin'):
            self.gpu_layers_spin.setValue(self.settings.value("gpu_layers", 33, type=int))
        
        last_model = self.settings.value("last_model", "")
        if last_model and hasattr(self, 'model_path_edit'):
            self.model_path_edit.setText(last_model)
    
    def change_repository_path(self):
        """Allow user to change the llama.cpp repository path"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select llama.cpp Repository Folder",
            str(self.project_root.parent),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        
        if not self._is_valid_llama_cpp_repo(folder_path):
            QMessageBox.warning(
                self,
                "Invalid Repository",
                f"The selected folder does not appear to be a valid llama.cpp repository.\n\n"
                f"Selected: {folder_path}\n\n"
                "The folder should contain:\n"
                "• CMakeLists.txt\n"
                "• include/llama.h"
            )
            return
        
        # Update paths
        self.project_root = folder_path
        self.build_dir = self.project_root / "build"
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Save and update managers
        self.settings.setValue("project_root", str(self.project_root))
        self.build_manager = BuildManager(self.project_root)
        self.model_downloader = ModelDownloader(self.models_dir)
        
        # Update UI
        if hasattr(self, 'repo_path_label'):
            self.repo_path_label.setText(str(self.project_root))
        
        # Refresh builds info
        self.refresh_builds_info()
        self.refresh_models_list()
        
        QMessageBox.information(
            self,
            "Repository Changed",
            f"Repository path changed to:\n{self.project_root}\n\n"
            "Build status and models list have been refreshed."
        )
            
    def check_project_build(self):
        """Check if llama.cpp is built and show warning if not"""
        # Check all available builds using multi-build system
        available_builds = self.get_available_builds()
        
        if available_builds:
            # Check if any build has llama-server
            for build_name, build_data in available_builds.items():
                if self._check_server_in_build(build_data["path"]):
                    return True
        
        # Fallback: check generic build folder paths
        possible_paths = [
            self.project_root / "build" / "bin" / "llama-server.exe",
            self.project_root / "build" / "bin" / "Release" / "llama-server.exe",
            self.project_root / "build" / "bin" / "Debug" / "llama-server.exe",
            self.project_root / "build" / "bin" / "llama-server",  # Linux/Mac
        ]
        
        for server_path in possible_paths:
            if server_path.exists():
                return True
        
        # No builds found - show warning
        reply = QMessageBox.warning(
            self,
            "Project Not Built",
            "llama.cpp project has not been built yet.\n\n"
            "You need to build the project first before running the server.\n\n"
            "Go to 'Build & Setup' tab and click 'Build' button.",
            QMessageBox.StandardButton.Ok
        )
        return False
            
    def save_settings(self):
        """Save settings"""
        # Save server settings first
        if hasattr(self, 'server_port_spin'):
            self.settings.setValue("server_port", self.server_port_spin.value())
        if hasattr(self, 'server_gpu_checkbox'):
            self.settings.setValue("server_gpu", self.server_gpu_checkbox.isChecked())
        if hasattr(self, 'server_cors_checkbox'):
            self.settings.setValue("server_cors", self.server_cors_checkbox.isChecked())
        if hasattr(self, 'server_api_key_edit'):
            self.settings.setValue("server_api_key", self.server_api_key_edit.text())
        if hasattr(self, 'server_backend_combo'):
            self.settings.setValue("server_backend", self.server_backend_combo.currentIndex())
        if hasattr(self, 'server_model_path_edit'):
            self.settings.setValue("server_model_path", self.server_model_path_edit.text())
        if hasattr(self, 'server_threads_spin'):
            self.settings.setValue("server_threads", self.server_threads_spin.value())
        if hasattr(self, 'server_ctx_slider'):
            self.settings.setValue("server_ctx", self.server_ctx_slider.value() * 8192)  # Save actual value
        if hasattr(self, 'server_batch_slider'):
            self.settings.setValue("server_batch", self.server_batch_slider.value() * 32)  # Save actual value
        if hasattr(self, 'server_gpu_layers_slider'):
            self.settings.setValue("server_gpu_layers", self.server_gpu_layers_slider.value())
        # Save Quick Select model selection
        if hasattr(self, 'server_models_list'):
            current_model = self.server_models_list.currentText()
            if current_model and current_model != "-- Select Model --":
                self.settings.setValue("server_quick_select_model", current_model)
            
        # Save inference settings
        if hasattr(self, 'n_predict_spin'):
            self.settings.setValue("n_predict", self.n_predict_spin.value())
        if hasattr(self, 'temp_spin'):
            self.settings.setValue("temperature", self.temp_spin.value())
        if hasattr(self, 'top_p_spin'):
            self.settings.setValue("top_p", self.top_p_spin.value())
        if hasattr(self, 'top_k_spin'):
            self.settings.setValue("top_k", self.top_k_spin.value())
        if hasattr(self, 'ctx_size_spin'):
            self.settings.setValue("ctx_size", self.ctx_size_spin.value())
        if hasattr(self, 'threads_spin'):
            self.settings.setValue("threads", self.threads_spin.value())
        if hasattr(self, 'gpu_layers_spin'):
            self.settings.setValue("gpu_layers", self.gpu_layers_spin.value())
        if hasattr(self, 'model_path_edit'):
            self.settings.setValue("last_model", self.model_path_edit.text())
        
    def closeEvent(self, event):
        """Handle window close"""
        self.save_settings()
        
        # Check if processes are running
        has_running_processes = False
        message_parts = []
        
        if self.server_thread and self.server_thread.isRunning():
            has_running_processes = True
            message_parts.append("Server is still running")
            
        if self.inference_thread and self.inference_thread.isRunning():
            has_running_processes = True
            message_parts.append("Inference is still running")
        
        if self.download_thread and self.download_thread.isRunning():
            has_running_processes = True
            message_parts.append("Download is in progress")
            
        if has_running_processes:
            reply = QMessageBox.question(
                self,
                "Confirmation",
                f"{'. '.join(message_parts)}. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_all_threads()
                event.accept()
            else:
                event.ignore()
        else:
            self._stop_all_threads()
            event.accept()
    
    def _stop_all_threads(self):
        """Safely stop all running threads"""
        # Stop download thread
        if self.download_thread and self.download_thread.isRunning():
            try:
                self.download_thread.stop()
                self.download_thread.wait(1000)  # Wait up to 1 second
            except:
                pass
        
        # Stop list files thread
        if self.list_files_thread and self.list_files_thread.isRunning():
            try:
                self.list_files_thread.stop()
                self.list_files_thread.wait(500)
            except:
                pass
        
        # Stop server thread
        if self.server_thread and self.server_thread.isRunning():
            try:
                self.server_thread.stop()
                self.server_thread.wait(2000)
            except:
                pass
        
        # Stop inference thread
        if self.inference_thread and self.inference_thread.isRunning():
            try:
                self.inference_thread.stop()
                self.inference_thread.wait(1000)
            except:
                pass
        
        # Stop configure thread
        if hasattr(self, 'configure_thread') and self.configure_thread and self.configure_thread.isRunning():
            try:
                self.configure_thread.terminate()
                self.configure_thread.wait(500)
            except:
                pass
        
        # Stop build thread
        if hasattr(self, 'build_thread') and self.build_thread and self.build_thread.isRunning():
            try:
                self.build_thread.terminate()
                self.build_thread.wait(500)
            except:
                pass


def main():
    # Check and install missing dependencies first
    if not init_dependencies():
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Современный стиль
    
    window = LlamaCppGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
