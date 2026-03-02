#!/usr/bin/env python3
"""
Test script for demonstrating the automatic dependency installer GUI

This script shows what happens when a user clicks "Install Dependencies"
with missing dependencies.
"""

import sys
from pathlib import Path

# Add GUI directory to path
gui_dir = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_dir))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
from PyQt6.QtCore import Qt

from dependency_installer import DependencyManager, DependencyInstallThread


def main():
    """Test the dependency installer"""
    app = QApplication(sys.argv)
    
    # Create a simple test window
    window = QMainWindow()
    window.setWindowTitle("Dependency Installer Test")
    window.setGeometry(100, 100, 800, 600)
    
    central = QWidget()
    window.setCentralWidget(central)
    layout = QVBoxLayout(central)
    
    # Create test log
    log = QTextEdit()
    log.setReadOnly(True)
    layout.addWidget(log)
    
    # Create buttons for testing
    check_btn = QPushButton("✅ Check Dependencies")
    install_btn = QPushButton("🚀 Install Missing")
    simulate_btn = QPushButton("📋 Simulate Installation")
    
    layout.addWidget(check_btn)
    layout.addWidget(install_btn)
    layout.addWidget(simulate_btn)
    
    def check_deps():
        """Check for missing dependencies"""
        log.clear()
        log.append("📦 Checking dependencies...\n")
        
        manager = DependencyManager()
        missing = manager.get_missing_dependencies("Vulkan")
        
        if missing:
            log.append(f"❌ Missing: {', '.join(missing)}\n")
        else:
            log.append("✅ All dependencies installed!\n")
    
    def simulate_install():
        """Simulate installation process without actually installing"""
        log.clear()
        log.append("🚀 Simulating installation process...\n\n")
        
        # Simulate installation of a few dependencies
        deps = ["CMake", "Git", "MSVC Build Tools", "Vulkan SDK"]
        
        for i, dep in enumerate(deps):
            percentage = int((i / len(deps)) * 100)
            log.append(f"[{percentage:3d}%] Installing {dep}...\n")
        
        log.append("[100%] ✅ Installation complete!\n")
    
    def start_install():
        """Start actual installation thread (won't run without deps)"""
        log.clear()
        log.append("📦 Checking for missing dependencies...\n")
        
        manager = DependencyManager()
        missing = manager.get_missing_dependencies("Vulkan")
        
        if not missing:
            log.append("✅ All dependencies are already installed!\n")
            return
        
        log.append(f"⚠️ Missing: {', '.join(missing)}\n\n")
        log.append("🚀 Starting installation thread...\n")
        log.append("(This will download and install dependencies)\n\n")
        
        # Create thread
        thread = DependencyInstallThread(missing, "Windows")
        
        def on_status(msg):
            log.append(f"📍 {msg}\n")
        
        def on_progress(msg, percentage):
            log.append(f"[{percentage:3d}%] {msg}\n")
        
        def on_finished(success, msg):
            if success:
                log.append(f"\n✅ {msg}\n")
            else:
                log.append(f"\n❌ {msg}\n")
        
        thread.status_update.connect(on_status)
        thread.progress.connect(on_progress)
        thread.finished_signal.connect(on_finished)
        thread.start()
    
    check_btn.clicked.connect(check_deps)
    install_btn.clicked.connect(start_install)
    simulate_btn.clicked.connect(simulate_install)
    
    # Show initial status
    check_deps()
    
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
