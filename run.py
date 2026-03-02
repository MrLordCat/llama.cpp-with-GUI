#!/usr/bin/env python3
"""
Quick launcher for llama.cpp GUI with automatic dependency checking
Run this script to start the GUI with automatic dependency installation
"""

import sys
from pathlib import Path

# Add GUI directory to path
gui_dir = Path(__file__).parent / "gui"
sys.path.insert(0, str(gui_dir))

from dependency_checker import init_dependencies

def main():
    print("\n🚀 LLaMA.cpp GUI Launcher\n")
    
    # Initialize and check dependencies
    if not init_dependencies():
        print("❌ Failed to initialize dependencies")
        print("Please fix the issues above and try again\n")
        sys.exit(1)
    
    print("✅ All dependencies ready! Starting GUI...\n")
    
    # Import and start GUI
    try:
        from llama_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
