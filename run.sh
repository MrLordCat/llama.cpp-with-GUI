#!/bin/bash
# Quick start script for llama.cpp GUI on Linux/macOS

cd "$(dirname "$0")" || exit 1

echo "🚀 Starting LLaMA.cpp GUI..."

# Use python3 on Linux (python may not exist)
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python not found! Please install Python 3.9+"
        echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "   Fedora: sudo dnf install python3 python3-pip"
        exit 1
    fi
fi

# Check for PyQt6 and install dependencies if needed
if ! $PYTHON_CMD -c "import PyQt6" 2>/dev/null; then
    echo "📦 Installing GUI dependencies..."
    PIP_ARGS="--user"
    # Handle PEP 668 externally-managed environments (Ubuntu 23+, Fedora 38+, etc.)
    if [ -f "$($PYTHON_CMD -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')/EXTERNALLY-MANAGED" ]; then
        PIP_ARGS="--user --break-system-packages"
    fi
    $PYTHON_CMD -m pip install -r gui/requirements-gui.txt $PIP_ARGS
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Try:"
        echo "   $PYTHON_CMD -m pip install -r gui/requirements-gui.txt --break-system-packages"
        exit 1
    fi
fi

# Ensure ROCm is in PATH if installed
if [ -d "/opt/rocm/bin" ] && [[ ":$PATH:" != *":/opt/rocm/bin:"* ]]; then
    export PATH="/opt/rocm/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
fi

$PYTHON_CMD run.py
