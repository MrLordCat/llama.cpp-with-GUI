#!/usr/bin/env python3
"""Test ROCm detection"""

import os
from pathlib import Path
from gui.build_manager import BuildManager

# Create builder
bm = BuildManager(Path.cwd())

# Test ROCm detection directly
rocm_installed = bm._check_rocm()
print(f"ROCm detected: {rocm_installed}")

# Check paths
print("\nChecking ROCm paths:")
rocm_paths = [
    Path("C:/Program Files/AMD/ROCm"),
    Path("C:/Program Files (x86)/AMD/ROCm"),
    Path(os.environ.get("HIP_PATH", "")),
    Path(os.environ.get("ROCM_PATH", "")),
]
for path in rocm_paths:
    path_str = str(path)
    exists = path.exists() if path_str else False
    print(f"  {path_str}: {exists}")

# Test full prerequisites check
print("\nFull prerequisites check for ROCm backend:")
prereqs = bm.check_build_prerequisites("ROCm")
for tool, available in prereqs.items():
    status = "✅" if available else "❌"
    print(f"  {status} {tool}: {available}")
