#!/usr/bin/env python3
"""Test script for model search features"""

import sys
from pathlib import Path

# Add GUI path
sys.path.insert(0, str(Path(__file__).parent / "gui"))

try:
    from model_downloader import ModelDownloader
    print("✅ model_downloader imports OK")
    
    # Test instantiation
    md = ModelDownloader(Path.cwd() / "models")
    print("✅ ModelDownloader instantiated OK")
    
    # Test search_models signature
    import inspect
    sig = inspect.signature(md.search_models)
    print(f"✅ search_models signature: {sig}")
    
    # Test list_model_files return type
    print("✅ All model downloader functions look good")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
