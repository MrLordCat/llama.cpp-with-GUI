#!/usr/bin/env python3
"""Test file sizes from HuggingFace API"""

import sys
from pathlib import Path

# Add GUI directory to path
sys.path.insert(0, str(Path(__file__).parent / "gui"))

from model_downloader import ModelDownloader


def test_file_sizes():
    """Test getting file sizes from HuggingFace"""
    print("Testing file size detection (using HfFileSystem)...")
    print()
    
    downloader = ModelDownloader(Path.cwd() / "models")
    
    # Test with GGUF repository
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    
    print(f"Repository: {repo_id}")
    print("=" * 60)
    
    try:
        files = downloader.list_model_files(repo_id)
        
        if files:
            print(f"[OK] Found {len(files)} GGUF files:")
            print()
            
            # Sort by size descending
            sorted_files = sorted(files.items(), key=lambda x: x[1], reverse=True)
            
            for filename, size in sorted_files:
                # Format size
                if size == 0:
                    size_str = "0 B"
                elif size >= 1024**3:
                    size_str = f"{size / (1024**3):.2f} GB"
                elif size >= 1024**2:
                    size_str = f"{size / (1024**2):.2f} MB"
                else:
                    size_str = f"{size} B"
                
                print(f"  {filename}")
                print(f"    Size: {size_str} ({size:,} bytes)")
                print()
                
        else:
            print("[WARNING] No GGUF files found")
                
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file_sizes()
