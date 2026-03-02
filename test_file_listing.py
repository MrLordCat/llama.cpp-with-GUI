#!/usr/bin/env python3
"""Test file listing with fixed RepoSibling issue"""

import sys
from pathlib import Path

# Add GUI directory to path
sys.path.insert(0, str(Path(__file__).parent / "gui"))

from model_downloader import ModelDownloader

def test_list_files():
    """Test the fixed list_model_files function"""
    print("Testing file listing with RepoSibling fix...\n")
    
    downloader = ModelDownloader(Path.cwd() / "models")
    
    # Test with a popular model
    test_repos = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-hf",
    ]
    
    for repo_id in test_repos:
        try:
            print(f"Testing repository: {repo_id}")
            files = downloader.list_model_files(repo_id)
            
            if files:
                print(f"✅ Found {len(files)} GGUF files:\n")
                for filename, size in list(files.items())[:5]:  # Show first 5
                    size_mb = size / (1024 * 1024) if size else 0
                    print(f"  • {filename} ({size_mb:.1f} MB)")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more files")
            else:
                print("⚠️ No GGUF files found in this repository")
                
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue
            
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    test_list_files()
