"""
Model downloader module for HuggingFace and other sources
"""

import os
import time
import requests
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from huggingface_hub import hf_hub_download, list_repo_files, HfApi, HfFileSystem
from PyQt6.QtCore import QThread, pyqtSignal


class DownloadThread(QThread):
    """Thread for downloading files in background with progress tracking"""
    # Use object type for large integers that may exceed int32
    progress = pyqtSignal(object, object, float, float)  # downloaded, total, speed_mbps, eta_seconds
    status = pyqtSignal(str)
    finished_signal = pyqtSignal(str)  # file_path
    error_signal = pyqtSignal(str)
    
    def __init__(self, repo_id: str, filename: str, local_dir: Path):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.local_dir = local_dir
        self.should_stop = False
        self._is_cancelled = False
        self._response = None
        
    def run(self):
        try:
            self.status.emit(f"Preparing download: {self.filename}...")
            print(f"[DEBUG] Starting download: {self.filename}")
            
            # Build direct download URL for HuggingFace
            # Format: https://huggingface.co/{repo_id}/resolve/main/{filename}
            download_url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
            print(f"[DEBUG] URL: {download_url}")
            
            # Create output path
            output_path = self.local_dir / self.filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for partial download to resume
            downloaded_size = 0
            if output_path.exists():
                downloaded_size = output_path.stat().st_size
                print(f"[DEBUG] Existing file size: {downloaded_size}")
            
            # Setup headers - important for HuggingFace
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) llama-cpp-gui/1.0'
            }
            if downloaded_size > 0:
                headers['Range'] = f'bytes={downloaded_size}-'
                self.status.emit(f"Resuming download from {downloaded_size / 1024 / 1024:.1f} MB...")
            
            # Start download with streaming and follow redirects
            self.status.emit(f"Connecting to HuggingFace...")
            print(f"[DEBUG] Connecting...")
            self._response = requests.get(
                download_url, 
                headers=headers, 
                stream=True, 
                timeout=60,
                allow_redirects=True
            )
            
            print(f"[DEBUG] Response status: {self._response.status_code}")
            print(f"[DEBUG] Response headers: {dict(self._response.headers)}")
            
            # Handle response
            if self._response.status_code == 416:  # Range not satisfiable - file complete
                self.status.emit(f"[OK] File already downloaded: {self.filename}")
                self.finished_signal.emit(str(output_path))
                return
            
            if self._response.status_code == 401:
                self.error_signal.emit("Authentication required. This model may require login to HuggingFace.")
                return
            
            if self._response.status_code == 404:
                self.error_signal.emit(f"File not found: {self.filename}")
                return
            
            self._response.raise_for_status()
            
            # Get total size
            if 'content-range' in self._response.headers:
                # Resuming: Content-Range: bytes 1234-5678/9999
                total_size = int(self._response.headers['content-range'].split('/')[-1])
            else:
                content_length = self._response.headers.get('content-length', '0')
                total_size = int(content_length) + downloaded_size
            
            print(f"[DEBUG] Total size: {total_size}")
            
            if total_size == 0:
                # Try to continue without known size (show indeterminate progress)
                self.status.emit(f"Downloading {self.filename} (size unknown)...")
                print(f"[DEBUG] Unknown size, continuing anyway")
            else:
                self.status.emit(f"Downloading {self.filename} ({total_size / 1024 / 1024 / 1024:.2f} GB)...")
            
            # Send initial progress
            self.progress.emit(downloaded_size, total_size, 0.0, 0.0)
            print(f"[DEBUG] Initial progress emitted: {downloaded_size}/{total_size}")
            
            # Download with progress tracking
            chunk_size = 1024 * 256  # 256 KB chunks for more frequent updates
            start_time = time.time()
            last_update_time = start_time
            bytes_since_last_update = 0
            chunk_count = 0
            
            mode = 'ab' if downloaded_size > 0 else 'wb'
            with open(output_path, mode) as f:
                for chunk in self._response.iter_content(chunk_size=chunk_size):
                    if self.should_stop:
                        self._is_cancelled = True
                        self.status.emit("Download cancelled by user")
                        if self._response:
                            self._response.close()
                        return
                    
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        bytes_since_last_update += len(chunk)
                        chunk_count += 1
                        
                        # Update progress every 0.3 seconds
                        current_time = time.time()
                        time_diff = current_time - last_update_time
                        
                        if time_diff >= 0.3:
                            # Calculate speed (MB/s)
                            speed_mbps = (bytes_since_last_update / 1024 / 1024) / time_diff if time_diff > 0 else 0
                            
                            # Calculate ETA
                            if total_size > 0:
                                remaining_bytes = total_size - downloaded_size
                                if speed_mbps > 0:
                                    eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024)
                                else:
                                    eta_seconds = 0
                            else:
                                eta_seconds = 0
                            
                            print(f"[DEBUG] Progress: {downloaded_size}/{total_size} ({speed_mbps:.1f} MB/s)")
                            self.progress.emit(downloaded_size, total_size, speed_mbps, eta_seconds)
                            
                            last_update_time = current_time
                            bytes_since_last_update = 0
            
            if not self.should_stop:
                elapsed = time.time() - start_time
                avg_speed = ((total_size - (output_path.stat().st_size if output_path.exists() else 0)) / 1024 / 1024) / elapsed if elapsed > 0 else 0
                self.progress.emit(total_size, total_size, avg_speed, 0)
                self.status.emit(f"[OK] Downloaded: {self.filename} (avg {avg_speed:.1f} MB/s)")
                self.finished_signal.emit(str(output_path))
            
        except requests.exceptions.Timeout:
            if not self.should_stop:
                self.error_signal.emit("Connection timeout. Please check your internet connection.")
        except requests.exceptions.ConnectionError:
            if not self.should_stop:
                self.error_signal.emit("Connection failed. Please check your internet connection.")
        except requests.exceptions.RequestException as e:
            if not self.should_stop:
                self.error_signal.emit(f"Network error: {str(e)}")
        except Exception as e:
            if not self.should_stop:
                self.error_signal.emit(f"Download error: {str(e)}")
        finally:
            if self._response:
                try:
                    self._response.close()
                except:
                    pass
            
    def stop(self):
        """Stop the download"""
        self.should_stop = True
        if self._response:
            try:
                self._response.close()
            except:
                pass
        self.should_stop = True
    
    def is_cancelled(self) -> bool:
        return self._is_cancelled


class ListFilesThread(QThread):
    """Thread for getting file list in background (prevents UI freeze)"""
    finished_signal = pyqtSignal(dict)  # {filename: size}
    error_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    
    def __init__(self, repo_id: str):
        super().__init__()
        self.repo_id = repo_id
        self.should_stop = False
        
    def run(self):
        try:
            self.status_signal.emit(f"Loading files from {self.repo_id}...")
            
            # Use HfFileSystem to get accurate file sizes
            fs = HfFileSystem()
            gguf_files = {}
            
            try:
                # Get all GGUF files in the repository
                pattern = f"{self.repo_id}/*.gguf"
                self.status_signal.emit("Searching for .gguf files...")
                file_paths = fs.glob(pattern)
                
                total_files = len(file_paths)
                for i, file_path in enumerate(file_paths):
                    if self.should_stop:
                        return
                    
                    try:
                        # Get file info using HfFileSystem
                        info = fs.info(file_path)
                        if info and 'size' in info:
                            filename = file_path.split('/')[-1]
                            size = info['size']
                            gguf_files[filename] = int(size) if size else 0
                            
                            # Update status every few files
                            if i % 3 == 0:
                                self.status_signal.emit(f"Found {i+1}/{total_files} files...")
                    except Exception:
                        continue
                
                if gguf_files:
                    self.finished_signal.emit(gguf_files)
                    return
                    
            except Exception as fs_error:
                self.status_signal.emit("Trying alternative method...")
            
            # Fallback to repo_info if HfFileSystem fails
            try:
                hf_api = HfApi()
                repo_info = hf_api.repo_info(repo_id=self.repo_id)
                
                if hasattr(repo_info, 'siblings') and repo_info.siblings:
                    for file_info in repo_info.siblings:
                        if self.should_stop:
                            return
                        
                        filename = getattr(file_info, 'rfilename', None) or \
                                  getattr(file_info, 'path', None) or \
                                  getattr(file_info, 'filename', None)
                        
                        if filename and filename.endswith('.gguf'):
                            size = getattr(file_info, 'size', 0) or 0
                            gguf_files[filename] = int(size) if size else 0
                
                self.finished_signal.emit(gguf_files)
                
            except Exception as e:
                self.error_signal.emit(f"Failed to get file list: {str(e)}")
                
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            
    def stop(self):
        self.should_stop = True


class ModelDownloader:
    """Class for downloading models"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_api = HfApi()
        
    def search_models(
        self,
        query: str = "gguf",
        sort: str = "downloads",
        limit: int = 50,
        filter_tag: str = "gguf",
        min_date: Optional[str] = None  # Format: "2025-07" (YYYY-MM)
    ) -> List[Dict[str, Any]]:
        """
        Search for models on HuggingFace
        
        Args:
            query: Search query
            sort: Sort by ('downloads', 'likes', 'trending', 'updated')
            limit: Maximum number of results
            filter_tag: Filter by tag
            min_date: Minimum date filter (format: "2025-07")
            
        Returns:
            List of models with information
        """
        try:
            models = self.hf_api.list_models(
                search=query,
                filter=filter_tag,
                sort=sort,
                limit=limit,
                full=True
            )
            
            results = []
            for model in models:
                # Filter by date if specified
                if min_date:
                    updated = str(getattr(model, 'lastModified', ''))
                    if updated and not updated.startswith(min_date) and updated < min_date:
                        continue
                
                results.append({
                    "id": model.id,
                    "author": model.author or "Unknown",
                    "model_name": model.id.split("/")[-1] if "/" in model.id else model.id,
                    "downloads": getattr(model, 'downloads', 0),
                    "likes": getattr(model, 'likes', 0),
                    "updated": str(getattr(model, 'lastModified', '')),
                    "tags": getattr(model, 'tags', []),
                })
                
            return results
            
        except Exception as e:
            raise Exception(f"Error searching models: {str(e)}")
            
    def get_popular_gguf_models(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get popular GGUF models
        
        Args:
            limit: Number of models to retrieve
            
        Returns:
            List of popular models
        """
        return self.search_models(
            query="gguf",
            sort="downloads",
            limit=limit,
            filter_tag="gguf"
        )
        
    def list_model_files(self, repo_id: str) -> Dict[str, int]:
        """Get list of files in HuggingFace repository with their sizes
        
        Returns:
            Dictionary with filename as key and size in bytes as value
        """
        try:
            # Use HfFileSystem to get accurate file sizes
            fs = HfFileSystem()
            gguf_files = {}
            
            try:
                # Get all GGUF files in the repository
                pattern = f"{repo_id}/*.gguf"
                file_paths = fs.glob(pattern)
                
                for file_path in file_paths:
                    try:
                        # Get file info using HfFileSystem
                        info = fs.info(file_path)
                        if info and 'size' in info:
                            # Extract just the filename
                            filename = file_path.split('/')[-1]
                            size = info['size']
                            gguf_files[filename] = int(size) if size else 0
                    except Exception as e:
                        # Skip files that can't be accessed
                        continue
                
                return gguf_files
                
            except Exception as fs_error:
                # Fallback to repo_info if HfFileSystem fails
                # This handles older API versions or special cases
                repo_info = self.hf_api.repo_info(repo_id=repo_id)
                gguf_files = {}
                
                if hasattr(repo_info, 'siblings') and repo_info.siblings:
                    for file_info in repo_info.siblings:
                        # Get filename - try different attribute names
                        filename = None
                        if hasattr(file_info, 'rfilename'):
                            filename = file_info.rfilename
                        elif hasattr(file_info, 'path'):
                            filename = file_info.path
                        elif hasattr(file_info, 'filename'):
                            filename = file_info.filename
                        else:
                            # Try converting to string
                            filename = str(file_info)
                        
                        # Check if it's a GGUF file
                        if filename and filename.endswith('.gguf'):
                            # Get size in bytes - try multiple attributes
                            size = 0
                            if hasattr(file_info, 'size') and file_info.size:
                                size = file_info.size
                            elif hasattr(file_info, 'blob_size') and file_info.blob_size:
                                size = file_info.blob_size
                            elif hasattr(file_info, 'lfs') and file_info.lfs:
                                # For LFS files
                                lfs = file_info.lfs
                                if hasattr(lfs, 'size'):
                                    size = lfs.size
                            
                            gguf_files[filename] = int(size) if size else 0
                
                return gguf_files
                
        except Exception as e:
            raise Exception(f"Failed to get file list: {str(e)}")
            
    def download_model(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Download model from HuggingFace
        
        Args:
            repo_id: Repository ID (e.g., "TheBloke/Llama-2-7B-Chat-GGUF")
            filename: Filename to download (optional)
            progress_callback: Function to track progress
            
        Returns:
            Path to downloaded file
        """
        try:
            # If filename not specified, try to find suitable one
            if not filename:
                files = self.list_model_files(repo_id)
                if not files:
                    raise Exception("No .gguf files found in repository")
                    
                # Select first file (largest) or can add selection logic
                # files is now a dict, get first key
                filename = list(files.keys())[0]
                
            # Download file
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            return file_path
            
        except Exception as e:
            raise Exception(f"Model download error: {str(e)}")
            
    def download_from_url(
        self,
        url: str,
        filename: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        Загрузка модели по прямой ссылке
        
        Args:
            url: URL для загрузки
            filename: Имя файла для сохранения
            progress_callback: Функция для отслеживания прогресса
            
        Returns:
            Путь к загруженному файлу
        """
        output_path = self.models_dir / filename
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
                            
            return str(output_path)
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise Exception(f"Ошибка загрузки: {str(e)}")
