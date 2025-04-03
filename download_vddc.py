#!/usr/bin/env python3
"""
VDD-C Dataset Downloader

This script downloads the Video Diver Detection Dataset (VDD-C) from the 
University of Minnesota repository. It supports selective downloads, progress tracking,
resume capability, and download verification.

Usage:
  python download_vddc.py --all
  python download_vddc.py --images
  python download_vddc.py --labels
  python download_vddc.py --images --labels
"""

import os
import sys
import argparse
import hashlib
import time
from pathlib import Path
import requests

# Conditional import of tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # Define tqdm as None if not available

# Base URL for the VDD-C dataset
BASE_URL = "https://conservancy.umn.edu/bitstream/handle/11299/219383"

# Dataset components with their URLs, expected sizes, and MD5 checksums
DATASET_FILES = {
    "images": {
        "url": f"{BASE_URL}/images.zip",
        "size": 7_630_000_000,  # Approximate size in bytes: 7.63 GB
        "md5": None,  # We don't have the official MD5, will be updated if available
        "description": "Processed images of VDD-C"
    },
    "yolo_labels": {
        "url": f"{BASE_URL}/yolo_labels.zip",
        "size": 27_820_000,  # Approximate size in bytes: 27.82 MB
        "md5": None,  # We don't have the official MD5, will be updated if available
        "description": "YOLO style labels for VDD-C"
    },
    "voc_labels": {
        "url": f"{BASE_URL}/voc_labels.zip",
        "size": 42_050_000,  # Approximate size in bytes: 42.05 MB
        "md5": None,  # We don't have the official MD5, will be updated if available
        "description": "VOC style labels for VDD-C"
    },
    "readme": {
        "url": f"{BASE_URL}/README_VDDC.txt",
        "size": 7_160,  # Approximate size in bytes: 7.16 KB
        "md5": None,  # We don't have the official MD5, will be updated if available
        "description": "README for VDD-C"
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download the VDD-C dataset")
    parser.add_argument('--all', action='store_true', help='Download all dataset components')
    parser.add_argument('--images', action='store_true', help='Download images.zip')
    parser.add_argument('--yolo-labels', action='store_true', help='Download yolo_labels.zip')
    parser.add_argument('--voc-labels', action='store_true', help='Download voc_labels.zip')
    parser.add_argument('--readme', action='store_true', help='Download README_VDDC.txt')
    parser.add_argument('--output-dir', type=str, default='sample_data/vdd-c/raw',
                      help='Directory to save the downloaded files')
    parser.add_argument('--retries', type=int, default=3,
                      help='Number of retries for failed downloads')
    parser.add_argument('--retry-delay', type=int, default=5,
                      help='Delay between retries in seconds')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars (requires tqdm library)')
    
    args = parser.parse_args()
    
    # If no specific component is selected, show help and exit
    if not (args.all or args.images or args.yolo_labels or args.voc_labels or args.readme):
        parser.print_help()
        sys.exit(1)
        
    return args

def calculate_md5(file_path, chunk_size=8192):
    """Calculate the MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def verify_download(file_path, expected_md5):
    """Verify the downloaded file using MD5 checksum."""
    if expected_md5 is None:
        print(f"No MD5 checksum available for {file_path}, skipping verification")
        return True
    
    print(f"Verifying {file_path}...")
    actual_md5 = calculate_md5(file_path)
    
    if actual_md5 == expected_md5:
        print(f"Verification successful for {file_path}")
        return True
    else:
        print(f"Verification failed for {file_path}")
        print(f"  Expected MD5: {expected_md5}")
        print(f"  Actual MD5:   {actual_md5}")
        return False

def download_file(url, output_path, expected_size=None, expected_md5=None, retries=3, retry_delay=5, no_progress=False):
    """
    Download a file with progress tracking and resume capability.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        expected_size: Expected file size in bytes (for progress bar)
        expected_md5: Expected MD5 checksum for verification
        retries: Number of retries for failed downloads
        retry_delay: Delay between retries in seconds
        no_progress: Disable progress bar
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists and get its size
    file_size = 0
    if output_path.exists():
        file_size = output_path.stat().st_size
        if expected_size and file_size >= expected_size:
            print(f"{output_path} already exists and appears complete.")
            if expected_md5:
                return verify_download(output_path, expected_md5)
            return True
    
    # Prepare for resume if file exists
    headers = {}
    mode = 'wb'
    if file_size > 0:
        headers['Range'] = f'bytes={file_size}-'
        mode = 'ab'
        print(f"Resuming download of {output_path} from {file_size} bytes")
    
    for attempt in range(retries):
        try:
            # Make a streaming request
            response = requests.get(url, headers=headers, stream=True)
            
            # Check if the server supports resume
            if file_size > 0 and response.status_code != 206:
                print("Server does not support resume. Starting from the beginning.")
                file_size = 0
                mode = 'wb'
                headers = {}
                response = requests.get(url, stream=True)
            
            total_size = int(response.headers.get('content-length', 0)) + file_size
            
            # Use expected_size if we couldn't get content-length
            if total_size == file_size:  # No content-length in response
                if expected_size:
                    total_size = expected_size
                else:
                    total_size = None  # Unknown size
            
            # Display progress bar or simple message
            use_progress_bar = TQDM_AVAILABLE and not no_progress
            
            if use_progress_bar:
                # Display progress bar using tqdm
                with open(output_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        initial=file_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=output_path.name
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Download without progress bar
                print(f"Downloading {output_path.name}...")
                downloaded_size = file_size
                with open(output_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size:
                                print(f"\rDownloaded {downloaded_size / (1024*1024):.2f} / {total_size / (1024*1024):.2f} MB", end="")
                            else:
                                print(f"\rDownloaded {downloaded_size / (1024*1024):.2f} MB", end="")
                print()  # Newline after download completes
            
            # Verify download if MD5 is provided
            if expected_md5:
                if verify_download(output_path, expected_md5):
                    return True
                else:
                    print(f"Download verification failed, retrying ({attempt+1}/{retries})...")
                    continue
            
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading {url}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {retry_delay} seconds... ({attempt+1}/{retries})")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download {url} after {retries} attempts")
                return False
    
    return False

def main():
    """Main function to download the dataset."""
    args = parse_args()
    
    # Determine which components to download
    components = []
    if args.all:
        components = list(DATASET_FILES.keys())
    else:
        if args.images:
            components.append("images")
        if args.yolo_labels:
            components.append("yolo_labels")
        if args.voc_labels:
            components.append("voc_labels")
        if args.readme:
            components.append("readme")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading VDD-C dataset components: {', '.join(components)}")
    print(f"Files will be saved to: {output_dir.absolute()}")
    
    # Download each component
    results = {}
    for component in components:
        if component not in DATASET_FILES:
            print(f"Unknown component: {component}")
            continue
        
        info = DATASET_FILES[component]
        output_path = output_dir / f"{component}.zip"
        if component == "readme":
            output_path = output_dir / "README_VDDC.txt"
        
        print(f"\nDownloading {component}: {info['description']}")
        success = download_file(
            info['url'],
            output_path,
            expected_size=info['size'],
            expected_md5=info['md5'],
            retries=args.retries,
            retry_delay=args.retry_delay,
            no_progress=args.no_progress
        )
        
        results[component] = success
    
    # Print summary
    print("\nDownload Summary:")
    all_success = True
    for component, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {component}: {status}")
        all_success = all_success and success
    
    if all_success:
        print("\nAll downloads completed successfully!")
        return 0
    else:
        print("\nSome downloads failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 