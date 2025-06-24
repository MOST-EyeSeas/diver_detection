#!/usr/bin/env python3
"""
Batch Dataset Enhancement Script

This script applies underwater image enhancement to the entire VDD-C dataset
using the aneris_enhance algorithm. It processes both training and validation
images while preserving the YOLO dataset structure.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Conditional import of tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # Define tqdm as None if not available

# Add the aneris_enhance module to the path
sys.path.append('aneris_enhance/python/src')

try:
    from image_processor import enhance_image
    ENHANCEMENT_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try to import the enhancement function directly
        import cv2
        import numpy as np
        
        def enhance_image(image):
            """
            Simple underwater enhancement function as fallback
            Red channel correction and contrast stretching
            """
            # Red channel correction
            enhanced = image.copy()
            enhanced[:, :, 2] = cv2.multiply(enhanced[:, :, 2], 1.2)  # Boost red channel
            
            # Contrast stretching
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        
        ENHANCEMENT_AVAILABLE = True
    except ImportError:
        ENHANCEMENT_AVAILABLE = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhance VDD-C dataset images for underwater detection")
    parser.add_argument('--input-dataset', type=str, default='sample_data/vdd-c/dataset',
                      help='Path to the original dataset directory')
    parser.add_argument('--output-dataset', type=str, default='sample_data/vdd-c/dataset_enhanced',
                      help='Path to save the enhanced dataset')
    parser.add_argument('--workers', type=int, default=min(4, cpu_count()),
                      help='Number of parallel workers for processing')
    parser.add_argument('--test-only', action='store_true',
                      help='Process only a small subset for testing')
    parser.add_argument('--force', action='store_true',
                      help='Overwrite existing enhanced dataset')
    parser.add_argument('--skip-validation', action='store_true',
                      help='Skip validation images, process only training images')
    parser.add_argument('--no-progress', action='store_true',
                      help='Disable progress bars (useful if tqdm is not available)')
    
    return parser.parse_args()

def enhance_single_image(args_tuple):
    """
    Enhance a single image file.
    
    Args:
        args_tuple: (input_path, output_path, use_aneris)
    
    Returns:
        tuple: (success, input_path, error_message)
    """
    input_path, output_path, use_aneris = args_tuple
    
    try:
        if use_aneris:
            # Use aneris_enhance via subprocess (more reliable)
            import subprocess
            result = subprocess.run([
                'python3', 'aneris_enhance/python/src/underwater_enhance.py',
                str(input_path), str(output_path)
            ], capture_output=True, text=True, cwd='/workspaces/diver_detection')
            
            if result.returncode != 0:
                return False, input_path, f"aneris_enhance failed: {result.stderr}"
        else:
            # Use fallback enhancement
            import cv2
            image = cv2.imread(str(input_path))
            if image is None:
                return False, input_path, "Could not read image"
            
            enhanced = enhance_image(image)
            cv2.imwrite(str(output_path), enhanced)
        
        return True, input_path, None
        
    except Exception as e:
        return False, input_path, str(e)

def copy_labels(input_dataset, output_dataset, no_progress=False):
    """Copy label files from input to output dataset (unchanged)."""
    input_labels = Path(input_dataset) / 'labels'
    output_labels = Path(output_dataset) / 'labels'
    
    if input_labels.exists():
        print("Copying label files...")
        
        # Count total files for progress tracking
        if TQDM_AVAILABLE and not no_progress:
            label_files = list(input_labels.rglob('*.txt'))
            print(f"Found {len(label_files)} label files to copy...")
        
        shutil.copytree(input_labels, output_labels, dirs_exist_ok=True)
        print(f"✅ Labels copied from {input_labels} to {output_labels}")
    else:
        print(f"⚠️  No labels directory found at {input_labels}")

def create_enhanced_dataset_yaml(input_dataset, output_dataset):
    """Create dataset.yaml file for the enhanced dataset."""
    input_yaml = Path(input_dataset) / 'dataset.yaml'
    output_yaml = Path(output_dataset) / 'dataset_enhanced.yaml'
    
    if input_yaml.exists():
        # Read original yaml and modify paths
        with open(input_yaml, 'r') as f:
            content = f.read()
        
        # Replace paths to point to enhanced dataset
        enhanced_content = content.replace(
            str(Path(input_dataset)),
            str(Path(output_dataset))
        )
        
        # Add comment about enhancement
        enhanced_content = f"# Enhanced underwater dataset using aneris_enhance\n# Original: {input_dataset}\n\n" + enhanced_content
        
        with open(output_yaml, 'w') as f:
            f.write(enhanced_content)
        
        print(f"✅ Created enhanced dataset configuration: {output_yaml}")
    else:
        print(f"⚠️  No dataset.yaml found at {input_yaml}")

def process_image_set(input_dir, output_dir, set_name, workers=4, test_only=False, use_aneris=True, no_progress=False):
    """Process a set of images (train or val)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"⚠️  Input directory {input_path} does not exist")
        return 0, 0
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if test_only:
        image_files = image_files[:20]  # Process only first 20 images for testing
    
    total_images = len(image_files)
    print(f"Processing {total_images} {set_name} images...")
    
    # Prepare arguments for parallel processing
    process_args = []
    for img_file in image_files:
        output_file = output_path / img_file.name
        process_args.append((img_file, output_file, use_aneris))
    
    # Process images in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    # Use progress bar if available and not disabled
    use_progress_bar = TQDM_AVAILABLE and not no_progress
    
    if use_progress_bar:
        progress_bar = tqdm(
            total=total_images,
            desc=f"Enhancing {set_name}",
            unit="img",
            unit_scale=False
        )
    
    if workers > 1:
        with Pool(workers) as pool:
            if use_progress_bar:
                # Use imap for progress tracking with multiprocessing
                results = []
                for result in pool.imap(enhance_single_image, process_args):
                    results.append(result)
                    progress_bar.update(1)
            else:
                results = pool.map(enhance_single_image, process_args)
    else:
        # Sequential processing for debugging
        results = []
        for args in process_args:
            result = enhance_single_image(args)
            results.append(result)
            if use_progress_bar:
                progress_bar.update(1)
    
    if use_progress_bar:
        progress_bar.close()
    
    # Count results
    for success, img_path, error in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(f"❌ Failed to enhance {img_path}: {error}")
    
    elapsed = time.time() - start_time
    fps = total_images / elapsed if elapsed > 0 else 0
    
    print(f"✅ {set_name} processing completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed:.1f}s ({fps:.1f} FPS)")
    
    return successful, failed

def main():
    """Main function."""
    args = parse_args()
    
    if not ENHANCEMENT_AVAILABLE:
        print("❌ Enhancement modules not available. Please check aneris_enhance installation.")
        return 1
    
    input_dataset = Path(args.input_dataset)
    output_dataset = Path(args.output_dataset)
    
    # Validate input dataset
    if not input_dataset.exists():
        print(f"❌ Input dataset not found: {input_dataset}")
        return 1
    
    # Check if output already exists
    if output_dataset.exists() and not args.force:
        print(f"❌ Output dataset already exists: {output_dataset}")
        print("Use --force to overwrite existing dataset")
        return 1
    
    # Create output directory structure
    output_dataset.mkdir(parents=True, exist_ok=True)
    (output_dataset / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dataset / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting dataset enhancement...")
    print(f"  Input: {input_dataset}")
    print(f"  Output: {output_dataset}")
    print(f"  Workers: {args.workers}")
    print(f"  Test mode: {args.test_only}")
    print(f"  Progress bars: {'Enabled' if TQDM_AVAILABLE and not args.no_progress else 'Disabled'}")
    
    # Determine enhancement method
    use_aneris = True
    try:
        # Test aneris_enhance
        import subprocess
        result = subprocess.run([
            'python3', 'aneris_enhance/python/src/underwater_enhance.py', '--help'
        ], capture_output=True, text=True, cwd='/workspaces/diver_detection')
        if result.returncode != 0:
            use_aneris = False
            print("⚠️  aneris_enhance not working, using fallback enhancement")
    except:
        use_aneris = False
        print("⚠️  Using fallback enhancement method")
    
    total_successful = 0
    total_failed = 0
    
    # Process training images
    train_input = input_dataset / 'images' / 'train'
    train_output = output_dataset / 'images' / 'train'
    train_success, train_failed = process_image_set(
        train_input, train_output, 'training', 
        workers=args.workers, test_only=args.test_only, use_aneris=use_aneris, no_progress=args.no_progress
    )
    total_successful += train_success
    total_failed += train_failed
    
    # Process validation images (unless skipped)
    if not args.skip_validation:
        val_input = input_dataset / 'images' / 'val'
        val_output = output_dataset / 'images' / 'val'
        val_success, val_failed = process_image_set(
            val_input, val_output, 'validation',
            workers=args.workers, test_only=args.test_only, use_aneris=use_aneris, no_progress=args.no_progress
        )
        total_successful += val_success
        total_failed += val_failed
    
    # Copy labels (unchanged)
    copy_labels(input_dataset, output_dataset, no_progress=args.no_progress)
    
    # Create enhanced dataset.yaml
    create_enhanced_dataset_yaml(input_dataset, output_dataset)
    
    print(f"\n🎉 Dataset enhancement completed!")
    print(f"  Total images processed: {total_successful + total_failed}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"  Enhanced dataset saved to: {output_dataset}")
    
    if total_failed > 0:
        print(f"⚠️  {total_failed} images failed to process")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 