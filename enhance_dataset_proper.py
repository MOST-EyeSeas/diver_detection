#!/usr/bin/env python3
"""
Enhance dataset with proper train/val/test split using aneris_enhance.
Creates enhanced version while maintaining the held-out test set structure.
"""

import os
import sys
import shutil
import subprocess
import argparse
import multiprocessing
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars disabled.")

def enhance_image(input_path, output_path):
    """
    Enhance a single image using aneris_enhance.
    Returns True if successful, False otherwise.
    """
    try:
        cmd = [
            'python3', 
            'aneris_dev/python/src/underwater_enhance.py',
            str(input_path),
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        return False

def enhance_image_wrapper(args):
    """
    Wrapper function for multiprocessing compatibility.
    Unpacks arguments and calls enhance_image.
    """
    return enhance_image(*args)

def enhance_split(input_split_dir, output_split_dir, split_name, workers=4, no_progress=False):
    """
    Enhance all images in a split directory using parallel processing.
    """
    # Create output directory
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(list(Path(input_split_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_split_dir).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {input_split_dir}")
        return 0, 0
    
    print(f"Enhancing {len(image_files)} images in {split_name} split...")
    
    # Prepare arguments for parallel processing
    enhance_args = []
    for img_path in image_files:
        output_path = Path(output_split_dir) / img_path.name
        enhance_args.append((str(img_path), str(output_path)))
    
    # Process in parallel
    enhanced_count = 0
    failed_count = 0
    
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            if TQDM_AVAILABLE and not no_progress:
                # Use imap_unordered for better tqdm compatibility
                results = list(tqdm(
                    pool.imap_unordered(enhance_image_wrapper, enhance_args),
                    desc=f"Enhancing {split_name}",
                    total=len(enhance_args)
                ))
            else:
                results = pool.starmap(enhance_image, enhance_args)
        
        enhanced_count = sum(results)
        failed_count = len(results) - enhanced_count
    else:
        # Sequential processing
        for input_path, output_path in (tqdm(enhance_args, desc=f"Enhancing {split_name}") if TQDM_AVAILABLE and not no_progress else enhance_args):
            if enhance_image(input_path, output_path):
                enhanced_count += 1
            else:
                failed_count += 1
    
    # Copy failed files as-is (fallback)
    if failed_count > 0:
        print(f"Copying {failed_count} failed enhancements as original images...")
        for input_path, output_path in enhance_args:
            if not os.path.exists(output_path):
                shutil.copy2(input_path, output_path)
    
    success_rate = enhanced_count / len(image_files) if image_files else 0
    print(f"Enhanced {enhanced_count}/{len(image_files)} images ({success_rate:.1%})")
    
    return enhanced_count, len(image_files)

def copy_labels(input_dataset_dir, output_dataset_dir):
    """
    Copy label files from input to output dataset (labels don't change).
    """
    print("Copying label files...")
    
    for split in ['train', 'val', 'test']:
        input_labels_dir = os.path.join(input_dataset_dir, 'labels', split)
        output_labels_dir = os.path.join(output_dataset_dir, 'labels', split)
        
        if os.path.exists(input_labels_dir):
            os.makedirs(output_labels_dir, exist_ok=True)
            
            label_files = list(Path(input_labels_dir).glob('*.txt'))
            for label_file in label_files:
                shutil.copy2(label_file, output_labels_dir)
            
            print(f"  Copied {len(label_files)} label files for {split} split")

def create_enhanced_dataset_yaml(input_yaml_path, output_dataset_dir):
    """
    Create dataset.yaml for enhanced dataset.
    """
    output_yaml_path = os.path.join(output_dataset_dir, 'dataset_enhanced.yaml')
    
    # Read original yaml
    with open(input_yaml_path, 'r') as f:
        content = f.read()
    
    # Update path and add enhanced info
    content = content.replace(
        f"path: {os.path.dirname(input_yaml_path)}", 
        f"path: {os.path.abspath(output_dataset_dir)}"
    )
    
    # Add enhancement info
    enhanced_info = """
# Enhancement info
# Images enhanced using aneris_enhance underwater image processing
# - Red channel correction (multiply by 1.2)
# - Contrast stretching using CLAHE
# - Labels remain unchanged from original dataset
"""
    content += enhanced_info
    
    with open(output_yaml_path, 'w') as f:
        f.write(content)
    
    print(f"Created enhanced dataset.yaml: {output_yaml_path}")
    return output_yaml_path

def verify_enhanced_dataset(dataset_dir):
    """
    Verify the enhanced dataset structure matches the original.
    """
    print("\nVerifying enhanced dataset structure...")
    
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    all_good = True
    split_counts = {}
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"  ✗ Missing: {dir_path}")
            all_good = False
        else:
            if 'images' in dir_path:
                count = len([f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                split_name = dir_path.split('/')[-1]
                split_counts[f'{split_name}_images'] = count
                print(f"  ✓ {dir_path}: {count} images")
            else:
                count = len([f for f in os.listdir(full_path) if f.endswith('.txt')])
                split_name = dir_path.split('/')[-1]
                split_counts[f'{split_name}_labels'] = count
                print(f"  ✓ {dir_path}: {count} labels")
    
    # Check image-label matching
    for split in ['train', 'val', 'test']:
        img_count = split_counts.get(f'{split}_images', 0)
        lbl_count = split_counts.get(f'{split}_labels', 0)
        if img_count == lbl_count:
            print(f"  ✓ {split}: {img_count} matched pairs")
        else:
            print(f"  ⚠ {split}: {img_count} images, {lbl_count} labels (mismatch)")
            all_good = False
    
    # Check yaml
    yaml_path = os.path.join(dataset_dir, 'dataset_enhanced.yaml')
    if os.path.exists(yaml_path):
        print(f"  ✓ dataset_enhanced.yaml exists")
    else:
        print(f"  ✗ dataset_enhanced.yaml missing")
        all_good = False
    
    if all_good:
        total_images = sum(split_counts[f'{split}_images'] for split in ['train', 'val', 'test'])
        print(f"\n✅ Enhanced dataset is ready! Total: {total_images} enhanced images")
    else:
        print(f"\n❌ Enhanced dataset has issues")
    
    return all_good

def main():
    parser = argparse.ArgumentParser(description='Enhance dataset with proper train/val/test split')
    parser.add_argument('--input-dataset', default='sample_data/vdd-c/dataset_proper',
                       help='Input dataset directory (with proper split)')
    parser.add_argument('--output-dataset', default='sample_data/vdd-c/dataset_proper_enhanced',
                       help='Output enhanced dataset directory')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers for enhancement')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Which splits to enhance (default: all)')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing enhanced dataset')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    parser.add_argument('--test-only', action='store_true',
                       help='Process only a small subset for testing')
    
    args = parser.parse_args()
    
    print("Dataset Enhancement with Proper Train/Val/Test Split")
    print("=" * 55)
    print("This script enhances images while maintaining the proper")
    print("dataset split with held-out test set.")
    print()
    
    # Check input dataset
    if not os.path.exists(args.input_dataset):
        print(f"Error: Input dataset not found: {args.input_dataset}")
        print("Please run prepare_vddc_proper.py first to create the proper dataset split.")
        return 1
    
    # Check dataset.yaml
    input_yaml = os.path.join(args.input_dataset, 'dataset.yaml')
    if not os.path.exists(input_yaml):
        print(f"Error: dataset.yaml not found in {args.input_dataset}")
        return 1
    
    # Check aneris_enhance
    enhance_script = 'aneris_enhance/python/src/underwater_enhance.py'
    if not os.path.exists(enhance_script):
        print(f"Error: aneris_enhance script not found: {enhance_script}")
        print("Please ensure aneris_enhance is properly installed.")
        return 1
    
    # Check output directory
    if os.path.exists(args.output_dataset):
        if args.force:
            print(f"Removing existing enhanced dataset: {args.output_dataset}")
            shutil.rmtree(args.output_dataset)
        else:
            print(f"Error: Output directory exists: {args.output_dataset}")
            print("Use --force to overwrite.")
            return 1
    
    try:
        # Create output directory structure
        os.makedirs(args.output_dataset, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(args.output_dataset, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(args.output_dataset, 'labels', split), exist_ok=True)
        
        # Enhance each split
        total_enhanced = 0
        total_images = 0
        
        for split in args.splits:
            input_images_dir = os.path.join(args.input_dataset, 'images', split)
            output_images_dir = os.path.join(args.output_dataset, 'images', split)
            
            if os.path.exists(input_images_dir):
                enhanced, total = enhance_split(
                    input_images_dir, 
                    output_images_dir, 
                    split, 
                    args.workers, 
                    args.no_progress
                )
                total_enhanced += enhanced
                total_images += total
            else:
                print(f"Warning: {split} split not found in input dataset")
        
        # Copy labels (unchanged)
        copy_labels(args.input_dataset, args.output_dataset)
        
        # Create enhanced dataset.yaml
        create_enhanced_dataset_yaml(input_yaml, args.output_dataset)
        
        # Verify result
        verify_enhanced_dataset(args.output_dataset)
        
        # Summary
        success_rate = total_enhanced / total_images if total_images > 0 else 0
        print(f"\n✅ Enhancement complete!")
        print(f"📊 Enhanced: {total_enhanced}/{total_images} images ({success_rate:.1%})")
        print(f"📁 Output: {args.output_dataset}")
        print(f"📝 Config: {os.path.join(args.output_dataset, 'dataset_enhanced.yaml')}")
        print()
        print("Next steps:")
        print("1. Train models on enhanced dataset")
        print("2. Test inference enhancement on held-out test set")
        print("3. Compare with models trained on original dataset")
        
        return 0
        
    except Exception as e:
        print(f"\nError during enhancement: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 