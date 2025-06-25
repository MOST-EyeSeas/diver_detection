#!/usr/bin/env python3
"""
Prepare VDD-C dataset with proper train/val/test split.
Creates a held-out test set that models never see during training.
"""

import os
import sys
import zipfile
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars disabled.")

def create_progress_bar(iterable, desc="Processing", disable=False):
    """Create progress bar if tqdm available."""
    if TQDM_AVAILABLE and not disable:
        return tqdm(iterable, desc=desc)
    else:
        return iterable

def verify_downloads(input_dir):
    """Verify that required files are downloaded."""
    required_files = {
        'images.zip': 'Main image dataset',
        'yolo_labels.zip': 'YOLO format labels'
    }
    
    print("Verifying downloaded files...")
    missing_files = []
    
    for filename, description in required_files.items():
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(f"  - {filename}: {description}")
        else:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename}: {size_mb:.1f} MB")
    
    if missing_files:
        print(f"\nMissing files:")
        for file_info in missing_files:
            print(file_info)
        print(f"\nPlease run download_vddc.py first:")
        print(f"  python download_vddc.py --images --yolo-labels")
        return False
    
    print("All required files found.")
    return True

def extract_with_progress(zip_path, extract_to, desc="Extracting", no_progress=False):
    """Extract ZIP file with progress bar."""
    print(f"\n{desc} {os.path.basename(zip_path)}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        
        for member in create_progress_bar(members, desc=desc, disable=no_progress):
            # Skip directories and hidden files
            if not member.filename.endswith('/') and not os.path.basename(member.filename).startswith('.'):
                zip_ref.extract(member, extract_to)

def find_image_label_pairs(images_dir, labels_dir):
    """Find all valid image-label pairs."""
    print("Finding image-label pairs...")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Find corresponding label files
    pairs = []
    label_search_dirs = []
    
    # Find all label directories
    for root, dirs, files in os.walk(labels_dir):
        if any(f.endswith('.txt') for f in files):
            label_search_dirs.append(root)
    
    print(f"Searching for labels in {len(label_search_dirs)} directories...")
    
    for image_path in create_progress_bar(image_files, desc="Matching labels", disable=False):
        # Extract image filename and directory components
        image_filename = os.path.basename(image_path)
        image_stem = os.path.splitext(image_filename)[0]
        
        # Get the relative path from images directory and extract directory name
        rel_path = os.path.relpath(image_path, images_dir)
        rel_dir = os.path.dirname(rel_path).replace('images/', '')
        
        label_found = False
        
        # Try different label naming conventions used by VDD-C
        possible_label_names = [
            f"{rel_dir}_{image_stem}.txt",  # Main pattern: directory_imagename.txt
            f"{image_stem}.txt",            # Direct match
            f"train_{image_stem}.txt",      # With train prefix
            f"val_{image_stem}.txt",        # With val prefix
            f"test_{image_stem}.txt"        # With test prefix
        ]
        
        for label_dir in label_search_dirs:
            for label_name in possible_label_names:
                label_path = os.path.join(label_dir, label_name)
                if os.path.exists(label_path):
                    pairs.append((image_path, label_path))
                    label_found = True
                    break
            if label_found:
                break
    
    print(f"Found {len(pairs)} valid image-label pairs")
    return pairs

def create_train_val_test_split(pairs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """Create train/val/test split from image-label pairs."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    print(f"\nCreating train/val/test split:")
    print(f"  Train: {train_ratio:.1%}")
    print(f"  Val:   {val_ratio:.1%}")  
    print(f"  Test:  {test_ratio:.1%}")
    print(f"  Random seed: {random_seed}")
    
    # Shuffle pairs deterministically
    random.seed(random_seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    total_count = len(shuffled_pairs)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # Remainder goes to test
    
    train_pairs = shuffled_pairs[:train_count]
    val_pairs = shuffled_pairs[train_count:train_count + val_count]
    test_pairs = shuffled_pairs[train_count + val_count:]
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_pairs)} pairs ({len(train_pairs)/total_count:.1%})")
    print(f"  Val:   {len(val_pairs)} pairs ({len(val_pairs)/total_count:.1%})")
    print(f"  Test:  {len(test_pairs)} pairs ({len(test_pairs)/total_count:.1%})")
    
    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

def copy_split_data(splits, output_dir, no_progress=False):
    """Copy image-label pairs to proper YOLO directory structure."""
    print(f"\nCopying data to YOLO directory structure...")
    
    for split_name, pairs in splits.items():
        if not pairs:
            continue
            
        print(f"\nProcessing {split_name} split ({len(pairs)} pairs)...")
        
        # Create directories
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Copy files
        for image_path, label_path in create_progress_bar(pairs, desc=f"Copying {split_name}", disable=no_progress):
            # Copy image
            image_name = os.path.basename(image_path)
            shutil.copy2(image_path, os.path.join(images_dir, image_name))
            
            # Copy label
            label_name = os.path.splitext(image_name)[0] + '.txt'
            shutil.copy2(label_path, os.path.join(labels_dir, label_name))

def create_dataset_yaml(output_dir, class_names=['diver']):
    """Create dataset.yaml file for YOLO training."""
    yaml_content = f"""# VDD-C Dataset Configuration (Proper Train/Val/Test Split)
# Generated by prepare_vddc_proper.py

path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}

# Dataset info
source: "VDD-C: A Video Dataset for Diver Classification"
url: "https://conservancy.umn.edu/handle/11299/206847"
license: "Creative Commons Attribution-ShareAlike 3.0"

# Split info
# This dataset includes a held-out test set for inference enhancement testing
# Test set was never seen during model training or validation
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml: {yaml_path}")
    return yaml_path

def verify_dataset_structure(output_dir):
    """Verify the created dataset has proper YOLO structure."""
    print(f"\nVerifying dataset structure...")
    
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    structure_valid = True
    counts = {}
    
    for dir_path in required_dirs:
        full_path = os.path.join(output_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"  ✗ Missing directory: {dir_path}")
            structure_valid = False
        else:
            if 'images' in dir_path:
                file_count = len([f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                split_name = dir_path.split('/')[-1]
                counts[f'{split_name}_images'] = file_count
                print(f"  ✓ {dir_path}: {file_count} images")
            else:
                file_count = len([f for f in os.listdir(full_path) if f.endswith('.txt')])
                split_name = dir_path.split('/')[-1]
                counts[f'{split_name}_labels'] = file_count
                print(f"  ✓ {dir_path}: {file_count} labels")
    
    # Check if image and label counts match
    for split in ['train', 'val', 'test']:
        img_count = counts.get(f'{split}_images', 0)
        lbl_count = counts.get(f'{split}_labels', 0)
        if img_count != lbl_count:
            print(f"  ⚠ Warning: {split} has {img_count} images but {lbl_count} labels")
        else:
            print(f"  ✓ {split}: {img_count} matched image-label pairs")
    
    # Check for dataset.yaml
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    if os.path.exists(yaml_path):
        print(f"  ✓ dataset.yaml exists")
    else:
        print(f"  ✗ dataset.yaml missing")
        structure_valid = False
    
    if structure_valid:
        print(f"\n✅ Dataset structure is valid and ready for YOLO training!")
        total_pairs = sum(counts[f'{split}_images'] for split in ['train', 'val', 'test'])
        print(f"📊 Total dataset: {total_pairs} image-label pairs")
        
        # Show split percentages
        for split in ['train', 'val', 'test']:
            count = counts[f'{split}_images']
            percentage = (count / total_pairs) * 100 if total_pairs > 0 else 0
            print(f"   {split.capitalize()}: {count} pairs ({percentage:.1f}%)")
    else:
        print(f"\n❌ Dataset structure has issues. Please check the errors above.")
    
    return structure_valid

def main():
    parser = argparse.ArgumentParser(description='Prepare VDD-C dataset with proper train/val/test split')
    parser.add_argument('--input-dir', default='sample_data/vdd-c/raw',
                       help='Directory containing downloaded VDD-C files')
    parser.add_argument('--output-dir', default='sample_data/vdd-c/dataset_proper',
                       help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Ratio for training set (default: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio for validation set (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Ratio for test set (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    parser.add_argument('--force', action='store_true',
                       help='Force overwrite existing output directory')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify downloads without extraction')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip final dataset verification')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    
    args = parser.parse_args()
    
    print("VDD-C Dataset Preparation (Proper Train/Val/Test Split)")
    print("=" * 60)
    print("This script creates a methodologically sound dataset split with:")
    print("- Training set: Used for model training")
    print("- Validation set: Used during training for model selection")  
    print("- Test set: Held-out set NEVER seen during training")
    print()
    print("The test set will be used for unbiased inference enhancement testing.")
    print()
    
    # Verify ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: Train, val, and test ratios must sum to 1.0")
        return 1
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        print("Please run download_vddc.py first to download the dataset.")
        return 1
    
    # Verify downloads
    if not verify_downloads(args.input_dir):
        return 1
    
    if args.verify_only:
        print("Verification complete. Use without --verify-only to extract and prepare dataset.")
        return 0
    
    # Check output directory
    if os.path.exists(args.output_dir):
        if args.force:
            print(f"Removing existing output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            print(f"Error: Output directory already exists: {args.output_dir}")
            print("Use --force to overwrite or choose a different output directory.")
            return 1
    
    try:
        # Create temporary extraction directory
        temp_dir = os.path.join(args.output_dir, 'temp_extraction')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract files
        images_zip = os.path.join(args.input_dir, 'images.zip')
        labels_zip = os.path.join(args.input_dir, 'yolo_labels.zip')
        
        extract_with_progress(images_zip, temp_dir, "Extracting images", args.no_progress)
        extract_with_progress(labels_zip, temp_dir, "Extracting labels", args.no_progress)
        
        # Find image and label files
        images_dir = os.path.join(temp_dir, 'images')
        labels_dir = os.path.join(temp_dir, 'yolo')
        
        pairs = find_image_label_pairs(images_dir, labels_dir)
        
        if not pairs:
            print("Error: No valid image-label pairs found.")
            return 1
        
        # Create splits
        splits = create_train_val_test_split(
            pairs, 
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio,
            args.random_seed
        )
        
        # Copy data to proper structure
        copy_split_data(splits, args.output_dir, args.no_progress)
        
        # Create dataset.yaml
        create_dataset_yaml(args.output_dir)
        
        # Clean up temp directory
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        # Verify structure
        if not args.skip_verification:
            verify_dataset_structure(args.output_dir)
        
        print(f"\n✅ Dataset preparation complete!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📝 Configuration file: {os.path.join(args.output_dir, 'dataset.yaml')}")
        print()
        print("Next steps:")
        print("1. Train models using train+val sets")
        print("2. Test inference enhancement on held-out test set")
        print("3. Compare enhancement benefits on unseen data")
        
        return 0
        
    except Exception as e:
        print(f"\nError during dataset preparation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 