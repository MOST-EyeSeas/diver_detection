#!/usr/bin/env python3
"""
Prepare Transect Line dataset with proper train/val/test split.
Creates a held-out test set that models never see during training.
"""

import os
import sys
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

def verify_transect_data(input_dir):
    """Verify that transect data structure exists."""
    print("Verifying transect dataset structure...")
    
    # Check main directories
    required_dirs = ['images', 'bounding_box']
    for req_dir in required_dirs:
        full_path = os.path.join(input_dir, req_dir)
        if not os.path.exists(full_path):
            print(f"Error: Missing {req_dir} directory in {input_dir}")
            return False
    
    # Find timestamp directories inside images directory
    images_dir = os.path.join(input_dir, 'images')
    timestamp_dirs = []
    
    for item in os.listdir(images_dir):
        item_path = os.path.join(images_dir, item)
        if os.path.isdir(item_path):
            timestamp_dirs.append(item)
    
    if not timestamp_dirs:
        print(f"Error: No timestamp directories found in {images_dir}")
        return False
    
    print(f"Found timestamp directories: {timestamp_dirs}")
    
    # Verify each timestamp directory has corresponding structure
    for ts_dir in timestamp_dirs:
        images_ts_path = os.path.join(input_dir, 'images', ts_dir)
        bbox_ts_path = os.path.join(input_dir, 'bounding_box', ts_dir)
        
        if not os.path.exists(bbox_ts_path):
            print(f"Error: Missing bounding_box directory for {ts_dir}")
            return False
        
        # Count files
        img_count = len([f for f in os.listdir(images_ts_path) if f.lower().endswith('.png')])
        bbox_count = len([f for f in os.listdir(bbox_ts_path) if f.endswith('.txt')])
        
        print(f"  ✓ {ts_dir}: {img_count} images, {bbox_count} annotations")
    
    print("Transect dataset structure verified.")
    return True

def find_transect_pairs(input_dir, include_negatives=False):
    """Find all valid image-label pairs from transect dataset."""
    print("Finding transect image-label pairs...")
    
    pairs = []
    positive_pairs = []
    negative_pairs = []
    
    # Find timestamp directories from images directory
    images_base_dir = os.path.join(input_dir, 'images')
    timestamp_dirs = [d for d in os.listdir(images_base_dir) 
                     if os.path.isdir(os.path.join(images_base_dir, d))]
    
    # Process each timestamp directory
    for ts_dir in timestamp_dirs:
        images_dir = os.path.join(input_dir, 'images', ts_dir)
        labels_dir = os.path.join(input_dir, 'bounding_box', ts_dir)
        
        if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
            print(f"Skipping {ts_dir}: missing images or labels directory")
            continue
        
        print(f"Processing {ts_dir}...")
        
        # Find all image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
        
        for image_file in create_progress_bar(image_files, desc=f"Processing {ts_dir}", disable=False):
            # Construct paths
            image_path = os.path.join(images_dir, image_file)
            
            # Find corresponding label file
            label_name = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            if os.path.exists(label_path):
                # Check if label file has content (positive sample) or is empty (negative sample)
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    
                    if content:  # Non-empty annotation = positive sample
                        positive_pairs.append((image_path, label_path, True))
                    else:  # Empty annotation = negative sample
                        negative_pairs.append((image_path, label_path, False))
                        
                except Exception as e:
                    print(f"Warning: Could not read {label_path}: {e}")
                    continue
            else:
                # Create empty label file for negative sample if image has no annotation
                negative_pairs.append((image_path, None, False))
    
    print(f"\nDataset summary:")
    print(f"  Positive samples (with transect lines): {len(positive_pairs)}")
    print(f"  Negative samples (background only): {len(negative_pairs)}")
    
    # Combine based on include_negatives flag
    if include_negatives:
        pairs = positive_pairs + negative_pairs
        print(f"  Total samples (including negatives): {len(pairs)}")
    else:
        pairs = positive_pairs
        print(f"  Total samples (positives only): {len(pairs)}")
    
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
        for image_path, label_path, has_annotation in create_progress_bar(pairs, desc=f"Copying {split_name}", disable=no_progress):
            # Copy image
            image_name = os.path.basename(image_path)
            shutil.copy2(image_path, os.path.join(images_dir, image_name))
            
            # Handle label file
            label_name = os.path.splitext(image_name)[0] + '.txt'
            target_label_path = os.path.join(labels_dir, label_name)
            
            if label_path and os.path.exists(label_path):
                # Copy existing label file
                shutil.copy2(label_path, target_label_path)
            else:
                # Create empty label file for negative samples
                with open(target_label_path, 'w') as f:
                    pass  # Create empty file

def create_dataset_yaml(output_dir, class_names=['transect_line'], include_negatives=False):
    """Create dataset.yaml file for YOLO training."""
    dataset_type = "with negatives" if include_negatives else "positives only"
    
    yaml_content = f"""# Transect Line Dataset Configuration (Proper Train/Val/Test Split)
# Generated by prepare_transect_proper.py
# Dataset type: {dataset_type}

path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}

# Dataset info
source: "Transect Line Detection Dataset"
task: "Underwater transect line detection"
license: "Research use"

# Split info
# This dataset includes a held-out test set for unbiased evaluation
# Test set was never seen during model training or validation
# Methodology follows proven approach from diver detection project
"""
    
    yaml_path = os.path.join(output_dir, 'transect_dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created transect_dataset.yaml: {yaml_path}")
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
                file_count = len([f for f in os.listdir(full_path) if f.lower().endswith('.png')])
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
    yaml_path = os.path.join(output_dir, 'transect_dataset.yaml')
    if os.path.exists(yaml_path):
        print(f"  ✓ transect_dataset.yaml exists")
    else:
        print(f"  ✗ transect_dataset.yaml missing")
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
    parser = argparse.ArgumentParser(description='Prepare Transect Line dataset with proper train/val/test split')
    parser.add_argument('--input-dir', default='transect_result',
                       help='Directory containing transect data')
    parser.add_argument('--output-dir', default='sample_data/transect_line/dataset_proper',
                       help='Output directory for prepared dataset')
    parser.add_argument('--include-negatives', action='store_true',
                       help='Include negative samples (background images without transect lines)')
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
                       help='Only verify data structure without preparation')
    parser.add_argument('--skip-verification', action='store_true',
                       help='Skip final dataset verification')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    
    args = parser.parse_args()
    
    print("Transect Line Dataset Preparation (Proper Train/Val/Test Split)")
    print("=" * 70)
    print("This script creates a methodologically sound dataset split with:")
    print("- Training set: Used for model training")
    print("- Validation set: Used during training for model selection")  
    print("- Test set: Held-out set NEVER seen during training")
    print()
    
    if args.include_negatives:
        print("🔍 Mode: Including negative samples (background images)")
        print("   This will include images without transect lines for better model robustness")
    else:
        print("🎯 Mode: Positive samples only (images with transect lines)")
        print("   This establishes baseline performance on transect line detection")
    print()
    
    # Verify ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: Train, val, and test ratios must sum to 1.0")
        return 1
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Verify transect data structure
    if not verify_transect_data(args.input_dir):
        return 1
    
    if args.verify_only:
        print("Verification complete. Use without --verify-only to prepare dataset.")
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
        # Find image-label pairs
        pairs = find_transect_pairs(args.input_dir, args.include_negatives)
        
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
        create_dataset_yaml(args.output_dir, include_negatives=args.include_negatives)
        
        # Verify structure
        if not args.skip_verification:
            verify_dataset_structure(args.output_dir)
        
        print(f"\n✅ Transect line dataset preparation complete!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📝 Configuration file: {os.path.join(args.output_dir, 'transect_dataset.yaml')}")
        
        if args.include_negatives:
            print("\n📊 Dataset includes negative samples for improved model robustness")
        else:
            print("\n🎯 Dataset contains positive samples only (baseline approach)")
        
        print()
        print("Next steps:")
        print("1. Train YOLOv11n model:")
        print(f"   yolo train model=yolo11n.pt data={args.output_dir}/transect_dataset.yaml epochs=50 batch=16")
        print("2. Evaluate on held-out test set")
        print("3. Compare with enhancement pipeline (future)")
        
        return 0
        
    except Exception as e:
        print(f"\nError during dataset preparation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 