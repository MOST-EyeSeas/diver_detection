#!/usr/bin/env python3
"""
VDD-C Dataset Preparation Script

This script prepares the downloaded VDD-C dataset for YOLO training by:
1. Verifying downloaded files
2. Creating appropriate directory structure
3. Extracting ZIP files to the correct locations
4. Creating a YOLO-compatible dataset.yaml file

Usage:
  python prepare_vddc.py
  python prepare_vddc.py --input-dir custom/input/path --output-dir custom/output/path
  python prepare_vddc.py --verify-only
  python prepare_vddc.py --force
"""

import os
import sys
import argparse
import zipfile
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm
import random

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare the VDD-C dataset for YOLO training")
    parser.add_argument('--input-dir', type=str, default='sample_data/vdd-c/raw',
                      help='Directory containing downloaded VDD-C ZIP files')
    parser.add_argument('--output-dir', type=str, default='sample_data/vdd-c/dataset',
                      help='Directory for the prepared dataset')
    parser.add_argument('--verify-only', action='store_true',
                      help='Only verify downloaded files without extraction')
    parser.add_argument('--force', action='store_true',
                      help='Overwrite existing extracted files')
    parser.add_argument('--train-val-split', type=float, default=0.8,
                      help='Ratio for train/validation split (default: 0.8)')
    parser.add_argument('--skip-verification', action='store_true',
                      help='Skip final YOLO compatibility verification')
    
    return parser.parse_args()

def verify_downloads(input_dir):
    """Verify that the required downloaded files exist with expected sizes."""
    print("Verifying downloaded files...")
    input_dir = Path(input_dir)
    
    required_files = {
        "images.zip": 7_630_000_000,  # ~7.63 GB
        "yolo_labels.zip": 27_820_000,  # ~27.82 MB
    }
    
    missing_files = []
    for filename, expected_size in required_files.items():
        file_path = input_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
            print(f"❌ Error: {filename} not found in {input_dir}")
            continue
            
        actual_size = file_path.stat().st_size
        size_diff_percent = abs(actual_size - expected_size) / expected_size * 100
        
        if size_diff_percent > 10:  # Allow 10% difference due to approximate expected sizes
            print(f"⚠️ Warning: {filename} size is {actual_size} bytes, expected ~{expected_size} bytes")
            print(f"   Size difference: {size_diff_percent:.2f}%")
        else:
            print(f"✅ {filename} found with acceptable size")
    
    return len(missing_files) == 0

def create_directory_structure(output_dir, force=False):
    """Create the directory structure for the dataset."""
    print("Creating directory structure...")
    output_dir = Path(output_dir)
    
    # Create main directories
    directories = [
        output_dir,
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]
    
    for directory in directories:
        if directory.exists() and force:
            print(f"Removing existing directory: {directory}")
            shutil.rmtree(directory)
        
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def extract_zip_with_progress(zip_path, extract_dir):
    """Extract a ZIP file with progress tracking."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        total_size = sum(m.file_size for m in members)
        extracted_size = 0
        
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for member in members:
                zip_ref.extract(member, extract_dir)
                extracted_size += member.file_size
                pbar.update(member.file_size)

def extract_dataset(input_dir, output_dir, force=False):
    """Extract the dataset ZIP files to the appropriate locations."""
    print("Extracting dataset files...")
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create temporary extraction directories
    temp_images_dir = output_dir / "temp_images"
    temp_labels_dir = output_dir / "temp_labels"
    
    # Remove temporary directories if they exist and force is enabled
    if temp_images_dir.exists() and force:
        shutil.rmtree(temp_images_dir)
    if temp_labels_dir.exists() and force:
        shutil.rmtree(temp_labels_dir)
    
    # Create temporary directories
    temp_images_dir.mkdir(exist_ok=True)
    temp_labels_dir.mkdir(exist_ok=True)
    
    # Extract images
    print("Extracting images.zip...")
    extract_zip_with_progress(input_dir / "images.zip", temp_images_dir)
    
    # Extract labels
    print("Extracting yolo_labels.zip...")
    extract_zip_with_progress(input_dir / "yolo_labels.zip", temp_labels_dir)
    
    print("Extraction completed successfully.")
    return temp_images_dir, temp_labels_dir

def split_dataset(temp_images_dir, temp_labels_dir, output_dir, train_val_split=0.8):
    """Split the dataset into train and validation sets."""
    print(f"Organizing dataset with {train_val_split:.0%} train, {1-train_val_split:.0%} validation split...")
    
    # Get all image files
    image_files = list(temp_images_dir.glob("**/*.jpg"))
    if not image_files:
        image_files = list(temp_images_dir.glob("**/*.png"))
    
    if not image_files:
        print("❌ Error: No image files found in the extracted dataset")
        return False
    
    # Shuffle the image files for random split
    random.shuffle(image_files)
    
    # Calculate split indices
    split_idx = int(len(image_files) * train_val_split)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Found {len(image_files)} images total")
    print(f"Assigning {len(train_images)} images to training set")
    print(f"Assigning {len(val_images)} images to validation set")
    
    # Process training images and labels
    train_success = process_dataset_split(train_images, temp_images_dir, temp_labels_dir, 
                         output_dir, "train")
    
    # Process validation images and labels
    val_success = process_dataset_split(val_images, temp_images_dir, temp_labels_dir, 
                         output_dir, "val")
    
    if not train_success or not val_success:
        print("⚠️ Warning: At least one split had no matching labels found.")
        print("Please check that the label files match the image files.")
    
    return True

def process_dataset_split(image_files, temp_images_dir, temp_labels_dir, output_dir, split_name):
    """Process and copy files for a specific dataset split (train/val)."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    
    print(f"Processing {split_name} split...")
    
    labels_found = 0
    total_images = len(image_files)
    
    for img_path in tqdm(image_files, desc=f"Copying {split_name} files"):
        # Get relative path from the temp images directory
        rel_path = img_path.relative_to(temp_images_dir)
        
        # Extract image filename and directory components
        image_filename = img_path.name
        image_stem = img_path.stem
        rel_dir = str(rel_path.parent).replace('images/', '')
        
        # Build the correct label path using the image filename pattern
        # The labels are organized in yolo/train, yolo/val, yolo/test directories
        possible_label_paths = [
            temp_labels_dir / "yolo" / "train" / f"{rel_dir}_{image_stem}.txt",
            temp_labels_dir / "yolo" / "val" / f"{rel_dir}_{image_stem}.txt",
            temp_labels_dir / "yolo" / "test" / f"{rel_dir}_{image_stem}.txt",
            # Also check directories without directory prefix in case of direct matches
            temp_labels_dir / "yolo" / "train" / f"{image_stem}.txt",
            temp_labels_dir / "yolo" / "val" / f"{image_stem}.txt",
            temp_labels_dir / "yolo" / "test" / f"{image_stem}.txt"
        ]
        
        # Copy image
        shutil.copy2(img_path, images_dir / image_filename)
        
        # Check all possible label paths and copy the first one found
        label_found = False
        for label_path in possible_label_paths:
            if label_path.exists():
                shutil.copy2(label_path, labels_dir / f"{image_stem}.txt")
                labels_found += 1
                label_found = True
                break
    
    print(f"Processed {total_images} images with {labels_found} matching labels for {split_name} set")
    return labels_found > 0

def create_yaml_config(output_dir, train_val_split=0.8):
    """Create a YOLO-compatible dataset.yaml file."""
    print("Creating dataset.yaml file...")
    output_dir = Path(output_dir)
    
    # Count files in each directory
    train_images = len(list((output_dir / "images" / "train").glob("*.jpg")))
    val_images = len(list((output_dir / "images" / "val").glob("*.jpg")))
    
    # Create dataset.yaml content
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes (diver)
        'names': ['diver']
    }
    
    # Write to file
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created dataset.yaml with {train_images} training and {val_images} validation images")
    return yaml_path

def verify_yolo_compatibility(dataset_yaml_path):
    """Verify that the dataset is compatible with YOLO."""
    print("Verifying YOLO compatibility...")
    yaml_path = Path(dataset_yaml_path)
    
    if not yaml_path.exists():
        print("❌ Error: dataset.yaml not found")
        return False
    
    try:
        # Load dataset.yaml
        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in dataset_config:
                print(f"❌ Error: Missing required key '{key}' in dataset.yaml")
                return False
        
        # Verify paths exist
        base_path = Path(dataset_config['path'])
        train_path = base_path / dataset_config['train']
        val_path = base_path / dataset_config['val']
        
        if not train_path.exists():
            print(f"❌ Error: Training path {train_path} does not exist")
            return False
        
        if not val_path.exists():
            print(f"❌ Error: Validation path {val_path} does not exist")
            return False
        
        # Verify images exist
        train_images = list(train_path.glob("*.jpg"))
        if not train_images:
            train_images = list(train_path.glob("*.png"))
        
        val_images = list(val_path.glob("*.jpg"))
        if not val_images:
            val_images = list(val_path.glob("*.png"))
        
        if not train_images:
            print("❌ Error: No training images found")
            return False
        
        if not val_images:
            print("❌ Error: No validation images found")
            return False
        
        # Check all training and validation images for corresponding labels
        train_labels_path = base_path / "labels" / "train"
        val_labels_path = base_path / "labels" / "val"
        
        train_labels = list(train_labels_path.glob("*.txt"))
        val_labels = list(val_labels_path.glob("*.txt"))
        
        print(f"Found {len(train_images)} training images and {len(train_labels)} training labels")
        print(f"Found {len(val_images)} validation images and {len(val_labels)} validation labels")
        
        # Check some random training images for corresponding labels
        labels_found = 0
        sample_size = min(20, len(train_images))
        for img_path in random.sample(train_images, sample_size):
            img_name = img_path.stem
            label_path = train_labels_path / f"{img_name}.txt"
            if label_path.exists():
                labels_found += 1
        
        if labels_found == 0:
            print("⚠️ Warning: No labels found for sample training images")
        else:
            print(f"✅ Found labels for {labels_found}/{sample_size} sample training images ({labels_found/sample_size:.1%})")
        
        # Check some random validation images for corresponding labels
        val_labels_found = 0
        val_sample_size = min(20, len(val_images))
        for img_path in random.sample(val_images, val_sample_size):
            img_name = img_path.stem
            label_path = val_labels_path / f"{img_name}.txt"
            if label_path.exists():
                val_labels_found += 1
        
        if val_labels_found == 0:
            print("⚠️ Warning: No labels found for sample validation images")
        else:
            print(f"✅ Found labels for {val_labels_found}/{val_sample_size} sample validation images ({val_labels_found/val_sample_size:.1%})")
        
        # Overall assessment
        if labels_found == 0 and val_labels_found == 0:
            print("❌ Error: No labels found for any sample images, dataset is not properly prepared")
            return False
        elif (labels_found / sample_size) < 0.5 or (val_labels_found / val_sample_size) < 0.5:
            print("⚠️ Warning: Less than 50% of sample images have matching labels")
            print("Dataset may have partial annotation coverage")
        else:
            print("✅ Dataset appears to be properly annotated and YOLO-compatible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during YOLO compatibility verification: {e}")
        return False

def cleanup_temp_directories(output_dir):
    """Clean up temporary directories after processing."""
    print("Cleaning up temporary directories...")
    output_dir = Path(output_dir)
    
    temp_dirs = [
        output_dir / "temp_images",
        output_dir / "temp_labels"
    ]
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            print(f"Removing {temp_dir}")
            shutil.rmtree(temp_dir)
    
    return True

def main():
    """Main function to prepare the VDD-C dataset."""
    args = parse_args()
    
    # Verify downloaded files
    if not verify_downloads(args.input_dir):
        print("❌ Error: Required files are missing. Please download them first.")
        return 1
    
    # If verify-only, exit after verification
    if args.verify_only:
        print("✅ Verification completed. Use without --verify-only to proceed with extraction.")
        return 0
    
    # Create directory structure
    if not create_directory_structure(args.output_dir, args.force):
        print("❌ Error: Failed to create directory structure.")
        return 1
    
    # Extract dataset
    try:
        temp_images_dir, temp_labels_dir = extract_dataset(args.input_dir, args.output_dir, args.force)
        
        # Verify label directory structure
        yolo_dir = temp_labels_dir / "yolo"
        if not yolo_dir.exists():
            print("⚠️ Warning: Expected 'yolo' directory not found in labels extraction.")
            print(f"Found instead: {list(temp_labels_dir.glob('*'))}")
        else:
            print(f"✅ Found YOLO labels directory with subdirectories: {list(yolo_dir.glob('*'))}")
            print(f"Label files count: {len(list(yolo_dir.glob('**/*.txt')))}")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 1
    
    # Split dataset into train and validation sets
    if not split_dataset(temp_images_dir, temp_labels_dir, args.output_dir, args.train_val_split):
        print("❌ Error: Failed to split dataset.")
        return 1
    
    # Create dataset.yaml
    yaml_path = create_yaml_config(args.output_dir, args.train_val_split)
    
    # Verify YOLO compatibility
    if not args.skip_verification:
        if not verify_yolo_compatibility(yaml_path):
            print("⚠️ Warning: Dataset may not be fully compatible with YOLO.")
    
    # Clean up temporary directories
    cleanup_temp_directories(args.output_dir)
    
    print("\n✅ Dataset preparation completed successfully!")
    print(f"Dataset is ready at: {Path(args.output_dir).absolute()}")
    print(f"Use the following path in your YOLO configuration: {yaml_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 