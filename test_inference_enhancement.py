#!/usr/bin/env python3
"""
Test inference-time enhancement benefits.
Compares model performance with and without aneris_enhance preprocessing during inference.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

def enhance_image(input_path, output_path):
    """
    Enhance a single image using aneris_enhance.
    Returns True if successful, False otherwise.
    """
    try:
        cmd = [
            'python3', 
            'aneris_enhance/python/src/underwater_enhance.py',
            str(input_path),
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"Enhancement failed for {input_path}: {e}")
        return False

def create_enhanced_dataset(original_images_dir, enhanced_images_dir):
    """
    Create enhanced version of validation dataset for testing.
    """
    os.makedirs(enhanced_images_dir, exist_ok=True)
    
    image_files = list(Path(original_images_dir).glob('*.jpg')) + list(Path(original_images_dir).glob('*.png'))
    
    enhanced_count = 0
    total_count = len(image_files)
    
    print(f"Enhancing {total_count} validation images...")
    
    for img_path in tqdm(image_files, desc="Enhancing images"):
        output_path = Path(enhanced_images_dir) / img_path.name
        if enhance_image(img_path, output_path):
            enhanced_count += 1
        else:
            # Copy original if enhancement fails
            shutil.copy2(img_path, output_path)
    
    print(f"Enhanced: {enhanced_count}/{total_count} images ({enhanced_count/total_count*100:.1f}%)")
    return enhanced_count, total_count

def run_validation(model_path, images_dir, labels_dir, dataset_yaml_path, test_name):
    """
    Run YOLO validation on a dataset and return metrics.
    """
    print(f"\nRunning validation: {test_name}")
    print(f"Model: {model_path}")
    print(f"Images: {images_dir}")
    
    try:
        model = YOLO(model_path)
        
        # Create temporary dataset.yaml for this test
        temp_yaml = f"temp_dataset_{test_name.lower().replace(' ', '_')}.yaml"
        
        with open(dataset_yaml_path, 'r') as f:
            content = f.read()
        
        # Replace validation path
        content = content.replace('val: images/val', f'val: {images_dir}')
        
        with open(temp_yaml, 'w') as f:
            f.write(content)
        
        # Run validation
        results = model.val(data=temp_yaml, split='val', save=False, plots=False)
        
        # Extract key metrics
        metrics = {
            'test_name': test_name,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr)
        }
        
        # Cleanup
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)
        
        return metrics
        
    except Exception as e:
        print(f"Validation failed for {test_name}: {e}")
        return None

def test_single_model(model_path, dataset_dir, model_name):
    """
    Test inference enhancement for a single model.
    """
    print(f"\n{'='*60}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*60}")
    
    # Paths
    original_images = os.path.join(dataset_dir, 'images', 'val')
    labels_dir = os.path.join(dataset_dir, 'labels', 'val')
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')
    
    # Create enhanced validation set
    enhanced_images = f"temp_enhanced_val_{model_name.lower().replace(' ', '_')}"
    
    try:
        enhanced_count, total_count = create_enhanced_dataset(original_images, enhanced_images)
        
        if enhanced_count == 0:
            print("No images were enhanced successfully. Skipping this test.")
            return None
        
        # Test 1: Original images
        print("\n" + "-"*40)
        print("Testing with ORIGINAL images")
        print("-"*40)
        original_metrics = run_validation(model_path, original_images, labels_dir, dataset_yaml, f"{model_name} Original")
        
        # Test 2: Enhanced images
        print("\n" + "-"*40)
        print("Testing with ENHANCED images")
        print("-"*40)
        enhanced_metrics = run_validation(model_path, enhanced_images, labels_dir, dataset_yaml, f"{model_name} Enhanced")
        
        # Calculate improvements
        if original_metrics and enhanced_metrics:
            improvements = {}
            for key in ['mAP50', 'mAP50-95', 'precision', 'recall']:
                original_val = original_metrics[key]
                enhanced_val = enhanced_metrics[key]
                delta = enhanced_val - original_val
                percent_change = (delta / original_val) * 100 if original_val > 0 else 0
                improvements[f'{key}_delta'] = delta
                improvements[f'{key}_percent'] = percent_change
            
            return {
                'model_name': model_name,
                'original': original_metrics,
                'enhanced': enhanced_metrics,
                'improvements': improvements,
                'enhancement_success_rate': enhanced_count / total_count
            }
        
        return None
        
    finally:
        # Cleanup enhanced images
        if os.path.exists(enhanced_images):
            shutil.rmtree(enhanced_images)

def print_results(results):
    """
    Print comprehensive results comparison.
    """
    print("\n" + "="*80)
    print("INFERENCE ENHANCEMENT RESULTS SUMMARY")
    print("="*80)
    
    if not results:
        print("No results to display.")
        return
    
    # Create comparison table
    comparison_data = []
    
    for result in results:
        if result is None:
            continue
            
        model_name = result['model_name']
        original = result['original']
        enhanced = result['enhanced']
        improvements = result['improvements']
        
        comparison_data.append({
            'Model': model_name,
            'Dataset': 'Original',
            'mAP50': f"{original['mAP50']:.3f}",
            'mAP50-95': f"{original['mAP50-95']:.3f}",
            'Precision': f"{original['precision']:.3f}",
            'Recall': f"{original['recall']:.3f}"
        })
        
        comparison_data.append({
            'Model': model_name,
            'Dataset': 'Enhanced',
            'mAP50': f"{enhanced['mAP50']:.3f}",
            'mAP50-95': f"{enhanced['mAP50-95']:.3f}",
            'Precision': f"{enhanced['precision']:.3f}",
            'Recall': f"{enhanced['recall']:.3f}"
        })
        
        # Print improvement summary
        print(f"\n{model_name} Enhancement Impact:")
        print(f"  mAP50:    {original['mAP50']:.3f} → {enhanced['mAP50']:.3f} (Δ{improvements['mAP50_delta']:+.3f}, {improvements['mAP50_percent']:+.1f}%)")
        print(f"  mAP50-95: {original['mAP50-95']:.3f} → {enhanced['mAP50-95']:.3f} (Δ{improvements['mAP50-95_delta']:+.3f}, {improvements['mAP50-95_percent']:+.1f}%)")
        print(f"  Precision: {original['precision']:.3f} → {enhanced['precision']:.3f} (Δ{improvements['precision_delta']:+.3f}, {improvements['precision_percent']:+.1f}%)")
        print(f"  Recall:   {original['recall']:.3f} → {enhanced['recall']:.3f} (Δ{improvements['recall_delta']:+.3f}, {improvements['recall_percent']:+.1f}%)")
        print(f"  Enhancement Success Rate: {result['enhancement_success_rate']:.1%}")
    
    # Create DataFrame for easy viewing
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print(f"\nDetailed Comparison Table:")
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_filename = "inference_enhancement_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to: {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='Test inference-time enhancement benefits')
    parser.add_argument('--models', nargs='+', 
                       default=['runs/comparison/v11n_original/weights/best.pt',
                               'runs/comparison/v12n_original/weights/best.pt'],
                       help='Paths to model weights to test')
    parser.add_argument('--dataset', default='sample_data/vdd-c/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model-names', nargs='+',
                       default=['YOLOv11n Original', 'YOLOv12n Original'],
                       help='Names for the models (for reporting)')
    
    args = parser.parse_args()
    
    print("INFERENCE ENHANCEMENT TESTING")
    print("="*50)
    print("This script tests whether applying aneris_enhance preprocessing")
    print("during inference improves model performance.")
    print()
    print("Test methodology:")
    print("1. Load models trained on original (non-enhanced) dataset")
    print("2. Run validation on original validation images")
    print("3. Run validation on enhanced validation images") 
    print("4. Compare results to quantify enhancement benefits")
    print()
    
    # Verify inputs
    if len(args.models) != len(args.model_names):
        print("Error: Number of models must match number of model names")
        return
    
    # Check if models exist
    missing_models = []
    for model_path in args.models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print("Error: The following model files were not found:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease check that training has completed and model weights exist.")
        return
    
    # Check dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found: {args.dataset}")
        return
    
    # Check aneris_enhance
    enhance_script = 'aneris_enhance/python/src/underwater_enhance.py'
    if not os.path.exists(enhance_script):
        print(f"Error: aneris_enhance script not found: {enhance_script}")
        return
    
    print(f"Testing {len(args.models)} models...")
    print(f"Dataset: {args.dataset}")
    print()
    
    # Run tests
    results = []
    for model_path, model_name in zip(args.models, args.model_names):
        result = test_single_model(model_path, args.dataset, model_name)
        results.append(result)
    
    # Print comprehensive results
    print_results(results)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("1. If enhancement shows benefits, test on external underwater video")
    print("2. Consider extended training (100 epochs) for models showing promise") 
    print("3. Test inference enhancement on challenging underwater conditions")
    print("="*80)

if __name__ == "__main__":
    main() 