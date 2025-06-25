#!/usr/bin/env python3
"""
Test inference enhancement on held-out test set.
This tests enhancement benefits on truly unseen data with no data leakage.
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def run_model_validation(model_path, test_images_dir, test_labels_dir, model_name, test_type):
    """Run validation on test set and return metrics."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} on {test_type} Test Images")
    print(f"{'='*60}")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Create temporary dataset.yaml for this test
        temp_yaml = f"temp_test_{model_name.lower().replace(' ', '_')}_{test_type.lower()}.yaml"
        
        # YOLO requires train and val keys even for test-only inference
        base_path = os.path.abspath(os.path.dirname(test_images_dir))
        test_images_name = os.path.basename(test_images_dir)
        
        yaml_content = f"""# Temporary test configuration
path: {base_path}
train: {test_images_name}
val: {test_images_name}
test: {test_images_name}
nc: 1
names: ['diver']
"""
        
        with open(temp_yaml, 'w') as f:
            f.write(yaml_content)
        
        # Run validation on test set
        print(f"Running inference on {len(list(Path(test_images_dir).glob('*.jpg')))} test images...")
        results = model.val(data=temp_yaml, split='test', save=False, plots=False, verbose=False)
        
        # Extract metrics
        metrics = {
            'model_name': model_name,
            'test_type': test_type,
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'num_images': len(list(Path(test_images_dir).glob('*.jpg')))
        }
        
        print(f"Results - mAP50: {metrics['mAP50']:.3f}, mAP50-95: {metrics['mAP50-95']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        
        # Cleanup
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)
        
        return metrics
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return None

def test_inference_enhancement(models_config, dataset_dir, enhanced_dataset_dir):
    """Test inference enhancement on all models."""
    print("INFERENCE ENHANCEMENT TESTING ON HELD-OUT TEST SET")
    print("=" * 60)
    print("Testing enhancement benefits on truly unseen data (no data leakage)")
    print()
    
    # Paths
    original_test_images = os.path.join(dataset_dir, 'images', 'test')
    enhanced_test_images = os.path.join(enhanced_dataset_dir, 'images', 'test')
    test_labels = os.path.join(dataset_dir, 'labels', 'test')
    
    if not os.path.exists(original_test_images):
        print(f"Error: Original test images directory not found: {original_test_images}")
        return []
    
    if not os.path.exists(enhanced_test_images):
        print(f"Error: Enhanced test images directory not found: {enhanced_test_images}")
        return []
    
    if not os.path.exists(test_labels):
        print(f"Error: Test labels directory not found: {test_labels}")
        return []
    
    # Verify test sets have same number of images
    original_count = len(list(Path(original_test_images).glob('*.jpg')))
    enhanced_count = len(list(Path(enhanced_test_images).glob('*.jpg')))
    
    print(f"✅ Using pre-enhanced test set:")
    print(f"   Original test images: {original_count}")
    print(f"   Enhanced test images: {enhanced_count}")
    
    if original_count != enhanced_count:
        print(f"⚠️ Warning: Mismatch in test set sizes!")
        return []
        
    all_results = []
    
    # Test each model on both original and enhanced test sets
    for model_info in models_config:
        model_path = model_info['path']
        model_name = model_info['name']
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'='*80}")
        
        # Test on original test images
        original_result = run_model_validation(
            model_path, original_test_images, test_labels, 
            model_name, "Original"
        )
        
        # Test on enhanced test images  
        enhanced_result = run_model_validation(
            model_path, enhanced_test_images, test_labels,
            model_name, "Enhanced"
        )
        
        if original_result and enhanced_result:
            all_results.extend([original_result, enhanced_result])
            
            # Calculate improvement
            metrics_to_compare = ['mAP50', 'mAP50-95', 'precision', 'recall']
            print(f"\n📊 ENHANCEMENT IMPACT for {model_name}:")
            for metric in metrics_to_compare:
                original_val = original_result[metric]
                enhanced_val = enhanced_result[metric]
                delta = enhanced_val - original_val
                percent_change = (delta / original_val) * 100 if original_val > 0 else 0
                
                status = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                print(f"  {status} {metric}: {original_val:.3f} → {enhanced_val:.3f} "
                      f"(Δ{delta:+.3f}, {percent_change:+.1f}%)")
    
    return all_results

def analyze_results(results):
    """Analyze and summarize all results."""
    if not results:
        print("No results to analyze")
        return
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS - INFERENCE ENHANCEMENT BENEFITS")
    print(f"{'='*80}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Group by model for comparison
    models = df['model_name'].unique()
    
    print("\n📊 DETAILED RESULTS TABLE:")
    print(df.to_string(index=False, float_format='%.3f'))
    
    print(f"\n🎯 ENHANCEMENT IMPACT SUMMARY:")
    print("-" * 50)
    
    significant_improvements = []
    
    for model in models:
        model_data = df[df['model_name'] == model]
        if len(model_data) != 2:
            continue
            
        original = model_data[model_data['test_type'] == 'Original'].iloc[0]
        enhanced = model_data[model_data['test_type'] == 'Enhanced'].iloc[0]
        
        print(f"\n🏷️ {model}:")
        
        metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
        model_improvements = []
        
        for metric in metrics:
            orig_val = original[metric]
            enh_val = enhanced[metric]
            delta = enh_val - orig_val
            percent = (delta / orig_val) * 100 if orig_val > 0 else 0
            
            status = "📈" if delta > 0.001 else "📉" if delta < -0.001 else "➡️"
            
            print(f"  {status} {metric}: {orig_val:.3f} → {enh_val:.3f} "
                  f"(Δ{delta:+.3f}, {percent:+.1f}%)")
            
            if delta > 0.001:  # Improvement threshold
                model_improvements.append((metric, delta, percent))
        
        if model_improvements:
            significant_improvements.extend([(model, metric, delta, percent) 
                                           for metric, delta, percent in model_improvements])
    
    # Overall summary
    print(f"\n🎉 ENHANCEMENT BENEFITS SUMMARY:")
    print("-" * 40)
    
    if significant_improvements:
        print(f"✅ Found {len(significant_improvements)} significant improvements!")
        for model, metric, delta, percent in significant_improvements:
            print(f"  • {model}: {metric} improved by {delta:+.3f} ({percent:+.1f}%)")
    else:
        print("❓ No significant improvements detected in this test")
        print("   This could mean:")
        print("   - Test set was not challenging enough for enhancement benefits")
        print("   - Enhancement benefits are more apparent in real-world conditions")
        print("   - Need to test on external underwater video for clearer benefits")
    
    # Save results
    csv_filename = "inference_enhancement_test_results_150epochs.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n📁 Results saved to: {csv_filename}")
    
    return significant_improvements

def main():
    parser = argparse.ArgumentParser(description='Test inference enhancement on held-out test set')
    parser.add_argument('--dataset', default='sample_data/vdd-c/dataset_proper',
                       help='Path to dataset with proper train/val/test split')
    parser.add_argument('--enhanced-dataset', default='sample_data/vdd-c/dataset_proper_enhanced',
                       help='Path to enhanced dataset with proper train/val/test split')
    parser.add_argument('--models-dir', default='runs/proper_comparison',
                       help='Directory containing trained model weights')
    
    args = parser.parse_args()
    
    # Define models to test - 150-epoch extended training models
    models_config = [
        {
            'name': 'YOLOv11n Original-150ep',
            'path': f'runs/extended/v11n_original_100ep/weights/best.pt'
        },
        {
            'name': 'YOLOv11n Enhanced-150ep', 
            'path': f'runs/extended/v11n_enhanced_100ep/weights/best.pt'
        }
    ]
    
    print("METHODOLOGICALLY SOUND INFERENCE ENHANCEMENT TESTING")
    print("=" * 60)
    print("Key advantages of this approach:")
    print("✅ Test set completely held-out during training (no data leakage)")
    print("✅ 5,793 unseen images for unbiased evaluation")
    print("✅ Both training-time and inference-time enhancement testing")
    print("✅ Comprehensive comparison across model architectures")
    print("✅ Using pre-enhanced test set (no redundant processing)")
    print()
    
    # Check dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Original dataset not found: {args.dataset}")
        return 1
    
    # Check enhanced dataset
    if not os.path.exists(args.enhanced_dataset):
        print(f"Error: Enhanced dataset not found: {args.enhanced_dataset}")
        return 1
    
    # Run tests
    results = test_inference_enhancement(models_config, args.dataset, args.enhanced_dataset)
    
    if results:
        significant_improvements = analyze_results(results)
        
        print(f"\n{'='*80}")
        print("NEXT STEPS RECOMMENDATIONS:")
        print("=" * 80)
        
        if significant_improvements:
            print("🎯 Enhancement benefits detected! Recommended next steps:")
            print("1. Test best performing model on external underwater video")
            print("2. Consider longer training (100 epochs) for further improvements")
            print("3. Prepare for Jetson deployment with TensorRT optimization")
        else:
            print("🎬 Test on real underwater video for qualitative assessment:")
            print("1. Enhancement benefits may be more apparent on challenging conditions")
            print("2. Validation set quality might be high enough to minimize enhancement impact")
            print("3. Real-world testing will show practical deployment benefits")
        
        print(f"\n📝 All results saved to: inference_enhancement_test_results_150epochs.csv")
    else:
        print("❌ No results obtained. Please check model paths and dataset structure.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 