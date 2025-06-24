#!/usr/bin/env python3
"""
Training Results Comparison Script

This script compares the results from all 4 training experiments:
1. YOLOv11n Original Dataset
2. YOLOv11n Enhanced Dataset  
3. YOLOv12n Original Dataset
4. YOLOv12n Enhanced Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml

def load_results(results_path):
    """Load results.csv from a training run."""
    csv_path = Path(results_path) / 'results.csv'
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Get best epoch metrics
        best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
        return {
            'path': results_path,
            'epochs_completed': len(df),
            'best_epoch': int(best_epoch['epoch']),
            'best_mAP50': float(best_epoch['metrics/mAP50(B)']),
            'best_mAP50_95': float(best_epoch['metrics/mAP50-95(B)']),
            'best_precision': float(best_epoch['metrics/precision(B)']),
            'best_recall': float(best_epoch['metrics/recall(B)']),
            'final_train_loss': float(df.iloc[-1]['train/box_loss']),
            'final_val_loss': float(df.iloc[-1]['val/box_loss']),
            'dataframe': df
        }
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def load_training_args(results_path):
    """Load training arguments from args.yaml."""
    args_path = Path(results_path) / 'args.yaml'
    if not args_path.exists():
        return {}
    
    try:
        with open(args_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {args_path}: {e}")
        return {}

def create_comparison_table(results_dict):
    """Create a comparison table of all experiments."""
    data = []
    
    for name, result in results_dict.items():
        if result is None:
            data.append({
                'Experiment': name,
                'Status': 'No Results',
                'Epochs': 0,
                'Best mAP50': 0,
                'Best mAP50-95': 0,
                'Precision': 0,
                'Recall': 0,
                'Train Loss': 0,
                'Val Loss': 0
            })
        else:
            data.append({
                'Experiment': name,
                'Status': f"Completed ({result['epochs_completed']}/50 epochs)",
                'Epochs': result['epochs_completed'],
                'Best mAP50': f"{result['best_mAP50']:.3f}",
                'Best mAP50-95': f"{result['best_mAP50_95']:.3f}",
                'Precision': f"{result['best_precision']:.3f}",
                'Recall': f"{result['best_recall']:.3f}",
                'Train Loss': f"{result['final_train_loss']:.3f}",
                'Val Loss': f"{result['final_val_loss']:.3f}"
            })
    
    return pd.DataFrame(data)

def plot_training_curves(results_dict, save_path='training_comparison.png'):
    """Create training curve plots for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Comparison: Original vs Enhanced Dataset', fontsize=16)
    
    # Define colors for each experiment
    colors = {
        'YOLOv11n Original': '#1f77b4',
        'YOLOv11n Enhanced': '#ff7f0e', 
        'YOLOv12n Original': '#2ca02c',
        'YOLOv12n Enhanced': '#d62728'
    }
    
    metrics = [
        ('metrics/mAP50(B)', 'mAP50'),
        ('metrics/mAP50-95(B)', 'mAP50-95'),
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        for name, result in results_dict.items():
            if result is not None and metric in result['dataframe'].columns:
                df = result['dataframe']
                ax.plot(df['epoch'], df[metric], 
                       label=name, color=colors.get(name, 'gray'), linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")

def analyze_enhancement_impact(results_dict):
    """Analyze the impact of image enhancement on performance."""
    print("\n🔍 Enhancement Impact Analysis:")
    print("=" * 50)
    
    # Compare YOLOv11n
    v11_orig = results_dict.get('YOLOv11n Original')
    v11_enh = results_dict.get('YOLOv11n Enhanced')
    
    if v11_orig and v11_enh:
        print("\n📊 YOLOv11n: Original vs Enhanced")
        print(f"mAP50:     {v11_orig['best_mAP50']:.3f} → {v11_enh['best_mAP50']:.3f} ({v11_enh['best_mAP50'] - v11_orig['best_mAP50']:+.3f})")
        print(f"mAP50-95:  {v11_orig['best_mAP50_95']:.3f} → {v11_enh['best_mAP50_95']:.3f} ({v11_enh['best_mAP50_95'] - v11_orig['best_mAP50_95']:+.3f})")
        print(f"Precision: {v11_orig['best_precision']:.3f} → {v11_enh['best_precision']:.3f} ({v11_enh['best_precision'] - v11_orig['best_precision']:+.3f})")
        print(f"Recall:    {v11_orig['best_recall']:.3f} → {v11_enh['best_recall']:.3f} ({v11_enh['best_recall'] - v11_orig['best_recall']:+.3f})")
    
    # Compare YOLOv12n
    v12_orig = results_dict.get('YOLOv12n Original')
    v12_enh = results_dict.get('YOLOv12n Enhanced')
    
    if v12_orig and v12_enh:
        print("\n📊 YOLOv12n: Original vs Enhanced")
        print(f"mAP50:     {v12_orig['best_mAP50']:.3f} → {v12_enh['best_mAP50']:.3f} ({v12_enh['best_mAP50'] - v12_orig['best_mAP50']:+.3f})")
        print(f"mAP50-95:  {v12_orig['best_mAP50_95']:.3f} → {v12_enh['best_mAP50_95']:.3f} ({v12_enh['best_mAP50_95'] - v12_orig['best_mAP50_95']:+.3f})")
        print(f"Precision: {v12_orig['best_precision']:.3f} → {v12_enh['best_precision']:.3f} ({v12_enh['best_precision'] - v12_orig['best_precision']:+.3f})")
        print(f"Recall:    {v12_orig['best_recall']:.3f} → {v12_enh['best_recall']:.3f} ({v12_enh['best_recall'] - v12_orig['best_recall']:+.3f})")

def recommend_best_model(results_dict):
    """Recommend the best model based on results."""
    print("\n🏆 Model Recommendation:")
    print("=" * 50)
    
    # Find best performing model
    best_model = None
    best_mAP50 = 0
    
    for name, result in results_dict.items():
        if result is not None and result['best_mAP50'] > best_mAP50:
            best_mAP50 = result['best_mAP50']
            best_model = name
    
    if best_model:
        result = results_dict[best_model]
        print(f"🥇 Best Overall Model: {best_model}")
        print(f"   mAP50: {result['best_mAP50']:.3f}")
        print(f"   mAP50-95: {result['best_mAP50_95']:.3f}")
        print(f"   Precision: {result['best_precision']:.3f}")
        print(f"   Recall: {result['best_recall']:.3f}")
        print(f"   Best weights: {result['path']}/weights/best.pt")
        
        # Jetson deployment considerations
        print(f"\n🚀 For Jetson Deployment:")
        if 'Enhanced' in best_model:
            print("   ⚠️  Remember to apply enhancement to new images during inference")
        print("   💡 Consider TensorRT optimization for faster inference")
    else:
        print("❌ No completed training runs found")

def main():
    parser = argparse.ArgumentParser(description='Compare YOLO training results')
    parser.add_argument('--results-dir', type=str, default='runs/comparison',
                      help='Directory containing training results')
    parser.add_argument('--save-plots', action='store_true',
                      help='Save comparison plots')
    
    args = parser.parse_args()
    
    # Define expected experiment directories
    experiments = {
        'YOLOv11n Original': 'v11n_original',
        'YOLOv11n Enhanced': 'v11n_enhanced',
        'YOLOv12n Original': 'v12n_original', 
        'YOLOv12n Enhanced': 'v12n_enhanced'
    }
    
    # Load results from all experiments
    results = {}
    results_dir = Path(args.results_dir)
    
    print("🔍 Loading training results...")
    
    for name, subdir in experiments.items():
        path = results_dir / subdir
        if path.exists():
            result = load_results(path)
            if result:
                print(f"✅ Loaded {name}: {result['epochs_completed']} epochs")
            else:
                print(f"⚠️  Failed to load {name}")
            results[name] = result
        else:
            print(f"❌ Not found: {name} ({path})")
            results[name] = None
    
    # Create comparison table
    print("\n📊 Results Summary:")
    print("=" * 80)
    comparison_df = create_comparison_table(results)
    print(comparison_df.to_string(index=False))
    
    # Save detailed results
    comparison_df.to_csv('training_comparison.csv', index=False)
    print(f"\n✅ Detailed results saved to training_comparison.csv")
    
    # Create visualizations if requested
    if args.save_plots:
        completed_results = {k: v for k, v in results.items() if v is not None}
        if completed_results:
            plot_training_curves(completed_results)
        else:
            print("⚠️  No completed training runs to plot")
    
    # Analyze enhancement impact
    analyze_enhancement_impact(results)
    
    # Recommend best model
    recommend_best_model(results)

if __name__ == "__main__":
    main() 