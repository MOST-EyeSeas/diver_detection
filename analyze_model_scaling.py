#!/usr/bin/env python3
"""
Model Scaling Analysis: Compare YOLOv11n vs YOLOv11s Enhancement Benefits

Analyzes how model size affects enhancement advantages in underwater diver detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_training_results(results_path):
    """Load results.csv from YOLO training run"""
    try:
        df = pd.read_csv(results_path)
        # Get final epoch results
        final_results = df.iloc[-1]
        return {
            'mAP50': final_results.get('metrics/mAP50(B)', 0),
            'mAP50-95': final_results.get('metrics/mAP50-95(B)', 0),
            'precision': final_results.get('metrics/precision(B)', 0),
            'recall': final_results.get('metrics/recall(B)', 0),
            'epochs': len(df)
        }
    except Exception as e:
        print(f"Error loading {results_path}: {e}")
        return None

def analyze_model_scaling():
    """Compare nano vs small model enhancement benefits"""
    
    # Define model experiments
    experiments = {
        'nano': {
            'enhanced': 'runs/proper_comparison/v11n_enhanced_FIXED/results.csv',
            'original': 'runs/proper_comparison/v11n_original/results.csv',
            'size_mb': 5.4
        },
        'small': {
            'enhanced': 'runs/larger_models/v11s_enhanced2/results.csv', 
            'original': 'runs/larger_models/v11s_original3/results.csv',
            'size_mb': 21.0
        }
    }
    
    print("=" * 60)
    print("📊 MODEL SCALING ANALYSIS: YOLOv11n vs YOLOv11s")
    print("=" * 60)
    
    results_summary = []
    
    for model_name, paths in experiments.items():
        print(f"\n🔍 {model_name.upper()} MODEL ANALYSIS:")
        
        # Load results
        enhanced_results = load_training_results(paths['enhanced'])
        original_results = load_training_results(paths['original'])
        
        if enhanced_results and original_results:
            # Calculate enhancement deltas
            map50_delta = enhanced_results['mAP50'] - original_results['mAP50']
            map50_95_delta = enhanced_results['mAP50-95'] - original_results['mAP50-95']
            precision_delta = enhanced_results['precision'] - original_results['precision']
            recall_delta = enhanced_results['recall'] - original_results['recall']
            
            print(f"  Enhanced:  mAP50={enhanced_results['mAP50']:.3f}, mAP50-95={enhanced_results['mAP50-95']:.3f}")
            print(f"  Original:  mAP50={original_results['mAP50']:.3f}, mAP50-95={original_results['mAP50-95']:.3f}")
            print(f"  🎯 DELTAS: mAP50={map50_delta:+.3f}, mAP50-95={map50_95_delta:+.3f}")
            print(f"            precision={precision_delta:+.3f}, recall={recall_delta:+.3f}")
            
            # Store for comparison
            results_summary.append({
                'model': model_name,
                'size_mb': paths['size_mb'],
                'enhanced_map50': enhanced_results['mAP50'],
                'enhanced_map50_95': enhanced_results['mAP50-95'],
                'original_map50': original_results['mAP50'],
                'original_map50_95': original_results['mAP50-95'],
                'map50_delta': map50_delta,
                'map50_95_delta': map50_95_delta,
                'precision_delta': precision_delta,
                'recall_delta': recall_delta
            })
        else:
            print(f"  ❌ Results not available yet")
            results_summary.append({
                'model': model_name,
                'size_mb': paths['size_mb'],
                'status': 'training_in_progress'
            })
    
    # Analysis and recommendations
    if len([r for r in results_summary if 'status' not in r]) >= 2:
        print("\n" + "=" * 60)
        print("🔬 SCALING ANALYSIS:")
        print("=" * 60)
        
        nano_results = next(r for r in results_summary if r['model'] == 'nano' and 'status' not in r)
        small_results = next(r for r in results_summary if r['model'] == 'small' and 'status' not in r)
        
        # Enhancement benefit scaling
        nano_benefit = nano_results['map50_95_delta']
        small_benefit = small_results['map50_95_delta']
        scaling_factor = small_benefit / nano_benefit if nano_benefit != 0 else 0
        
        print(f"📈 Enhancement Benefit Scaling:")
        print(f"   Nano (5.4MB):  {nano_benefit:+.3f}% mAP50-95 improvement")
        print(f"   Small (21MB):  {small_benefit:+.3f}% mAP50-95 improvement")
        print(f"   🚀 Scaling Factor: {scaling_factor:.2f}x")
        
        # Absolute performance scaling
        nano_abs = nano_results['enhanced_map50_95']
        small_abs = small_results['enhanced_map50_95']
        abs_improvement = small_abs - nano_abs
        
        print(f"\n📊 Absolute Performance Scaling:")
        print(f"   Nano Enhanced:  {nano_abs:.3f} mAP50-95")
        print(f"   Small Enhanced: {small_abs:.3f} mAP50-95")
        print(f"   🎯 Absolute Gain: {abs_improvement:+.3f}")
        
        # Jetson deployment analysis
        print(f"\n🚀 Deployment Recommendations:")
        if small_benefit > nano_benefit * 1.2:  # 20% better enhancement benefits
            print(f"   ✅ YOLOv11s shows significant enhancement amplification ({scaling_factor:.2f}x)")
            print(f"   ✅ Recommended for production (21MB still Jetson-compatible)")
        elif abs_improvement > 0.01:  # 1% absolute improvement
            print(f"   ✅ YOLOv11s provides meaningful absolute improvement")
            print(f"   ⚖️  Consider accuracy vs speed trade-off for deployment")
        else:
            print(f"   ⚠️  YOLOv11s gains may not justify increased complexity")
            print(f"   💡 YOLOv11n remains optimal for resource-constrained deployment")
    
    # Save detailed results
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv('model_scaling_analysis.csv', index=False)
    print(f"\n💾 Detailed results saved to: model_scaling_analysis.csv")
    
    return results_summary

def plot_scaling_comparison(results_summary):
    """Create visualization comparing model scaling"""
    completed_results = [r for r in results_summary if 'status' not in r]
    
    if len(completed_results) < 2:
        print("⏳ Waiting for both models to complete training for visualization")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = [r['model'] for r in completed_results]
    sizes = [r['size_mb'] for r in completed_results]
    
    # Plot 1: Enhancement deltas
    map50_95_deltas = [r['map50_95_delta'] for r in completed_results]
    ax1.bar(models, map50_95_deltas, color=['skyblue', 'orange'])
    ax1.set_title('Enhancement Benefits by Model Size')
    ax1.set_ylabel('mAP50-95 Delta')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute performance
    enhanced_map50_95 = [r['enhanced_map50_95'] for r in completed_results]
    original_map50_95 = [r['original_map50_95'] for r in completed_results]
    
    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - width/2, original_map50_95, width, label='Original', color='lightcoral')
    ax2.bar(x + width/2, enhanced_map50_95, width, label='Enhanced', color='lightgreen')
    ax2.set_title('Absolute Performance Comparison')
    ax2.set_ylabel('mAP50-95')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model size vs enhancement benefit
    ax3.scatter(sizes, map50_95_deltas, s=100, color='purple')
    ax3.set_title('Model Size vs Enhancement Benefit')
    ax3.set_xlabel('Model Size (MB)')
    ax3.set_ylabel('mAP50-95 Delta')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis (performance per MB)
    efficiency = [r['enhanced_map50_95'] / r['size_mb'] for r in completed_results]
    ax4.bar(models, efficiency, color=['lightblue', 'lightsalmon'])
    ax4.set_title('Performance Efficiency (mAP50-95 per MB)')
    ax4.set_ylabel('mAP50-95 / Model Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_scaling_comparison.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: model_scaling_comparison.png")

if __name__ == "__main__":
    results = analyze_model_scaling()
    plot_scaling_comparison(results) 