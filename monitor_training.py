#!/usr/bin/env python3
"""
Training Progress Monitor for YOLOv11s Enhanced vs Original

Real-time monitoring of training progress with comparison to nano baseline.
"""

import pandas as pd
import time
import os
from pathlib import Path

def check_training_progress():
    """Check progress of current training runs"""
    
    experiments = {
        'YOLOv11s Enhanced': 'runs/larger_models/v11s_enhanced2/results.csv',
        'YOLOv11s Original': 'runs/larger_models/v11s_original3/results.csv'
    }
    
    # Baseline nano results for comparison
    nano_baseline = {
        'enhanced_map50_95': 0.754,
        'original_map50_95': 0.748,
        'enhancement_delta': 0.006
    }
    
    print("🔍 YOLOV11S TRAINING PROGRESS MONITOR")
    print("=" * 50)
    print(f"📊 Nano Baseline: Enhanced={nano_baseline['enhanced_map50_95']:.3f}, Delta={nano_baseline['enhancement_delta']:+.3f}")
    print("=" * 50)
    
    for exp_name, results_path in experiments.items():
        print(f"\n📈 {exp_name}:")
        
        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path)
                current_epoch = len(df)
                latest = df.iloc[-1]
                
                map50 = latest.get('metrics/mAP50(B)', 0)
                map50_95 = latest.get('metrics/mAP50-95(B)', 0)
                
                print(f"   Epoch: {current_epoch}/150 ({current_epoch/150*100:.1f}%)")
                print(f"   Current: mAP50={map50:.3f}, mAP50-95={map50_95:.3f}")
                
                # Compare to nano at same epoch if available
                if map50_95 > nano_baseline['enhanced_map50_95']:
                    print(f"   🎯 Already exceeding nano enhanced baseline!")
                elif map50_95 > nano_baseline['original_map50_95']:
                    print(f"   ✅ Above nano original baseline")
                else:
                    print(f"   ⏳ Still below nano baselines")
                    
            except Exception as e:
                print(f"   ❌ Error reading results: {e}")
        else:
            print(f"   ⏳ Training not started yet (no results.csv)")
    
    # Check if both are complete for analysis
    both_complete = all(
        os.path.exists(path) and len(pd.read_csv(path)) >= 150 
        for path in experiments.values() 
        if os.path.exists(path)
    )
    
    if both_complete:
        print(f"\n🎉 BOTH TRAINING RUNS COMPLETE! Ready for scaling analysis.")
        print(f"   Run: python analyze_model_scaling.py")
    else:
        incomplete = sum(1 for path in experiments.values() if not os.path.exists(path) or len(pd.read_csv(path)) < 150)
        print(f"\n⏳ {incomplete} training run(s) still in progress...")

if __name__ == "__main__":
    check_training_progress() 