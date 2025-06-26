#!/bin/bash

# YOLOv11s Enhanced vs Original Training Comparison Script
# Tests whether larger models amplify enhancement benefits

echo "🚀 YOLOV11S TRAINING COMPARISON SCRIPT"
echo "======================================"
echo "Testing enhancement scaling from nano (5.4MB) to small (21MB)"
echo "Baseline: YOLOv11n Enhanced achieved +0.6% mAP50-95 improvement"
echo "Target: YOLOv11s should show >1.0% enhancement benefit"
echo ""

# Check if yolo11s.pt exists, download if needed
if [ ! -f "yolo11s.pt" ]; then
    echo "📥 Downloading YOLOv11s weights..."
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
    
    # Handle potential .pt.1 extension issue
    if [ -f "yolo11s.pt.1" ] && [ ! -f "yolo11s.pt" ]; then
        mv yolo11s.pt.1 yolo11s.pt
        echo "✅ Renamed yolo11s.pt.1 to yolo11s.pt"
    fi
fi

# Create larger_models directory if it doesn't exist
mkdir -p runs/larger_models

echo "🔬 Starting YOLOv11s Enhanced Dataset Training (150 epochs)..."
echo "Expected time: ~12-16 hours"
echo "Dataset: Enhanced underwater images with aneris preprocessing"
echo ""

# Training command 1: Enhanced dataset
yolo train model=yolo11s.pt \
  data=sample_data/vdd-c/dataset_proper_enhanced/dataset_enhanced.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/larger_models \
  name=v11s_enhanced

echo ""
echo "✅ YOLOv11s Enhanced training completed!"
echo ""
echo "🔬 Starting YOLOv11s Original Dataset Training (150 epochs)..."
echo "Expected time: ~12-16 hours"
echo "Dataset: Original underwater images (no preprocessing)"
echo ""

# Training command 2: Original dataset  
yolo train model=yolo11s.pt \
  data=sample_data/vdd-c/dataset_proper/dataset.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=runs/larger_models \
  name=v11s_original

echo ""
echo "🎉 BOTH YOLOV11S TRAINING RUNS COMPLETED!"
echo "========================================"
echo ""
echo "📊 To analyze results and compare to nano baseline:"
echo "   python analyze_model_scaling.py"
echo ""
echo "📈 To monitor progress during training:"
echo "   python monitor_training.py"
echo ""
echo "🎯 Expected outcomes:"
echo "   - YOLOv11s should show >1.0% mAP50-95 enhancement benefit"
echo "   - Absolute performance >76% mAP50-95"
echo "   - Confirm larger models amplify underwater enhancement advantages"
echo ""
echo "🚀 Next steps after analysis:"
echo "   - If successful: Consider YOLOv11m (50MB) for maximum accuracy"
echo "   - If marginal: YOLOv11n remains optimal for Jetson deployment"
echo "   - Real-world video testing with enhanced model" 