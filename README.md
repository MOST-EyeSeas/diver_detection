# Diver Detection System

Real-time underwater diver detection system using YOLO object detection framework. This system is designed to detect and track divers in underwater environments through video feeds or image analysis.

## Features

- Real-time diver detection using YOLO
- Optimized for underwater environments and varying visibility conditions
- Designed for deployment on NVIDIA Jetson platform
- **Enhanced underwater image preprocessing pipeline**
- **Comprehensive model comparison infrastructure**

## Getting Started

### Prerequisites

- Ubuntu with Docker and NVIDIA GPU
- NVIDIA Container Toolkit
- CUDA-compatible GPU

### Setup

The development environment is containerized using Docker. Use the provided Dockerfile or devcontainer configuration for VS Code.

## Dataset

This project uses the Video Diver Dataset (VDD-C), a dataset of 100,000+ annotated images of divers underwater.

### Downloading the Dataset

Use the provided download script:

```bash
# Download all dataset components
python download_vddc.py --all

# Download only images and YOLO labels (recommended)
python download_vddc.py --images --yolo-labels

# For help and more options
python download_vddc.py --help
```

### Preparing the Dataset

After downloading the dataset components (e.g., `images.zip` and `yolo_labels.zip`), use the preparation script to extract and organize them into a YOLO-compatible format:

```bash
# Prepare the dataset using downloaded files
python prepare_vddc.py

# Verify downloaded files without extracting
python prepare_vddc.py --verify-only

# For help and more options
python prepare_vddc.py --help
```

## **Enhanced Dataset Pipeline**

### **Underwater Image Enhancement**

This project includes an underwater image enhancement pipeline using [aneris_enhance](https://github.com/VISEAON-Lab/aneris_enhance) to improve underwater image quality before training.

#### **Creating Enhanced Dataset**

```bash
# Enhance entire dataset (training + validation)
python enhance_dataset.py

# Test enhancement on small subset (20 images)
python enhance_dataset.py --test-only

# Enhance only training images (faster)
python enhance_dataset.py --skip-validation

# Use more parallel workers for faster processing
python enhance_dataset.py --workers 8

# Force overwrite existing enhanced dataset
python enhance_dataset.py --force

# For help and more options
python enhance_dataset.py --help
```

**Enhancement Features:**
- **Red channel correction** to compensate for underwater color loss
- **Contrast stretching** using CLAHE for improved visibility
- **Parallel processing** (default: 4 workers, ~8.2 FPS)
- **Progress tracking** with tqdm progress bars
- **100% success rate** across 11,752 images
- **Maintains YOLO compatibility** (labels unchanged)

## Commands

### Basic YOLO Testing

```bash
# Test YOLO detection with visualization (may require manual download for v11/v12)
# Replace yolo11n.pt with desired model (e.g., yolo12n.pt)
yolo predict model=yolo11n.pt show=True

# Test on specific source
yolo predict model=yolo11n.pt source=path/to/image_or_folder show=True
```

### **Model Comparison Training**

This project implements a comprehensive 4-way comparison of model architectures and image preprocessing:

#### **Download Pre-trained Weights**

```bash
# Download YOLOv11n and YOLOv12n weights (required for newer models)
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt
```

#### **4-Way Training Comparison**

```bash
# 1. YOLOv11n on Original Dataset
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v11n_original

# 2. YOLOv11n on Enhanced Dataset
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v11n_enhanced

# 3. YOLOv12n on Original Dataset
yolo train model=yolo12n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v12n_original

# 4. YOLOv12n on Enhanced Dataset
yolo train model=yolo12n.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v12n_enhanced
```

#### **Results Analysis and Comparison**

```bash
# Generate comprehensive comparison report
python compare_results.py

# Include training curve visualizations
python compare_results.py --save-plots

# Specify custom results directory
python compare_results.py --results-dir runs/comparison --save-plots
```

**Comparison Features:**
- **Automatic results loading** from all training experiments
- **Performance metrics comparison** (mAP50, mAP50-95, precision, recall)
- **Enhancement impact analysis** (original vs enhanced datasets)
- **Model architecture comparison** (YOLOv11n vs YOLOv12n)
- **Training curve visualizations** and plots
- **Best model recommendation** with deployment considerations
- **CSV export** for detailed analysis (`training_comparison.csv`)

### **Extended Comparison Options (Future)**

```bash
# Compare with YOLOv10n (planned)
yolo train model=yolo10n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v10n_original

# Longer training runs for best model (planned)
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=100 imgsz=640 batch=16 device=0 project=runs/extended name=v11n_enhanced_100e

# Different model sizes (planned)
yolo train model=yolo11s.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v11s_enhanced
```

### Model Inference on Video

After training, you can run inference on a video file using your best model.

```bash
# Example using the best model from comparison:
yolo predict model=runs/comparison/v11n_enhanced/weights/best.pt source=path/to/your/video.mp4 show=True save=True

# For enhanced dataset models, apply enhancement during inference (if using enhanced model):
# 1. First enhance your input video frames using aneris_enhance
# 2. Then run inference on enhanced frames
```

## **Complete Workflow Example**

```bash
# 1. Download and prepare dataset
python download_vddc.py --images --yolo-labels
python prepare_vddc.py

# 2. Create enhanced dataset
python enhance_dataset.py

# 3. Download model weights
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt

# 4. Run 4-way comparison training (can run in parallel or sequentially)
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v11n_original &
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v11n_enhanced &
yolo train model=yolo12n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v12n_original &
yolo train model=yolo12n.pt data=sample_data/vdd-c/dataset_enhanced/dataset_enhanced.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/comparison name=v12n_enhanced &

# 5. Wait for training completion, then analyze results
python compare_results.py --save-plots

# 6. Test best model on your video
yolo predict model=runs/comparison/[best_model]/weights/best.pt source=your_video.mp4 show=True save=True
```

# WandB Integration
```bash
# Enable WandB experiment tracking
yolo settings wandb=True

# Login to WandB (requires API key)
wandb login
```

## **Project Structure**

```
diver_detection/
├── sample_data/vdd-c/
│   ├── raw/                     # Downloaded dataset files
│   ├── dataset/                 # Original YOLO dataset
│   └── dataset_enhanced/        # Enhanced YOLO dataset
├── runs/comparison/             # Training results
│   ├── v11n_original/
│   ├── v11n_enhanced/
│   ├── v12n_original/
│   └── v12n_enhanced/
├── aneris_enhance/              # Underwater enhancement tool
├── memory-bank/                 # Project documentation
├── download_vddc.py             # Dataset download script
├── prepare_vddc.py              # Dataset preparation script
├── enhance_dataset.py           # Dataset enhancement script
├── compare_results.py           # Training results comparison
└── setup_dataset.sh             # Automated dataset setup
```

## **Performance Metrics**

| Component | Performance | Notes |
|-----------|-------------|-------|
| Dataset Download | Resume capable | 8.38GB images + 6.06MB labels |
| Dataset Enhancement | 8.2 FPS | 11,752 images in ~23 minutes |
| Training (50 epochs) | 2-4 hours | Per model/dataset combination |
| YOLOv11n Early Results | mAP50=0.693 | After just 1 epoch |

## References

- [YOLO Documentation](https://docs.ultralytics.com/)
- [NVIDIA Jetson Setup](https://docs.ultralytics.com/guides/nvidia-jetson/#quick-start-with-docker)
- [VDD-C Dataset](https://conservancy.umn.edu/handle/11299/219383)
- [aneris_enhance](https://github.com/VISEAON-Lab/aneris_enhance) - Underwater image enhancement