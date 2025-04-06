# Diver Detection System

Real-time underwater diver detection system using YOLO object detection framework. This system is designed to detect and track divers in underwater environments through video feeds or image analysis.

## Features

- Real-time diver detection using YOLO
- Optimized for underwater environments and varying visibility conditions
- Designed for deployment on NVIDIA Jetson platform

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

## Commands

### Basic YOLO Testing

```bash
# Test YOLO detection with visualization (may require manual download for v11/v12)
# Replace yolo11n.pt with desired model (e.g., yolo12n.pt)
yolo predict model=yolo11n.pt show=True

# Test on specific source
yolo predict model=yolo11n.pt source=path/to/image_or_folder show=True
```

### Model Training

```bash
# Example: Fine-tune YOLOv8n for 100 epochs
yolo train model=yolov8n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=100 imgsz=640

# ---- Current Comparison Training Runs (YOLOv11 vs YOLOv12) ----

# Train YOLOv11n (50 epochs)
yolo train model=yolo11n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/train_v11n_e50 name=diver_detection

# Train YOLOv12n (50 epochs)
yolo train model=yolo12n.pt data=sample_data/vdd-c/dataset/dataset.yaml epochs=50 imgsz=640 batch=16 device=0 project=runs/train_v12n_e50 name=diver_detection
```


# Wandb
```bash
yolo settings wandb=True
```


# Example using the hypothetical best YOLOv11n model:
```bash
yolo predict model=runs/train_v11n_e50/diver_detection/weights/best.pt source=path/to/your/video.mp4 show=True save=True
```


## References

- [YOLO Documentation](https://docs.ultralytics.com/)
- [NVIDIA Jetson Setup](https://docs.ultralytics.com/guides/nvidia-jetson/#quick-start-with-docker)
- [VDD-C Dataset](https://conservancy.umn.edu/handle/11299/219383)