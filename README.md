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

## Commands

### Basic YOLO Testing

```bash
# Test YOLO detection with visualization
yolo predict model=yolo11n.pt show=True

# Test on specific source
yolo predict model=yolo11n.pt source=path/to/image_or_folder show=True
```

## References

- [YOLO Documentation](https://docs.ultralytics.com/)
- [NVIDIA Jetson Setup](https://docs.ultralytics.com/guides/nvidia-jetson/#quick-start-with-docker)
- [VDD-C Dataset](https://conservancy.umn.edu/handle/11299/219383)