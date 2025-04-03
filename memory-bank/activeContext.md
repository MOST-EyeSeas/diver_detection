# Active Context: Diver Detection System

## Current Focus
We are now performing a **comparative evaluation of YOLOv11 and YOLOv12** for the diver detection task. The goal is to identify the best model based on accuracy and efficiency using the prepared VDD-C dataset. Training runs for `yolov11n` and `yolov12n` have been initiated.

## Recent Changes
1. Set up the development container with required dependencies
2. Configured Docker environment with GPU support
3. Established proper X11 forwarding for visualization tools
4. Created Memory Bank for project documentation
5. Resolved CUDA initialization issues
6. Successfully tested basic YOLO functionality
7. Created download script for VDD-C dataset
8. Identified VDD-C as an excellent dataset for diver detection
9. Created dataset preparation script (prepare_vddc.py) to extract and organize dataset
10. Successfully downloaded and prepared the VDD-C dataset for YOLO training
11. **Decided to compare YOLOv11 vs YOLOv12 instead of initial v8 plan.**
12. **Created specification files for YOLOv11 and YOLOv12 in `memory-bank/model-specs/`.**
13. **Updated `ultralytics` package to latest version (`8.3.100`).**
14. **Manually downloaded `yolov11n.pt` and `yolov12n.pt` pre-trained weights.**
15. **Initiated training runs for YOLOv11n and YOLOv12n (50 epochs).**

## Current Tasks
- [x] Set up Docker development environment
- [x] Configure GPU access in container
- [x] Fix X11 forwarding for visualization
- [x] Initialize project documentation
- [x] Test basic YOLO functionality
- [x] Identify source for diver image dataset
- [x] Create download script for diver dataset
- [x] Download and extract VDD-C dataset
- [x] Create script for YOLO-compatible dataset structure
- [x] Document YOLOv11 and YOLOv12 specs
- [x] Update ultralytics package
- [x] Download required pre-trained weights (v11n, v12n)
- [▶️] Run comparative training for YOLOv11n and YOLOv12n (In Progress)
- [ ] Evaluate training results (mAP, speed, efficiency)
- [ ] Select best model based on comparison
- [ ] Document comparison results and decision

## Next Steps

### Immediate Next Steps (Current Sprint)
1. **Monitor Training Runs**
   - Observe progress of YOLOv11n and YOLOv12n training.
   - Ensure runs complete successfully.

2. **Results Evaluation & Comparison**
   - Once training completes, collect performance metrics (mAP, Precision, Recall, Speed, Size, GFLOPs) from `runs/train_v11n_e50/diver_detection/` and `runs/train_v12n_e50/diver_detection/`.
   - Create comparison table (e.g., in `memory-bank/model-comparison-v11-v12.md`).
   - Analyze trade-offs between accuracy and efficiency.

3. **Model Selection & Documentation**
   - Choose the best performing model for the diver detection task, considering Jetson deployment constraints.
   - Update Memory Bank (`activeContext.md`, `progress.md`, `.clinerules`, comparison file) with the decision and results.

### Upcoming Priorities
1. **Further Model Optimization (If needed)**
   - Based on comparison results, potentially fine-tune the chosen model further (e.g., more epochs, hyperparameter tuning).
   - Consider data augmentation specific to underwater conditions.

2. **Jetson Deployment Preparation**
   - Prepare testing environment for Jetson deployment
   - Identify optimization requirements for edge deployment
   - Establish benchmarking process
   - Plan for TensorRT conversion

## Active Decisions and Considerations

### Current Decision Points
1. **Dataset Selection**
   - ✅ Selected VDD-C dataset.
   - ✅ Successfully prepared dataset (5,997 train / 5,763 val images).

2. **YOLO Version Selection**
   - ✅ Decided to compare **YOLOv11n vs YOLOv12n** based on latest advancements and potential performance benefits.
   - Documentation created for both versions.
   - Comparison training initiated.

3. **Training Approach**
   - ✅ Fine-tuning from pre-trained `yolov11n.pt` and `yolov12n.pt` weights.
   - ✅ Single-class (diver) detection.
   - ✅ Initial comparison run: 50 epochs, batch size 16, imgsz 640.

### Known Challenges
1. **Underwater Image Characteristics**
   - Color distortion and limited visibility
   - Variable lighting conditions
   - Bubbles and particulates creating visual noise
   - VDD-C dataset provides good representation of these challenges

2. **Class Imbalance**
   - Some underwater scenes may have multiple divers while others have single divers
   - Need to ensure balanced loss function and evaluation metrics
   - Consider specialized detection approaches for small/distant divers

3. **Jetson Deployment Considerations**
   - Nano ('n') variants of YOLOv11/v12 chosen for efficiency.
   - Final comparison needs to weigh accuracy against inference speed and model size suitable for Jetson.

4. **Real-time Processing Requirements**
   - Balancing detection accuracy with processing speed
   - Managing input pipeline for consistent frame rate
   - Handling variable processing loads

5. **Model Weight Availability**
   - ❗ Newer models (YOLOv11, YOLOv12) might require manual download of `.pt` files, unlike older versions automatically fetched by the library. Need to use direct download links. 