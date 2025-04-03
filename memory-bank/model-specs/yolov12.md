# YOLO12 Specifications

## Overview
YOLO12 represents a significant shift towards an attention-centric architecture, aiming for state-of-the-art object detection accuracy while maintaining real-time performance. It departs from purely CNN-based approaches seen in most prior YOLO versions. [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]

## Key Features
- **Attention-Centric Design:**
  - **Area Attention Mechanism:** Efficient self-attention approach processing large receptive fields by dividing feature maps into regions, reducing computational cost. [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]
  - **Residual Efficient Layer Aggregation Networks (R-ELAN):** Improved feature aggregation module addressing optimization challenges in larger attention models, using block-level residual connections and a bottleneck-like structure. [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]
- **Optimized Attention Architecture:**
  - Optional **FlashAttention** integration to minimize memory access overhead (requires compatible NVIDIA GPU: Turing or newer). [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]
  - **Removed Positional Encoding:** Simplifies the model.
  - **Adjusted MLP Ratio:** Balances computation between attention and feed-forward layers (ratio 1.2 or 2 vs typical 4).
  - **Implicit Positional Info:** Adds a 7x7 separable convolution ("position perceiver") to the attention mechanism.
- **Comprehensive Task Support:** Object Detection, Instance Segmentation, Image Classification, Pose Estimation, Oriented Object Detection (OBB). [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]
- **Performance:** Achieves higher accuracy than YOLO11 across scales (e.g., +0.4% to +1.0% mAP for m/l/x variants) but potentially slightly slower inference speed (-3% to -8% on T4). YOLO12n achieves +2.1% mAP vs YOLOv10n. [Source: [https://docs.ultralytics.com/models/yolo12/](https://docs.ultralytics.com/models/yolo12/)]
- **Variants:** Available in scaled versions (n, s, m, l, x).

## Usage in Project
- Proposed for comparative evaluation against YOLO11 on the VDD-C dataset.
- Potential benefits include the highest accuracy among the YOLO series based on COCO benchmarks.

## Known Considerations
- Might be slightly slower than YOLO11 according to initial benchmarks.
- The attention-centric approach is newer compared to the established CNN-based YOLO architectures.
- Optional FlashAttention requires specific hardware if enabled. 