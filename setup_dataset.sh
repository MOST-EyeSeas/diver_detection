#!/bin/bash

# Script to download and prepare the VDD-C dataset without progress bars

set -e  # Exit immediately if a command exits with a non-zero status.

# Step 1: Download Images and YOLO Labels
echo "Starting dataset download..."
python3 download_vddc.py --images --yolo-labels --no-progress
echo "Download finished."

# Step 2: Prepare Dataset
echo "Starting dataset preparation..."
python3 prepare_vddc.py --no-progress
echo "Preparation finished."

echo "Dataset setup complete." 