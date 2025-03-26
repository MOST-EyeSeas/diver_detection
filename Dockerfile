# t=ultralytics/ultralytics:latest-jetson-jetpack5
# sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t

# FROM ultralytics/ultralytics:latest-jetson-jetpack5

FROM ultralytics/ultralytics:latest

# Install GTK dependencies for OpenCV GUI support
RUN apt-get update && apt-get install -y \
    libgtk2.0-dev \
    libgtk-3-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    x11-apps \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV QT_X11_NO_MITSHM=1

# Install Git and SSH for repository operations
RUN apt-get update && apt-get install -y \
    openssh-client \
    git \
    && rm -rf /var/lib/apt/lists/*

# Reinstall OpenCV with GUI support
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python

# Verify CUDA is properly configured
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    && rm -rf /var/lib/apt/lists/*

# Test CV2 has GUI support
RUN python -c "import cv2; print('CV2 version:', cv2.__version__); print('CV2 GUI support:', cv2.getBuildInformation().find('GTK') > 0)"