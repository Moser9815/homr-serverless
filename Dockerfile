# HOMR OMR — RunPod Serverless Endpoint
# Licensed under AGPL-3.0 (required by HOMR's license)
#
# Builds a GPU-enabled container that runs HOMR optical music recognition
# as a RunPod serverless handler.

FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all

# System dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN 9 for ONNX Runtime GPU
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    wget -qO /tmp/cudnn.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn9-cuda-12_9.0.0.312-1_amd64.deb && \
    dpkg -i /tmp/cudnn.deb && \
    rm /tmp/cudnn.deb && \
    apt-get remove -y wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download HOMR models during build (avoids cold-start latency)
RUN python -c "from homr import main; print('HOMR imported successfully')" && \
    homr --init 2>/dev/null || true

# If --init doesn't exist, force model download by running on a tiny test image
RUN python -c "\
from PIL import Image; \
img = Image.new('RGB', (100, 50), 'white'); \
img.save('/tmp/test_init.png')" && \
    homr --gpu no /tmp/test_init.png 2>/dev/null || true && \
    rm -f /tmp/test_init.png /tmp/test_init.musicxml

# Copy handler code
COPY handler.py .
COPY parse_musicxml.py .

CMD ["python", "-u", "handler.py"]
