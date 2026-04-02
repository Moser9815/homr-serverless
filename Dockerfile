# HOMR OMR — RunPod Serverless Endpoint
# Licensed under AGPL-3.0 (required by HOMR's license)
#
# Lightweight build: python-slim + onnxruntime-gpu (bundles its own CUDA libs).
# No full CUDA base image needed since HOMR is ONNX-only.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all

# Minimal system deps for OpenCV headless + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download HOMR models during build (avoids cold-start latency)
# Force model download by running on a tiny test image
RUN python -c "\
from PIL import Image; \
img = Image.new('RGB', (100, 50), 'white'); \
img.save('/tmp/test_init.png')" && \
    homr --gpu no /tmp/test_init.png 2>/dev/null || true && \
    rm -f /tmp/test_init.png /tmp/test_init.musicxml /tmp/test_init.xml

# Copy handler code
COPY handler.py .
COPY parse_musicxml.py .

CMD ["python", "-u", "handler.py"]
