# HOMR OMR — RunPod Serverless Endpoint
# Licensed under AGPL-3.0 (required by HOMR's license)
#
# Uses runpod/base with CUDA for GPU inference via onnxruntime-gpu.

FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all

# System deps for OpenCV headless + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies using the same python3 that will run the handler
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download HOMR models during build
RUN python3 -c "from homr.main import download_weights; download_weights(use_gpu_inference=True)"

# Copy handler code
COPY handler.py .
COPY parse_musicxml.py .
COPY detect_voltas.py .

CMD ["python3", "-u", "handler.py"]
