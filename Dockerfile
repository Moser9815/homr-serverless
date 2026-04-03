# HOMR OMR — RunPod Serverless Endpoint
# Licensed under AGPL-3.0 (required by HOMR's license)

FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# System deps for OpenCV + cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    && wget -qO /tmp/cudnn.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn9-cuda-12_9.0.0.312-1_amd64.deb \
    && dpkg -i /tmp/cudnn.deb \
    && rm /tmp/cudnn.deb \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Verify Python version (HOMR requires >=3.10,<3.13)
RUN python3 -c "import sys; v=sys.version_info; print(f'Python {v.major}.{v.minor}.{v.micro}'); assert (3,10)<=v<(3,13), f'Need 3.10-3.12, got {v}'"

# Install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Fix onnxruntime GPU: the CPU and GPU packages conflict (same namespace).
# Force-remove both, then install GPU-only with --no-deps to prevent
# the CPU variant from sneaking back as a transitive dependency.
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu && \
    python3 -m pip install --no-cache-dir --no-deps onnxruntime-gpu==1.23.0

# Verify onnxruntime-gpu installed correctly (GPU provider check is runtime-only
# since Docker builds don't have GPU access — but we can verify the package is
# the GPU variant by checking for CUDA-related symbols)
RUN python3 -c "\
import onnxruntime as ort; \
print('ORT version:', ort.__version__); \
print('Available providers:', ort.get_available_providers()); \
# On a non-GPU build machine, CUDAExecutionProvider won't show up, \
# but we can verify the GPU .so is present \
import os, glob; \
ort_dir = os.path.dirname(ort.__file__); \
cuda_libs = glob.glob(os.path.join(ort_dir, 'capi', '*cuda*')) + \
            glob.glob(os.path.join(ort_dir, 'capi', '*cudnn*')); \
print('CUDA libs found:', len(cuda_libs)); \
assert len(cuda_libs) > 0, 'No CUDA libs found in onnxruntime — GPU package not installed correctly'"

# Pre-download HOMR models during build
RUN python3 -c "from homr.main import download_weights; download_weights(use_gpu_inference=True)"

# Copy handler code
COPY handler.py .
COPY parse_musicxml.py .
COPY detect_repeats.py .
COPY detect_voltas.py .

CMD ["python3", "-u", "handler.py"]
