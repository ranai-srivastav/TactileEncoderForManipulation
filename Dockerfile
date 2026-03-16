# ── Base image ────────────────────────────────────────────────────────────────
# CUDA 12.1 + cuDNN 8 on Ubuntu 22.04.
# Change the tag if your AWS GPU instance uses a different CUDA version.
# Full tag list: https://hub.docker.com/r/nvidia/cuda/tags
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System setup ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        # libjpeg / libpng needed by Pillow
        libjpeg-turbo8-dev \
        libpng-dev \
        # OpenCV-style lib sometimes pulled in by torchvision
        libgl1 \
        libglib2.0-0 \
        # general utilities
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && python3 -m pip install --upgrade pip

# ── Python dependencies ───────────────────────────────────────────────────────
# Install PyTorch with CUDA 12.1 wheels first (separate index), then the rest.
RUN pip install --no-cache-dir \
        torch==2.3.1+cu121 \
        torchvision==0.18.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        numpy \
        Pillow \
        wandb          # optional — train.py gracefully handles ImportError

# ── Working directory & code ──────────────────────────────────────────────────
WORKDIR /workspace

# Copy the entire repo into the image.
# Mount your dataset at runtime with -v /host/data:/workspace/data
COPY . /workspace/

# ── Default command ───────────────────────────────────────────────────────────
# Override this at `docker run` time with the actual training command, e.g.:
#   docker run --gpus all -v /data:/workspace/data <image> \
#       python train.py --split random --n_iters 600
CMD ["python", "train.py", "--help"]
