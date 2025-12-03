FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# Clone and checkout specific Qlib version
RUN git clone https://github.com/microsoft/qlib.git /workspace/qlib \
    && cd /workspace/qlib \
    && git fetch \
    && git reset 3e72593b8c985f01979bebcf646658002ac43b00 --hard

WORKDIR /workspace/qlib

# Consolidated pip installs for better layer caching
# Note: scipy version is not pinned to avoid conflicts with pyqlib dependencies
RUN python -m pip install --no-cache-dir --upgrade cython \
    && python -m pip install --no-cache-dir -e . \
    && pip install --no-cache-dir catboost xgboost scipy tables
