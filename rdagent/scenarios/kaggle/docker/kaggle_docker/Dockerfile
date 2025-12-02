FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Consolidated pip installs for better layer caching
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    torch_geometric \
    pytorch_lightning \
    ogb \
    networkx \
    scikit-learn \
    catboost \
    xgboost \
    sparse \
    lightgbm==3.3.5 \
    pyarrow \
    fastparquet \
    optuna
