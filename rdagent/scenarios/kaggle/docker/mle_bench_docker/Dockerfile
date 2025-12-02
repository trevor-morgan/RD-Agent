FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \
    curl \
    vim \
    git \
    build-essential \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Clone and install mle-bench (consolidated for better caching)
RUN git clone https://github.com/openai/mle-bench.git /workspace/mle-bench \
    && cd /workspace/mle-bench \
    && git lfs fetch --all \
    && git lfs pull \
    && python -m pip install --no-cache-dir -e .

WORKDIR /workspace
