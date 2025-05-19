FROM python:3.9-slim

WORKDIR /app

# Install git and build essentials
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install q8_kernels
WORKDIR /app/q8_kernels
RUN git submodule init && \
    git submodule update && \
    python setup.py install && \
    pip install .

# Clone and install LTXVideo
WORKDIR /app
WORKDIR /app/LTXVideo
RUN python -m pip install -e .[inference-script]

# Download model from Hugging Face
RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; \
    model_path = '/app/models'; \
    snapshot_download('konakona/ltxvideo_q8', local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')"

# Set working directory to LTXVideo
WORKDIR /app/LTXVideo

# Command to run when container starts
CMD ["bash"]
