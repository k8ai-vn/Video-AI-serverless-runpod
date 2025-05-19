FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/workspace/.cache/huggingface/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p downloaded_models_gradio_cpu_init

# Download models during build
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    import yaml; \
    with open('configs/ltxv-13b-0.9.7-distilled.yaml', 'r') as file: \
        config = yaml.safe_load(file); \
    hf_hub_download(repo_id='Lightricks/LTX-Video', \
                   filename=config['checkpoint_path'], \
                   local_dir='downloaded_models_gradio_cpu_init', \
                   local_dir_use_symlinks=False); \
    hf_hub_download(repo_id='Lightricks/LTX-Video', \
                   filename=config['spatial_upscaler_model_path'], \
                   local_dir='downloaded_models_gradio_cpu_init', \
                   local_dir_use_symlinks=False)"

# Expose port for Gradio
EXPOSE 7860

# Command to run the application
CMD ["python3", "app.py"]
