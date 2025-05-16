# Base image: Đã có sẵn Python 3.12, CUDA 12.1, cuDNN, PyTorch, Transformers
FROM huggingface/transformers-pytorch-gpu:4.40.0-py3.12-cu121-ubuntu22.04

# Set HuggingFace cache để tận dụng shared volume của RunPod (tăng tốc tải model)
ENV HF_HOME="/runpod-volume/.cache/huggingface"

# Tùy chọn: Ngăn apt yêu cầu prompt
ENV DEBIAN_FRONTEND=noninteractive

# Cài các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /workspace

# Copy trước requirements.txt để tận dụng cache của Docker layer
COPY requirements.txt /workspace/requirements.txt

# Cài Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: Nếu bạn dùng flash-attention (và không cần build CUDA riêng)
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN pip install packaging ninja && \
    pip install flash-attn==2.7.0.post2 --no-build-isolation

# Copy toàn bộ source code vào container
COPY . /workspace

# Entry point cho RunPod (RunPod yêu cầu file `rp_handler.py` hoặc hàm `handler`)
CMD ["python", "-u", "rp_handler.py"]
