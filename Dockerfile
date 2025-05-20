# Sử dụng image FastVideo làm base
FROM ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest

# Copy script của bạn vào container
WORKDIR /workspace
COPY . /workspace

RUN pip install -r requirements.txt
# Cài đặt thêm các phụ thuộc nếu cần
# RUN pip install <package_name>

# Tải trước model
RUN python -c "from fastvideo import VideoGenerator, PipelineConfig; \
    model_name = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'; \
    config = PipelineConfig.from_pretrained(model_name); \
    config.vae_precision = 'fp16'; \
    config.use_cpu_offload = True; \
    VideoGenerator.from_pretrained(model_name, pipeline_config=config)"

RUN python /FastVideo/scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan-diffusers --local_dir=/workspace/data/FastHunyuan-diffusers --repo_type=model

# Expose port for FastAPI
EXPOSE 8000

# Lệnh chạy script khi container khởi động
CMD ["python", "/workspace/handler.py"]