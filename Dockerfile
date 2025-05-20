# Sử dụng image FastVideo làm base
FROM ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest

# Copy script của bạn vào container
WORKDIR /workspace
COPY . /workspace

RUN pip install -r requirements.txt
# Cài đặt thêm các phụ thuộc nếu cần
# RUN pip install <package_name>

# Lệnh chạy script khi container khởi động
CMD ["python", "/workspace/handler.py"]