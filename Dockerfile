FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py .

# 复制目录（如果存在）
COPY --chown=root:root . .

# 创建必要的目录（如果不存在）
RUN mkdir -p /app/static && \
    mkdir -p /app/templates && \
    mkdir -p /app/generated_images

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "main.py"]
