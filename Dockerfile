FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖和编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p generated_images

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "-u", "main.py"]