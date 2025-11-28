FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 先创建必要的目录
RUN mkdir -p /app/static /app/templates /app/generated_images

# 复制所有文件（会覆盖上面创建的空目录，如果源目录存在的话）
COPY . .

# 确保目录存在（防止 COPY 时源目录不存在导致目录被删除）
RUN mkdir -p /app/static /app/templates /app/generated_images

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "main.py"]
