# Gemini Link System

一个将 Gemini Business API 转换为 OpenAI 兼容接口的网关服务，支持多账号负载均衡、API 密钥管理、图片生成和思考过程显示等功能。
<div align="center">
  <img width="642" height="540" alt="78e0edf7-a521-483e-bfc8-2e185594ea66" src="https://github.com/user-attachments/assets/20dccf7f-6dc0-446e-b57e-d492448f979c" />
</div>

<div align="center">
  <img width="593" height="532" alt="f49c2af4-2036-4895-a50f-b665b51cf33a" src="https://github.com/user-attachments/assets/69d44114-cac5-4e65-a52f-b01ec6333a7c" />
</div>

<div align="center">
  <img width="622" height="484" alt="8c87f393-b3f1-4eec-8059-e040487e28ee" src="https://github.com/user-attachments/assets/566a1872-837f-43eb-a05f-6dfe8e6e885f" />
</div>

<div align="center">
  <img width="1912" height="954" alt="3aabf85c-6dd8-4626-a574-001ab30003ec" src="https://github.com/user-attachments/assets/25230990-cebb-41e7-8a7f-a54f8873ad61" />
</div>

<div align="center">
  <img width="886" height="721" alt="27eabdac-1c57-4dfb-8b98-7069524e6aa2" src="https://github.com/user-attachments/assets/4f7885b1-eb51-4414-b3a5-0c02dd4a7ee4" />
</div>

<div align="center">
  <img width="896" height="793" alt="4f259f9d-7e80-4fdd-b342-717c9ba01524" src="https://github.com/user-attachments/assets/ef6a08c4-7191-46af-b60d-8b1d6c287f77" />
</div>
## ✨ 功能特性

- 🚀 **OpenAI 兼容接口**：完全兼容 OpenAI Chat Completions API
- 🔑 **API 密钥管理**：支持生成、管理和撤销 API 密钥
- 📊 **使用统计**：详细的 API 调用日志和统计信息
- 🖼️ **图片生成**：支持 Gemini 图片生成输入功能，自动下载和保存
- 💭 **思考过程显示**：支持显示模型的思考过程（可折叠）
- 🔄 **多账号支持**：支持配置多个 Gemini Business 账号，自动负载均衡
- 🛡️ **账号容错**：自动检测账号配额限制，切换到备用账号
- 📝 **管理员面板**：Web 界面管理 API 密钥和查看统计

## 📋 系统要求

- Python 3.10+
- SQLite（默认）或 PostgreSQL
- Gemini Business 账号凭证

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd geminibusiness
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的 Gemini Business 账号信息。

### 4. 运行服务

```bash
python main.py
```

服务将在 `http://localhost:5000` 启动。

## 🐳 Docker 部署

### 使用 Docker Compose（推荐）

```bash
docker-compose up -d
```

### 使用 Docker

```bash
docker build -t geminibusiness .
docker run -d \
  --name geminibusiness \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/geminibusiness.db:/app/geminibusiness.db \
  -v $(pwd)/generated_images:/app/generated_images \
  geminibusiness
```

## ⚙️ 配置说明

### 单账号配置

```env
SECURE_C_SES=your_secure_c_ses_value
CSESIDX=your_csesidx_value
CONFIG_ID=your_config_id_value
HOST_C_OSES=your_host_c_oses_value  # 可选
```

### 多账号配置

支持配置多个账号，系统会自动进行负载均衡：

```env
# 账号 1
ACCOUNT1_SECURE_C_SES=your_secure_c_ses_1
ACCOUNT1_CSESIDX=your_csesidx_1
ACCOUNT1_CONFIG_ID=your_config_id_1
ACCOUNT1_NAME=account-1  # 可选，默认 account-1
ACCOUNT1_HOST_C_OSES=your_host_c_oses_1  # 可选

# 账号 2
ACCOUNT2_SECURE_C_SES=your_secure_c_ses_2
ACCOUNT2_CSESIDX=your_csesidx_2
ACCOUNT2_CONFIG_ID=your_config_id_2
ACCOUNT2_NAME=account-2
```

### 其他配置

```env
# 代理设置（可选）
PROXY=http://proxy.example.com:8080

# 请求超时时间（秒，默认 600）
TIMEOUT_SECONDS=600
```

## 📖 API 使用

### 获取 API 密钥

1. 访问 `http://localhost:5000/static/index.html`
2. 使用默认账号登录（用户名：`admin`，密码：`admin123456`）
3. 在管理面板中生成 API 密钥

### 调用示例

#### Python

```python
import requests

url = "http://localhost:5000/v1/chat/completions"
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}
data = {
    "model": "gemini-3-pro-preview",
    "messages": [
        {"role": "user", "content": "你好"}
    ],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode())
```

#### cURL

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

#### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:5000/v1"
)

response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=[{"role": "user", "content": "生成一张图片"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 🎯 支持的模型

- `gemini-auto` - 自动选择模型
- `gemini-2.5-flash` - 快速响应模型
- `gemini-2.5-pro` - 平衡推理模型
- `gemini-3-pro-preview` - 预览版旗舰模型

## 🖼️ 图片生成

服务支持 Gemini 的图片生成功能。生成的图片会：

1. 自动下载并保存到 `generated_images` 目录
2. 以 base64 格式包含在 API 响应中
3. 支持 Markdown 格式显示

### 图片生成示例

```python
response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=[{"role": "user", "content": "生成一张关于人工智能的图片"}],
    stream=False
)

# 响应中会包含图片的 base64 数据
if hasattr(response, 'images'):
    for img in response.images:
        print(f"图片文件名: {img['file_name']}")
```

## 💭 思考过程

服务会自动提取并显示模型的思考过程：

- 思考过程会以可折叠的 HTML `<details>` 标签格式显示
- 支持通过 `thinking` 字段单独获取思考内容
- 思考标题（如 "Assessing User Intent"）会自动从正文中提取并折叠显示

## 📊 管理面板

访问 `http://localhost:5000/static/index.html` 进入管理面板，可以：

- 生成和管理 API 密钥
- 查看 API 使用统计
- 查看调用日志
- 撤销 API 密钥

### 默认管理员账号

- 用户名：`admin`
- 密码：`admin123456`

**⚠️ 生产环境请务必修改默认密码！**

## 🔧 项目结构

```
geminibusiness/
├── main.py                 # 主应用文件
├── auth.py                 # 认证和授权模块
├── database.py             # 数据库模型
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 镜像配置
├── docker-compose.yml      # Docker Compose 配置
├── .env.example            # 环境变量示例
├── geminibusiness.db       # SQLite 数据库（自动生成）
├── generated_images/       # 生成的图片存储目录
└── static/                 # 静态文件（管理面板）
    ├── index.html
    ├── dashboard.html
    ├── style.css
    └── app.js
```

## 🔐 安全建议

1. **修改默认密码**：首次登录后立即修改管理员密码
2. **使用强密码**：为 API 密钥设置合理的过期时间
3. **保护环境变量**：不要将 `.env` 文件提交到版本控制
4. **使用 HTTPS**：生产环境建议使用反向代理（如 Nginx）配置 HTTPS
5. **限制访问**：使用防火墙限制管理面板的访问

## 🐛 故障排除

### 问题：无法连接到 Gemini API

- 检查环境变量配置是否正确
- 确认账号凭证是否有效
- 检查网络连接和代理设置

### 问题：图片生成失败

- 检查 `generated_images` 目录权限
- 查看日志中的错误信息
- 确认账号是否有图片生成权限

### 问题：多账号切换不工作

- 确认所有账号配置完整
- 检查账号是否被标记为不可用（查看日志）
- 等待账号冷却期结束（默认 5 分钟）

## 📝 日志

日志输出格式：

```
时间 | 级别 | 消息内容 [账号名称]
```

示例：
```
01:38:27 | INFO | 🆕 开启新对话 [gemini-3-pro-preview] 使用账号 account-1
01:38:28 | INFO | JWT 刷新成功 [account-1]
01:38:32 | INFO | 📝 提取到 7 个思考步骤 [account-1]
01:38:35 | INFO | 🖼️  检测到 1 个生成图片 [account-1]
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的 Web 框架
- [Gemini Business](https://business.gemini.google/) - Google 的 Gemini Business API

## 📞 支持

如有问题或建议，请提交 Issue。




