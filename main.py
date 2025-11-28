import json
import time
import hmac
import hashlib
import base64
import os
import asyncio
import uuid
import ssl
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect  # noqa: F401
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, get_db, Admin, APIKey, APICallLog
from auth import (
    hash_password, verify_password, create_access_token, 
    generate_api_key, hash_api_key, get_current_admin, init_admin,
    encrypt_api_key, decrypt_api_key
)


# ---------- 本地 .env 加载（便于直接 python main.py 运行） ----------
def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # 本地加载失败直接忽略，保持与原环境一致
        pass


_load_env_file()


# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gemini")

# ---------- 时区配置 ----------
# 北京时间 UTC+8
BEIJING_TZ = timezone(timedelta(hours=8))

def get_beijing_time():
    """获取当前北京时间"""
    return datetime.now(BEIJING_TZ)

def ensure_aware(dt: datetime) -> datetime:
    """确保 datetime 是 aware（有时区信息）"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # 如果是 naive datetime，假设它是北京时间并添加时区信息
        return dt.replace(tzinfo=BEIJING_TZ)
    return dt

def ensure_naive(dt: datetime) -> datetime:
    """确保 datetime 是 naive（无时区信息），转换为北京时间后移除时区"""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # 转换为北京时间后移除时区信息
        return dt.astimezone(BEIJING_TZ).replace(tzinfo=None)
    return dt

# ---------- 配置 ----------
SECURE_C_SES = os.getenv("SECURE_C_SES")
HOST_C_OSES = os.getenv("HOST_C_OSES")
CSESIDX = os.getenv("CSESIDX")
CONFIG_ID = os.getenv("CONFIG_ID")
PROXY = os.getenv("PROXY") or None
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "600"))

# ---------- 图片生成相关常量 ----------
BASE_DIR = Path(__file__).resolve().parent
IMAGE_SAVE_DIR = BASE_DIR / "generated_images"
LIST_FILE_METADATA_URL = "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetListSessionFileMetadata"

# ---------- 图片数据类 ----------
@dataclass
class ChatImage:
    """表示生成的图片"""
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    base64_data: Optional[str] = None
    url: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: str = "image/png"

    def save_to_file(self, directory: Optional[Path] = None) -> str:
        """保存图片到本地文件，返回文件路径"""
        if self.local_path and os.path.exists(self.local_path):
            return self.local_path

        save_dir = directory or IMAGE_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        ext = ".png"
        if self.mime_type:
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
            }
            ext = ext_map.get(self.mime_type, ".png")

        if self.file_name:
            filename = self.file_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_{timestamp}_{uuid.uuid4().hex[:8]}{ext}"

        filepath = os.path.join(save_dir, filename)

        if self.base64_data:
            image_data = base64.b64decode(self.base64_data)
            with open(filepath, "wb") as f:
                f.write(image_data)
            self.local_path = filepath

        return filepath

# ---------- 模型映射配置 ----------
MODEL_MAPPING: Dict[str, Optional[str]] = {
    "gemini-auto": None,
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro-preview": "gemini-3-pro-preview"
}

# ---------- 全局 Session 缓存 ----------
# key: conversation_key -> {"session_id": str, "updated_at": float, "account": str}
SESSION_CACHE: Dict[str, Dict[str, Any]] = {}

# ---------- WebSocket 管理 ----------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# ---------- HTTP 客户端 ----------
http_client = httpx.AsyncClient(
    proxies=PROXY,
    verify=False,
    http2=False,
    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=60.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
)


# ---------- 通用工具函数 ----------
def get_common_headers(jwt: str) -> dict:
    return {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "authorization": f"Bearer {jwt}",
        "content-type": "application/json",
        "origin": "https://business.gemini.google",
        "referer": "https://business.gemini.google/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "x-server-timeout": "1800",
        "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
    }


def urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def kq_encode(s: str) -> str:
    b = bytearray()
    for ch in s:
        v = ord(ch)
        if v > 255:
            b.append(v & 255)
            b.append(v >> 8)
        else:
            b.append(v)
    return urlsafe_b64encode(bytes(b))


def create_jwt(key_bytes: bytes, key_id: str, csesidx: str) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT", "kid": key_id}
    payload = {
        "iss": "https://business.gemini.google",
        "aud": "https://biz-discoveryengine.googleapis.com",
        "sub": f"csesidx/{csesidx}",
        "iat": now,
        "exp": now + 300,
        "nbf": now,
    }
    header_b64 = kq_encode(json.dumps(header, separators=(",", ":")))
    payload_b64 = kq_encode(json.dumps(payload, separators=(",", ":")))
    message = f"{header_b64}.{payload_b64}"
    sig = hmac.new(key_bytes, message.encode(), hashlib.sha256).digest()
    return f"{message}.{urlsafe_b64encode(sig)}"


# ---------- JWT 与账号管理 ----------
class JWTManager:
    def __init__(self, account: "Account") -> None:
        self.account = account
        self.jwt: str = ""
        self.expires: float = 0
        self._lock = asyncio.Lock()

    async def get(self) -> str:
        async with self._lock:
            if time.time() > self.expires:
                await self._refresh()
            return self.jwt

    async def _refresh(self) -> None:
        cookie = f"__Secure-C_SES={self.account.secure_c_ses}"
        if self.account.host_c_oses:
            cookie += f"; __Host-C_OSES={self.account.host_c_oses}"

        logger.debug(f"🔑 正在刷新 JWT... 账号={self.account.name}")
        r = await http_client.get(
            "https://business.gemini.google/auth/getoxsrf",
            params={"csesidx": self.account.csesidx},
            headers={
                "cookie": cookie,
                "user-agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/140.0.0.0 Safari/537.36"
                ),
                "referer": "https://business.gemini.google/",
            },
        )
        if r.status_code != 200:
            logger.error(
                f"getoxsrf 失败 [{self.account.name}]: {r.status_code} {r.text}"
            )
            if r.status_code in (401, 403, 429):
                self.account.mark_quota_error(r.status_code, r.text)
            raise HTTPException(r.status_code, "getoxsrf failed")

        txt = r.text[4:] if r.text.startswith(")]}'") else r.text
        data = json.loads(txt)

        key_bytes = base64.urlsafe_b64decode(data["xsrfToken"] + "==")
        self.jwt = create_jwt(key_bytes, data["keyId"], self.account.csesidx)
        self.expires = time.time() + 270
        logger.info(f"JWT 刷新成功 [{self.account.name}]")


class Account:
    def __init__(
        self,
        name: str,
        secure_c_ses: str,
        csesidx: str,
        config_id: str,
        host_c_oses: Optional[str] = None,
    ) -> None:
        self.name = name
        self.secure_c_ses = secure_c_ses
        self.host_c_oses = host_c_oses
        self.csesidx = csesidx
        self.config_id = config_id
        self.jwt_mgr = JWTManager(self)
        self.disabled_until: float = 0.0

    def is_available(self) -> bool:
        return time.time() >= self.disabled_until

    def mark_quota_error(self, status_code: int, detail: str = "") -> None:
        cooldown_seconds = 300  # 暂停 5 分钟
        self.disabled_until = max(self.disabled_until, time.time() + cooldown_seconds)
        logger.warning(f"账号[{self.name}] 暂时标记为不可用 (status={status_code})")
        if detail:
            logger.debug(f"账号[{self.name}] 错误详情: {detail[:200]}")


class AccountPool:
    def __init__(self, accounts: List[Account]) -> None:
        if not accounts:
            raise RuntimeError("No Gemini business accounts configured")
        self.accounts = accounts
        self._rr_index = 0

    def _next_round_robin(self) -> Account:
        n = len(self.accounts)
        for _ in range(n):
            acc = self.accounts[self._rr_index % n]
            self._rr_index = (self._rr_index + 1) % n
            if acc.is_available():
                return acc
        # 如果都被标记为不可用，仍然返回一个账号避免完全瘫痪
        return self.accounts[0]

    def get_for_conversation(self, conv_key: str) -> Account:
        cached = SESSION_CACHE.get(conv_key)
        if cached:
            acc_name = cached.get("account")
            for acc in self.accounts:
                if acc.name == acc_name and acc.is_available():
                    return acc
        # 没有缓存或账号不可用，走轮询
        return self._next_round_robin()

    def get_alternative(self, exclude_name: str) -> Optional[Account]:
        for acc in self.accounts:
            if acc.name != exclude_name and acc.is_available():
                return acc
        return None


def load_accounts_from_env() -> List[Account]:
    accounts: List[Account] = []

    # 支持 ACCOUNT1_*, ACCOUNT2_*... 多账号配置
    account_indices = set()
    for key in os.environ.keys():
        if key.startswith("ACCOUNT") and key.endswith("_SECURE_C_SES"):
            idx_str = key[len("ACCOUNT") : -len("_SECURE_C_SES")]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            account_indices.add(idx)

    for idx in sorted(account_indices):
        prefix = f"ACCOUNT{idx}_"
        secure = os.getenv(prefix + "SECURE_C_SES")
        csesidx = os.getenv(prefix + "CSESIDX")
        config_id = os.getenv(prefix + "CONFIG_ID")
        host = os.getenv(prefix + "HOST_C_OSES")
        if not (secure and csesidx and config_id):
            logger.warning(f"账号索引 {idx} 配置不完整，已跳过")
            continue
        name = os.getenv(prefix + "NAME") or f"account-{idx}"
        accounts.append(
            Account(
                name=name,
                secure_c_ses=secure,
                csesidx=csesidx,
                config_id=config_id,
                host_c_oses=host,
            )
        )

    # 兼容旧的单账号环境变量
    if not accounts and SECURE_C_SES and CSESIDX and CONFIG_ID:
        accounts.append(
            Account(
                name="default",
                secure_c_ses=SECURE_C_SES,
                csesidx=CSESIDX,
                config_id=CONFIG_ID,
                host_c_oses=HOST_C_OSES,
            )
        )

    return accounts


ACCOUNTS: List[Account] = load_accounts_from_env()
ACCOUNT_POOL: Optional["AccountPool"]
if ACCOUNTS:
    ACCOUNT_POOL = AccountPool(ACCOUNTS)
else:
    ACCOUNT_POOL = None


# ---------- Session & File 管理 ----------
async def create_google_session(account: Account) -> str:
    jwt = await account.jwt_mgr.get()
    headers = get_common_headers(jwt)
    body = {
        "configId": account.config_id,
        "additionalParams": {"token": "-"},
        "createSessionRequest": {"session": {"name": "", "displayName": ""}},
    }

    logger.debug(f"🌐 申请 Session... 账号={account.name}")
    r = await http_client.post(
        "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetCreateSession",
        headers=headers,
        json=body,
    )
    if r.status_code != 200:
        logger.error(
            f"createSession 失败 [{account.name}]: {r.status_code} {r.text}"
        )
        if r.status_code in (401, 403, 429):
            account.mark_quota_error(r.status_code, r.text)
        raise HTTPException(r.status_code, "createSession failed")
    sess_name = r.json()["session"]["name"]
    return sess_name


async def upload_context_file(
    account: Account, session_name: str, mime_type: str, base64_content: str
) -> str:
    """上传文件到指定 Session，返回 fileId"""
    jwt = await account.jwt_mgr.get()
    headers = get_common_headers(jwt)

    ext = mime_type.split("/")[-1] if "/" in mime_type else "bin"
    file_name = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}.{ext}"

    body = {
        "configId": account.config_id,
        "additionalParams": {"token": "-"},
        "addContextFileRequest": {
            "name": session_name,
            "fileName": file_name,
            "mimeType": mime_type,
            "fileContents": base64_content,
        },
    }

    logger.info(f"📤 上传图片 [{mime_type}] 到 Session，账号={account.name}")
    r = await http_client.post(
        "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetAddContextFile",
        headers=headers,
        json=body,
    )

    if r.status_code != 200:
        logger.error(
            f"上传文件失败 [{account.name}]: {r.status_code} {r.text}"
        )
        if r.status_code in (401, 403, 429):
            account.mark_quota_error(r.status_code, r.text)
        raise HTTPException(r.status_code, f"Upload failed: {r.text}")

    data = r.json()
    file_id = data.get("addContextFileResponse", {}).get("fileId")
    logger.info(f"图片上传成功, ID: {file_id}, 账号={account.name}")
    return file_id


# ---------- 消息处理逻辑 ----------
def get_conversation_key(messages: List[dict]) -> str:
    if not messages:
        return "empty"
    first_msg = messages[0].copy()
    if isinstance(first_msg.get("content"), list):
        text_part = "".join(
            [x.get("text", "") for x in first_msg["content"] if x.get("type") == "text"]
        )
        first_msg["content"] = text_part

    key_str = json.dumps(first_msg, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def parse_last_message(messages: List["Message"]):
    """解析最后一条消息，分离文本和图片"""
    if not messages:
        return "", []

    last_msg = messages[-1]
    content = last_msg.content

    text_content = ""
    images = []  # List of {"mime": str, "data": str_base64}

    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                text_content += part.get("text", "")
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                match = re.match(r"data:(image/[^;]+);base64,(.+)", url)
                if match:
                    images.append(
                        {"mime": match.group(1), "data": match.group(2)}
                    )
                else:
                    logger.warning(
                        f"暂不支持非 Base64 图片链接: {url[:30]}..."
                    )

    return text_content, images


def build_full_context_text(messages: List["Message"]) -> str:
    """仅拼接历史文本，图片只处理当次请求的"""
    prompt = ""
    for msg in messages:
        role = "User" if msg.role in ["user", "system"] else "Assistant"
        content_str = ""
        if isinstance(msg.content, str):
            content_str = msg.content
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") == "text":
                    content_str += part.get("text", "")
                elif part.get("type") == "image_url":
                    content_str += "[图片]"

        prompt += f"{role}: {content_str}\n\n"
    return prompt


# ---------- 图片生成处理方法 ----------
async def get_session_file_metadata(account: Account, session_name: str) -> dict:
    """获取 session 中的文件元数据，包括下载链接"""
    jwt = await account.jwt_mgr.get()
    headers = get_common_headers(jwt)
    body = {
        "configId": account.config_id,
        "additionalParams": {"token": "-"},
        "listSessionFileMetadataRequest": {
            "name": session_name,
            "filter": "file_origin_type = AI_GENERATED"
        }
    }

    async with httpx.AsyncClient(proxy=PROXY, verify=False, timeout=30) as cli:
        resp = await cli.post(LIST_FILE_METADATA_URL, headers=headers, json=body)

        if resp.status_code == 401:
            # JWT 过期，刷新后重试
            jwt = await account.jwt_mgr.get()
            headers = get_common_headers(jwt)
            resp = await cli.post(LIST_FILE_METADATA_URL, headers=headers, json=body)

        if resp.status_code != 200:
            logger.warning(f"获取文件元数据失败 [{account.name}]: {resp.status_code}")
            return {}

        data = resp.json()
        result = {}
        file_metadata_list = data.get("listSessionFileMetadataResponse", {}).get("fileMetadata", [])
        for fm in file_metadata_list:
            fid = fm.get("fileId")
            if fid:
                result[fid] = fm

        return result


def build_image_download_url(session_name: str, file_id: str) -> str:
    """构造正确的图片下载 URL"""
    return f"https://biz-discoveryengine.googleapis.com/v1alpha/{session_name}:downloadFile?fileId={file_id}&alt=media"


async def download_image_with_jwt(account: Account, session_name: str, file_id: str) -> bytes:
    """使用 JWT 认证下载图片"""
    url = build_image_download_url(session_name, file_id)
    jwt = await account.jwt_mgr.get()
    headers = get_common_headers(jwt)

    async with httpx.AsyncClient(proxy=PROXY, verify=False, timeout=120) as cli:
        resp = await cli.get(url, headers=headers, follow_redirects=True)

        if resp.status_code == 401:
            # JWT 过期，刷新后重试
            jwt = await account.jwt_mgr.get()
            headers = get_common_headers(jwt)
            resp = await cli.get(url, headers=headers, follow_redirects=True)

        resp.raise_for_status()
        content = resp.content

        # 检测是否为 base64 编码的内容
        try:
            text_content = content.decode("utf-8", errors="ignore").strip()
            if text_content.startswith("iVBORw0KGgo") or text_content.startswith("/9j/"):
                # 是 base64 编码，需要解码
                return base64.b64decode(text_content)
        except Exception:
            pass

        return content


async def save_generated_image(account: Account, session_name: str, file_id: str, file_name: Optional[str], mime_type: str, chat_id: str, image_index: int = 1) -> ChatImage:
    """下载并保存生成的图片，按 chat_id 命名"""
    img = ChatImage(
        file_id=file_id,
        file_name=file_name,
        mime_type=mime_type,
    )

    try:
        image_data = await download_image_with_jwt(account, session_name, file_id)
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

        ext = ".png"
        ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
        ext = ext_map.get(mime_type, ".png")

        # 按 {chat_id}_{序号}.png 命名
        filename = f"{chat_id}_{image_index}{ext}"
        filepath = IMAGE_SAVE_DIR / filename

        # 如果文件已存在，添加时间戳避免覆盖
        if filepath.exists():
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{chat_id}_{image_index}_{timestamp}{ext}"
            filepath = IMAGE_SAVE_DIR / filename

        with open(filepath, "wb") as f:
            f.write(image_data)

        img.local_path = str(filepath)
        img.file_name = filename
        img.base64_data = base64.b64encode(image_data).decode("utf-8")
        logger.info(f"图片已保存 [{account.name}]: {filepath}")
    except Exception as e:
        logger.error(f"下载图片失败 [{account.name}]: {e}")

    return img


def parse_images_from_response(data_list: list) -> tuple[list, Optional[str]]:
    """
    从 API 响应中解析图片文件引用
    返回: (file_ids_list, current_session)
    file_ids_list: [{"fileId": str, "mimeType": str}, ...]
    """
    file_ids = []
    current_session = None

    for data in data_list:
        sar = data.get("streamAssistResponse")
        if not sar:
            continue

        # 获取 session 信息
        session_info = sar.get("sessionInfo", {})
        if session_info.get("session"):
            current_session = session_info["session"]

        answer = sar.get("answer") or {}
        replies = answer.get("replies") or []

        for reply in replies:
            gc = reply.get("groundedContent", {})
            content = gc.get("content", {})

            # 检查 file 字段（图片生成的关键）
            file_info = content.get("file")
            if file_info and file_info.get("fileId"):
                file_ids.append({
                    "fileId": file_info["fileId"],
                    "mimeType": file_info.get("mimeType", "image/png")
                })

    return file_ids, current_session


# ---------- OpenAI 兼容接口 ----------
app = FastAPI(title="Gemini-Business OpenAI Gateway")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

security_bearer = HTTPBearer()


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatRequest(BaseModel):
    model: str = "gemini-auto"
    messages: List[Message]
    stream: bool = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0


def create_chunk(
    id: str, created: int, model: str, delta: dict, finish_reason: Optional[str]
) -> str:
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return json.dumps(chunk, ensure_ascii=False)


@app.get("/v1/models")
async def list_models():
    data = []
    now = int(time.time())
    for m in MODEL_MAPPING.keys():
        data.append(
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "google",
                "permission": [],
            }
        )
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    return {"status": "ok", "time": get_beijing_time().isoformat()}


@app.get("/")
async def root():
    """重定向到登录页面"""
    return RedirectResponse(url="/static/index.html")


@app.websocket("/ws/admin/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------- 管理员认证接口 ----------
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/admin/login", response_model=LoginResponse)
async def admin_login(req: LoginRequest, db: Session = Depends(get_db)):
    """管理员登录"""
    admin = db.query(Admin).filter(Admin.username == req.username).first()
    
    if not admin or not verify_password(req.password, admin.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误"
        )
    
    access_token = create_access_token({"sub": admin.username})
    return LoginResponse(access_token=access_token)


# ---------- API 密钥管理接口 ----------
class GenerateKeysRequest(BaseModel):
    count: int = 1
    expires_days: int = 30
    name_prefix: str = "API Key"


class APIKeyResponse(BaseModel):
    id: int
    key: Optional[str] = None  # 仅在生成时返回明文
    name: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
    usage_count: int
    last_used_at: Optional[datetime]


@app.post("/admin/api-keys", response_model=List[APIKeyResponse])
async def generate_api_keys(
    req: GenerateKeysRequest,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """批量生成 API 密钥"""
    if req.count < 1 or req.count > 100:
        raise HTTPException(status_code=400, detail="数量必须在 1-100 之间")
    
    if req.expires_days < 1 or req.expires_days > 3650:
        raise HTTPException(status_code=400, detail="有效期必须在 1-3650 天之间")
    
    keys = []
    # 存储时使用 naive datetime（数据库兼容）
    expires_at = ensure_naive(get_beijing_time() + timedelta(days=req.expires_days))
    
    for i in range(req.count):
        # 生成 UUID 格式密钥
        plain_key = generate_api_key()
        key_hash = hash_api_key(plain_key)
        encrypted = encrypt_api_key(plain_key)  # 加密存储
        
        name = f"{req.name_prefix} #{i+1}" if req.count > 1 else req.name_prefix
        
        api_key = APIKey(
            key_hash=key_hash,
            encrypted_key=encrypted,  # 存储加密后的密钥
            name=name,
            expires_at=expires_at,
            is_active=True
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        keys.append(APIKeyResponse(
            id=api_key.id,
            key=plain_key,  # 仅在生成时返回明文
            name=api_key.name,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
            is_active=api_key.is_active,
            usage_count=api_key.usage_count,
            last_used_at=api_key.last_used_at
        ))
    
    return keys


@app.get("/admin/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """列出所有 API 密钥（仅显示活跃的）"""
    # 按创建时间升序排列（最老的在前）
    keys = db.query(APIKey).filter(APIKey.is_active == True).order_by(APIKey.created_at.asc()).all()
    return [
        APIKeyResponse(
            id=k.id,
            name=k.name,
            created_at=k.created_at,
            expires_at=k.expires_at,
            is_active=k.is_active,
            usage_count=k.usage_count,
            last_used_at=k.last_used_at
        )
        for k in keys
    ]


@app.delete("/admin/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """撤销 API 密钥"""
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    api_key.is_active = False
    db.commit()
    return {"message": "密钥已撤销"}


@app.get("/admin/api-keys/{key_id}/view")
async def view_api_key(
    key_id: int,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """查看 API 密钥明文"""
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    try:
        decrypted_key = decrypt_api_key(api_key.encrypted_key)
        return {"key": decrypted_key}
    except Exception as e:
        logger.error(f"Failed to decrypt key: {e}")
        raise HTTPException(status_code=500, detail="解密失败")


@app.get("/admin/stats")
async def get_stats(
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取统计信息"""
    from sqlalchemy import func
    
    total_keys = db.query(APIKey).count()
    # 获取当前北京时间（naive 格式用于数据库比较）
    now_naive = get_beijing_time().replace(tzinfo=None)
    active_keys = db.query(APIKey).filter(
        APIKey.is_active == True,
        APIKey.expires_at > now_naive
    ).count()
    total_usage = db.query(func.sum(APIKey.usage_count)).filter(
        APIKey.is_active == True
    ).scalar() or 0
    
    return {
        "total_keys": total_keys,
        "active_keys": active_keys,
        "total_usage": total_usage
    }


@app.get("/admin/api-keys/{key_id}/logs")
async def get_api_key_logs(
    key_id: int,
    page: int = 1,
    page_size: int = 50,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取指定 API 密钥的调用日志"""
    # 验证密钥是否存在
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    # 查询日志
    offset = (page - 1) * page_size
    logs = db.query(APICallLog).filter(
        APICallLog.api_key_id == key_id
    ).order_by(
        APICallLog.timestamp.desc()
    ).offset(offset).limit(page_size).all()
    
    # 获取总数
    total = db.query(APICallLog).filter(APICallLog.api_key_id == key_id).count()
    
    return {
        "key_id": key_id,
        "key_name": api_key.name,
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "model": log.model,
                "status": log.status,
                "error_message": log.error_message,
                "ip_address": log.ip_address,
                "endpoint": log.endpoint,
                "response_time": log.response_time
            }
            for log in logs
        ]
    }


@app.get("/admin/api-keys/{key_id}/stats")
async def get_api_key_stats(
    key_id: int,
    admin: Admin = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """获取指定 API 密钥的统计信息"""
    from sqlalchemy import func
    
    # 验证密钥是否存在
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not api_key:
        raise HTTPException(status_code=404, detail="密钥不存在")
    
    # 总调用次数
    total_calls = db.query(APICallLog).filter(APICallLog.api_key_id == key_id).count()
    
    # 成功/失败统计
    success_calls = db.query(APICallLog).filter(
        APICallLog.api_key_id == key_id,
        APICallLog.status == "success"
    ).count()
    
    error_calls = db.query(APICallLog).filter(
        APICallLog.api_key_id == key_id,
        APICallLog.status == "error"
    ).count()
    
    # 按模型统计
    model_stats = db.query(
        APICallLog.model,
        func.count(APICallLog.id).label("count")
    ).filter(
        APICallLog.api_key_id == key_id
    ).group_by(APICallLog.model).all()
    
    # 平均响应时间
    avg_response_time = db.query(
        func.avg(APICallLog.response_time)
    ).filter(
        APICallLog.api_key_id == key_id,
        APICallLog.response_time.isnot(None)
    ).scalar() or 0
    
    # 最近 7 天的调用趋势（使用 naive datetime 用于数据库查询）
    seven_days_ago = ensure_naive(get_beijing_time() - timedelta(days=7))
    
    daily_stats = db.query(
        func.date(APICallLog.timestamp).label("date"),
        func.count(APICallLog.id).label("count")
    ).filter(
        APICallLog.api_key_id == key_id,
        APICallLog.timestamp >= seven_days_ago
    ).group_by(
        func.date(APICallLog.timestamp)
    ).order_by(
        func.date(APICallLog.timestamp)
    ).all()
    
    return {
        "key_id": key_id,
        "key_name": api_key.name,
        "total_calls": total_calls,
        "success_calls": success_calls,
        "error_calls": error_calls,
        "success_rate": round(success_calls / total_calls * 100, 2) if total_calls > 0 else 0,
        "avg_response_time": round(avg_response_time, 2),
        "model_stats": [
            {"model": m[0], "count": m[1]}
            for m in model_stats
        ],
        "daily_stats": [
            {"date": str(d[0]), "count": d[1]}
            for d in daily_stats
        ]
    }


# ---------- API 密钥验证中间件 ----------
async def verify_api_key_middleware(request: Request, call_next):
    """验证 API 密钥"""
    from fastapi.responses import JSONResponse
    import time as time_module
    
    # 只对 /v1/ 开头的路径进行验证
    if request.url.path.startswith("/v1/"):
        auth_header = request.headers.get("Authorization")
        start_time = time_module.time()
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少 API 密钥"}
            )
        
        api_key = auth_header.replace("Bearer ", "")
        key_hash = hash_api_key(api_key)
        
        # 获取数据库会话
        db = next(get_db())
        db_key = None
        client_ip = request.client.host if request.client else "unknown"
        
        try:
            db_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            
            if not db_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "无效的 API 密钥"}
                )
            
            if not db_key.is_active:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API 密钥已被撤销"}
                )
            
            # 确保 expires_at 是 aware datetime 以便比较
            expires_at = ensure_aware(db_key.expires_at)
            if expires_at < get_beijing_time():
                return JSONResponse(
                    status_code=401,
                    content={"detail": "API 密钥已过期"}
                )
            
            # 更新使用统计（存储时使用 naive datetime）
            db_key.usage_count += 1
            db_key.last_used_at = ensure_naive(get_beijing_time())
            db.commit()
            
            # 存储到请求状态，用于后续记录
            request.state.api_key_id = db_key.id
            request.state.start_time = start_time
            request.state.client_ip = client_ip
            
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "服务器内部错误"}
            )
        finally:
            db.close()
    
    # 执行实际请求
    response = await call_next(request)
    
    # 记录调用日志
    if request.url.path.startswith("/v1/") and hasattr(request.state, "api_key_id"):
        db = next(get_db())
        try:
            # 尝试从请求体获取模型信息
            model = "unknown"
            if hasattr(request.state, "model"):
                model = request.state.model
            
            response_time = int((time_module.time() - request.state.start_time) * 1000)
            
            log_entry = APICallLog(
                api_key_id=request.state.api_key_id,
                timestamp=ensure_naive(get_beijing_time()),
                model=model,
                status="success" if response.status_code < 400 else "error",
                error_message=None if response.status_code < 400 else f"HTTP {response.status_code}",
                ip_address=request.state.client_ip,
                endpoint=request.url.path,
                response_time=response_time
            )
            db.add(log_entry)
            db.commit()
            
            # 广播更新消息
            await manager.broadcast("update")
        except Exception as e:
            logger.error(f"Failed to log API call: {e}")
        finally:
            db.close()
    
    return response

app.middleware("http")(verify_api_key_middleware)


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    # 记录模型到请求状态
    request.state.model = req.model
    
    # 1. 模型校验
    if req.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=404, detail=f"Model '{req.model}' not found."
        )

    if ACCOUNT_POOL is None:
        raise HTTPException(
            status_code=500,
            detail="No Gemini business accounts configured",
        )

    # 2. 解析请求内容
    last_text, current_images = parse_last_message(req.messages)

    # 3. 锚定对话 & 账号
    conv_key = get_conversation_key([m.dict() for m in req.messages])
    account = ACCOUNT_POOL.get_for_conversation(conv_key)
    cached = SESSION_CACHE.get(conv_key)

    if cached:
        google_session = cached["session_id"]
        text_to_send = last_text
        logger.info(
            f"♻️ 延续旧对话[{req.model}] 账号={account.name} session={google_session[-12:]}"
        )
        cached["updated_at"] = time.time()
        is_retry_mode = False
    else:
        logger.info(f"🆕 开启新对话 [{req.model}] 使用账号 {account.name}")

        google_session: Optional[str] = None
        first_error: Optional[Exception] = None
        tried_accounts = set()

        while True:
            tried_accounts.add(account.name)
            try:
                google_session = await create_google_session(account)
                break
            except HTTPException as e:
                if e.status_code in (401, 403, 429):
                    account.mark_quota_error(e.status_code, str(e.detail))
                    alt = ACCOUNT_POOL.get_alternative(account.name)
                    if not alt or alt.name in tried_accounts:
                        first_error = e
                        break
                    logger.info(
                        f"createSession 配额受限，切换账号 {account.name} -> {alt.name}"
                    )
                    account = alt
                    continue
                first_error = e
                break
            except Exception as e:
                first_error = e
                break

        if not google_session:
            if isinstance(first_error, HTTPException):
                raise first_error
            raise HTTPException(status_code=500, detail="No available account")

        text_to_send = build_full_context_text(req.messages)
        SESSION_CACHE[conv_key] = {
            "session_id": google_session,
            "updated_at": time.time(),
            "account": account.name,
        }
        is_retry_mode = True

    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # 4. 封装生成逻辑（含图片上传、重试和账号切换）
    async def response_wrapper():
        nonlocal account

        retry_count = 0
        max_retries = 2

        current_text = text_to_send
        current_retry_mode = is_retry_mode
        current_file_ids: List[str] = []

        while retry_count <= max_retries:
            try:
                cached_session = SESSION_CACHE.get(conv_key)
                if not cached_session:
                    new_sess = await create_google_session(account)
                    SESSION_CACHE[conv_key] = {
                        "session_id": new_sess,
                        "updated_at": time.time(),
                        "account": account.name,
                    }
                    cached_session = SESSION_CACHE[conv_key]

                current_session = cached_session["session_id"]

                # A. 如果有图片且还没上传到当前 Session，先上传
                if current_images and not current_file_ids:
                    for img in current_images:
                        fid = await upload_context_file(
                            account, current_session, img["mime"], img["data"]
                        )
                        current_file_ids.append(fid)

                # B. 准备文本 (重试模式下发全文)
                if current_retry_mode:
                    current_text = build_full_context_text(req.messages)

                # C. 发起对话
                async for chunk in stream_chat_generator(
                    account,
                    current_session,
                    current_text,
                    current_file_ids,
                    req.model,
                    chat_id,
                    created_time,
                    req.stream,
                ):
                    yield chunk
                break

            except (httpx.ConnectError, httpx.ReadTimeout, ssl.SSLError, HTTPException) as e:
                retry_count += 1
                logger.warning(
                    f"⚠️ 请求异常 (重试 {retry_count}/{max_retries}) 账号={account.name}: {e}"
                )

                status_code = getattr(e, "status_code", None)

                # 先判定配额/权限类错误，尝试切换账号
                if isinstance(e, HTTPException) and status_code in (401, 403, 429):
                    account.mark_quota_error(status_code, str(e.detail))
                    alt = ACCOUNT_POOL.get_alternative(account.name)
                    if alt:
                        logger.info(f"🔁 切换到备用账号 {alt.name}")
                        account = alt
                        try:
                            new_sess = await create_google_session(account)
                            SESSION_CACHE[conv_key] = {
                                "session_id": new_sess,
                                "updated_at": time.time(),
                                "account": account.name,
                            }
                            current_retry_mode = True
                            current_file_ids = []
                            continue
                        except Exception as create_err:
                            logger.error(
                                f"备用账号创建 Session 失败: {create_err}"
                            )
                            if req.stream:
                                yield "data: " + json.dumps(
                                    {
                                        "error": {
                                            "message": "All accounts exhausted"
                                        }
                                    }
                                ) + "\n\n"
                                return
                            raise

                # 非配额错误或切换失败，尝试当前账号重建 session
                if retry_count <= max_retries:
                    logger.info("🔄 尝试重建 Session...")
                    try:
                        new_sess = await create_google_session(account)
                        SESSION_CACHE[conv_key] = {
                            "session_id": new_sess,
                            "updated_at": time.time(),
                            "account": account.name,
                        }
                        current_retry_mode = True
                        current_file_ids = []
                    except Exception as create_err:
                        logger.error(f"Session 重建失败: {create_err}")
                        if req.stream:
                            yield "data: " + json.dumps(
                                {
                                    "error": {
                                        "message": "Session Recovery Failed"
                                    }
                                }
                            ) + "\n\n"
                            return
                        raise
                else:
                    if req.stream:
                        yield "data: " + json.dumps(
                            {"error": {"message": f"Final Error: {e}"}}
                        ) + "\n\n"
                        return
                    raise

    if req.stream:
        return StreamingResponse(response_wrapper(), media_type="text/event-stream")

    full_content = ""
    async for chunk_str in response_wrapper():
        if chunk_str.startswith("data: [DONE]"):
            break
        if chunk_str.startswith("data: "):
            try:
                data = json.loads(chunk_str[6:])
                delta = data["choices"][0]["delta"]
                if "content" in delta:
                    full_content += delta["content"]
            except Exception:
                pass

    response_data = {
        "id": chat_id,
        "object": "chat.completion",
        "created": created_time,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    return response_data


async def stream_chat_generator(
    account: Account,
    session: str,
    text_content: str,
    file_ids: List[str],
    model_name: str,
    chat_id: str,
    created_time: int,
    is_stream: bool = True,
):
    jwt = await account.jwt_mgr.get()
    headers = get_common_headers(jwt)

    body: Dict[str, Any] = {
        "configId": account.config_id,
        "additionalParams": {"token": "-"},
        "streamAssistRequest": {
            "session": session,
            "query": {"parts": [{"text": text_content}]},
            "filter": "",
            "fileIds": file_ids,
            "answerGenerationMode": "NORMAL",
            "toolsSpec": {
                "webGroundingSpec": {},
                "toolRegistry": "default_tool_registry",
                "imageGenerationSpec": {},
                "videoGenerationSpec": {},
            },
            "languageCode": "zh-CN",
            "userMetadata": {"timeZone": "Asia/Shanghai"},
            "assistSkippingMode": "REQUEST_ASSIST",
        },
    }

    target_model_id = MODEL_MAPPING.get(model_name)
    if target_model_id:
        body["streamAssistRequest"]["assistGenerationConfig"] = {
            "modelId": target_model_id
        }

    if is_stream:
        chunk = create_chunk(
            chat_id, created_time, model_name, {"role": "assistant"}, None
        )
        yield f"data: {chunk}\n\n"

    r = await http_client.post(
        "https://biz-discoveryengine.googleapis.com/v1alpha/locations/global/widgetStreamAssist",
        headers=headers,
        json=body,
    )

    if r.status_code != 200:
        logger.error(
            f"widgetStreamAssist 失败 [{account.name}]: {r.status_code} {r.text}"
        )
        if r.status_code in (401, 403, 429):
            account.mark_quota_error(r.status_code, r.text)
        raise HTTPException(status_code=r.status_code, detail=f"Upstream Error {r.text}")

    try:
        data_list = r.json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"JSON 解析失败 [{account.name}]: {e}")
        raise HTTPException(status_code=502, detail="Invalid JSON response")

    # ========== 第一步：收集文本内容和思考过程 ==========
    full_content = ""
    generated_images: List[ChatImage] = []
    thinking_parts = []

    for data in data_list:
        for reply in (
            data.get("streamAssistResponse", {})
            .get("answer", {})
            .get("replies", [])
        ):
            # 提取 API 返回的 thought 字段
            thought = reply.get("thought", "")
            if thought:
                thinking_parts.append(thought)
            
            # 提取正文内容
            text = (
                reply.get("groundedContent", {})
                .get("content", {})
                .get("text", "")
            )
            if text and not thought:
                full_content += text

    # ========== 第二步：从正文中提取 **标题** 格式的思考标题 ==========
    lines = full_content.splitlines()
    filtered_lines = []
    thinking_titles = []
    
    for line in lines:
        # 匹配单独成行的 **标题** 格式
        m = re.match(r'^\s*\*\*([^*]+)\*\*\s*$', line)
        if m:
            title_text = m.group(1).strip()
            # 判断是否是思考标题：英文短语，长度适中，不超过6个单词
            if (re.match(r'^[A-Za-z\s\'\-\(\)]+$', title_text) and 
                len(title_text) < 100 and 
                len(title_text.split()) <= 6):
                thinking_titles.append(title_text)
                continue  # 跳过思考标题行
        filtered_lines.append(line)
    
    if thinking_titles:
        thinking_parts.extend(thinking_titles)
    
    # 更新正文内容（已移除思考标题）
    full_content = "\n".join(filtered_lines).strip()

    # ========== 第三步：处理并发送思考过程 ==========
    if thinking_parts:
        thinking_content = "\n\n".join(thinking_parts)
        thinking_html = f"<details><summary>显示思路</summary>\n\n{thinking_content}\n\n</details>\n\n"
        full_content = thinking_html + full_content
        
        if is_stream:
            # 发送 thinking 字段（兼容支持该字段的前端）
            thinking_chunk = create_chunk(
                chat_id, created_time, model_name, {"thinking": thinking_content}, None
            )
            yield f"data: {thinking_chunk}\n\n"
        logger.info(f"📝 提取到 {len(thinking_parts)} 个思考步骤 [{account.name}]")

    # ========== 第四步：发送正文内容 ==========
    if full_content:
        chunk = create_chunk(
            chat_id, created_time, model_name, {"content": full_content}, None
        )
        if is_stream:
            yield f"data: {chunk}\n\n"

    # ========== 第五步：解析并下载生成的图片 ==========
    image_file_ids, response_session = parse_images_from_response(data_list)
    
    if response_session and response_session != session:
        logger.info(f"🔄 检测到新 Session [{account.name}]: ...{response_session[-15:]}")

    if image_file_ids:
        logger.info(f"🖼️  检测到 {len(image_file_ids)} 个生成图片 [{account.name}]")
        session_for_download = response_session or session

        try:
            file_metadata = await get_session_file_metadata(account, session_for_download)
            existing_image_count = 0

            for idx, finfo in enumerate(image_file_ids):
                fid = finfo["fileId"]
                mime = finfo["mimeType"]
                meta = file_metadata.get(fid, {})
                file_name = meta.get("name")
                session_path = meta.get("session") or session_for_download

                image_index = existing_image_count + idx + 1
                img = await save_generated_image(account, session_path, fid, file_name, mime, chat_id, image_index)
                generated_images.append(img)

        except Exception as e:
            logger.error(f"❌ 处理生成图片失败 [{account.name}]: {e}", exc_info=True)

    # ========== 第六步：将图片链接添加到文本内容中 ==========
    if generated_images:
        image_markdown = "\n\n"
        for img in generated_images:
            if img.base64_data:
                image_markdown += f"![生成的图片](data:{img.mime_type};base64,{img.base64_data})\n\n"
        
        if image_markdown.strip():
            full_content += image_markdown
            if is_stream:
                chunk = create_chunk(
                    chat_id, created_time, model_name, {"content": image_markdown}, None
                )
                yield f"data: {chunk}\n\n"
            logger.info(f"✅ 已将 {len(generated_images)} 张图片添加到回复中 [{account.name}]")

    if is_stream:
        final_chunk = create_chunk(
            chat_id, created_time, model_name, {}, "stop"
        )
        yield f"data: {final_chunk}\n\n"
        yield "data: [DONE]\n\n"


@app.on_event("startup")
async def _startup_event() -> None:
    """启动时初始化数据库和管理员账号"""
    init_db()
    db = next(get_db())
    try:
        init_admin(db)
    finally:
        db.close()


@app.on_event("shutdown")
async def _shutdown_event() -> None:
    try:
        await http_client.aclose()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    # 至少需要有一个账号配置
    if not ACCOUNTS:
        print("Error: No Gemini business accounts configured.")
        print(
            "Set SECURE_C_SES/CSESIDX/CONFIG_ID or ACCOUNT1_* env variables first."
        )
        raise SystemExit(1)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
