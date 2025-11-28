"""
认证和授权工具
"""
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db, Admin, APIKey

# 硬编码管理员账号
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123456"  # 请修改为您的密码

# JWT 配置
JWT_SECRET = "your-secret-key-change-this-in-production"  # 生产环境请修改
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# 密钥加密配置（用于加密存储API密钥）
ENCRYPTION_KEY = "your-encryption-key-change-this-32ch"  # 必须是32字节
from cryptography.fernet import Fernet
import base64

# 生成Fernet密钥（基于固定密钥）
def get_fernet_key():
    # 将固定密钥转换为32字节并编码为base64
    key = ENCRYPTION_KEY.ljust(32)[:32].encode()
    return base64.urlsafe_b64encode(key)

cipher = Fernet(get_fernet_key())

def encrypt_api_key(key: str) -> str:
    """加密API密钥"""
    return cipher.encrypt(key.encode()).decode()

def decrypt_api_key(encrypted_key: str) -> str:
    """解密API密钥"""
    return cipher.decrypt(encrypted_key.encode()).decode()

security = HTTPBearer()


def hash_password(password: str) -> str:
    """哈希密码"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建 JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """验证 JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_api_key() -> str:
    """生成 UUID 格式的 API 密钥"""
    return str(uuid.uuid4())


def hash_api_key(api_key: str) -> str:
    """哈希 API 密钥"""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Admin:
    """获取当前登录的管理员（用于保护管理接口）"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    admin = db.query(Admin).filter(Admin.username == username).first()
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin user not found"
        )
    
    return admin


def init_admin(db: Session):
    """初始化默认管理员账号"""
    existing_admin = db.query(Admin).filter(Admin.username == ADMIN_USERNAME).first()
    if not existing_admin:
        admin = Admin(
            username=ADMIN_USERNAME,
            hashed_password=hash_password(ADMIN_PASSWORD)
        )
        db.add(admin)
        db.commit()
        print(f"✅ 创建默认管理员账号: {ADMIN_USERNAME}")
    else:
        print(f"ℹ️  管理员账号已存在: {ADMIN_USERNAME}")
