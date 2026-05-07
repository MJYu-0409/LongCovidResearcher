"""
storage/postgres/user_store.py

users 表：用户账号持久化。
- password_hash：bcrypt 单向哈希
- phone_encrypted：Fernet (AES-128-CBC) 可逆加密，key 来自 FIELD_ENCRYPTION_KEY 环境变量
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import MetaData, Table, Column, String, Text, DateTime, inspect
from sqlalchemy.dialects.postgresql import UUID

from config import FIELD_ENCRYPTION_KEY
from infra.clients import get_pg_engine

logger = logging.getLogger(__name__)

TABLE_NAME = "users"
_metadata = MetaData()

users = Table(
    TABLE_NAME,
    _metadata,
    Column("user_id",         UUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("username",        String(64),  nullable=False, unique=True),
    Column("password_hash",   String(256), nullable=False),
    Column("phone_encrypted", Text(),      nullable=True),
    Column("created_at",      DateTime(timezone=True), nullable=False),
    Column("updated_at",      DateTime(timezone=True), nullable=False),
)


def create_table_if_not_exists(engine=None) -> None:
    """建表（已存在则跳过）。"""
    eng = engine or get_pg_engine()
    _metadata.create_all(eng)
    logger.info("users 表已就绪")


# ── 加密工具 ──────────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    if not FIELD_ENCRYPTION_KEY:
        raise ValueError(
            "FIELD_ENCRYPTION_KEY 未配置。"
            "请用 python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" 生成并写入 .env"
        )
    return Fernet(FIELD_ENCRYPTION_KEY.encode())


def encrypt_phone(phone: str) -> str:
    """将手机号加密为 base64 密文字符串。"""
    return _get_fernet().encrypt(phone.encode()).decode()


def decrypt_phone(cipher: str) -> str:
    """将 Fernet 密文解密为原始手机号。解密失败时抛出 InvalidToken。"""
    try:
        return _get_fernet().decrypt(cipher.encode()).decode()
    except InvalidToken:
        logger.error("手机号解密失败，密钥可能已轮换")
        raise


# ── 密码工具 ──────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """bcrypt 哈希密码，返回可直接存库的字符串。"""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码与 bcrypt 哈希是否匹配。"""
    return bcrypt.checkpw(plain.encode(), hashed.encode())
