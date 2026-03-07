"""
infra/clients.py

统一创建 OpenAI、Qdrant、PostgreSQL 等连接，从 config 读环境变量。
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, DATABASE_URL

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None
_qdrant_client: Optional[QdrantClient] = None
_pg_engine: Optional[Engine] = None


def get_openai_client() -> OpenAI:
    """返回 OpenAI 客户端（单例）。未配置 OPENAI_API_KEY 时抛出 ValueError。"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未配置")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    """返回 Qdrant 客户端（单例）。未配置 QDRANT_URL 时抛出 ValueError。"""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_URL:
            raise ValueError("QDRANT_URL 未配置")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    return _qdrant_client


def get_pg_engine() -> Engine:
    """
    返回 PostgreSQL 的 SQLAlchemy Engine（单例，复用连接池）。
    未配置 DATABASE_URL 时抛出 ValueError。
    """
    global _pg_engine
    if _pg_engine is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL 未配置")
        _pg_engine = create_engine(DATABASE_URL)
    return _pg_engine
