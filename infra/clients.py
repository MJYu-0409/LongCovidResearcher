"""
infra/clients.py

统一创建 OpenAI、Qdrant、PostgreSQL、稀疏向量模型、Qwen Chat 等连接/单例，从 config 读配置。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import (
    DATABASE_URL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    RERANK_MODEL,
    SPARSE_MODEL,
)

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None
_qdrant_client: Optional[QdrantClient] = None
_pg_engine: Optional[Engine] = None
_sparse_model: Any = None
_rerank_model: Any = None
_qwen_chat_cache: dict[Tuple[float, int], Any] = {}


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


def get_sparse_embedding_model():
    """
    返回稀疏向量模型（FastEmbed SPLADE）单例，供 pipeline 与 retrieval 共用。
    首次调用时加载模型（约 50MB），之后复用。
    """
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        logger.info("加载稀疏向量模型（首次运行会下载模型文件）")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def get_rerank_model():
    """
    返回 Rerank 模型（sentence_transformers Cross-Encoder）单例，供 retrieval 精排使用。
    首次调用时加载模型（约 90MB），之后复用。
    """
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        logger.info("加载 Reranking 模型（首次运行会下载模型文件）")
        _rerank_model = CrossEncoder(RERANK_MODEL)
    return _rerank_model


def get_qwen_chat_model(*, temperature: float = 0.1, max_tokens: int = 1500):
    """
    返回 Qwen Chat 模型（LangChain ChatOpenAI，OpenAI 兼容接口）单例。
    按 (temperature, max_tokens) 缓存，相同参数返回同一实例。
    供 Agent 编排、问答、综述使用。
    """
    global _qwen_chat_cache
    key = (temperature, max_tokens)
    if key not in _qwen_chat_cache:
        from langchain_openai import ChatOpenAI
        _qwen_chat_cache[key] = ChatOpenAI(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _qwen_chat_cache[key]
