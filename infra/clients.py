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
    DENSE_MODEL,
    HTTPS_PROXY,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    RERANK_MODEL,
    SPARSE_MODEL,
    REDIS_URL
)

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAI] = None
_qdrant_client: Optional[QdrantClient] = None
_pg_engine: Optional[Engine] = None
_dense_model: Any = None
_sparse_model: Any = None
_rerank_model: Any = None
_qwen_chat_cache: dict[Tuple[float, int], Any] = {}
_redis_client: Optional[Any] = None

def get_openai_client() -> OpenAI:
    """返回 OpenAI 客户端（单例）。未配置 OPENAI_API_KEY 时抛出 ValueError。"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 未配置")
        kwargs: dict = {"api_key": OPENAI_API_KEY}
        if HTTPS_PROXY:
            import httpx
            kwargs["http_client"] = httpx.Client(proxy=HTTPS_PROXY)
        _openai_client = OpenAI(**kwargs)
    return _openai_client


def get_dense_embedding_model():
    """
    返回本地 PubMedBERT dense embedding 模型单例（SentenceTransformer）。
    首次调用时下载模型（~440MB），之后复用。
    """
    global _dense_model
    if _dense_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("加载 dense embedding 模型（首次运行会下载模型文件）: %s", DENSE_MODEL)
        _dense_model = SentenceTransformer(DENSE_MODEL)
    return _dense_model


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


def get_redis_client():
    """返回 Redis 客户端单例。未配置 REDIS_URL 时抛出 ValueError。"""
    global _redis_client
    if _redis_client is None:
        from redis import Redis
        if not REDIS_URL:
            raise ValueError("REDIS_URL 未配置")
        _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client

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
        try:
            # Use local cache after first download; avoids HuggingFace Hub network checks.
            _rerank_model = CrossEncoder(RERANK_MODEL, local_files_only=True)
            logger.info("Reranking 模型已从本地缓存加载: %s", RERANK_MODEL)
        except (OSError, ValueError):
            logger.info("本地缓存未找到，下载 Reranking 模型: %s", RERANK_MODEL)
            _rerank_model = CrossEncoder(RERANK_MODEL)
    return _rerank_model


def get_qwen_chat_model(*, temperature: float = 0.1, max_tokens: int = 1500, timeout: int = 60):
    """
    返回 Qwen Chat 模型（LangChain ChatOpenAI，OpenAI 兼容接口）单例。
    按 (temperature, max_tokens, timeout) 缓存，相同参数返回同一实例。
    供 Agent 编排、问答、综述使用。
    """
    global _qwen_chat_cache
    key = (temperature, max_tokens, timeout)
    if key not in _qwen_chat_cache:
        from langchain_openai import ChatOpenAI
        kwargs: dict = dict(
            model=QWEN_MODEL,
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_BASE,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            model_kwargs={"extra_body": {"enable_thinking": False}},  # Disable Qwen3 thinking tokens
        )
        if HTTPS_PROXY:
            import httpx
            kwargs["http_client"] = httpx.Client(proxy=HTTPS_PROXY)
            kwargs["http_async_client"] = httpx.AsyncClient(proxy=HTTPS_PROXY)
        _qwen_chat_cache[key] = ChatOpenAI(**kwargs)
    return _qwen_chat_cache[key]
