"""
infra - 基础设施：外部服务连接、日志等。

统一提供 OpenAI、Qdrant 等客户端，供 data_pipeline 与 retrieval 共用。
日志在入口调用 configure_logging() 一次即可。
"""

from infra.clients import (
    get_openai_client,
    get_pg_engine,
    get_qdrant_client,
    get_qwen_chat_model,
    get_rerank_model,
    get_sparse_embedding_model,
)
from infra.logging_config import configure_logging

__all__ = [
    "get_openai_client",
    "get_pg_engine",
    "get_qdrant_client",
    "get_qwen_chat_model",
    "get_rerank_model",
    "get_sparse_embedding_model",
    "configure_logging",
]
