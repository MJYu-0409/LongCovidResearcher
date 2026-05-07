"""
storage/qdrant/memory_store.py

用户跨会话记忆：每轮对话结束后将 summary 向量化存入 user_memories collection，
新会话开始时按 user_id 过滤检索最相关历史记忆注入 LLM 上下文。
仅使用 dense 向量（1536维，COSINE），无需 sparse。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import USER_MEMORIES_COLLECTION
from infra.clients import get_qdrant_client

logger = logging.getLogger(__name__)

DENSE_DIM = 1536


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_point_id(user_id: str, session_id: str) -> str:
    """同一 user+session 永远映射到同一 UUID，保证 upsert 幂等。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{session_id}"))


def ensure_collection() -> None:
    """确保 user_memories collection 存在，幂等。"""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if USER_MEMORIES_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=USER_MEMORIES_COLLECTION,
        vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
    )
    logger.info("Qdrant collection %s 创建成功", USER_MEMORIES_COLLECTION)


def upsert_memory(user_id: str, session_id: str, summary: str) -> None:
    """
    将本轮 summary 向量化后 upsert 到 user_memories。
    相同 user_id+session_id 的条目会被覆盖（幂等），每轮写入自动更新。
    """
    if not summary.strip():
        return
    try:
        from retrieval.dense import embed_query
        ensure_collection()
        vector = embed_query(summary)
        point_id = _make_point_id(user_id, session_id)
        get_qdrant_client().upsert(
            collection_name=USER_MEMORIES_COLLECTION,
            points=[PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "user_id":    user_id,
                    "session_id": session_id,
                    "summary":    summary,
                    "created_at": _now_iso(),
                },
            )],
        )
        logger.info("跨会话记忆已写入 Qdrant: user_id=%s session_id=%s", user_id, session_id)
    except Exception as e:
        logger.warning("写入 user_memories 失败（不影响主流程）: %s", e)


def retrieve_memories(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """
    检索该用户最相关的历史 session summary，返回文本列表。
    user_memories 为空或检索失败时返回空列表。
    """
    if not query.strip():
        return []
    try:
        from retrieval.dense import embed_query
        vector = embed_query(query)
        results = get_qdrant_client().search(
            collection_name=USER_MEMORIES_COLLECTION,
            query_vector=vector,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            with_payload=True,
        )
        summaries = [r.payload["summary"] for r in results if r.payload.get("summary")]
        if summaries:
            logger.info("检索到 %d 条历史记忆: user_id=%s", len(summaries), user_id)
        return summaries
    except Exception as e:
        logger.warning("检索 user_memories 失败: %s", e)
        return []
