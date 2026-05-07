"""
storage/redis/session_store.py

活跃会话快速缓存：每轮结束写入 Redis，TTL 默认 24h。
Key 格式：session:{user_id}:{session_id}，支持多用户扩展。
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from config import REDIS_SESSION_TTL
from infra.clients import get_redis_client

logger = logging.getLogger(__name__)
_KEY_PREFIX = "session:"


def _make_key(user_id: str, session_id: str) -> str:
    return f"{_KEY_PREFIX}{user_id.strip()}:{session_id.strip()}"


def save(
    session_id: str,
    user_id: str,
    summary: str,
    history: list[BaseMessage],
    retrieved_chunks: list[dict],
) -> None:
    """每轮结束后调用：upsert 会话状态，TTL 自动续期。"""
    if not (session_id or "").strip():
        logger.warning("session_id 为空，跳过持久化")
        return
    uid = (user_id or "default").strip() or "default"
    try:
        key = _make_key(uid, session_id)
        payload = json.dumps(
            {
                "user_id": uid,
                "summary": summary or "",
                "history": messages_to_dict(history) if history else [],
                "retrieved_chunks": retrieved_chunks if isinstance(retrieved_chunks, list) else [],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        get_redis_client().setex(key, REDIS_SESSION_TTL, payload)
        logger.info("会话已缓存至 Redis: user_id=%s session_id=%s ttl=%ds", uid, session_id, REDIS_SESSION_TTL)
    except Exception as e:
        logger.warning("Redis 会话持久化失败（不影响本轮对话）: %s", e)


def load(session_id: str, user_id: str = "default") -> Optional[dict[str, Any]]:
    """
    按 user_id + session_id 加载会话状态。

    Returns:
        {"summary": str, "history": list[BaseMessage], "retrieved_chunks": list} 或 None
    """
    if not (session_id or "").strip():
        return None
    uid = (user_id or "default").strip() or "default"
    try:
        raw = get_redis_client().get(_make_key(uid, session_id))
        if not raw:
            return None
        data = json.loads(raw)
        history_data = data.get("history", [])
        try:
            history = messages_from_dict(history_data) if history_data else []
        except Exception as e:
            logger.warning("反序列化 history 失败，当作空: %s", e)
            history = []
        return {
            "summary": data.get("summary", ""),
            "history": history,
            "retrieved_chunks": list(data.get("retrieved_chunks", [])),
        }
    except Exception as e:
        logger.warning("Redis 加载会话失败: %s", e)
        return None
