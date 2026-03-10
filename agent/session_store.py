"""
agent/session_store.py

跨会话记忆：每轮对话结束后将 summary + history + retrieved_chunks 持久化到 PostgreSQL，
下次启动时按 session_id 加载。表与 CRUD 均在 agent 内，后续可再抽到 postgres 层。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from sqlalchemy import MetaData, Table, Column, String, Text, DateTime, inspect, select
from sqlalchemy.dialects.postgresql import JSONB

from infra.clients import get_pg_engine

logger = logging.getLogger(__name__)

TABLE_NAME = "agent_sessions"

_metadata = MetaData()
agent_sessions = Table(
    TABLE_NAME,
    _metadata,
    Column("session_id", String(64), primary_key=True),
    Column("summary", Text(), nullable=False),
    Column("history", JSONB(), nullable=False),
    Column("retrieved_chunks", JSONB(), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def _table_exists(engine) -> bool:
    return inspect(engine).has_table(TABLE_NAME)


def create_table_if_not_exists(engine=None):
    """建表（已存在则跳过）。"""
    eng = engine or get_pg_engine()
    _metadata.create_all(eng)
    logger.debug("表 %s 已就绪", TABLE_NAME)


def save(
    session_id: str,
    summary: str,
    history: list[BaseMessage],
    retrieved_chunks: list[dict],
) -> None:
    """
    每轮对话结束后调用：将当前状态写入 PostgreSQL，session_id 相同则覆盖（upsert）。
    网络或进程异常时最多丢当前轮，之前轮次已落库。
    """
    if not (session_id or "").strip():
        logger.warning("session_id 为空，跳过持久化")
        return
    try:
        engine = get_pg_engine()
        if not _table_exists(engine):
            create_table_if_not_exists(engine)
        history_data = messages_to_dict(history) if history else []
        chunks_data = retrieved_chunks if isinstance(retrieved_chunks, list) else []
        now = datetime.now(timezone.utc)
        with engine.connect() as conn:
            stmt = agent_sessions.insert().values(
                session_id=session_id.strip(),
                summary=summary,
                history=history_data,
                retrieved_chunks=chunks_data,
                updated_at=now,
            ).on_conflict_do_update(
                index_elements=["session_id"],
                set_={
                    "summary": summary,
                    "history": history_data,
                    "retrieved_chunks": chunks_data,
                    "updated_at": now,
                },
            )
            conn.execute(stmt)
            conn.commit()
        logger.info("会话已持久化: session_id=%s", session_id)
    except Exception as e:
        logger.warning("会话持久化失败（不影响本轮对话）: %s", e)


def load(session_id: str) -> Optional[dict[str, Any]]:
    """
    按 session_id 加载最近一次持久化的会话状态，供新会话恢复。

    Returns:
        {"summary": str, "history": list[BaseMessage], "retrieved_chunks": list} 或 None
    """
    if not (session_id or "").strip():
        return None
    try:
        engine = get_pg_engine()
        if not _table_exists(engine):
            return None
        with engine.connect() as conn:
            row = conn.execute(
                select(agent_sessions).where(agent_sessions.c.session_id == session_id.strip()).limit(1)
            ).fetchone()
        if row is None:
            return None
        history_data = row.history if row.history is not None else []
        try:
            history = messages_from_dict(history_data) if history_data else []
        except Exception as e:
            logger.warning("反序列化 history 失败，当作空: %s", e)
            history = []
        chunks = list(row.retrieved_chunks) if row.retrieved_chunks is not None else []
        return {
            "summary": row.summary or "",
            "history": history,
            "retrieved_chunks": chunks,
        }
    except Exception as e:
        logger.warning("加载会话失败: %s", e)
        return None
