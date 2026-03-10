"""
api/app.py

FastAPI 应用：Agent 对话 API、检索 API、运维健康检查。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run as run_agent
from storage.postgres.session_store import load as load_session, save as save_session
from agent.summarizer import run_summarizer
from retrieval.search import search as retrieval_search

logger = logging.getLogger(__name__)

# 与 main.py CLI 一致：每轮保留最近 N 条消息
RECENT_MESSAGES_KEEP = 3

app = FastAPI(
    title="Long COVID Researcher API",
    description="Agent 对话、文献检索、运维健康检查",
    version="0.1.0",
)


# ── 请求/响应模型 ───────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="用户输入的问题")
    session_id: str = Field(default="default", description="会话 ID，用于跨轮次记忆")


class ChatResponse(BaseModel):
    answer: str
    iterations: int
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索查询文本")
    top_k: int = Field(default=20, ge=1, le=100, description="混合检索召回条数")
    top_n: int = Field(default=5, ge=1, le=50, description="Rerank 后保留条数")
    filters: Optional[dict[str, Any]] = Field(default=None, description="过滤条件，如 pub_year, journal")


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded")
    postgres: str = Field(..., description="ok | error")
    qdrant: str = Field(..., description="ok | error")


# ── 路由 ───────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Agent 对话：单轮问答，支持 session_id 多轮记忆。
    内部会加载/保存会话状态（摘要 + 最近 N 条），断线后重连同 session_id 可恢复上下文。
    """
    session_id = (req.session_id or "default").strip() or "default"
    loaded = load_session(session_id)
    if loaded:
        summary = loaded.get("summary", "") or ""
        history = loaded.get("history")
        retrieved_chunks = loaded.get("retrieved_chunks")
    else:
        summary = ""
        history = None
        retrieved_chunks = None

    result = run_agent(
        req.user_input,
        history=history,
        retrieved_chunks=retrieved_chunks,
        summary=summary or None,
    )
    summary, history = run_summarizer(
        result.get("summary", ""),
        result.get("messages", []),
        keep_last_n=RECENT_MESSAGES_KEEP,
    )
    retrieved_chunks = result.get("retrieved_chunks") or []
    save_session(session_id, summary, history, retrieved_chunks)

    return ChatResponse(
        answer=result.get("answer", ""),
        iterations=result.get("iterations", 0),
        session_id=session_id,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """文献检索：混合检索 + Rerank，返回 Top-N 结果（payload + rerank_score）。"""
    try:
        results = retrieval_search(
            query=req.query,
            top_k=req.top_k,
            top_n=req.top_n,
            filters=req.filters,
        )
        return SearchResponse(results=results)
    except Exception as e:
        logger.exception("检索失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """运维健康检查：检测 PostgreSQL、Qdrant 是否可用。"""
    postgres_status = "ok"
    qdrant_status = "ok"

    try:
        from sqlalchemy import text
        from infra.clients import get_pg_engine
        engine = get_pg_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.debug("PostgreSQL 健康检查失败: %s", e)
        postgres_status = "error"

    try:
        from infra.clients import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
    except Exception as e:
        logger.debug("Qdrant 健康检查失败: %s", e)
        qdrant_status = "error"

    status = "ok" if (postgres_status == "ok" and qdrant_status == "ok") else "degraded"
    return HealthResponse(status=status, postgres=postgres_status, qdrant=qdrant_status)
