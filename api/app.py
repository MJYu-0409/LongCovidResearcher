"""
api/app.py

FastAPI 应用：Agent 对话 API、流式对话 API、检索 API、运维健康检查。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import run as run_agent
from agent.runner_async import stream_agent
# from storage.postgres.session_store import load as load_session, save as save_session
from storage.redis.session_store import load as load_session, save as save_session
from agent.summarizer import run_summarizer
from retrieval.search import search as retrieval_search

logger = logging.getLogger(__name__)

RECENT_MESSAGES_KEEP = 6

app = FastAPI(
    title="Long COVID Researcher API",
    description="Agent 对话、文献检索、运维健康检查",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── 请求/响应模型 ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    user_id: str = Field(default="default")


class ChatStreamRequest(BaseModel):
    user_input: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    user_id: str = Field(default="default")


class ChatResponse(BaseModel):
    answer: str
    iterations: int
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    top_n: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    postgres: str
    qdrant: str
    redis: str


# ── 路由 ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the chat UI."""
    path = os.path.join(_STATIC_DIR, "index.html")
    return FileResponse(path)


@app.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest) -> StreamingResponse:
    """
    流式 Agent 对话：SSE（text/event-stream）格式推送 token 和工具状态。

    事件格式：data: {"type": "...", ...}\\n\\n
      - tool_start: {"type":"tool_start","tool":"search_literature","query":"..."}
      - tool_end:   {"type":"tool_end","tool":"search_literature"}
      - token:      {"type":"token","content":"..."}
      - done:       {"type":"done","iterations":3}
      - error:      {"type":"error","message":"..."}
    """
    session_id = (req.session_id or "default").strip() or "default"
    user_id = (req.user_id or "default").strip() or "default"

    loaded = await asyncio.to_thread(load_session, session_id, user_id)
    history = loaded.get("history") if loaded else None
    retrieved_chunks = loaded.get("retrieved_chunks") if loaded else None
    summary = (loaded.get("summary") or "") if loaded else ""

    async def event_generator():
        try:
            async for evt in stream_agent(
                user_input=req.user_input,
                session_id=session_id,
                user_id=user_id,
                history=history,
                retrieved_chunks=retrieved_chunks,
                summary=summary,
            ):
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("chat_stream generator error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """同步 Agent 对话（保留兼容性）。"""
    session_id = (req.session_id or "default").strip() or "default"
    user_id = (req.user_id or "default").strip() or "default"
    loaded = load_session(session_id, user_id)
    if loaded:
        summary = loaded.get("summary", "") or ""
        history = loaded.get("history")
        retrieved_chunks = loaded.get("retrieved_chunks")
    else:
        summary, history, retrieved_chunks = "", None, None

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
    save_session(session_id, user_id, summary, history, [])

    return ChatResponse(
        answer=result.get("answer", ""),
        iterations=result.get("iterations", 0),
        session_id=session_id,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """文献检索：混合检索 + Rerank。"""
    try:
        return SearchResponse(results=retrieval_search(
            query=req.query, top_k=req.top_k, top_n=req.top_n, filters=req.filters,
        ))
    except Exception as e:
        logger.exception("检索失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """运维健康检查。"""
    pg, qdrant, redis = "ok", "ok", "ok"
    try:
        from sqlalchemy import text
        from infra.clients import get_pg_engine
        with get_pg_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pg = "error"
    try:
        from infra.clients import get_qdrant_client
        get_qdrant_client().get_collections()
    except Exception as _qe:
        logger.error("Qdrant health check failed: %s", _qe, exc_info=True)
        qdrant = "error"
    try:
        from infra.clients import get_redis_client
        get_redis_client().ping()
    except Exception as _re:
        logger.error("Redis health check failed: %s", _re, exc_info=True)
        redis = "error"

    all_ok = pg == "ok" and qdrant == "ok" and redis == "ok"
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        postgres=pg,
        qdrant=qdrant,
        redis=redis,
    )
