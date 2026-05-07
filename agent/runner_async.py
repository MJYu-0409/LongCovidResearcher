"""
agent/runner_async.py

Async streaming entry point for the agent.
Wraps agent_graph.astream_events and yields SSE-compatible event dicts.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, List, Optional

from langchain_core.messages import HumanMessage, BaseMessage

from agent.graph import agent_graph
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def stream_agent(
    user_input: str,
    session_id: str,
    user_id: str = "default",
    history: Optional[List[BaseMessage]] = None,
    retrieved_chunks: Optional[List[dict]] = None,
    summary: str = "",
) -> AsyncGenerator[dict, None]:
    """
    Async generator yielding event dicts for SSE.

    Yields:
        {"type": "tool_start", "tool": str, "query": str}
        {"type": "tool_end",   "tool": str}
        {"type": "token",      "content": str}
        {"type": "done",       "iterations": int}
        {"type": "error",      "message": str}
    """
    # Pre-flight: verify both Qdrant and OpenAI embedding API are reachable.
    # Fail fast with a clean error so Qwen never receives a raw exception.
    try:
        from infra.clients import get_qdrant_client
        await asyncio.to_thread(lambda: get_qdrant_client().get_collections())
    except Exception as exc:
        logger.warning("Qdrant pre-flight check failed: %s", exc)
        yield {"type": "error", "message": "向量数据库不可用，请确认 Qdrant 服务正常运行后重试。"}
        return

    try:
        from retrieval.dense import embed_query
        await asyncio.to_thread(lambda: embed_query("preflight"))
    except Exception as exc:
        logger.warning("Dense embedding pre-flight check failed: %s", exc)
        yield {"type": "error", "message": "文献检索所需的本地向量模型加载失败，请确认模型文件完整后重试。"}
        return

    # 检索用户历史跨会话记忆，注入 summary 前缀供 LLM 感知用户研究兴趣
    try:
        from storage.qdrant.memory_store import retrieve_memories
        memories = await asyncio.to_thread(retrieve_memories, user_id, user_input)
        if memories:
            mem_text = "\n".join(f"- {m[:300]}" for m in memories)
            memory_prefix = f"【用户历史研究兴趣（供参考）】\n{mem_text}"
            summary = f"{memory_prefix}\n\n{summary}".strip() if summary else memory_prefix
    except Exception as exc:
        logger.warning("历史记忆检索失败（忽略）: %s", exc)

    messages: List[BaseMessage] = list(history or [])
    messages.append(HumanMessage(content=user_input))

    initial_state = AgentState(
        messages=messages,
        retrieved_chunks=list(retrieved_chunks or []),
        iteration_count=0,
        summary=summary or "",
    )

    last_full_state: dict = {}
    _in_think_block = False  # track <think>...</think> spans in stream

    try:
        async for event in agent_graph.astream_events(initial_state, version="v2"):
            etype: str = event.get("event", "")
            name: str = event.get("name", "")
            data: dict = event.get("data") or {}

            if etype == "on_tool_start":
                inp = data.get("input") or {}
                query = ""
                if isinstance(inp, dict):
                    query = (
                        inp.get("query")
                        or inp.get("question")
                        or inp.get("topic")
                        or ""
                    )
                yield {"type": "tool_start", "tool": name, "query": str(query)[:120]}

            elif etype == "on_tool_end":
                tool_output = data.get("output", "")
                # Normalise to string regardless of whether ToolNode wraps it
                tool_output_str = (
                    tool_output.content
                    if hasattr(tool_output, "content")
                    else str(tool_output)
                )
                logger.debug(
                    "tool_end name=%s type=%s preview=%s",
                    name, type(tool_output).__name__, tool_output_str[:200],
                )
                if name == "search_literature":
                    _lo = tool_output_str.lower()
                    if (
                        "retrieval_system_unavailable" in _lo
                        or "currently unavailable" in _lo
                        or "unexpected response" in _lo
                        or "502" in tool_output_str
                        or "timed out" in _lo
                        or "connecttimeout" in _lo
                    ):
                        logger.warning("search_literature error intercepted, aborting: %s", tool_output_str[:200])
                        yield {
                            "type": "error",
                            "message": "检索系统当前不可用，无法基于文献回答。\n"
                                       "请检查：① OpenAI API 代理是否正常 ② Qdrant 容器是否运行",
                        }
                        return
                yield {"type": "tool_end", "tool": name}

            elif etype == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk:
                    content = getattr(chunk, "content", "")
                    tool_call_chunks = getattr(chunk, "tool_call_chunks", [])
                    if content and isinstance(content, str) and not tool_call_chunks:
                        # Filter Qwen3 thinking tokens (<think>...</think>)
                        if "<think>" in content:
                            _in_think_block = True
                        if _in_think_block:
                            if "</think>" in content:
                                _in_think_block = False
                            continue
                        yield {"type": "token", "content": content}

            elif etype == "on_chain_end":
                output = data.get("output") or {}
                if isinstance(output, dict) and "messages" in output:
                    last_full_state = output

    except Exception as exc:
        logger.exception("stream_agent error")
        yield {"type": "error", "message": str(exc)}
        return

    # Persist session after streaming completes
    final_msgs = last_full_state.get("messages", messages)
    iters = last_full_state.get("iteration_count", 0)

    try:
        from agent.summarizer import run_summarizer
        from storage.redis.session_store import save as save_session
        new_summary, trimmed_msgs = await asyncio.to_thread(
            run_summarizer, summary, final_msgs, 6
        )
        await asyncio.to_thread(save_session, session_id, user_id, new_summary, trimmed_msgs, [])
        # 异步双写 Qdrant user_memories（fire-and-forget，不阻塞响应）
        if new_summary.strip():
            async def _write_memory(uid=user_id, sid=session_id, s=new_summary):
                try:
                    from storage.qdrant.memory_store import upsert_memory
                    await asyncio.to_thread(upsert_memory, uid, sid, s)
                except Exception as mem_exc:
                    logger.warning("用户记忆写入失败（不影响响应）: %s", mem_exc)
            asyncio.create_task(_write_memory())
    except Exception as exc:
        logger.warning("Session save failed after streaming: %s", exc)

    yield {"type": "done", "iterations": iters}
