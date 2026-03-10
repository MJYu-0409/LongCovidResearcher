"""
agent/runner.py

对外暴露的统一入口。
支持单轮调用和多轮对话（通过传入 history 维持上下文）。
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage

from agent.graph import agent_graph
from agent.state import AgentState

logger = logging.getLogger(__name__)


def run(
    user_input: str,
    history: Optional[list] = None,
    retrieved_chunks: Optional[list] = None,
    summary: Optional[str] = None,
) -> dict:
    """
    运行 Agent，处理单轮或多轮对话。

    Args:
        user_input:       用户输入的问题
        history:          历史消息列表（多轮对话时传入上一轮 summarizer 保留的最近 N 条）
        retrieved_chunks: 历史检索结果（多轮对话时传入，避免重复检索）
        summary:          此前对话的摘要（由 main 在每轮结束后通过 summarizer 更新后传入）

    Returns:
        {
          "answer":          最终回答（字符串）
          "messages":        完整消息历史（供下一轮 summarizer 与 history 使用）
          "retrieved_chunks": 本轮累积的所有检索结果
          "iterations":      实际迭代次数
          "summary":         当前轮使用的摘要（与传入的 summary 一致，供 main 传给 summarizer）
        }
    """
    # 构建初始 State
    messages = list(history or [])
    messages.append(HumanMessage(content=user_input))

    initial_state: AgentState = {
        "messages":        messages,
        "retrieved_chunks": list(retrieved_chunks or []),
        "iteration_count": 0,
    }
    if summary:
        initial_state["summary"] = summary

    # 运行图
    logger.info("Agent 开始：'%s'", user_input[:60])
    final_state = agent_graph.invoke(initial_state)

    # 提取最终回答（最后一条 AI 消息，无工具调用的那条）
    answer = ""
    for msg in reversed(final_state["messages"]):
        if msg.type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                answer = msg.content if msg.content is not None else ""
                break

    logger.info("Agent 完成：迭代 %d 次，回答 %d 字",
                final_state.get("iteration_count", 0), len(answer))

    return {
        "answer":           answer,
        "messages":         final_state["messages"],
        "retrieved_chunks": final_state.get("retrieved_chunks", []),
        "iterations":       final_state.get("iteration_count", 0),
        "summary":          final_state.get("summary", ""),
    }
