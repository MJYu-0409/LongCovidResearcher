"""
agent/state.py

LangGraph Agent 的共享状态定义。
所有节点读写同一个 AgentState 实例，通过 State 传递上下文。
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class _AgentStateRequired(TypedDict):
    # 完整对话历史，add_messages 确保追加而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    # 本轮会话累积的所有检索 chunk（完整 payload，供工具直接使用）
    retrieved_chunks: list[dict]
    # 当前迭代次数，达到上限时强制结束
    iteration_count: int


class AgentState(_AgentStateRequired, total=False):
    # 此前对话的摘要（会话内压缩，供下一轮注入；由 main 在每轮结束后调用 summarizer 更新）
    summary: str
