"""
agent/summarizer.py

LangGraph 风格会话内摘要：在每轮会话结束时将较早的 messages 压成摘要，保留最近 N 条完整消息，供下一轮使用。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.messages import BaseMessage, HumanMessage

from infra.clients import get_qwen_chat_model

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """请将下面的对话历史压缩成一段简洁的中文摘要，保留：用户主要问题、使用过的工具与结论、关键文献或 pmcid。
不要编造内容，只概括已有信息。若内容为空或无需摘要，直接回复「无」。
---
{content}
---
摘要："""


def _messages_to_text(messages: List[BaseMessage], max_chars: int = 12000) -> str:
    """将消息列表转成纯文本，便于送给 LLM 做摘要。"""
    parts = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "") or ""
        if isinstance(content, str) and content.strip():
            parts.append(f"[{role}]: {content.strip()[:2000]}")
    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text or "（无内容）"


def run_summarizer(
    prev_summary: str,
    messages: List[BaseMessage],
    keep_last_n: int = 3,
) -> Tuple[str, List[BaseMessage]]:
    """
    在会话轮次结束时调用：将「此前摘要 + 除最近 keep_last_n 条外的消息」压成新摘要，保留最近 keep_last_n 条消息。

    Args:
        prev_summary: 上一轮已有的摘要（可为空）
        messages: 本轮结束后的完整消息列表
        keep_last_n: 保留最近几条完整消息不参与摘要

    Returns:
        (new_summary, recent_messages)：新摘要字符串，以及最近 keep_last_n 条消息（供下一轮作为 history）
    """
    if not messages:
        return prev_summary or "", []

    recent = messages[-keep_last_n:] if len(messages) > keep_last_n else list(messages)
    to_summarize = messages[:-keep_last_n] if len(messages) > keep_last_n else []

    if not to_summarize and not (prev_summary or "").strip():
        logger.debug("summarizer: 无历史可摘要，保留 %d 条", len(recent))
        return "", recent

    content_parts = []
    if (prev_summary or "").strip():
        content_parts.append("【此前摘要】\n" + prev_summary.strip())
    if to_summarize:
        content_parts.append("【对话内容】\n" + _messages_to_text(to_summarize))
    content = "\n\n".join(content_parts)

    try:
        llm = get_qwen_chat_model(temperature=0, max_tokens=800)
        response = llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(content=content))])
        new_summary = (response.content or "").strip()
        if new_summary.lower() in ("无", "无。", "无。"):
            new_summary = ""
        logger.info("summarizer: 已压缩为 %d 字摘要，保留最近 %d 条消息", len(new_summary), len(recent))
        return new_summary, recent
    except Exception as e:
        logger.warning("summarizer 调用失败，保留原摘要与最近消息: %s", e)
        return prev_summary or "", recent
