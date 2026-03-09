"""
agent/tools/qa.py

工具四：answer_question
调用 Qwen，基于已检索的 chunk 做事实性问答。
适合具体、精确的问题，不需要综合多篇生成长文。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from infra.clients import get_qwen_chat_model

logger = logging.getLogger(__name__)

# Qwen 单例（从 infra 统一入口获取）
_qwen = get_qwen_chat_model(temperature=0.1, max_tokens=1500)

_QA_SYSTEM = """你是 Long COVID 领域的专业研究助手。
根据提供的文献片段，准确回答用户问题。

要求：
- 答案必须基于提供的文献内容，不要臆造
- 如果文献内容不足以回答问题，明确说明
- 引用具体来源时注明 pmcid
- 使用中文回答，专业术语保留英文"""


@tool
def answer_question(
    question: str,
    context_chunks: str,
) -> str:
    """
    基于检索到的文献内容回答具体问题。
    适用于：需要从文献中提取具体事实、数据、结论的问题。

    Args:
        question:       用户的具体问题
        context_chunks: JSON 字符串，文献片段列表：
                        '[{"pmcid":"...", "section":"...", "text":"..."}]'
                        通常直接使用 State 中的 retrieved_chunks

    Returns:
        基于文献的回答（字符串）
    """
    try:
        chunks = json.loads(context_chunks)
    except json.JSONDecodeError:
        return "context_chunks 格式错误，需要 JSON 列表"

    if not chunks:
        return "没有可用的文献内容，请先使用 search_literature 检索相关文献"

    # 构建上下文，每个 chunk 附上来源
    context_parts = []
    for c in chunks[:15]:   # 最多15个 chunk，避免 token 超限
        p = c.get("payload", c)   # 兼容两种格式
        pmcid   = p.get("pmcid",   "")
        section = p.get("section", "")
        text    = p.get("text",    "")
        context_parts.append(f"[{pmcid} / {section}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system",  "content": _QA_SYSTEM},
        {"role": "user",    "content": f"文献内容：\n\n{context}\n\n问题：{question}"},
    ]

    try:
        response = _qwen.invoke(messages)
        answer   = response.content
        logger.info("answer_question: '%s' → %d 字", question[:40], len(answer))
        return answer
    except Exception as e:
        logger.error("Qwen 调用失败: %s", e)
        return f"问答服务暂时不可用: {e}"
