"""
agent/tools/synthesis.py

工具五：synthesize_review
调用 GPT-4o，将多篇文献综合生成结构化文献综述。
只在 Orchestrator 判断信息收集充分时才触发，避免频繁调用。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_gpt4o = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    max_tokens=3000,
)

_SYNTHESIS_SYSTEM = """你是 Long COVID 领域的资深综述专家。
根据提供的文献片段，生成一篇结构清晰的学术综述。

综述结构：
1. 研究背景与问题（2-3句）
2. 主要发现（按主题分类，每类2-4句，注明来源 pmcid）
3. 研究共识与争议
4. 局限性与未来方向
5. 结论（2-3句）

要求：
- 忠实于原文献内容，不要添加未在文献中出现的信息
- 每个重要观点注明来源 pmcid
- 使用学术语言，中英文混合（专业术语保留英文）
- 如果文献数量不足（少于3篇），说明综述的局限性"""


@tool
def synthesize_review(
    topic: str,
    context_chunks: str,
    focus: Optional[str] = None,
) -> str:
    """
    基于检索到的多篇文献生成结构化文献综述。
    适用于：用户需要某主题的综合性分析、系统梳理某领域的研究现状。
    注意：此工具消耗较多 token，仅在信息收集充分时使用。

    Args:
        topic:          综述主题，如 "long covid 的免疫机制"
        context_chunks: JSON 字符串，文献片段列表：
                        '[{"pmcid":"...", "section":"...", "text":"..."}]'
                        通常直接使用 State 中的 retrieved_chunks
        focus:          可选，重点关注的方面，如 "治疗方案" 或 "流行病学数据"

    Returns:
        结构化文献综述（字符串）
    """
    try:
        chunks = json.loads(context_chunks)
    except json.JSONDecodeError:
        return "context_chunks 格式错误，需要 JSON 列表"

    if not chunks:
        return "没有可用的文献内容，请先使用 search_literature 检索相关文献"

    # 构建上下文
    context_parts = []
    for c in chunks[:20]:   # 最多20个 chunk
        p       = c.get("payload", c)
        pmcid   = p.get("pmcid",   "")
        section = p.get("section", "")
        text    = p.get("text",    "")
        pub_year= p.get("pub_year","")
        context_parts.append(f"[{pmcid} {pub_year} / {section}]\n{text}")

    context  = "\n\n---\n\n".join(context_parts)
    focus_str = f"\n重点关注：{focus}" if focus else ""

    user_content = (
        f"综述主题：{topic}{focus_str}\n\n"
        f"文献内容（共 {len(chunks)} 个片段）：\n\n{context}"
    )

    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    try:
        response = _gpt4o.invoke(messages)
        review   = response.content
        logger.info("synthesize_review: topic='%s' → %d 字，使用 %d 个 chunk",
                    topic[:40], len(review), len(chunks))
        return review
    except Exception as e:
        logger.error("GPT-4o 综述生成失败: %s", e)
        return f"综述生成服务暂时不可用: {e}"
