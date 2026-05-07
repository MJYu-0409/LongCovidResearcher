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

from config import GROUNDING_CHECK
from infra.clients import get_qwen_chat_model

logger = logging.getLogger(__name__)

_qwen = get_qwen_chat_model(temperature=0.1, max_tokens=1500)
_grounding_llm = get_qwen_chat_model(temperature=0, max_tokens=1000)

_QA_SYSTEM = """你是 Long COVID 领域的专业研究助手。
根据提供的文献片段，准确回答用户问题。

要求：
1. 答案中每条断言必须来自提供的文献内容，禁止使用模型自身训练知识补充
2. 重要结论注明研究设计类型：[RCT] / [Meta-analysis] / [Cohort] / [Case series] / [Review]
3. 引用来源注明 pmcid；关键结论可在括号内引用原文关键句（英文）
4. 若文献未直接覆盖某子问题，明确写出"当前检索文献未提供此方面的直接证据"，不推断或延伸
5. 使用中文回答，专业术语保留英文
6. 末尾附：[证据充分度：高/中/低，来自 N 篇文献]"""

_GROUNDING_SYSTEM = """你是文献准确性审核员。核查回答中每条事实性断言是否有来源文献的直接支持。

审核规则：
- 有原文依据的断言：保持不变
- 超出原文的推断或延伸（文献未明确说明、但听起来"合理"的内容）：在该断言末尾追加 [⚠ 文献中无直接依据]
- 保持原有 pmcid 引用、结构和语言风格不变，不删减内容

只输出修正后的完整回答，不要任何解释。"""


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
        基于文献的回答（字符串），启用 GROUNDING_CHECK 时附带忠实度标注
    """
    try:
        chunks = json.loads(context_chunks)
    except json.JSONDecodeError:
        return "context_chunks 格式错误，需要 JSON 列表"

    if not chunks:
        return "⚠ 无可用文献内容：检索系统未返回任何文献片段，无法基于文献回答。请确认检索系统正常后重试，本系统不提供脱离文献的回答。"

    context_parts = []
    for c in chunks[:15]:
        p = c.get("payload", c)
        pmcid   = p.get("pmcid",   "")
        section = p.get("section", "")
        text    = p.get("text",    "")
        context_parts.append(f"[{pmcid} / {section}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": _QA_SYSTEM},
        {"role": "user",   "content": f"文献内容：\n\n{context}\n\n问题：{question}"},
    ]

    try:
        response = _qwen.invoke(messages)
        answer   = response.content or ""
        logger.info("answer_question: '%s' → %d 字", question[:40], len(answer))

        if GROUNDING_CHECK:
            try:
                grounding_msgs = [
                    {"role": "system", "content": _GROUNDING_SYSTEM},
                    {"role": "user",   "content": (
                        f"以下是基于检索文献生成的回答，请核查：\n\n{answer}\n\n"
                        f"参考文献原文：\n\n{context}"
                    )},
                ]
                verified = _grounding_llm.invoke(grounding_msgs)
                answer = verified.content or answer
                logger.info("grounding check 完成")
            except Exception as ge:
                logger.warning("grounding check 失败（使用原始答案）: %s", ge)

        return answer
    except Exception as e:
        logger.error("Qwen 调用失败: %s", e)
        return f"问答服务暂时不可用: {e}"
