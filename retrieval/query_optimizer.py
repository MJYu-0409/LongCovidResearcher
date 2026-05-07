"""
retrieval/query_optimizer.py

查询优化：
- HyDE：为普通问题生成假设答案文本，增强语义召回
- Decomposition：将复杂问题拆成 2-4 个可检索子问题
"""

from __future__ import annotations

import json
import logging
import re

from infra.clients import get_qwen_chat_model

logger = logging.getLogger(__name__)

_DECOMPOSE_MARKERS = (
    "compare", "comparison", "versus", "vs", "difference between",
    "contrast", "similarities", "tradeoff", "trade-off",
    # 中文复杂度关键词
    "比较", "区别", "对比", "异同", "优劣", "差异", "相比",
)

# Matches patterns like "A and B and C", "whether X or Y", or multiple question marks
_MULTI_INTENT_RE = re.compile(
    r"\band\b.{10,}\band\b"
    r"|\bwhether\b.+\bor\b"
    r"|[?].*[?]"
    # 中文多意图模式
    r"|与.{2,20}(的|之)(区别|比较|对比|关系|异同)"   # "X与Y的区别"
    r"|[，,].{2,20}(还是|或者|versus)"               # "X，还是Y"
    r"|[？].*[？]",                                   # 中文双问号
    re.IGNORECASE,
)


def classify_query_complexity(query: str) -> str:
    """
    Classify query into retrieval strategy tier.
    Returns: "direct" | "simple" | "complex"
      - direct:  short / entity lookup → plain hybrid search
      - simple:  single-intent question → HyDE expansion
      - complex: multi-intent / comparison → decompose into subqueries
    """
    q = (query or "").strip()
    if not q:
        return "direct"

    lowered = q.lower()
    # Count CJK characters individually; Latin/numeric words by space-split.
    # re.findall(r"\w+") treats an entire Chinese sentence as 1 token (no spaces),
    # which always misclassifies Chinese queries as "direct".
    cjk_count = sum(1 for c in lowered if '一' <= c <= '鿿')
    non_cjk_count = len(re.findall(r'[a-z0-9]+', re.sub(r'[一-鿿]', ' ', lowered)))
    token_count = cjk_count + non_cjk_count

    if token_count <= 5:
        return "direct"

    decompose_hits = sum(1 for m in _DECOMPOSE_MARKERS if m in lowered)
    if decompose_hits >= 1 or _MULTI_INTENT_RE.search(q):
        return "complex"

    return "simple"


def build_hyde_query(query: str) -> str:
    """
    生成 HyDE 假设文档并拼接原 query。
    失败时返回原 query。
    """
    q = (query or "").strip()
    if not q:
        return q

    prompt = (
        'You are a biomedical researcher specializing in Long COVID and post-acute sequelae '
        'of SARS-CoV-2 (PASC). Write an 80-150 word passage that a PubMed abstract would '
        'contain to answer the question below. Include relevant clinical terms, biomarkers, '
        'cytokines, anatomical systems, or study design keywords where appropriate. '
        'Do not fabricate statistics or trial names. Return only the passage, no explanation. '
        'Write the passage in English regardless of the language of the question.\n\n'
        f'Question: {q}'
    )
    try:
        llm = get_qwen_chat_model(temperature=0, max_tokens=240, timeout=15)
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        hypo = (resp.content or "").strip()
        if not hypo:
            return q
        logger.info("HyDE passage:\n%s", hypo)
        return f"{q}\n\nHypothetical answer passage:\n{hypo}"
    except Exception as e:
        logger.warning("HyDE 生成失败，回退原 query: %s", e)
        return q


def _extract_subqueries(raw: str, fallback_query: str, max_subqueries: int) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return [fallback_query]

    try:
        data = json.loads(text)
        items = data.get("subqueries", [])
        subs = [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        # 容错：尝试按行提取
        subs = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if line:
                subs.append(line)

    dedup: list[str] = []
    seen = set()
    for s in subs:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)

    if not dedup:
        return [fallback_query]
    result = dedup[:max_subqueries]
    logger.info("decompose subqueries: %s", result)
    return result


def decompose_query(query: str, max_subqueries: int = 4) -> list[str]:
    """
    将复杂问题拆分为子查询。
    失败时返回 [query]。
    """
    q = (query or "").strip()
    if not q:
        return [q]

    prompt = (
        'You are an academic search query rewriter for a Long COVID / PASC literature database. '
        'Split the question into 2-4 independent subqueries that can each be searched separately. '
        'Each subquery should be concise, self-contained, and suitable for PubMed-style retrieval. '
        'Write all subqueries in English regardless of the input language. '
        'Return only JSON: {"subqueries": ["...", "..."]}\n\n'
        f'Question: {q}'
    )
    try:
        llm = get_qwen_chat_model(temperature=0, max_tokens=300, timeout=15)
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        return _extract_subqueries(resp.content if hasattr(resp, "content") else "", q, max_subqueries)
    except Exception as e:
        logger.warning("Decomposition 失败，回退原 query: %s", e)
        return [q]


_ZH_RE = re.compile(r'[一-鿿]')


def translate_query_to_english(query: str) -> str:
    """
    若 query 含中文，用 Qwen 将其翻译为简洁的英文 PubMed 检索式，供 SPLADE sparse 路径使用。
    翻译失败时静默返回原 query（不中断检索流程）。
    英文 query 直接原样返回，零开销。
    """
    q = (query or "").strip()
    if not _ZH_RE.search(q):
        return q
    prompt = (
        'Translate the following question to concise English suitable for PubMed search. '
        'Preserve medical terms. Return only the translated text, no explanation.\n\n'
        f'Question: {q}'
    )
    try:
        llm = get_qwen_chat_model(temperature=0, max_tokens=120, timeout=8)
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        translated = (resp.content or "").strip()
        if translated:
            logger.info("query 翻译: '%s' → '%s'", q[:40], translated[:60])
            return translated
    except Exception as e:
        logger.warning("query 翻译失败，使用原 query: %s", e)
    return q
