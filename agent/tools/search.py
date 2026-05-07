"""
agent/tools/search.py

工具一：search_literature
包装 retrieval/search.py，向 Orchestrator 返回摘要视图，
完整 chunk 存入 State 供其他工具消费。
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Optional

from langchain_core.tools import tool

from retrieval.search import search as _search

logger = logging.getLogger(__name__)


def search_literature_fn(
    query: str,
    top_k: int = 40,
    top_n: int = 12,
    pub_year: Optional[str] = None,
    journal: Optional[str] = None,
) -> dict:
    """
    检索 Long COVID 相关文献。

    Args:
        query:    自然语言查询
        top_k:    混合检索初始召回数（默认40）
        top_n:    rerank 后保留数（默认12）
        pub_year: 可选，按发表年份过滤，如 "2023"
        journal:  可选，按期刊过滤

    Returns:
        {
          "summary":  摘要视图列表（给 Orchestrator 看，控制 token 消耗）
          "chunks":   完整 chunk 列表（存入 State，供其他工具使用）
          "count":    召回数量
        }
    """
    filters: dict = {}
    if pub_year:
        filters["pub_year"] = pub_year
    if journal:
        filters["journal"] = journal

    try:
        results = _search(query, top_k=top_k, top_n=top_n,
                          filters=filters if filters else None)
    except Exception as exc:
        logger.error("search_literature retrieval failed: %s", exc)
        return {
            "error": "retrieval_unavailable",
            "message": "The literature retrieval system is currently unavailable. "
                       "Do NOT answer from memory — inform the user that the search "
                       "backend is down and ask them to retry later.",
            "summary": [], "chunks": [], "count": 0,
        }

    # 摘要视图：Orchestrator 只看这部分，text 截断为 150 字符
    summary = []
    for r in results:
        p = r["payload"]
        summary.append({
            "pmcid":        p.get("pmcid", ""),
            "section":      p.get("section", ""),
            "source_type":  p.get("source_type", ""),
            "pub_year":     p.get("pub_year", ""),
            "rerank_score": round(r.get("rerank_score", r.get("rrf_score", 0)), 3),
            "text_preview": p.get("text", "")[:150],
        })

    logger.info("search_literature: query='%s' → %d 条", query[:50], len(results))

    return {
        "summary": summary,
        "chunks":  results,   # 完整内容，由 nodes.py 写入 State
        "count":   len(results),
    }


_RETRIEVAL_ERROR_MSG = (
    "RETRIEVAL_SYSTEM_UNAVAILABLE: The literature retrieval system is currently "
    "unavailable. Do NOT answer from memory — inform the user that the search "
    "backend is down and ask them to retry later."
)

# ContextVar isolates per-request chunk cache across concurrent async tasks.
# Each asyncio.Task (= one user request) gets its own copy, preventing cross-request
# contamination when multiple users call search_literature concurrently.
_last_chunks_var: ContextVar[list] = ContextVar("_last_chunks", default=[])


@tool
def search_literature(
    query: str,
    top_n: int = 10,
    pub_year: Optional[str] = None,
    journal: Optional[str] = None,
) -> str:
    """
    检索 Long COVID 相关学术文献。
    适用于：查找特定主题的研究、获取相关论文片段、为后续问答或综述收集素材。

    Args:
        query:    研究问题或关键词，用英文效果更好
        top_n:    返回结果数量（默认10）
        pub_year: 可选，只检索某年发表的论文，如 "2023"
        journal:  可选，只检索特定期刊

    Returns:
        检索结果摘要（JSON字符串），包含 pmcid、section、相关性分数和文本预览
    """
    try:
        result = search_literature_fn(query, top_n=top_n,
                                      pub_year=pub_year, journal=journal)
        if "error" in result:
            logger.error("search_literature returning error: %s", result.get("message", ""))
            _last_chunks_var.set([])
            return _RETRIEVAL_ERROR_MSG
        _last_chunks_var.set(result["chunks"])   # context_text 已由 _expand_neighbors 写入 payload
        return json.dumps(result["summary"], ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("search_literature unhandled exception (returning error marker): %s", exc)
        _last_chunks_var.set([])
        return _RETRIEVAL_ERROR_MSG
