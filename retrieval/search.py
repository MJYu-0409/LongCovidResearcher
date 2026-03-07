"""
retrieval/search.py

对外统一入口：Agent 工具调用这一个函数即可，
内部封装了 hybrid_search → rerank 的完整流程。
"""

from __future__ import annotations

import logging
from typing import Optional

from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank

logger = logging.getLogger(__name__)


def search(
    query: str,
    top_k: int = 20,
    top_n: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    完整检索流程：混合检索 → RRF 融合 → Reranking。

    Args:
        query:   用户查询文本（建议先经过 Query 改写）
        top_k:   混合检索召回条数，默认 20
        top_n:   Reranking 后保留条数，默认 5（送给 LLM）
        filters: 可选过滤条件，例如 {"pub_year": "2023"}

    Returns:
        list[dict]，Top-N 最相关结果，每条包含：
          - payload.pmcid
          - payload.text
          - payload.section
          - payload.source_type
          - payload.pub_year
          - payload.journal
          - rerank_score
    """
    if not (query or "").strip():
        return []
    candidates = hybrid_search(query, top_k=top_k, filters=filters)
    results    = rerank(query, candidates, top_n=top_n)

    logger.info(
        "检索完成：query='%s'，召回 %d 条，Reranking 后 %d 条",
        (query or "")[:50], len(candidates), len(results),
    )
    return results