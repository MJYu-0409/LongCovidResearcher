"""
retrieval/hybrid.py

混合检索：融合语义检索和 BM25 检索结果，用 RRF 算法排序。

RRF（Reciprocal Rank Fusion）原理：
  每条结果的得分 = sum(1 / (k + rank_i))，k=60 是平滑参数。
  两路都靠前的结果得分最高，体现"多路共同认可"的置信度。
  不需要手动设置权重，稳定性好。
"""

from __future__ import annotations

import logging
from typing import Optional

from retrieval.dense import dense_search
from retrieval.sparse import sparse_search

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 平滑参数，业界标准值


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    top_k: int = 20,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    混合检索：语义检索 + BM25 稀疏检索，RRF 融合排序。

    Args:
        query:   用户查询文本
        top_k:   最终返回条数（RRF 排序后取前 top_k）
        filters: 可选 payload 过滤条件，同时作用于两路检索

    Returns:
        list[dict]，按 RRF 分数降序排列，每条包含：
          - id:        Qdrant point id
          - rrf_score: 融合后的 RRF 分数
          - payload:   pmcid / text / section / source_type / pub_year / journal
    """
    # 两路各召回 top_k * 2，给 RRF 更多候选
    fetch_k = top_k * 2

    dense_results  = dense_search(query,  top_k=fetch_k, filters=filters)
    sparse_results = sparse_search(query, top_k=fetch_k, filters=filters)

    # 按 id 建索引，合并 payload
    all_hits: dict[str, dict] = {}

    for rank, hit in enumerate(dense_results):
        hit_id = hit["id"]
        all_hits.setdefault(hit_id, {"id": hit_id, "payload": hit["payload"], "rrf_score": 0.0})
        all_hits[hit_id]["rrf_score"] += _rrf_score(rank)

    for rank, hit in enumerate(sparse_results):
        hit_id = hit["id"]
        all_hits.setdefault(hit_id, {"id": hit_id, "payload": hit["payload"], "rrf_score": 0.0})
        all_hits[hit_id]["rrf_score"] += _rrf_score(rank)

    # 按 RRF 分数降序排列，取前 top_k
    ranked = sorted(all_hits.values(), key=lambda x: x["rrf_score"], reverse=True)
    return ranked[:top_k]