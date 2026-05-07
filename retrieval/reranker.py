"""
retrieval/reranker.py

Cross-Encoder Reranking：对混合检索召回的 Top-K 结果精细重排，
选出最相关的 Top-N 送给 LLM 生成回答。

Cross-Encoder 工作原理：
  Bi-Encoder（embedding 检索）把 Query 和文档分别编码，速度快但精度有限。
  Cross-Encoder 把 Query 和文档拼接后一起编码，精度更高但速度慢。
  所以用 Bi-Encoder 先粗筛 Top-20，再用 Cross-Encoder 精选 Top-5。
"""

from __future__ import annotations

import logging

from infra.clients import get_rerank_model

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    hits: list[dict],
    top_n: int = 5,
    max_per_paper: int = 2,
) -> list[dict]:
    """
    对检索结果重排序，返回最相关的 Top-N 条。
    选取时每篇论文最多保留 max_per_paper 条，与 RRF 层的多样性约束形成级联。

    Args:
        query:          用户查询文本
        hits:           hybrid_search 返回的候选列表，每条含 payload.text
        top_n:          重排后保留的条数，默认 5
        max_per_paper:  每篇论文最多保留条数，默认 2

    Returns:
        list[dict]，按相关性降序，每条新增 "rerank_score" 字段
    """
    if not hits:
        return []

    model = get_rerank_model()

    # Cross-Encoder 对全部候选打分，不提前截断
    pairs = [(query, hit["payload"].get("text", "")) for hit in hits]
    scores = model.predict(pairs)

    ranked = sorted(
        ({**hit, "rerank_score": float(s)} for hit, s in zip(hits, scores)),
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    # 贪心选取：按分数高低，每篇最多取 max_per_paper 条
    results: list[dict] = []
    paper_counts: dict[str, int] = {}
    for hit in ranked:
        pmcid = hit["payload"].get("pmcid", hit["id"])
        if paper_counts.get(pmcid, 0) < max_per_paper:
            paper_counts[pmcid] = paper_counts.get(pmcid, 0) + 1
            results.append(hit)
        if len(results) >= top_n:
            break

    logger.info(
        "rerank top-%d from %d papers: %s",
        top_n,
        len(paper_counts),
        [(r["payload"].get("pmcid", "?"), round(r["rerank_score"], 3)) for r in results],
    )
    return results