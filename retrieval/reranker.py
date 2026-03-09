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
) -> list[dict]:
    """
    对检索结果重排序，返回最相关的 Top-N 条。

    Args:
        query:  用户查询文本
        hits:   hybrid_search 返回的候选列表，每条含 payload.text
        top_n:  重排后保留的条数，默认 5（送给 LLM 的 context）

    Returns:
        list[dict]，按相关性降序，每条新增 "rerank_score" 字段
    """
    if not hits:
        return []

    model = get_rerank_model()

    # Cross-Encoder 输入格式：[query, document_text] 对
    pairs = [(query, hit["payload"].get("text", "")) for hit in hits]
    scores = model.predict(pairs)

    # # 把分数写回 hit，排序后取 Top-N
    # for hit, score in zip(hits, scores):
    #     hit["rerank_score"] = float(score)

    # ranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    # return ranked[:top_n]

    # 为每条 hit 建浅拷贝并写入 rerank_score，不修改传入的 hits
    ranked = [
        {**hit, "rerank_score": float(score)}
        for hit, score in zip(hits, scores)
    ]
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]