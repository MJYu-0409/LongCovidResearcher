"""
retrieval/dense.py

语义检索：将 Query 向量化后在 Qdrant dense 向量空间里做近邻搜索。
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_COLLECTION, DENSE_MODEL
from infra.clients import get_openai_client, get_qdrant_client

logger = logging.getLogger(__name__)


def embed_query(query: str) -> list[float]:
    """将 Query 文本向量化，返回稠密向量。"""
    client = get_openai_client()
    response = client.embeddings.create(model=DENSE_MODEL, input=[query])
    return response.data[0].embedding


def dense_search(
    query: str,
    top_k: int = 20,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    语义检索，返回 Top-K 结果。

    Args:
        query:   用户查询文本
        top_k:   返回条数，默认 20（后续 Reranking 会精选 Top-5）
        filters: 可选的 Qdrant payload 过滤条件，例如：
                 {"pub_year": "2023"} 或 {"source_type": "abstract"}

    Returns:
        list[dict]，每条包含 score / payload（pmcid/text/section 等）
    """
    qdrant = get_qdrant_client()
    query_vector = embed_query(query)

    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ])

    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using="dense",
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id":      str(hit.id),
            "score":   hit.score,
            "payload": hit.payload or {},
        }
        for hit in response.points
    ]