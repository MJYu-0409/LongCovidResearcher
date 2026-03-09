"""
retrieval/sparse.py

BM25 稀疏向量检索：将 Query 转为稀疏向量后在 Qdrant sparse 向量空间里检索。
捕捉关键词精确匹配，与语义检索互补。
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client.models import SparseVector
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_COLLECTION
from infra.clients import get_qdrant_client, get_sparse_embedding_model

logger = logging.getLogger(__name__)


def embed_query_sparse(query: str) -> SparseVector:
    """将 Query 转为稀疏向量。"""
    model = get_sparse_embedding_model()
    result = list(model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def sparse_search(
    query: str,
    top_k: int = 20,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    BM25 稀疏向量检索，返回 Top-K 结果。

    Args:
        query:   用户查询文本
        top_k:   返回条数
        filters: 可选 payload 过滤条件

    Returns:
        list[dict]，每条包含 score / payload
    """
    qdrant = get_qdrant_client()
    query_vector = embed_query_sparse(query)

    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ])

    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using="sparse",
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