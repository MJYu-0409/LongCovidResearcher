"""
retrieval/sparse.py

BM25 稀疏向量检索：将 Query 转为稀疏向量后在 Qdrant sparse 向量空间里检索。
捕捉关键词精确匹配，与语义检索互补。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, NamedSparseVector

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

_sparse_model: Optional[SparseTextEmbedding] = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def embed_query_sparse(query: str) -> SparseVector:
    """将 Query 转为稀疏向量。"""
    model = _get_sparse_model()
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
    qdrant = _get_qdrant_client()
    query_vector = embed_query_sparse(query)

    from qdrant_client.models import Filter, FieldCondition, MatchValue
    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(must=[
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ])

    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=NamedSparseVector(name="sparse", vector=query_vector),
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id":      str(hit.id),
            "score":   hit.score,
            "payload": hit.payload,
        }
        for hit in results
    ]