"""
retrieval/dense.py

语义检索：将 Query 向量化后在 Qdrant dense 向量空间里做近邻搜索。
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest

from config import OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

DENSE_MODEL = "text-embedding-3-small"


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def embed_query(query: str) -> list[float]:
    """将 Query 文本向量化，返回稠密向量。"""
    client = _get_openai_client()
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
    qdrant = _get_qdrant_client()
    query_vector = embed_query(query)

    qdrant_filter = _build_filter(filters) if filters else None

    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=("dense", query_vector),
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


def _build_filter(filters: dict) -> Filter:
    """把简单 key-value dict 转成 Qdrant Filter 对象。"""
    from qdrant_client.models import FieldCondition, MatchValue, Filter as QFilter
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return QFilter(must=conditions)