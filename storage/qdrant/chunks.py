"""
storage/qdrant/chunks.py

Chunk 向量写入 Qdrant：ensure_collection、upsert_chunks。
集合内同时存稠密向量与稀疏向量，payload 为 pmcid / source_type / section / text 等。
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import QDRANT_COLLECTION
from infra.clients import get_qdrant_client

logger = logging.getLogger(__name__)

DENSE_DIM  = 1536
BATCH_SIZE = 100


def _make_point_id(pmcid: str, source_type: str, section: str, chunk_index: int) -> str:
    """uuid5 把字符串稳定映射为合法 UUID，相同输入永远得到相同结果（幂等）。"""
    section_slug = section.lower().replace(" ", "_")[:30]
    raw = f"{pmcid}_{source_type}_{section_slug}_{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))


def ensure_collection(client: Optional[QdrantClient] = None):
    """
    确保集合存在，不存在则创建（同时配置稠密+稀疏向量）。
    已存在时不做任何修改，幂等。
    """
    c = client or get_qdrant_client()
    existing = [col.name for col in c.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        logger.info("集合 %s 已存在，跳过创建", QDRANT_COLLECTION)
        return

    c.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    logger.info("集合 %s 创建成功（dense + sparse）", QDRANT_COLLECTION)


def upsert_chunks(chunks: list[dict], client: Optional[QdrantClient] = None):
    """
    批量写入 Qdrant，每条 chunk 同时写入稠密和稀疏向量。
    dense_embedding 或 sparse_embedding 为 None 的条目自动跳过。
    upsert 语义：有则更新，无则插入，重复运行幂等。
    """
    c = client or get_qdrant_client()
    ensure_collection(c)

    valid = [
        ch for ch in chunks
        if ch.get("dense_embedding") is not None
        and ch.get("sparse_embedding") is not None
    ]
    skipped = len(chunks) - len(valid)
    if skipped:
        logger.warning("跳过 %d 条向量不完整的 chunk", skipped)
    if not valid:
        logger.warning("没有有效 chunk 可写入")
        return

    total = len(valid)
    logger.info("开始写入 Qdrant，共 %d 条", total)

    for start in range(0, total, BATCH_SIZE):
        batch = valid[start: start + BATCH_SIZE]
        points = []

        for ch in batch:
            point_id = _make_point_id(
                pmcid=ch["pmcid"],
                source_type=ch["source_type"],
                section=ch.get("section", "unknown"),
                chunk_index=ch.get("chunk_index", 0),
            )
            sparse = ch["sparse_embedding"]
            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense":  ch["dense_embedding"],
                    "sparse": SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },
                payload={
                    "pmcid":       ch["pmcid"],
                    "source_type": ch["source_type"],
                    "section":     ch.get("section", ""),
                    "chunk_index": ch.get("chunk_index", 0),
                    "text":        ch.get("text", ""),
                    "pub_year":    ch.get("pub_year", ""),
                    "journal":     ch.get("journal", ""),
                },
            ))

        c.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info("写入进度：%d / %d", min(start + BATCH_SIZE, total), total)

    logger.info("Qdrant 写入完成，共 %d 条", total)
