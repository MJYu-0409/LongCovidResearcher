"""
data_pipeline/storage/qdrant/db.py

封装 Qdrant 写入操作：建集合、批量 upsert 向量。

集合设计：
  - 集合名：longcovid_papers
  - 向量维度：1536（text-embedding-3-small）
  - 距离度量：Cosine
  - payload 字段：pmcid / source_type / section / chunk_index / text / pub_year / journal
    其中 text 冗余存一份（摘要原文或正文片段），检索命中后直接取，无需回查文件

ID 生成规则：
  - 摘要：  {pmcid}_abstract_0
  - 正文：  {pmcid}_{section_slug}_{chunk_index}
  Qdrant 需要 UUID 或无符号整数作为 point id，这里用 uuid5 把字符串 id 转为 UUID。
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

VECTOR_DIM = 1536
BATCH_SIZE = 100   # 每批 upsert 的 point 数


def _get_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("config.QDRANT_URL 未配置")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def _make_point_id(pmcid: str, source_type: str, section: str, chunk_index: int) -> str:
    """生成稳定唯一的字符串 id，用于 Qdrant point id（转为 UUID）。"""
    section_slug = section.lower().replace(" ", "_")[:30]
    raw = f"{pmcid}_{source_type}_{section_slug}_{chunk_index}"
    # uuid5 把任意字符串映射为合法 UUID，相同输入永远得到相同 UUID（幂等）
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))


def ensure_collection(client: Optional[QdrantClient] = None):
    """
    确保集合存在，不存在则创建。
    幂等：已存在时不会报错，也不会删除现有数据。
    """
    c = client or _get_client()
    existing = [col.name for col in c.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        logger.info("集合 %s 已存在，跳过创建", QDRANT_COLLECTION)
        return
    c.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    logger.info("集合 %s 创建成功", QDRANT_COLLECTION)


def upsert_chunks(chunks: list[dict], client: Optional[QdrantClient] = None):
    """
    将已向量化的 chunk 列表批量写入 Qdrant。
    embedding 为 None 的条目自动跳过。
    使用 upsert（有则更新，无则插入），重复运行幂等。

    Args:
        chunks: embedder.embed_chunks() 处理后的列表，每条含 embedding 字段
        client: 可选，传入已有客户端（测试或复用连接）
    """
    c = client or _get_client()
    ensure_collection(c)

    # 过滤掉向量化失败的条目
    valid = [ch for ch in chunks if ch.get("embedding") is not None]
    skipped = len(chunks) - len(valid)
    if skipped:
        logger.warning("跳过 %d 条 embedding 为 None 的 chunk", skipped)
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
            payload = {
                "pmcid":       ch["pmcid"],
                "source_type": ch["source_type"],
                "section":     ch.get("section", ""),
                "chunk_index": ch.get("chunk_index", 0),
                "text":        ch.get("text", ""),   # 摘要或正文片段，检索命中后直接取
                "pub_year":    ch.get("pub_year", ""),
                "journal":     ch.get("journal", ""),
            }
            points.append(PointStruct(
                id=point_id,
                vector=ch["embedding"],
                payload=payload,
            ))

        c.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info("写入进度: %d / %d", min(start + BATCH_SIZE, total), total)

    logger.info("Qdrant 写入完成，共 %d 条", total)
