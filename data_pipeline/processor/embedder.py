"""
data_pipeline/processor/embedder.py

同时生成稠密向量（dense）和稀疏向量（sparse）：
  - 稠密向量：OpenAI text-embedding-3-small，捕捉语义相似性
  - 稀疏向量：FastEmbed SPLADE，捕捉关键词精确匹配
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI
from fastembed import SparseTextEmbedding

from config import DENSE_MODEL, SPARSE_MODEL
from infra.clients import get_openai_client

logger = logging.getLogger(__name__)

BATCH_SIZE   = 200
MAX_RETRIES  = 3
BACKOFF_BASE = 2.0

_sparse_model: Optional[SparseTextEmbedding] = None


def _get_sparse_model() -> SparseTextEmbedding:
    """懒加载 SPLADE 模型，首次运行自动下载（约50MB），之后复用。"""
    global _sparse_model
    if _sparse_model is None:
        logger.info("加载稀疏向量模型（首次运行会下载模型文件）")
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def _embed_dense_batch(client: OpenAI, texts: list[str]) -> list[Optional[list[float]]]:
    """单批稠密向量，带指数退避重试。失败返回 None 占位。"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=DENSE_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning("稠密向量请求失败（%d/%d）: %s，%.1fs 后重试",
                           attempt + 1, MAX_RETRIES, e, wait)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    logger.error("稠密向量批次彻底失败")
    return [None] * len(texts)


def _embed_sparse_batch(texts: list[str]) -> list[Optional[dict]]:
    """
    单批稀疏向量。
    返回 {"indices": [...], "values": [...]} 与 Qdrant SparseVector 对应。
    """
    try:
        model = _get_sparse_model()
        embeddings = list(model.embed(texts))
        return [
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
            for emb in embeddings
        ]
    except Exception as e:
        logger.error("稀疏向量批次失败: %s", e)
        return [None] * len(texts)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    为 chunk 列表同时生成稠密和稀疏向量，in-place 写入：
      - "dense_embedding":  list[float] | None
      - "sparse_embedding": {"indices": [...], "values": [...]} | None
    """
    if not chunks:
        return chunks

    client = get_openai_client()
    texts  = [c["text"] for c in chunks]
    total  = len(texts)
    logger.info("开始向量化，共 %d 条", total)

    for start in range(0, total, BATCH_SIZE):
        batch_texts  = texts[start: start + BATCH_SIZE]
        batch_chunks = chunks[start: start + BATCH_SIZE]

        dense_embs  = _embed_dense_batch(client, batch_texts)
        sparse_embs = _embed_sparse_batch(batch_texts)

        for chunk, dense, sparse in zip(batch_chunks, dense_embs, sparse_embs):
            chunk["dense_embedding"]  = dense
            chunk["sparse_embedding"] = sparse

        logger.info("向量化进度：%d / %d", min(start + BATCH_SIZE, total), total)

    success = sum(
        1 for c in chunks
        if c.get("dense_embedding") is not None
        and c.get("sparse_embedding") is not None
    )
    logger.info("向量化完成：%d 成功 / %d 总计", success, total)
    return chunks