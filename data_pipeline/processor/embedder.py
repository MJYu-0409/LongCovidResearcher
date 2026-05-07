"""
data_pipeline/processor/embedder.py

同时生成稠密向量（dense）和稀疏向量（sparse）：
  - 稠密向量：NeuML/pubmedbert-base-embeddings（本地 SentenceTransformer），捕捉语义相似性
  - 稀疏向量：FastEmbed SPLADE，捕捉关键词精确匹配
"""

from __future__ import annotations

import logging
from typing import Optional

from infra.clients import get_dense_embedding_model, get_sparse_embedding_model

logger = logging.getLogger(__name__)

BATCH_SIZE = 256  # 批量推理最优大小（CPU 下减少 per-call 开销）


def _embed_dense_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """单批稠密向量，本地 SentenceTransformer 推理。失败返回 None 列表。"""
    try:
        model = get_dense_embedding_model()
        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
        return [v.tolist() for v in vectors]
    except Exception as e:
        logger.error("稠密向量批次失败: %s", e)
        return [None] * len(texts)


def _embed_sparse_batch(texts: list[str]) -> list[Optional[dict]]:
    """
    单批稀疏向量。
    返回 {"indices": [...], "values": [...]} 与 Qdrant SparseVector 对应。
    """
    try:
        model = get_sparse_embedding_model()
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

    texts = [c["text"] for c in chunks]
    total = len(texts)
    logger.info("开始向量化，共 %d 条", total)

    for start in range(0, total, BATCH_SIZE):
        batch_texts  = texts[start: start + BATCH_SIZE]
        batch_chunks = chunks[start: start + BATCH_SIZE]

        dense_embs  = _embed_dense_batch(batch_texts)
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