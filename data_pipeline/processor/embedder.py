"""
data_pipeline/processor/embedder.py

调用 OpenAI text-embedding-3-small 批量生成向量。

特性：
  - 批量请求，每批最多 BATCH_SIZE 条，避免单次请求过大
  - 指数退避重试，网络抖动或限流时自动重试
  - 返回与输入等长的向量列表，失败的批次对应位置填 None
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# 每批最多发送的文本数量（OpenAI 单次最多 2048 条，保守取 200）
BATCH_SIZE = 200
# 向量维度（text-embedding-3-small 固定 1536）
EMBEDDING_DIM = 1536
# 重试次数和退避基数
MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("config.OPENAI_API_KEY 未配置，请在 .env 中设置 OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[Optional[list[float]]]:
    """
    对单批文本调用 OpenAI embedding API，带指数退避重试。
    返回与 texts 等长的列表，成功返回向量，失败返回 None。
    """
    for attempt in range(MAX_RETRIES):
        try:
            #text-embedding-3-small是openai托管在自家服务器上的模型
            #text-embedding-3-small 是通用模型，在文献摘要/全文这类相对规范的医学文本上，很多场景下已经够用。
            #1 token ≈ 4 chars, 1 word ≈ 4-5 chars 即 1~1.25 tokens，摘要一般在400words以下，全文分chunk的上限是800words，都在该模型的单次输入上限token数范围内
            response = client.embeddings.create(
                model="text-embedding-3-small",   #1536 维
                input=texts,
            )
            # response.data 顺序与 input 对应  item是每个response的输出data包，里面还包了embedding
            return [item.embedding for item in response.data]
        except Exception as e:
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "Embedding 请求失败（attempt %d/%d）: %s，%.1f 秒后重试",
                attempt + 1, MAX_RETRIES, e, wait
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                logger.error("Embedding 批次彻底失败，返回 None 列表")
                return [None] * len(texts)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    为 chunk 列表批量生成向量，将 embedding 写入每条 dict 的 "embedding" 字段。
    embedding 为 None 表示该条向量化失败，写入 Qdrant 时会跳过。

    Args:
        chunks: 摘要 chunk（metadata_parser）或全文 chunk（chunker），每条含 "text" 字段

    Returns:
        原 chunks 列表（in-place 添加 "embedding" 字段）
    """
    if not chunks:
        return chunks

    client = _get_client()
    texts = [c["text"] for c in chunks]
    total = len(texts)
    logger.info("开始向量化，共 %d 条", total)

    for start in range(0, total, BATCH_SIZE):
        batch_texts = texts[start: start + BATCH_SIZE]
        batch_chunks = chunks[start: start + BATCH_SIZE]

        #批次行为，要么一起成功要么一起失败，三次重试机会，三次都失败则返回None写入库
        embeddings = _embed_batch(client, batch_texts)

        #zip：按照batch_chunks中chunk的顺序，返回对应embedding的向量化结果。由于两个数组排序都是一致的，才能做到匹配。
        #循环里 把对应向量化结果赋值给chunk新的字段embedding
        for chunk, emb in zip(batch_chunks, embeddings):
            chunk["embedding"] = emb

        logger.info(
            "向量化进度: %d / %d",
            min(start + BATCH_SIZE, total), total
        )

    success = sum(1 for c in chunks if c.get("embedding") is not None)
    logger.info("向量化完成：%d 成功 / %d 总计", success, total)
    return chunks
