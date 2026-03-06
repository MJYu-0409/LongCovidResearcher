"""
data_pipeline/pipeline.py

三阶段数据处理 pipeline，各阶段独立可重试：

  Stage 1: fetch_raw
    ESearch 获取 PMCID 列表 → EFetch 批量拉取摘要 + 全文 → 落盘 raw/

  Stage 2: process_meta
    单次遍历 raw/metadata JSON：
      → 全量元数据写入 PostgreSQL（papers 表）
      → 摘要批量向量化写入 Qdrant（source_type=abstract）

  Stage 3: process_fulltext
    遍历 raw/fulltext XML：
      → xml_parser 提取干净段落
      → chunker 按 section 切分
      → 批量向量化写入 Qdrant（source_type=fulltext）

运行方式（main.py 里按需调用）：
  from data_pipeline.pipeline import run_fetch_raw, run_process_meta, run_process_fulltext
"""

from __future__ import annotations

import json
import logging

from config import PROGRESS_FILE, FULLTEXT_DIR, TEST_MODE, TEST_LIMIT

from data_pipeline.fetcher.pmc_search import search_pmcids
from data_pipeline.fetcher.pmc_fetcher import fetch_all
from data_pipeline.storage.raw.progress import ProgressTracker
from data_pipeline.storage.postgres.db import create_tables, insert_papers, fetch_meta_by_pmcids
from data_pipeline.processor.metadata_parser import parse_metadata
from data_pipeline.processor.xml_parser import parse_fulltext_xml
from data_pipeline.processor.chunker import chunk_fulltext
from data_pipeline.processor.embedder import embed_chunks
from data_pipeline.storage.qdrant.db import upsert_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PMCID_CACHE_FILE = PROGRESS_FILE.parent / "pmcid_list.json"


# ══════════════════════════════════════════════════════════════
# Stage 1
# ══════════════════════════════════════════════════════════════

def run_fetch_raw():
    """ESearch 获取 PMCID 列表，批量 EFetch 拉取摘要与全文，落盘到 raw/，支持断点续传。"""
    tracker = ProgressTracker(PROGRESS_FILE)

    if PMCID_CACHE_FILE.exists():
        logger.info("发现 PMCID 缓存，直接加载")
        with open(PMCID_CACHE_FILE) as f:
            pmcids = json.load(f)
    else:
        logger.info("ESearch 获取 PMCID 列表...")
        pmcids = search_pmcids()
        with open(PMCID_CACHE_FILE, "w") as f:
            json.dump(pmcids, f)

    logger.info("共 %d 个 PMCID", len(pmcids))
    tracker.set_total(len(pmcids))

    if TEST_MODE:
        logger.info("⚠️  测试模式，只处理前 %d 篇", TEST_LIMIT)
        pmcids = pmcids[:TEST_LIMIT]

    fetch_all(pmcids, tracker)
    logger.info("Stage 1 完成：%s", tracker.summary())


# ══════════════════════════════════════════════════════════════
# Stage 2
# ══════════════════════════════════════════════════════════════

def run_process_meta():
    """
    单次遍历 raw/metadata JSON，同时产出两份数据：
      - 全量元数据 → PostgreSQL
      - 摘要文本   → 批量向量化 → Qdrant
    """
    logger.info("Stage 2 开始：解析 metadata JSON")

    result = parse_metadata()

    # ── 写入 PostgreSQL ──
    logger.info("写入 PostgreSQL，共 %d 条", len(result.db_records))
    create_tables()
    inserted = insert_papers(result.db_records)
    logger.info("PostgreSQL 写入完成：%d 行", inserted)

    # ── 摘要向量化 → Qdrant ──
    if result.abstract_chunks:
        logger.info("摘要向量化，共 %d 条", len(result.abstract_chunks))
        embed_chunks(result.abstract_chunks)
        upsert_chunks(result.abstract_chunks)
        # 记录摘要向量化失败的 PMCID，便于只重试这批
        failed_abstract = list({c["pmcid"] for c in result.abstract_chunks if c.get("dense_embedding") is None or c.get("sparse_embedding") is None})
        if failed_abstract:
            tracker = ProgressTracker(PROGRESS_FILE)
            tracker.mark_abstract_embed_failed(failed_abstract)
            logger.warning("摘要向量化失败 %d 条，已写入 progress.abstract_embed_failed_pmcids", len(failed_abstract))
        logger.info("摘要写入 Qdrant 完成")
    else:
        logger.warning("没有有效摘要，跳过向量化")

    logger.info("Stage 2 完成")


# ══════════════════════════════════════════════════════════════
# Stage 3
# ══════════════════════════════════════════════════════════════

def _process_fulltext_files(xml_files: list, meta_map: dict[str, dict], label: str):
    """
    内部辅助：遍历 xml_files，解析 → 切分 → 补 metadata → 向量化 → 写入 Qdrant。
    meta_map: {pmcid: {"pub_year": ..., "journal": ...}}
    label:    日志前缀，区分首次运行和重建
    """
    total = len(xml_files)
    tracker = ProgressTracker(PROGRESS_FILE)

    for i, xml_path in enumerate(xml_files, 1):
        pmcid = xml_path.stem

        paragraphs = parse_fulltext_xml(xml_path)
        if not paragraphs:
            logger.debug("[%d/%d] %s 无有效段落，跳过", i, total, pmcid)
            continue

        chunks = chunk_fulltext(pmcid, paragraphs)
        if not chunks:
            logger.debug("[%d/%d] %s chunk 为空，跳过", i, total, pmcid)
            continue

        # 补入 pub_year / journal（从 Postgres 预取）
        paper_meta = meta_map.get(pmcid, {"pub_year": "", "journal": ""})
        for chunk in chunks:
            chunk["pub_year"] = paper_meta["pub_year"]
            chunk["journal"]  = paper_meta["journal"]

        embed_chunks(chunks)
        upsert_chunks(chunks)

        failed = [c for c in chunks if c.get("dense_embedding") is None
                                    or c.get("sparse_embedding") is None]
        if failed:
            tracker.mark_fulltext_embed_failed([pmcid])
            logger.warning("%s 向量化失败 %d 个 chunk", pmcid, len(failed))

        if i % 50 == 0:
            logger.info("%s 进度：%d / %d", label, i, total)

    logger.info("%s 完成，共处理 %d 篇", label, total)


def run_process_fulltext():
    """
    Stage 3：遍历 raw/fulltext XML，解析 → 切分 → 向量化 → 写入 Qdrant。
    每篇独立处理，失败不影响其他篇。fulltext chunk 同时携带 pub_year / journal。
    """
    xml_files = sorted(FULLTEXT_DIR.glob("*.xml"))
    logger.info("Stage 3 开始：共 %d 篇全文", len(xml_files))

    pmcids = [f.stem for f in xml_files]
    meta_map = fetch_meta_by_pmcids(pmcids)
    logger.info("已从 PostgreSQL 预取 %d 篇 metadata", len(meta_map))

    _process_fulltext_files(xml_files, meta_map, label="Stage 3")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════

def run():
    """默认全流程，按阶段顺序执行。可在 main.py 里单独调用某个阶段。"""
    # run_fetch_raw()
    # run_process_meta()
    run_process_fulltext()


if __name__ == "__main__":
    run()
