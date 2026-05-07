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

from config import PROGRESS_FILE, FULLTEXT_DIR, TEST_MODE, TEST_LIMIT, QDRANT_COLLECTION_PC

from data_pipeline.fetcher.pmc_search import search_pmcids
from data_pipeline.fetcher.pmc_fetcher import fetch_all
from data_pipeline.raw.progress import ProgressTracker
from storage.postgres.papers import create_tables, insert_papers, fetch_meta_by_pmcids
from data_pipeline.processor.metadata_parser import parse_metadata
from data_pipeline.processor.xml_parser import parse_fulltext_xml
from data_pipeline.processor.chunker import chunk_fulltext_paragraphs
from data_pipeline.processor.embedder import embed_chunks
from storage.qdrant.chunks import upsert_chunks

logger = logging.getLogger(__name__)

PMCID_CACHE_FILE = PROGRESS_FILE.parent / "pmcid_list.json"


def _get_embedded_pmcids() -> set[str]:
    """滚动扫描 Qdrant，返回 longcovid_papers_pc 中已有的 pmcid 集合（用于断点续传）。"""
    from infra.clients import get_qdrant_client
    client = get_qdrant_client()
    embedded: set[str] = set()
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION_PC,
            scroll_filter=None,
            with_payload=["pmcid"],
            limit=1000,
            offset=offset,
        )
        for point in result:
            if point.payload and "pmcid" in point.payload:
                embedded.add(point.payload["pmcid"])
        if next_offset is None:
            break
        offset = next_offset
    logger.info("Qdrant 已有 %d 篇，将跳过", len(embedded))
    return embedded


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
    内部辅助：遍历 xml_files，解析 → 切分 → 补 metadata → 批量向量化 → 写入 Qdrant。
    跳过 Qdrant 中已有数据的 pmcid（断点续传）。
    meta_map: {pmcid: {"pub_year": ..., "journal": ...}}
    label:    日志前缀，区分首次运行和重建
    """
    total = len(xml_files)
    tracker = ProgressTracker(PROGRESS_FILE)

    already_done = _get_embedded_pmcids()
    xml_files = [f for f in xml_files if f.stem not in already_done]
    logger.info("%s：跳过已完成 %d 篇，剩余 %d 篇待处理",
                label, total - len(xml_files), len(xml_files))
    total = len(xml_files)

    PAPERS_PER_BATCH = 1000
    batch_chunks: list[dict] = []
    batch_start_idx = 1

    def _flush(up_to_i: int):
        nonlocal batch_chunks, batch_start_idx
        if not batch_chunks:
            return
        embed_chunks(batch_chunks)
        upsert_chunks(batch_chunks)
        failed_pmcids = list({
            c["pmcid"] for c in batch_chunks
            if c.get("dense_embedding") is None or c.get("sparse_embedding") is None
        })
        if failed_pmcids:
            tracker.mark_fulltext_embed_failed(failed_pmcids)
            logger.warning("批次向量化失败 %d 篇，已记录到 progress", len(failed_pmcids))
        logger.info("%s 进度：%d / %d（本批 %d chunks，论文 %d~%d）",
                    label, up_to_i, total, len(batch_chunks), batch_start_idx, up_to_i)
        batch_chunks = []
        batch_start_idx = up_to_i + 1

    for i, xml_path in enumerate(xml_files, 1):
        pmcid = xml_path.stem

        paragraphs = parse_fulltext_xml(xml_path)
        if not paragraphs:
            logger.debug("[%d/%d] %s 无有效段落，跳过", i, total, pmcid)
            continue

        chunks = chunk_fulltext_paragraphs(pmcid, paragraphs)
        if not chunks:
            logger.debug("[%d/%d] %s chunk 为空，跳过", i, total, pmcid)
            continue

        paper_meta = meta_map.get(pmcid, {"pub_year": "", "journal": ""})
        for chunk in chunks:
            chunk["pub_year"] = paper_meta["pub_year"]
            chunk["journal"]  = paper_meta["journal"]
        batch_chunks.extend(chunks)

        if i % PAPERS_PER_BATCH == 0:
            _flush(i)

    _flush(total)
    logger.info("%s 完成，共处理 %d 篇", label, total)


def run_process_fulltext():
    """
    Stage 3：遍历 raw/fulltext XML，解析 → 切分 → 向量化 → 写入 Qdrant。
    每篇独立处理，失败不影响其他篇。fulltext chunk 同时携带 pub_year / journal。

    依赖：必须先执行 run_process_meta()，确保 papers 表存在且有数据；
    否则 fetch_meta_by_pmcids 会报错（表不存在）或返回空元数据（表为空）。
    """
    xml_files = sorted(FULLTEXT_DIR.glob("*.xml"))
    logger.info("Stage 3 开始：共 %d 篇全文（依赖 Stage 2 已执行，papers 表需存在）", len(xml_files))

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
    from infra import configure_logging
    configure_logging()
    run()
