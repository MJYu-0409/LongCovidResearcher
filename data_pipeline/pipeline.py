"""
data_pipeline/pipeline.py
数据拉取主流程入口：
  Step 1: ESearch 获取所有 PMCID
  Step 2: 批量 EFetch 摘要 + 全文，落盘到本地，断点续传
第二步：清洗数据、分别入库
  Step 1: 过滤文献
  Step 2: 解析元数据json,提取元数据写入关系型数据库postgresql
"""

import json
import logging

from config import PROGRESS_FILE, TEST_MODE, TEST_LIMIT

from data_pipeline.fetcher.pmc_search import search_pmcids
from data_pipeline.fetcher.pmc_fetcher import fetch_all
from data_pipeline.storage.raw.progress import ProgressTracker
from data_pipeline.storage.postgres.db import create_tables, insert_papers
from data_pipeline.processor.metadata_parser import parse_metadata_records_for_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# PMCID 列表缓存文件（避免重复调用 ESearch）
PMCID_CACHE_FILE = PROGRESS_FILE.parent / "pmcid_list.json"


def run_metadata_to_postgres():
    """解析 raw/metadata，写入 PostgreSQL papers 表；表不存在则先建表。"""
    create_tables()
    records = parse_metadata_records_for_db()
    logger.info("解析通过筛选的记录数: %d", len(records))
    if not records:
        return
    n = insert_papers(records)
    logger.info("写入/更新 papers 表行数: %d", n)


def run_fetch_raw():
    """ESearch 获取 PMCID 列表，批量 EFetch 摘要与全文，落盘到 raw，断点续传。"""
    tracker = ProgressTracker(PROGRESS_FILE)

    # ── Step 1: 获取 PMCID 列表 ──────────────────────────────
    if PMCID_CACHE_FILE.exists():
        logger.info("发现 PMCID 缓存文件，直接加载，跳过 ESearch")
        with open(PMCID_CACHE_FILE, "r") as f:
            pmcids = json.load(f)
        logger.info(f"加载 {len(pmcids)} 个 PMCID")
    else:
        logger.info("开始 ESearch 获取 PMCID 列表...")
        pmcids = search_pmcids()
        # 缓存到本地，下次直接用
        with open(PMCID_CACHE_FILE, "w") as f:
            json.dump(pmcids, f)
        logger.info(f"ESearch 完成，共 {len(pmcids)} 个 PMCID，已缓存到 {PMCID_CACHE_FILE}")

    tracker.set_total(len(pmcids))

    # 测试模式：只取前 N 篇
    if TEST_MODE:
        logger.info(f"⚠️  测试模式：只处理前 {TEST_LIMIT} 篇，全量请将 config.py 中 TEST_MODE 改为 False")
        pmcids = pmcids[:TEST_LIMIT]

    # ── Step 3: 批量拉取摘要 + 全文 ──────────────────────────
    logger.info("开始批量拉取论文数据...")
    fetch_all(pmcids, tracker)

    logger.info("=" * 50)
    logger.info("拉取全部完成！%s", tracker.summary())


def run():
    """主流程：拉取文献数据，再写入库"""
    # run_fetch_raw()
    run_metadata_to_postgres()


if __name__ == "__main__":
    run()
