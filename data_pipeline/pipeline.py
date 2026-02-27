"""
data_pipeline/pipeline.py
数据拉取主流程入口：
  Step 1: ESearch 获取所有 PMCID
  Step 2: 批量 EFetch 摘要 + 全文，落盘到本地，断点续传
"""

import json
import logging
from config import PROGRESS_FILE, TEST_MODE, TEST_LIMIT

from data_pipeline.fetcher.pmc_search import search_pmcids
from data_pipeline.fetcher.pmc_fetcher import fetch_all
from data_pipeline.storage.progress import ProgressTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# PMCID 列表缓存文件（避免重复调用 ESearch）
PMCID_CACHE_FILE = PROGRESS_FILE.parent / "pmcid_list.json"


def run():
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

    # ── Step 2: 批量拉取摘要 + 全文 ──────────────────────────
    logger.info("开始批量拉取论文数据...")
    fetch_all(pmcids, tracker)

    logger.info("=" * 50)
    logger.info("数据拉取全部完成！")
    logger.info(tracker.summary())
    

if __name__ == "__main__":
    run()