"""
data_pipeline/scripts/reprocess_failed_fulltext.py

增量补跑：只对上次 eval/scan_parser_quality 报告中 FAIL 的文章重新向量化写入 Qdrant。
已有数据的文章不受影响（upsert 幂等）。

运行：
    python data_pipeline/scripts/reprocess_failed_fulltext.py
    python data_pipeline/scripts/reprocess_failed_fulltext.py --scan-report eval/output/scan_report.json
    python data_pipeline/scripts/reprocess_failed_fulltext.py --dry-run   # 只打印会处理哪些文章，不实际执行
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
# 把项目根目录加入 path，便于从任意位置运行脚本
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))
from config import FULLTEXT_DIR, PROGRESS_FILE
from data_pipeline.processor.xml_parser import parse_fulltext_xml
from data_pipeline.processor.chunker import chunk_fulltext
from data_pipeline.processor.embedder import embed_chunks
from data_pipeline.storage.postgres.db import fetch_meta_by_pmcids
from data_pipeline.storage.qdrant.db import upsert_chunks
from data_pipeline.storage.raw.progress import ProgressTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_SCAN_REPORT = _root / "eval" / "output" / "scan_report.json"

def load_fail_pmcids(scan_report_path: Path) -> list[str]:
    """从 scan_report.json 读取 FAIL 状态的 pmcid 列表。"""
    with open(scan_report_path, encoding="utf-8") as f:
        report = json.load(f)
    fail_pmcids = [
        r["pmcid"] for r in report["reports"] if r["status"] == "FAIL"
    ]
    logger.info("从扫描报告读取到 %d 个 FAIL pmcid", len(fail_pmcids))
    return fail_pmcids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-report", type=Path, default=DEFAULT_SCAN_REPORT,
        help=f"scan_parser_quality 输出的报告路径（默认：{DEFAULT_SCAN_REPORT}）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印会处理的文章列表，不实际向量化和写入",
    )
    args = parser.parse_args()

    if not args.scan_report.exists():
        logger.error("找不到扫描报告：%s，请先运行 eval/scan_parser_quality.py", args.scan_report)
        return

    # ── 1. 读取 FAIL pmcid ────────────────────────────────────────────
    fail_pmcids = load_fail_pmcids(args.scan_report)
    if not fail_pmcids:
        logger.info("没有 FAIL 文章，无需补跑")
        return

    # ── 2. 过滤出本地 XML 存在的文件 ──────────────────────────────────
    xml_files = []
    missing   = []
    for pmcid in fail_pmcids:
        xml_path = Path(FULLTEXT_DIR) / f"{pmcid}.xml"
        if xml_path.exists():
            xml_files.append(xml_path)
        else:
            missing.append(pmcid)

    if missing:
        logger.warning("%d 篇 XML 文件不存在，跳过：%s…", len(missing), missing[:5])
    logger.info("实际可处理：%d 篇", len(xml_files))

    if args.dry_run:
        print(f"\n[dry-run] 将处理以下 {len(xml_files)} 篇：")
        for p in xml_files:
            print(f"  {p.stem}")
        return

    if not xml_files:
        logger.info("没有可处理的文件")
        return

    # ── 3. 预取 metadata ──────────────────────────────────────────────
    pmcids   = [f.stem for f in xml_files]
    meta_map = fetch_meta_by_pmcids(pmcids)
    logger.info("已从 PostgreSQL 预取 %d 篇 metadata", len(meta_map))

    # ── 4. 逐篇处理 ───────────────────────────────────────────────────
    total   = len(xml_files)
    tracker = ProgressTracker(PROGRESS_FILE)
    success = 0
    still_empty = []

    for i, xml_path in enumerate(xml_files, 1):
        pmcid = xml_path.stem

        paragraphs = parse_fulltext_xml(xml_path)
        if not paragraphs:
            logger.debug("[%d/%d] %s 仍无有效段落，跳过", i, total, pmcid)
            still_empty.append(pmcid)
            continue

        chunks = chunk_fulltext(pmcid, paragraphs)
        if not chunks:
            logger.debug("[%d/%d] %s chunk 为空，跳过", i, total, pmcid)
            still_empty.append(pmcid)
            continue

        paper_meta = meta_map.get(pmcid, {"pub_year": "", "journal": ""})
        for chunk in chunks:
            chunk["pub_year"] = paper_meta["pub_year"]
            chunk["journal"]  = paper_meta["journal"]

        embed_chunks(chunks)
        upsert_chunks(chunks)

        failed_embed = [c for c in chunks
                        if c.get("dense_embedding") is None
                        or c.get("sparse_embedding") is None]
        if failed_embed:
            tracker.mark_fulltext_embed_failed([pmcid])
            logger.warning("%s 向量化失败 %d 个 chunk", pmcid, len(failed_embed))
        else:
            success += 1

        if i % 50 == 0:
            logger.info("进度：%d / %d", i, total)

    # ── 5. 汇总 ───────────────────────────────────────────────────────
    logger.info("补跑完成：%d 篇成功写入，%d 篇仍为空（无正文可提取）",
                success, len(still_empty))
    if still_empty:
        logger.info("仍为空的 pmcid（这类文章本身无正文，属正常）：%s…",
                    still_empty[:10])


if __name__ == "__main__":
    main()
