"""
data_pipeline/processor/metadata_parser.py

解析 raw/metadata 下 JSON，产出可写入 papers 表的记录（list[dict]）。
单次遍历：读文件 → 判 pmid / 类型 / title → 不符合则写 progress 对应 key 并跳过。

会写入 progress 的 key：article_types_invalid、no_title_pmcids、parse_error_pmcids。
行结构与 db.papers 表字段一致，供 db.insert_papers(records) 直接使用。
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from config import EXCLUDED_ARTICLE_TYPES, METADATA_DIR, PROGRESS_FILE
from data_pipeline.storage.raw.progress import ProgressTracker

logger = logging.getLogger(__name__)

# 月份名 → 数字（PubMed 常见格式）
_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _parse_pub_date(meta: dict) -> date | None:
    """从 pub_year/pub_month/pub_day 解析为 date，解析失败返回 None。"""
    y = (meta.get("pub_year") or "").strip()
    m = (meta.get("pub_month") or "").strip()
    d = (meta.get("pub_day") or "").strip()
    if not y:
        return None
    if m and m.isdigit() and len(m) <= 2:
        month = m.zfill(2)
    elif m:
        month = _MONTH_MAP.get(m[:3].lower())
        if not month:
            return None
    else:
        return None
    if not d or not d.isdigit():
        day = "01"
    else:
        day = d.zfill(2) if len(d) <= 2 else d[:2]
    try:
        return date(int(y), int(month), min(int(day), 28))
    except (ValueError, TypeError):
        return None


def _try_build_row(meta: dict, pmcid: str, tracker: ProgressTracker) -> dict | None:
    """
    判断是否应入库并拼成一行：有 pmid、类型符合、有 title 则返回 row dict，否则按情况写 progress 并返回 None。
    """
    if meta.get("no_pubmed_record") or meta.get("pmid") is None:
        return None
    if set(meta.get("article_types") or []) & EXCLUDED_ARTICLE_TYPES:
        tracker.mark_article_types_invalid(pmcid)
        return None
    title = (meta.get("title") or "").strip()
    if not title:
        logger.debug("跳过无标题记录 %s", pmcid)
        tracker.mark_no_title(pmcid)
        return None
    return {
        "pmcid": meta.get("pmcid") or pmcid,
        "pmid": str(meta["pmid"]).strip(),
        "doi": (meta.get("doi") or "").strip() or None,
        "title": title,
        # "abstract": (meta.get("abstract") or "").strip() or None,
        "authors": list(meta.get("authors") or []),
        "journal": (meta.get("journal") or "").strip() or None,
        "pub_year": (meta.get("pub_year") or "").strip() or None,
        "pub_month": (meta.get("pub_month") or "").strip() or None,
        "pub_day": (meta.get("pub_day") or "").strip() or None,
        "pub_date": _parse_pub_date(meta),
        "keywords": list(meta.get("keywords") or []),
        "article_types": list(meta.get("article_types") or []),
    }


def parse_metadata_records_for_db(
    metadata_dir: Path | None = None,
    progress_file: Path | None = None,
) -> list[dict]:
    """
    单次遍历 metadata 下所有 JSON，返回可入库的记录列表。
    跳过项会写入 progress（article_types_invalid / no_title_pmcids / parse_error_pmcids）。
    返回的每条 dict 与 papers 表字段一致，可直接传入 db.insert_papers(records)。
    """
    directory = metadata_dir or METADATA_DIR
    if not directory.exists():
        logger.warning("metadata 目录不存在: %s", directory)
        return []

    tracker = ProgressTracker(progress_file or PROGRESS_FILE)
    records = []

    for path in sorted(directory.glob("*.json")):
        pmcid = path.stem
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("跳过无效文件 %s: %s", path.name, e)
            tracker.mark_parse_error(pmcid)
            continue

        row = _try_build_row(meta, pmcid, tracker)
        if row is not None:
            records.append(row)

    return records
