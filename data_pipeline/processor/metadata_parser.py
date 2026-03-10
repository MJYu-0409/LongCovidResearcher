"""
data_pipeline/processor/metadata_parser.py

单次遍历 raw/metadata 下所有 JSON，同时产出：
  1. db_records:       list[dict]  → 直接写入 PostgreSQL papers 表
  2. abstract_chunks:  list[dict]  → 直接送 embedder 向量化后写入 Qdrant

设计原则：
  - 一次遍历，两份产出，避免重复读文件
  - 过滤逻辑集中在 _is_valid()，不散落到 pipeline
  - 摘要不切分，整体作为一个 chunk（source_type="abstract"）
  - abstract_chunk 携带检索时需要的 payload 字段（pub_year / journal）
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import NamedTuple

from config import EXCLUDED_ARTICLE_TYPES, METADATA_DIR, PROGRESS_FILE
from data_pipeline.raw.progress import ProgressTracker

logger = logging.getLogger(__name__)

_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


class ParseResult(NamedTuple):
    """parse_metadata 的返回值，两份产出打包在一起。"""
    db_records: list[dict]
    abstract_chunks: list[dict]


# ── 内部工具函数 ───────────────────────────────────────────────

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
    day = d.zfill(2) if (d and d.isdigit() and len(d) <= 2) else "01"
    try:
        return date(int(y), int(month), min(int(day), 28))
    except (ValueError, TypeError):
        return None


def _is_valid(meta: dict, pmcid: str, tracker: ProgressTracker) -> bool:
    """集中所有过滤逻辑，不符合条件的记录写 progress 并返回 False。"""
    if meta.get("no_pubmed_record") or meta.get("pmid") is None:
        return False
    if set(meta.get("article_types") or []) & EXCLUDED_ARTICLE_TYPES:
        tracker.mark_article_types_invalid(pmcid)
        return False
    if not (meta.get("title") or "").strip():
        tracker.mark_no_title(pmcid)
        return False
    return True


def _build_db_record(meta: dict, pmcid: str) -> dict:
    """构建 PostgreSQL papers 表的一行，字段与 db.papers 完全对应。"""
    return {
        "pmcid":         meta.get("pmcid") or pmcid,
        "pmid":          str(meta["pmid"]).strip(),
        "doi":           (meta.get("doi") or "").strip() or None,
        "title":         meta["title"].strip(),
        "authors":       list(meta.get("authors") or []),
        "journal":       (meta.get("journal") or "").strip() or None,
        "pub_year":      (meta.get("pub_year") or "").strip() or None,
        "pub_month":     (meta.get("pub_month") or "").strip() or None,
        "pub_day":       (meta.get("pub_day") or "").strip() or None,
        "pub_date":      _parse_pub_date(meta),
        "keywords":      list(meta.get("keywords") or []),
        "article_types": list(meta.get("article_types") or []),
    }


def _build_abstract_chunk(meta: dict, pmcid: str) -> dict | None:
    """
    构建摘要 chunk，供 embedder 直接使用。
    摘要整体作为一个 chunk，不切分。
    没有摘要文本则返回 None。
    """
    abstract = (meta.get("abstract") or "").strip()
    if not abstract:
        return None
    return {
        "pmcid":       pmcid,
        "source_type": "abstract",
        "section":     "abstract",
        "chunk_index": 0,
        #统一需要向量化的文本的原文字段为text，通用于摘要和切片后的全文
        "text":        abstract,
        # payload 字段：检索时过滤用，不需要回查 PostgreSQL
        "pub_year":    (meta.get("pub_year") or "").strip(),
        "journal":     (meta.get("journal") or "").strip(),
    }


# ── 对外接口 ──────────────────────────────────────────────────

def parse_metadata(
    metadata_dir: Path | None = None,
    progress_file: Path | None = None,
) -> ParseResult:
    """
    单次遍历 metadata 目录，同时产出 db_records 和 abstract_chunks。

    Returns:
        ParseResult(db_records, abstract_chunks)
        - db_records:      写入 PostgreSQL 用
        - abstract_chunks: 向量化后写入 Qdrant 用
    """
    directory = metadata_dir or METADATA_DIR
    if not directory.exists():
        logger.warning("metadata 目录不存在: %s", directory)
        return ParseResult([], [])

    tracker = ProgressTracker(progress_file or PROGRESS_FILE)
    db_records: list[dict] = []
    abstract_chunks: list[dict] = []

    files = sorted(directory.glob("*.json"))
    logger.info("开始解析 metadata，共 %d 个文件", len(files))

    for path in files:
        pmcid = path.stem
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("跳过无效文件 %s: %s", path.name, e)
            tracker.mark_parse_error(pmcid)
            continue

        if not _is_valid(meta, pmcid, tracker):
            continue

        db_records.append(_build_db_record(meta, pmcid))

        chunk = _build_abstract_chunk(meta, pmcid)
        #摘要存在就需要向量化，为空也不需要跟踪写入tracker
        if chunk:
            abstract_chunks.append(chunk)
        else:
            logger.debug("%s 无摘要文本，跳过摘要向量化", pmcid)

    logger.info(
        "解析完成：%d 条入库记录 / %d 条摘要 chunk",
        len(db_records), len(abstract_chunks)
    )
    return ParseResult(db_records, abstract_chunks)
