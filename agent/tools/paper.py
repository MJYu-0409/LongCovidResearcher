"""
agent/tools/paper.py

工具二：get_paper_detail
查询 PostgreSQL，返回论文完整元数据。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool
from sqlalchemy import select, text

from infra.clients import get_pg_engine

logger = logging.getLogger(__name__)


def _fetch_paper(pmcid: str) -> dict | None:
    engine = get_pg_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT pmcid, title, authors, journal, pub_year,
                       doi, abstract
                FROM papers
                WHERE pmcid = :pmcid
                LIMIT 1
            """),
            {"pmcid": pmcid},
        ).fetchone()

    if row is None:
        return None

    return {
        "pmcid":    row.pmcid,
        "title":    row.title    or "",
        "authors":  row.authors  or "",
        "journal":  row.journal  or "",
        "pub_year": row.pub_year or "",
        "doi":      row.doi      or "",
        "abstract": row.abstract or "",
    }


@tool
def get_paper_detail(pmcid: str) -> str:
    """
    根据 PMC ID 获取论文的完整元数据，包括标题、作者、期刊、年份、DOI 和摘要。
    适用于：用户追问某篇具体文章的信息、需要引用文献来源时。

    Args:
        pmcid: PMC 论文 ID，格式如 "PMC10502909"

    Returns:
        论文元数据（JSON字符串），找不到时返回错误提示
    """
    paper = _fetch_paper(pmcid)
    if paper is None:
        return json.dumps({"error": f"未找到 {pmcid}，请确认 pmcid 格式正确"})

    logger.info("get_paper_detail: %s → %s", pmcid, paper.get("title", "")[:50])
    return json.dumps(paper, ensure_ascii=False, indent=2)
