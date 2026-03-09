"""
data_pipeline/storage/postgres/db.py
papers 表定义、建表、写入。将 metadata_parser 返回的 list[dict] 写入 PostgreSQL。
"""

from __future__ import annotations

import logging

from sqlalchemy import MetaData, Table, Column, String, Text, Date, inspect, select
from sqlalchemy.dialects.postgresql import ARRAY, insert

from infra.clients import get_pg_engine

logger = logging.getLogger(__name__)


def _papers_table_exists(engine):
    """判断 papers 表是否已存在。"""
    return inspect(engine).has_table("papers")


_metadata = MetaData()
papers = Table(
    "papers",
    _metadata,
    Column("pmcid", String(20), primary_key=True),
    Column("pmid", String(20)),
    Column("doi", Text()),
    Column("title", Text(), nullable=False),
    Column("authors", ARRAY(Text())),
    Column("journal", Text()),
    Column("pub_year", String(4)),
    Column("pub_month", String(10)),
    Column("pub_day", String(2)),
    Column("pub_date", Date()),
    Column("keywords", ARRAY(Text())),
    Column("article_types", ARRAY(Text())),
)


def create_tables(engine=None):
    """根据 _metadata 在库中建表（已存在则跳过）。可单独执行一次。"""
    eng = engine or get_pg_engine()
    _metadata.create_all(eng)
    logger.info("papers 表已就绪（已存在则未改动）")


def insert_papers(records: list[dict]) -> int:
    """
    将 records（来自 metadata_parser.parse_metadata()的db_records）写入表 papers。
    ON CONFLICT (pmcid) DO UPDATE，重复运行幂等。返回写入/更新的行数。
    表不存在时自动建表。
    """
    if not records:
        return 0

    engine = get_pg_engine()
    if not _papers_table_exists(engine):
        create_tables(engine)
    count = 0
    with engine.connect() as conn:
        for row in records:
            # pub_date 若为 date 对象需保持，若为 None 亦可
            stmt = insert(papers).values(**row).on_conflict_do_update(
                index_elements=["pmcid"],
                set_={
                    "pmid": row["pmid"],
                    "doi": row["doi"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "journal": row["journal"],
                    "pub_year": row["pub_year"],
                    "pub_month": row["pub_month"],
                    "pub_day": row["pub_day"],
                    "pub_date": row["pub_date"],
                    "keywords": row["keywords"],
                    "article_types": row["article_types"],
                },
            )
            conn.execute(stmt)
            count += 1
        conn.commit()
    return count


def fetch_meta_by_pmcids(pmcids: list[str]) -> dict[str, dict]:
    """
    从 papers 表批量读取 pub_year / journal。
    返回 {pmcid: {"pub_year": str, "journal": str}}。
    找不到的 pmcid 以空字符串占位，保证调用方无需判断 key 是否存在。
    表不存在时直接 raise。
    """
    if not pmcids:
        return {}

    engine = get_pg_engine()
    if not _papers_table_exists(engine):
        raise RuntimeError(
            "papers 表不存在，请先执行 create_tables() 或 "
            "python -m data_pipeline.storage.postgres.db"
        )
    with engine.connect() as conn:
        rows = conn.execute(
            select(papers.c.pmcid, papers.c.pub_year, papers.c.journal)
            .where(papers.c.pmcid.in_(pmcids))
        ).fetchall()

    result = {row.pmcid: {"pub_year": row.pub_year or "", "journal": row.journal or ""}
              for row in rows}

    # 找不到的 pmcid 补空占位
    for pmcid in pmcids:
        result.setdefault(pmcid, {"pub_year": "", "journal": ""})

    return result


def fetch_paper_by_pmcid(pmcid: str) -> dict | None:
    """
    按 pmcid 查询单篇论文的完整元数据（供应用层如 Agent 调用）。

    Returns:
        包含 pmcid, title, authors, journal, pub_year, doi, abstract 的 dict；
        找不到或表不存在时返回 None。
    """
    engine = get_pg_engine()
    if not _papers_table_exists(engine):
        return None
    with engine.connect() as conn:
        row = conn.execute(
            select(
                papers.c.pmcid,
                papers.c.title,
                papers.c.authors,
                papers.c.journal,
                papers.c.pub_year,
                papers.c.doi,
                papers.c.abstract,
            ).where(papers.c.pmcid == pmcid).limit(1)
        ).fetchone()
    if row is None:
        return None
    authors = row.authors
    if authors is None:
        authors = []
    return {
        "pmcid": row.pmcid or "",
        "title": row.title or "",
        "authors": authors,
        "journal": row.journal or "",
        "pub_year": row.pub_year or "",
        "doi": row.doi or "",
        "abstract": row.abstract or "",
    }


if __name__ == "__main__":
    from infra import configure_logging
    configure_logging()
    create_tables()
