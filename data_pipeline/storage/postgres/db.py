"""
data_pipeline/storage/postgres/db.py
papers 表定义、建表、写入。将 metadata_parser 返回的 list[dict] 写入 PostgreSQL。
"""

from __future__ import annotations

import logging

from sqlalchemy import MetaData, Table, Column, String, Text, Date, create_engine, select
from sqlalchemy.dialects.postgresql import ARRAY, insert

from config import DATABASE_URL

logger = logging.getLogger(__name__)

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
    if not DATABASE_URL:
        raise ValueError("config.DATABASE_URL 未配置")
    eng = engine or create_engine(DATABASE_URL)
    _metadata.create_all(eng)
    logger.info("papers 表已就绪（已存在则未改动）")


def insert_papers(records: list[dict]) -> int:
    """
    将 records（来自 metadata_parser.parse_metadata()的db_records）写入表 papers。
    ON CONFLICT (pmcid) DO UPDATE，重复运行幂等。返回写入/更新的行数。
    """
    if not DATABASE_URL:
        raise ValueError("config.DATABASE_URL 未配置")
    if not records:
        return 0

    engine = create_engine(DATABASE_URL)
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
    """
    if not DATABASE_URL:
        raise ValueError("config.DATABASE_URL 未配置")
    if not pmcids:
        return {}

    engine = create_engine(DATABASE_URL)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_tables()
