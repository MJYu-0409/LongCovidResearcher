"""
agent/tools/paper.py

工具二：get_paper_detail
应用层：调用数据层按 pmcid 查论文，整合为 Agent 所需的 JSON 返回。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from storage.postgres.papers import fetch_paper_by_pmcid

logger = logging.getLogger(__name__)


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
    paper = fetch_paper_by_pmcid(pmcid)
    if paper is None:
        return json.dumps({"error": f"未找到 {pmcid}，请确认 pmcid 格式正确"})

    logger.info("get_paper_detail: %s → %s", pmcid, paper.get("title", "")[:50])
    return json.dumps(paper, ensure_ascii=False, indent=2)
