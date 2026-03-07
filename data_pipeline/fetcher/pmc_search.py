"""
data_pipeline/fetcher/pmc_search.py
使用 ESearch 获取符合条件的全部 PMCID 列表
"""

import time
import logging
import requests

from config import (
    NCBI_API_KEY, NCBI_EMAIL, NCBI_TOOL,
    ESEARCH_URL,
    PMC_SEARCH_QUERY, PMC_DATE_MIN, PMC_DATE_MAX,
    BATCH_SIZE, REQUEST_INTERVAL,
)

logger = logging.getLogger(__name__)


def _base_params() -> dict:
    """所有 NCBI 请求的公共参数"""
    return {
        "tool":    NCBI_TOOL,
        "email":   NCBI_EMAIL,
        "api_key": NCBI_API_KEY,
        "retmode": "json",
    }


def search_pmcids() -> list[str]:
    """
    分页调用 ESearch，返回所有匹配的 PMCID 列表（字符串，如 "PMC9387111"）。
    ESearch 针对 db=pmc 返回的 id 已经是 PMC 数字 id，需要加 "PMC" 前缀。
    """
    all_ids: list[str] = []
    retstart = 0

    # 第一次请求：获取总数
    params = {
        **_base_params(),
        "db":       "pmc",
        "term":     PMC_SEARCH_QUERY,
        "mindate":  PMC_DATE_MIN,
        "maxdate":  PMC_DATE_MAX,
        "datetype": "pdat",
        "retmax":   BATCH_SIZE,
        "retstart": 0,
        "usehistory": "y",   # 使用 history server，避免超长 URL
    }

    logger.info("ESearch 第一次请求，获取总数...")
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["esearchresult"]

    total_count = int(data["count"])
    if total_count == 0:
        logger.info("未找到符合条件的文章")
        return []

    web_env = data["webenv"]
    query_key = data["querykey"]

    logger.info(f"共找到 {total_count} 篇文章，开始分页获取 PMCID...")

    # 收集第一批
    ids_batch = [f"PMC{i}" for i in data["idlist"]]
    all_ids.extend(ids_batch)
    retstart += len(ids_batch)

    # 后续分页：复用 WebEnv + query_key，不重复传 term
    while retstart < total_count:
        time.sleep(REQUEST_INTERVAL)

        page_params = {
            **_base_params(),
            "db":        "pmc",
            "WebEnv":    web_env,
            "query_key": query_key,
            "retmax":    BATCH_SIZE,
            "retstart":  retstart,
        }

        try:
            resp = requests.get(ESEARCH_URL, params=page_params, timeout=30)
            resp.raise_for_status()
            page_data = resp.json()["esearchresult"]
            ids_batch = [f"PMC{i}" for i in page_data["idlist"]]

            if not ids_batch:
                logger.warning(f"retstart={retstart} 返回空列表，提前终止")
                break

            all_ids.extend(ids_batch)
            retstart += len(ids_batch)
            logger.info(f"已获取 {retstart}/{total_count} 个 PMCID")

        except requests.RequestException as e:
            logger.error(f"ESearch 分页请求失败 (retstart={retstart}): {e}")
            time.sleep(5)   # 遇到错误等待后重试一次
            continue

    logger.info(f"PMCID 获取完成，共 {len(all_ids)} 条")
    return all_ids