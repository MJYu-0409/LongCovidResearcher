"""
data_pipeline/fetcher/pmc_fetcher.py

每篇论文拉取两样东西：
  1. 全文 XML       →  fulltext/{PMCID}.xml
     用 EFetch db=pmc rettype=full retmode=xml 拉取 JATS 格式全文

  2. 摘要 + 元数据  →  metadata/{PMCID}.json
    先 ELink 拿 PMID，再 EFetch db=pubmed 拿摘要和元数据。
    ELink 无结果时会尝试从已保存的全文 XML（article-meta）中提取 PMID 再拉元数据。
    若仍无 PMID，返回只含 pmcid 的最小 dict。

支持断点续传：任意位置中断后重新运行，自动跳过已完成的文章。
"""

import json
import time
import logging
import requests
import xml.etree.ElementTree as ET
from pathlib import Path

from config import (
    NCBI_API_KEY, NCBI_EMAIL, NCBI_TOOL,
    EFETCH_URL, ELINK_URL,
    METADATA_DIR, FULLTEXT_DIR,
    REQUEST_INTERVAL,
)
from data_pipeline.storage.raw.progress import ProgressTracker

logger = logging.getLogger(__name__)


# ── 公共参数 ──────────────────────────────────────────────────

def _base_params() -> dict:
    p = {
        "tool":  NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


# ── PMCID → PMID 转换 ─────────────────────────────────────────

def pmcid_to_pmid(pmcid: str) -> str | None:
    """
    用 ELink 把单个 PMCID（纯数字）转换为 PMID。
    PMC 文章大多数有对应的 PubMed 记录，少数没有则返回 None。
    """
    time.sleep(REQUEST_INTERVAL)
    pmc_numeric = pmcid.replace("PMC", "")
    params = {
        **_base_params(),
        "dbfrom":  "pmc",
        "db":      "pubmed",
        "id":      pmc_numeric,
        "retmode": "json",
    }
    try:
        resp = requests.get(ELINK_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        linksets = data.get("linksets", [])
        if not linksets:
            return None
        linksetdbs = linksets[0].get("linksetdbs", [])
        for db in linksetdbs:
            if db.get("linkname") == "pmc_pubmed":
                ids = db.get("links", [])
                return str(ids[0]) if ids else None
        return None

    except (requests.RequestException, KeyError, IndexError) as e:
        logger.debug(f"ELink 转换失败 {pmcid}: {e}")
        return None


# ── Elink转换id接口返回异常时，兜底做法从全文中提取pmid ───────────

def _pmid_from_fulltext_xml(xml_text: str) -> str | None:
    """从 JATS 全文 XML 的 article-meta 中提取 PMID（ELink 延迟时的兜底）。"""
    try:
        root = ET.fromstring(xml_text)
        for aid in root.findall(".//article-meta/article-id[@pub-id-type='pmid']"):
            if aid.text:
                return aid.text.strip()
        return None
    except ET.ParseError:
        return None


# ── 全文 XML（JATS 格式） ─────────────────────────────────────

def fetch_fulltext_xml(pmcid: str) -> str | None:
    """
    EFetch db=pmc rettype=full retmode=xml 拉取 JATS 格式全文。
    返回 XML 字符串；若无法获取返回 None（不影响摘要流程）。
    """
    time.sleep(REQUEST_INTERVAL)
    pmc_numeric = pmcid.replace("PMC", "")
    params = {
        **_base_params(),
        "db":      "pmc",
        "id":      pmc_numeric,
        "rettype": "full",
        "retmode": "xml",
    }

    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=60)
        resp.raise_for_status()

        if "<article" not in resp.text and "<pmc-articleset" not in resp.text:
            logger.warning(f"全文内容异常 {pmcid}，跳过")
            return None

        return resp.text
    except requests.RequestException as e:
        logger.error(f"全文拉取失败 {pmcid}: {e}")
        return None


# ── 摘要 + 元数据（via PubMed）────────────────────────────────

def fetch_metadata(pmcid: str) -> dict | None:
    """
    先 ELink 拿 PMID，再 EFetch db=pubmed 拿摘要和元数据。
    如果 ELink 失败（没有 PubMed 记录），直接返回只含 pmcid 的最小 dict，
    后续全文 XML 里也能提取部分信息。
    """
    pmid = pmcid_to_pmid(pmcid)

    # 如果ELink转换id接口返回异常，则从全文中提取pmid
    if pmid is None:
        fulltext_path = FULLTEXT_DIR / f"{pmcid}.xml"
        if fulltext_path.exists():
            try:
                with open(fulltext_path, "r", encoding="utf-8") as f:
                    xml_text = f.read()
                pmid = _pmid_from_fulltext_xml(xml_text)
                if pmid is not None:
                    logger.info(f"全文提取 PMID 成功 {pmcid}: {pmid}")
            except Exception as e:
                logger.debug(f"读取全文XML / 全文提取 PMID 失败 {pmcid}: {e}")

        if pmid is None:
            logger.warning(f"{pmcid} 没有对应PMID，无法拉取完整元数据，将只保存全文。")
            return {"pmcid": pmcid, "pmid": None, "no_pubmed_record": True} # 区别于API调用失败的情况

    params = {
        **_base_params(),
        "db":      "pubmed",
        "id":      pmid,
        "rettype": "xml",
        "retmode": "xml",
    }
    time.sleep(REQUEST_INTERVAL)
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        return _parse_metadata_xml(resp.text, pmcid, pmid)
    except requests.RequestException as e:
        logger.error(f"PubMed EFetch 失败 {pmcid} (pmid={pmid}): {e}")
        return None


def _text(node, xpath: str) -> str:
    """安全地从 XML 节点提取文本"""
    el = node.find(xpath)
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_metadata_xml(xml_text: str, pmcid: str, pmid: str) -> dict:
    """解析 PubMed EFetch 返回的 XML，提取所有需要的元数据字段"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"PubMed XML 解析失败 {pmcid}: {e}")
        return {"pmcid": pmcid, "pmid": pmid, "parse_error": str(e)}

    article = root.find(".//PubmedArticle")
    if article is None:
        return {"pmcid": pmcid, "pmid": pmid, "parse_error": "No PubmedArticle node"}

    # ── 摘要（可能分多段，每段有 Label） ──
    abstract_parts = []
    for ab in article.findall(".//AbstractText"):
        label = ab.get("Label", "")
        content = "".join(ab.itertext()).strip()
        if not content:
            continue
        abstract_parts.append(f"{label}: {content}" if label else content)
    abstract = "\n".join(abstract_parts)

    # ── 作者 ──
    authors = []
    for author in article.findall(".//Author"):
        last = _text(author, "LastName")
        fore = _text(author, "ForeName")
        if last:
            authors.append(f"{fore} {last}".strip())

    # ── 发表日期（优先 PubDate，其次 ArticleDate） ──
    pub_year  = _text(article, ".//PubDate/Year")  or _text(article, ".//ArticleDate/Year")
    pub_month = _text(article, ".//PubDate/Month") or _text(article, ".//ArticleDate/Month")
    pub_day   = _text(article, ".//PubDate/Day")   or _text(article, ".//ArticleDate/Day")

    # ── DOI ──
    doi = ""
    for id_el in article.findall(".//ArticleId"):
        if id_el.get("IdType") == "doi":
            doi = (id_el.text or "").strip()
            break

    # ── 关键词 ──
    keywords = [kw.text.strip() for kw in article.findall(".//Keyword") if kw.text]

    # ── 文章类型 ──
    article_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]

    return {
        "pmcid":         pmcid,
        "pmid":          pmid,
        "doi":           doi,
        "title":         _text(article, ".//ArticleTitle"),
        "abstract":      abstract,
        "authors":       authors,
        "journal":       _text(article, ".//Journal/Title"),
        "pub_year":      pub_year,
        "pub_month":     pub_month,
        "pub_day":       pub_day,
        "keywords":      keywords,
        "article_types": article_types,
    }



# ── 文件读写 ──────────────────────────────────────────────────

def save_metadata(pmcid: str, data: dict):
    path: Path = METADATA_DIR / f"{pmcid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_fulltext(pmcid: str, xml_text: str):
    path: Path = FULLTEXT_DIR / f"{pmcid}.xml"
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_text)


def metadata_exists(pmcid: str) -> bool:
    return (METADATA_DIR / f"{pmcid}.json").exists()


def fulltext_exists(pmcid: str) -> bool:
    return (FULLTEXT_DIR / f"{pmcid}.xml").exists()


# ── 批量拉取主函数 ────────────────────────────────────────────

def fetch_all(pmcids: list[str], tracker: ProgressTracker):
    """
    遍历 pmcids，跳过已完成的，依次拉取摘要和全文并落盘。
    摘要落盘成功 → 尝试拉全文 → 标记完成。
    全文失败不影响整体流程（记录警告，仍标记该篇完成）。
    """
    total = len(pmcids)
    logger.info(f"开始批量拉取，共 {total} 篇 | 已完成 {tracker.get_fetched_count()} 篇")

    for i, pmcid in enumerate(pmcids, 1):
        # 如果已处理过/无需重试，跳过
        if tracker.is_fetched(pmcid) or tracker.is_no_pmid(pmcid):
            continue

        logger.info(f"[{i}/{total}] {pmcid}")


        # ── Step 1: 全文 XML ──
        if not fulltext_exists(pmcid):
            xml = fetch_fulltext_xml(pmcid)
            if xml is None:
                logger.warning(f"  全文不可获取，标记为失败(可重试)")
                tracker.mark_failed(pmcid)
                continue
            else:
                try:
                    save_fulltext(pmcid, xml)
                    logger.info(f"  ✓ 全文已保存")
                except OSError as e:
                    logger.error(f"  全文文件保存失败 {pmcid}: {e}")
                    tracker.mark_failed(pmcid)
                    continue
        else:
            logger.debug(f"  全文已存在，跳过")

        # ── Step 2: 摘要 + 元数据 ──
        if not metadata_exists(pmcid):
            meta = fetch_metadata(pmcid)
            # 如果元数据拉取失败，标记为失败(可重试)
            if meta is None:
                logger.warning(f"  元数据拉取失败，标记为失败(可重试)")
                tracker.mark_failed(pmcid)
                continue
            # 如果无对应 PMID，归类为 no_pmid（不重试）
            if meta.get("no_pubmed_record") or meta.get("pmid") is None:
                logger.warning(f"  无对应 PMID，归类为 no_pmid（不重试）")
                tracker.mark_no_pmid(pmcid)
                continue
            if meta.get("parse_error"):
                logger.warning(f"  {pmcid} 元数据解析异常，标记为失败(可重试)")
                tracker.mark_failed(pmcid)
                continue
            # 如果元数据拉取成功，保存到本地
            try:
                save_metadata(pmcid, meta)
                logger.info(f"  ✓ 元数据已保存")
            except OSError as e:
                logger.error(f"  元数据文件保存失败 {pmcid}: {e}")
                tracker.mark_failed(pmcid)
                continue
        else:
            logger.debug(f"  元数据已存在，跳过")

        # ── Step 3: 标记完成 ──
        tracker.mark_fetched(pmcid)

        if i % 50 == 0:
            logger.info(f"── 进度 ── {tracker.summary()}")

    logger.info(f"批量拉取结束 | {tracker.summary()}")