"""
data_pipeline/storage/progress.py
断点续传进度管理 - 记录已完成、失败的 PMCID 和没有 PMID 的 PMCID，支持任意位置中断后继续
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    维护 progress.json，结构如下：
    {
        "total_pmcids": 4943,
        "fetched_pmcids": ["PMC123", ...],
        "failed_pmcids":  ["PMC456", ...],
        "no_pmid_pmcids": ["PMC789", ...],
        "last_updated": "2026-02-24T10:00:00"
    }
    """

    def __init__(self, progress_file: Path):
        self.path = progress_file
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(
                    f"进度文件已加载：已完成 {len(data.get('fetched_pmcids', []))} 篇，"
                    f"失败 {len(data.get('failed_pmcids', []))} 篇，"
                    f"没有 PMID {len(data.get('no_pmid_pmcids', []))} 篇"
                )
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"进度文件损坏，重新开始：{e}")
        return {
            "total_pmcids": 0,
            "fetched_pmcids": [],
            "failed_pmcids": [],
            "no_pmid_pmcids": [],
            "last_updated": None,
        }

    def _save(self):
        self._data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    # ── 对外接口 ──────────────────────────────────────────────

    def set_total(self, total: int):
        self._data["total_pmcids"] = total
        self._save()

    def mark_fetched(self, pmcid: str):
        if pmcid not in self._data["fetched_pmcids"]:
            self._data["fetched_pmcids"].append(pmcid)
        # 如果之前失败过\没有 PMID，从失败列表移除
        if pmcid in self._data["failed_pmcids"]:
            self._data["failed_pmcids"].remove(pmcid)
        if pmcid in self._data.get("no_pmid_pmcids", []):
            self._data["no_pmid_pmcids"].remove(pmcid)
        self._save()

    def mark_failed(self, pmcid: str):
        if pmcid not in self._data["failed_pmcids"]:
            self._data["failed_pmcids"].append(pmcid)
        self._save()

    def mark_no_pmid(self, pmcid: str):
        if pmcid not in self._data.get("no_pmid_pmcids", []):
            self._data.setdefault("no_pmid_pmcids", []).append(pmcid)
        if pmcid in self._data.get("failed_pmcids", []):
            self._data["failed_pmcids"].remove(pmcid)
        self._save()

    def is_fetched(self, pmcid: str) -> bool:
        return pmcid in self._data["fetched_pmcids"]

    def get_failed(self) -> list[str]:
        return list(self._data["failed_pmcids"])
        
    def is_no_pmid(self, pmcid: str) -> bool:
        return pmcid in self._data.get("no_pmid_pmcids", [])

    def get_fetched_count(self) -> int:
        return len(self._data["fetched_pmcids"])

    def get_total(self) -> int:
        return self._data["total_pmcids"]

    def summary(self) -> str:
        total = self._data["total_pmcids"]
        fetched = len(self._data["fetched_pmcids"])
        failed = len(self._data["failed_pmcids"])
        no_pmid = len(self._data.get("no_pmid_pmcids", []))
        remaining = total - fetched
        return (
            f"总计 {total} | 已完成 {fetched} | 失败 {failed} | 没有 PMID {no_pmid} | 剩余 {remaining}"
        )
