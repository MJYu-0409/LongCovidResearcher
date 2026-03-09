"""
agent/tools/sentiment.py

工具三：analyze_sentiment
对指定 pmcid 列表实时调用情感分析 API，汇总返回结果。
不依赖预存数据，按需分析。

情感分析 API：单条摘要用 /predict，多条用 /predict/batch（见 SENTIMENT_API_SINGLE / SENTIMENT_API_BATCH）。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests
from langchain_core.tools import tool

from config import SENTIMENT_API_BATCH, SENTIMENT_API_SINGLE, SENTIMENT_API_TIMEOUT

logger = logging.getLogger(__name__)


def _call_sentiment_single(pmcid: str, text: str) -> dict:
    """调用单条情感分析 API。预期返回：{"pmcid": str, "label": str, "score": float, ...}"""
    try:
        resp = requests.post(
            SENTIMENT_API_SINGLE,
            json={"pmcid": pmcid, "text": text},
            timeout=SENTIMENT_API_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("情感分析 API 调用失败 pmcid=%s: %s", pmcid, e)
        return {"pmcid": pmcid, "error": str(e)}


def _call_sentiment_batch(pairs: list[dict]) -> list[dict]:
    """调用批量情感分析 API。预期请求体为列表，返回为结果列表（顺序与请求一致）。"""
    if not pairs:
        return []
    try:
        resp = requests.post(
            SENTIMENT_API_BATCH,
            json=pairs,
            timeout=SENTIMENT_API_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, list):
            return [{"pmcid": p.get("pmcid", ""), "error": "API 返回格式非列表"} for p in pairs]
        # 若 API 返回数量与请求不一致，按索引对齐或补 error
        out = []
        for i, p in enumerate(pairs):
            pmcid = p.get("pmcid", "")
            if i < len(results):
                r = results[i] if isinstance(results[i], dict) else {"pmcid": pmcid, "raw": results[i]}
                r.setdefault("pmcid", pmcid)
                out.append(r)
            else:
                out.append({"pmcid": pmcid, "error": "API 未返回该条"})
        return out
    except requests.RequestException as e:
        logger.warning("批量情感分析 API 调用失败: %s", e)
        return [{"pmcid": p.get("pmcid", ""), "error": str(e)} for p in pairs]


@tool
def analyze_sentiment(
    pmcid_text_pairs: str,
    topic: Optional[str] = None,
) -> str:
    """
    对指定论文进行情感分析，判断学界对某治疗方案或研究议题的态度（正面/负面/中性）。
    适用于：了解学界对某疗法的总体评价、跟踪某议题的情感趋势。

    Args:
        pmcid_text_pairs: JSON 字符串，格式为论文列表：
                          '[{"pmcid": "PMC...", "text": "摘要或关键段落"}]'
                          通常从 search_literature 的结果中提取
        topic:            可选，分析的主题（用于结果汇总标注）

    Returns:
        各论文的情感标签和置信度，以及整体汇总（JSON字符串）
    """
    try:
        pairs = json.loads(pmcid_text_pairs)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"输入格式错误，需要 JSON 列表: {e}"})

    if not pairs:
        return json.dumps({"error": "输入为空"})

    # 过滤掉无 text 的项，保留 pmcid 占位
    valid_pairs = [{"pmcid": item.get("pmcid", ""), "text": item.get("text", "") or ""} for item in pairs]
    valid_pairs = [p for p in valid_pairs if p["text"]]
    if not valid_pairs:
        return json.dumps({"error": "没有有效摘要文本可分析"})

    # 单条走 /predict，多条走 /predict/batch
    if len(valid_pairs) == 1:
        results = [_call_sentiment_single(valid_pairs[0]["pmcid"], valid_pairs[0]["text"])]
    else:
        results = _call_sentiment_batch(valid_pairs)

    # 汇总统计（排除调用失败的）
    valid = [r for r in results if "error" not in r]
    summary: dict = {"topic": topic or "未指定", "total": len(results), "valid": len(valid)}

    if valid:
        from collections import Counter
        label_counts = Counter(r.get("label", "unknown") for r in valid)
        summary["distribution"] = dict(label_counts)
        summary["dominant"] = label_counts.most_common(1)[0][0] if label_counts else "unknown"

    logger.info("analyze_sentiment: %d 篇 → %d 成功", len(pairs), len(valid))

    return json.dumps({
        "summary": summary,
        "details": results,
    }, ensure_ascii=False, indent=2)
