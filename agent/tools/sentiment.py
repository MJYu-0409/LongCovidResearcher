"""
agent/tools/sentiment.py

工具三：analyze_sentiment
对指定 pmcid 列表实时调用情感分析 API，汇总返回结果。
不依赖预存数据，按需分析。

你的情感分析 API 接口：替换 SENTIMENT_API_URL 即可。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 替换为你的情感分析 API 地址 ──────────────────────────────────────────
SENTIMENT_API_URL = "http://localhost:8000/sentiment"   # TODO: 替换为实际地址
SENTIMENT_API_TIMEOUT = 30  # 秒
# ─────────────────────────────────────────────────────────────────────────


def _call_sentiment_api(pmcid: str, text: str) -> dict:
    """
    调用情感分析 API，返回单篇文章的情感结果。

    预期 API 接受：{"pmcid": str, "text": str}
    预期 API 返回：{"pmcid": str, "label": str, "score": float, ...}

    如果你的 API 格式不同，在这里调整即可。
    """
    try:
        resp = requests.post(
            SENTIMENT_API_URL,
            json={"pmcid": pmcid, "text": text},
            timeout=SENTIMENT_API_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("情感分析 API 调用失败 pmcid=%s: %s", pmcid, e)
        return {"pmcid": pmcid, "error": str(e)}


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

    # 逐篇调用情感 API
    results = []
    for item in pairs:
        pmcid = item.get("pmcid", "")
        text  = item.get("text", "")
        if not text:
            continue
        result = _call_sentiment_api(pmcid, text)
        results.append(result)

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
