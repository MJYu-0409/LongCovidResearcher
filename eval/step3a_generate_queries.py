"""
eval/step3a_generate_queries.py

第三步 Part A：从语料自动生成评估 query 集

从 Qdrant 随机采样论文摘要，用 GPT-4o 为每篇生成 3 个自然语言问题，
并记录"来源论文"作为 known relevant（relevance=2）。

这是 LLM 自动标注法的第一步，生成的 query_set.json 供 step3b 使用。

运行：
  python eval/step3a_generate_queries.py --sample 50
  python eval/step3a_generate_queries.py --sample 100  # 更完整的评估集
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# 保证从项目根可找到 config（支持 python eval/step3a_... 或 -m eval.step3a_...）
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval/output")

QUERY_GEN_PROMPT = """\
你是 Long COVID 领域的研究助理。给定以下论文摘要，生成 3 个研究人员可能会提问的自然语言问题。

要求：
- 问题自然，像真实用户提问，不要照抄摘要原文
- 覆盖不同意图（机制 / 症状 / 治疗 / 流行病学 / 综述）
- 难度适中：不能太宽泛（"什么是 long covid"），也不能太具体（只有这篇能答）
- 用英文提问

摘要：
{abstract}

只返回 JSON，格式如下，不要其他内容：
{{"questions":[
  {{"query":"问题1","intent":"机制"}},
  {{"query":"问题2","intent":"治疗"}},
  {{"query":"问题3","intent":"流行病学"}}
]}}"""


def sample_abstracts(n: int) -> list[dict]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    logger.info("从 Qdrant 拉取 abstract 列表…")

    pool, offset = [], None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="source_type", match=MatchValue(value="abstract"))
            ]),
            limit=500, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        pool.extend(points)
        if offset is None:
            break

    logger.info("共 %d 篇 abstract，随机采样 %d 篇", len(pool), min(n, len(pool)))
    sampled = random.sample(pool, min(n, len(pool)))
    return [
        {
            "pmcid":    p.payload.get("pmcid", ""),
            "abstract": p.payload.get("text",  ""),
            "pub_year": p.payload.get("pub_year", ""),
            "journal":  p.payload.get("journal",  ""),
        }
        for p in sampled
    ]


def gen_questions(client: OpenAI, paper: dict) -> list[dict]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[{"role": "user",
                       "content": QUERY_GEN_PROMPT.format(
                           abstract=paper["abstract"][:1500])}],
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw).get("questions", [])
    except Exception as e:
        logger.warning("pmcid=%s 生成失败: %s", paper["pmcid"], e)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = sample_abstracts(args.sample)
    gpt4   = OpenAI(api_key=OPENAI_API_KEY)
    rows   = []

    for i, paper in enumerate(papers, 1):
        logger.info("[%d/%d] pmcid=%s", i, len(papers), paper["pmcid"])
        questions = gen_questions(gpt4, paper)
        for q in questions:
            rows.append({
                "query":           q.get("query", ""),
                "intent":          q.get("intent", ""),
                # 来源论文作为已知相关文档（relevance=2 高度相关）
                "relevant_pmcids": [{"pmcid": paper["pmcid"], "relevance": 2}],
                "source_pmcid":    paper["pmcid"],
                "source_abstract": paper["abstract"][:300],
            })

    out = OUTPUT_DIR / "query_set.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("完成：%d 篇 → %d 个 query，已保存到 %s", len(papers), len(rows), out)
    print("\n示例（前3条）：")
    for r in rows[:3]:
        print(f"  [{r['intent']}] {r['query']}")
        print(f"   来源 pmcid: {r['source_pmcid']}")


if __name__ == "__main__":
    main()
