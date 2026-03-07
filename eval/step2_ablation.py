"""
eval/step2_ablation.py

第二步：消融实验（无需标注，人眼对比）

并排对比四种检索策略：
  A. Dense only    — 语义检索
  B. Sparse only   — 关键词检索
  C. Hybrid        — RRF 融合（无 rerank）
  D. Hybrid+Rerank — 完整流程

同时统计两路互补性（Jaccard 重叠率），
重叠率低说明 hybrid 有价值，重叠率高说明两路冗余。

运行：
  python eval/step2_ablation.py                          # 跑全部预设 query
  python eval/step2_ablation.py --query "your question"  # 指定单个 query
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 保证从项目根可找到 config / retrieval（支持 python eval/step2_... 或 -m eval.step2_...）
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.WARNING)

from retrieval.dense  import dense_search
from retrieval.sparse import sparse_search
from retrieval.hybrid import hybrid_search
from retrieval.search import search

OUTPUT_DIR = Path("eval/output")

ABLATION_QUERIES = [
    "autonomic dysfunction treatment long covid",
    "post-exertional malaise mechanism pathophysiology",
    "long covid prevalence omicron variant 2022 2023",
    "brain fog cognitive impairment intervention",
    "mast cell activation long covid symptoms",
    "antivirals paxlovid long covid clinical trial results",
    "gut microbiome dysbiosis long covid",
    "long covid systematic review meta-analysis outcomes",
]

TOP_N = 5   # 每路展示 top-N
TOP_K = 20  # hybrid+rerank 先召回 top-K 再精选


def _fmt(rank: int, hit: dict) -> str:
    p     = hit.get("payload", {})
    score = hit.get("rerank_score", hit.get("rrf_score", hit.get("score", 0)))
    text  = p.get("text", "")[:90].replace("\n", " ")
    return (
        f"  [{rank}] {score:.4f} | {p.get('source_type','?'):8s} | "
        f"{p.get('pub_year','?')} | pmcid={p.get('pmcid','?')}\n"
        f"       sec: {p.get('section','?')[:48]}\n"
        f"       {text}…"
    )


def _pmcids(results: list[dict]) -> set[str]:
    return {r["payload"].get("pmcid", "") for r in results}


def ablate(query: str) -> dict:
    print(f"\n{'='*68}")
    print(f"Query: {query}")
    print(f"{'='*68}")

    try:
        a = dense_search(query,  top_k=TOP_N)
        b = sparse_search(query, top_k=TOP_N)
        c = hybrid_search(query, top_k=TOP_N)
        d = search(query, top_k=TOP_K, top_n=TOP_N)
    except Exception as e:
        print(f"  ✗ 检索失败: {e}")
        return {}

    for label, results in [
        ("A. Dense only", a), ("B. Sparse only", b),
        ("C. Hybrid RRF", c), ("D. Hybrid + Rerank", d),
    ]:
        print(f"\n── {label} ──")
        for i, hit in enumerate(results, 1):
            print(_fmt(i, hit))

    # 互补性
    sa, sb = _pmcids(a), _pmcids(b)
    union  = sa | sb
    inter  = sa & sb
    jaccard = len(inter) / len(union) if union else 0

    rerank_new  = _pmcids(d) - _pmcids(c)
    rerank_drop = _pmcids(c) - _pmcids(d)

    print(f"\n── 互补性 ──")
    print(f"  Dense 独有: {len(sa-sb)}  Sparse 独有: {len(sb-sa)}  "
          f"两路共有: {len(inter)}  Jaccard: {jaccard:.1%}")
    if rerank_new:
        print(f"  Rerank 新晋 top-{TOP_N}: {rerank_new}")
    if rerank_drop:
        print(f"  Rerank 降出 top-{TOP_N}: {rerank_drop}")

    return {
        "query": query,
        "jaccard_dense_sparse": jaccard,
        "rerank_new_count": len(rerank_new),
        "strategies": {
            "A_dense":         [r["payload"].get("pmcid") for r in a],
            "B_sparse":        [r["payload"].get("pmcid") for r in b],
            "C_hybrid":        [r["payload"].get("pmcid") for r in c],
            "D_hybrid_rerank": [r["payload"].get("pmcid") for r in d],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    queries = [args.query] if args.query else ABLATION_QUERIES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [ablate(q) for q in queries]
    results = [r for r in results if r]

    if len(results) > 1:
        avg_j = sum(r["jaccard_dense_sparse"] for r in results) / len(results)
        avg_new = sum(r["rerank_new_count"] for r in results) / len(results)
        print(f"\n{'='*68}")
        print(f"汇总  |  Dense/Sparse 平均 Jaccard: {avg_j:.1%}  |  "
              f"Rerank 平均新晋: {avg_new:.1f} 条")
        if avg_j < 0.3:
            print("  ✓ 两路互补性好，hybrid 融合有价值")
        elif avg_j > 0.7:
            print("  ⚠ 两路高度重叠，hybrid 收益有限，可考虑只用 dense")

    out = OUTPUT_DIR / "ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
