"""
eval/step1_health_check.py

第一步：健康检查（无需标注，5-10 分钟运行完）

检查检索系统是否存在系统性偏差：
  - source_type 分布是否均衡（abstract / fulltext 都被召回）
  - pub_year 分布是否合理（近年论文应占主体）
  - 结果多样性（是否少数文章垄断所有 query 的结果）
  - 是否有 query 返回空结果

运行：python eval/step1_health_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

# 保证从项目根可找到 config / retrieval（支持 python eval/step1_... 或 -m eval.step1_...）
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.WARNING)

from retrieval.search import search

OUTPUT_DIR = Path("eval/output")

# 覆盖不同意图类型的代表性 query
QUERIES = {
    "机制": [
        "autonomic dysfunction mechanism in long covid",
        "SARS-CoV-2 spike protein persistence chronic symptoms",
        "microbiome dysbiosis long covid pathophysiology",
    ],
    "症状": [
        "post-exertional malaise fatigue long covid",
        "brain fog cognitive impairment long covid",
        "long covid cardiovascular symptoms palpitations",
    ],
    "治疗": [
        "antivirals treatment long covid clinical trial",
        "low dose naltrexone long covid",
        "rehabilitation exercise therapy post-covid",
    ],
    "流行病学": [
        "long covid prevalence omicron variant",
        "long covid risk factors age sex comorbidities",
        "long covid incidence vaccination effect",
    ],
    "综述": [
        "long covid systematic review meta-analysis",
        "post-acute sequelae SARS-CoV-2 overview",
    ],
}


def run():
    print("=" * 68)
    print("Long COVID 检索系统健康检查")
    print("=" * 68)

    all_results: list[dict] = []
    failed_queries: list[str] = []

    for intent, queries in QUERIES.items():
        print(f"\n── 意图：{intent} ──")
        for query in queries:
            print(f"\n  Query: {query}")
            try:
                results = search(query, top_k=20, top_n=5)
            except Exception as e:
                print(f"    ✗ 检索失败: {e}")
                failed_queries.append(query)
                continue

            if not results:
                print("    ⚠ 返回空结果")
                failed_queries.append(query)
                continue

            all_results.extend(results)

            for i, r in enumerate(results, 1):
                p = r["payload"]
                score = r.get("rerank_score", r.get("rrf_score", 0))
                text  = p.get("text", "")[:80].replace("\n", " ")
                print(f"    [{i}] {score:.3f} | {p.get('source_type','?'):8s} | "
                      f"{p.get('pub_year','?')} | {p.get('section','?')[:28]:28s}")
                print(f"         {text}…")

    # ── 全局统计 ───────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("全局统计")
    print(f"{'='*68}")

    if not all_results:
        print("无结果，请检查 Qdrant 连接和数据是否写入")
        return

    source_cnt = Counter(r["payload"].get("source_type", "?") for r in all_results)
    year_cnt   = Counter(r["payload"].get("pub_year",    "?") for r in all_results)
    pmcids     = [r["payload"].get("pmcid", "") for r in all_results]
    unique_r   = len(set(pmcids)) / len(pmcids)

    print(f"\nsource_type 分布（abstract 和 fulltext 均应出现）：")
    for k, v in source_cnt.most_common():
        bar = "█" * v
        print(f"  {k:12s}: {v:3d}  {bar}")

    print(f"\npub_year 分布（近年应占主体）：")
    for k, v in sorted(year_cnt.items(), reverse=True)[:8]:
        bar = "█" * v
        print(f"  {k}: {v:3d}  {bar}")

    print(f"\n结果多样性：{len(set(pmcids))}/{len(pmcids)} 唯一 pmcid（{unique_r:.1%}）")

    # ── 异常预警 ───────────────────────────────────────────────────────
    print(f"\n── 异常检测 ──")
    warnings: list[str] = []

    if "abstract" not in source_cnt:
        warnings.append("abstract 结果从未出现，abstract 向量可能未写入 Qdrant")
    if "fulltext" not in source_cnt:
        warnings.append("fulltext 结果从未出现，fulltext 向量可能未写入 Qdrant")
    if unique_r < 0.5:
        warnings.append(f"结果重复率高（唯一率 {unique_r:.1%}），少数文章垄断检索结果")

    recent = sum(v for k, v in year_cnt.items() if k.isdigit() and int(k) >= 2022)
    if recent / len(all_results) < 0.3:
        warnings.append(f"近三年论文占比低（{recent/len(all_results):.1%}），存在时间偏差")
    if failed_queries:
        warnings.append(f"{len(failed_queries)} 个 query 失败或返回空结果")

    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  ✓ 未发现明显异常")

    # ── 保存原始结果 ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "health_check.json"
    with open(out, "w", encoding="utf-8") as f:
        # payload 里的内容已是可序列化的 dict，直接存
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_q = sum(len(v) for v in QUERIES.values())
    print(f"\n健康检查完成：{total_q} 个 query，{len(all_results)} 条结果，已保存到 {out}")


if __name__ == "__main__":
    run()
