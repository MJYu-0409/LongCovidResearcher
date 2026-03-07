"""
eval/step3b_evaluate.py

第三步 Part B：LLM 打分 + 用 ranx 计算检索指标

依赖：
    pip install ranx

流程：
  1. 读取 step3a 生成的 query_set.json
  2. 四路检索（dense / sparse / hybrid / hybrid+rerank）
  3. GPT-4o-mini 对所有 (query, chunk) 对打相关性分 0/1/2
     已知的 source_pmcid 直接标注 relevance=2，跳过 LLM 调用
  4. 用 ranx 计算 NDCG@5 / Recall@5 / Precision@5 / MRR
  5. 输出四路对比报告和改进建议

运行：
  python eval/step3b_evaluate.py              # 完整模式（调 GPT-4o-mini）
  python eval/step3b_evaluate.py --fast       # 快速模式（只用已知标注）
  python eval/step3b_evaluate.py --limit 20   # 只跑前20条（调试）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 保证从项目根可找到 config（支持 python eval/step3b_... 或 -m eval.step3b_...）
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from openai import OpenAI
from ranx import Qrels, Run, evaluate, compare

from config import OPENAI_API_KEY
from retrieval.dense  import dense_search
from retrieval.sparse import sparse_search
from retrieval.hybrid import hybrid_search
from retrieval.search import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval/output")
QUERY_SET  = OUTPUT_DIR / "query_set.json"

TOP_K = 20  # hybrid+rerank 初始召回数
TOP_N = 5   # 最终评估的 top-N

METRICS = ["ndcg@5", "recall@5", "precision@5", "mrr"]

RELEVANCE_PROMPT = """\
你是 Long COVID 领域专家。判断以下检索结果对给定问题的相关性。

问题：{query}

检索文本：
{text}

评分标准：
  2 = 高度相关，直接回答问题
  1 = 部分相关，涉及相关话题但不直接回答
  0 = 不相关

只返回数字 0、1 或 2，不要任何其他内容。"""


# ── LLM 打分 ─────────────────────────────────────────────────────────────

def score_hits(
    client: OpenAI,
    query: str,
    hits: list[dict],
    known: dict[str, int],
    fast: bool,
) -> dict[str, int]:
    """
    返回 {pmcid: relevance(0/1/2)}。
    fast=True 时未知 pmcid 全部给 0，不调 LLM。
    """
    rel_map = dict(known)

    if fast:
        for r in hits:
            rel_map.setdefault(r["payload"].get("pmcid", ""), 0)
        return rel_map

    for r in hits:
        pmcid = r["payload"].get("pmcid", "")
        if pmcid in rel_map:
            continue
        text = r["payload"].get("text", "")[:600]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[{"role": "user",
                           "content": RELEVANCE_PROMPT.format(
                               query=query, text=text)}],
            )
            raw = resp.choices[0].message.content.strip()
            rel_map[pmcid] = int(raw) if raw in ("0", "1", "2") else 0
        except Exception as e:
            logger.warning("打分失败 pmcid=%s: %s", pmcid, e)
            rel_map[pmcid] = 0

    return rel_map


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true",
                        help="快速模式：只用已知标注，不调 LLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 个 query（调试用）")
    args = parser.parse_args()

    if not QUERY_SET.exists():
        print(f"找不到 {QUERY_SET}，请先运行 step3a_generate_queries.py")
        return

    with open(QUERY_SET, encoding="utf-8") as f:
        queries = json.load(f)
    if args.limit:
        queries = queries[:args.limit]

    gpt4 = OpenAI(api_key=OPENAI_API_KEY)

    # ranx 需要的数据结构：
    #   qrels_dict[query_id][pmcid] = relevance_int
    #   run_dict[strategy][query_id][pmcid] = score_float
    qrels_dict: dict[str, dict[str, int]] = {}
    runs_dict: dict[str, dict[str, dict[str, float]]] = {
        "A_dense":         {},
        "B_sparse":        {},
        "C_hybrid":        {},
        "D_hybrid_rerank": {},
    }

    for idx, item in enumerate(queries, 1):
        query    = item["query"]
        qid      = f"q{idx:04d}"   # ranx 要求字符串 ID
        known    = {r["pmcid"]: r["relevance"]
                    for r in item.get("relevant_pmcids", [])}

        logger.info("[%d/%d] %s", idx, len(queries), query[:60])

        try:
            a = dense_search(query,  top_k=TOP_N)
            b = sparse_search(query, top_k=TOP_N)
            c = hybrid_search(query, top_k=TOP_N)
            d = search(query, top_k=TOP_K, top_n=TOP_N)
        except Exception as e:
            logger.warning("检索失败，跳过: %s", e)
            continue

        # 合并去重，批量打分（每个 pmcid 只打一次）
        all_hits = {
            r["payload"].get("pmcid", ""): r
            for bucket in [a, b, c, d]
            for r in bucket
        }
        rel_map = score_hits(gpt4, query, list(all_hits.values()),
                             known, args.fast)

        # 写入 qrels（只保留 relevance > 0 的，ranx 约定）
        qrels_dict[qid] = {
            pmcid: rel for pmcid, rel in rel_map.items() if rel > 0
        }
        # 如果这个 query 没有任何相关文档（全0），跳过——
        # 无相关文档时 NDCG/Recall 无意义
        if not qrels_dict[qid]:
            del qrels_dict[qid]
            logger.debug("query %s 无相关文档，跳过", qid)
            continue

        # 写入各策略的 run（pmcid → score）
        def to_run(results: list[dict]) -> dict[str, float]:
            """把检索结果转为 ranx Run 格式：{pmcid: score}。"""
            out = {}
            for r in results:
                pmcid = r["payload"].get("pmcid", "")
                # 优先用 rerank_score，其次 rrf_score，最后 score
                score = r.get("rerank_score",
                               r.get("rrf_score",
                                     r.get("score", 0.0)))
                out[pmcid] = float(score)
            return out

        runs_dict["A_dense"][qid]         = to_run(a)
        runs_dict["B_sparse"][qid]        = to_run(b)
        runs_dict["C_hybrid"][qid]        = to_run(c)
        runs_dict["D_hybrid_rerank"][qid] = to_run(d)

    if not qrels_dict:
        print("没有有效的评估数据，请检查 query_set.json 和检索连接")
        return

    # ── 构建 ranx 对象 ────────────────────────────────────────────────
    qrels = Qrels(qrels_dict)
    runs = {
        name: Run(run_data, name=name)
        for name, run_data in runs_dict.items()
        if run_data  # 跳过空的
    }

    # ── 用 ranx 计算指标 ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("检索评估报告")
    print(f"{'='*68}")
    mode_tag = "快速模式（已知标注）" if args.fast else "GPT-4o-mini 打分"
    print(f"有效 query 数：{len(qrels_dict)}  模式：{mode_tag}\n")

    results_table = {}
    for name, run in runs.items():
        scores = evaluate(qrels, run, METRICS)
        results_table[name] = scores
        print(f"  {name}")
        for metric, val in scores.items():
            print(f"    {metric:15s}: {val:.4f}")
        print()

    # compare() 生成多策略对比表（含统计显著性检验）
    if len(runs) > 1:
        print("── 多策略对比（ranx compare）──")
        report = compare(
            qrels,
            runs=list(runs.values()),
            metrics=METRICS,
            # stat_test="student"  # 可选：开启统计显著性检验
        )
        print(report)

    # ── 改进建议 ──────────────────────────────────────────────────────
    print("── 建议 ──")
    full = results_table.get("D_hybrid_rerank", {})
    dns  = results_table.get("A_dense", {})
    hyb  = results_table.get("C_hybrid", {})

    ndcg_gain_hybrid  = full.get("ndcg@5", 0) - dns.get("ndcg@5", 0)
    ndcg_gain_rerank  = full.get("ndcg@5", 0) - hyb.get("ndcg@5", 0)

    if ndcg_gain_hybrid > 0.05:
        print("  ✓ Hybrid+Rerank 明显优于 Dense only（NDCG 提升 "
              f"{ndcg_gain_hybrid:+.3f}），混合策略有效")
    else:
        print(f"  ⚠ Hybrid+Rerank vs Dense only NDCG 提升 {ndcg_gain_hybrid:+.3f}，"
              "不明显，检查 sparse 向量质量")

    if ndcg_gain_rerank > 0.03:
        print(f"  ✓ Reranking 有效提升排序（NDCG +{ndcg_gain_rerank:.3f}）")
    else:
        print(f"  ⚠ Reranking 提升有限（NDCG +{ndcg_gain_rerank:.3f}），"
              "可考虑替换更强的 reranker 模型")

    if full.get("recall@5", 0) < 0.4:
        print("  ⚠ Recall@5 < 0.4，建议增大 top_k 或检查 embedding 质量")

    # ── 保存 ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "eval_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "mode":         "fast" if args.fast else "llm",
            "query_count":  len(qrels_dict),
            "results":      results_table,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到 {out}")


if __name__ == "__main__":
    main()
