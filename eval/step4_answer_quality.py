"""
eval/step4_answer_quality.py

第四步：答案质量评估（LLM-as-judge）

指标：
  - Faithfulness（忠实度 0-2）：答案中的断言是否均有检索文献的直接支持
  - Citation Accuracy（引用准确率）：答案引用的 pmcid 有多少实际出现在检索结果中（程序计算）
  - Completeness（完整度 0-2）：答案是否覆盖了问题的所有方面

运行前提：Qdrant、OpenAI Embedding API、Qwen API 均可用。

运行：
  python eval/step4_answer_quality.py              # 全量（读 query_set.json）
  python eval/step4_answer_quality.py --limit 10   # 前10题（调试）
  python eval/step4_answer_quality.py --queries "brain fog" "fatigue mechanisms"  # 指定题目
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from infra.clients import get_openai_client
from agent.runner import run as agent_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval/output")
QUERY_SET  = OUTPUT_DIR / "query_set.json"


_FAITHFULNESS_PROMPT = """\
你是一名学术评审员，评估回答相对于原始文献片段的忠实度。

问题：{query}

回答：
{answer}

参考文献片段（评审时以此为唯一事实依据）：
{chunks}

忠实度标准：
  2 = 所有事实性断言均有文献原文直接支持，无无依据的推断或延伸
  1 = 大部分断言有依据，但存在 1-2 处超出文献的推断
  0 = 多处断言无文献依据，或与文献内容矛盾

只返回数字 0、1 或 2，不要其他内容。"""

_COMPLETENESS_PROMPT = """\
你是一名学术评审员，评估回答对问题的覆盖完整度。

问题：{query}

回答：
{answer}

完整度标准：
  2 = 全面覆盖了问题的所有方面，无明显遗漏
  1 = 覆盖了主要方面，但遗漏了部分子问题
  0 = 仅涉及问题的次要方面，或未实质性回答

只返回数字 0、1 或 2，不要其他内容。"""


def _llm_score(prompt: str) -> int:
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"[012]", raw)
        return int(m.group()) if m else 0
    except Exception as e:
        logger.warning("LLM judge 失败: %s", e)
        return 0


def _extract_cited_pmcids(answer: str) -> set[str]:
    """从答案文本中提取所有 PMC\d+ 格式的 pmcid。"""
    return set(re.findall(r"PMC\d+", answer, re.IGNORECASE))


def _citation_accuracy(answer: str, retrieved_chunks: list[dict]) -> float:
    """
    引用准确率 = 答案中出现的 pmcid 里有多少实际在检索结果中。
    返回 0.0-1.0，无引用时返回 None。
    """
    cited = _extract_cited_pmcids(answer)
    if not cited:
        return None
    retrieved_pmcids = set()
    for c in retrieved_chunks:
        p = c.get("payload", c)
        pmcid = p.get("pmcid", "")
        if pmcid:
            retrieved_pmcids.add(pmcid)
    matched = cited & retrieved_pmcids
    return len(matched) / len(cited)


def evaluate_one(query: str) -> dict:
    """运行一条 query，返回答案质量指标。"""
    logger.info("评估中：%s", query[:60])

    try:
        result = agent_run(user_input=query)
    except Exception as e:
        logger.error("agent_run 失败: %s", e)
        return {"query": query, "error": str(e)}

    answer = result.get("answer", "")
    chunks = result.get("retrieved_chunks", [])
    iterations = result.get("iterations", 0)

    if not answer:
        return {"query": query, "error": "agent 未返回答案", "iterations": iterations}

    chunks_text_parts = []
    for c in chunks[:10]:
        p = c.get("payload", c)
        chunks_text_parts.append(
            f"[{p.get('pmcid','')}] {p.get('text','')[:400]}"
        )
    chunks_preview = "\n\n".join(chunks_text_parts) or "（无检索结果）"

    faithfulness  = _llm_score(_FAITHFULNESS_PROMPT.format(
        query=query, answer=answer[:2000], chunks=chunks_preview
    ))
    completeness  = _llm_score(_COMPLETENESS_PROMPT.format(
        query=query, answer=answer[:2000]
    ))
    citation_acc  = _citation_accuracy(answer, chunks)

    return {
        "query":            query,
        "answer_preview":   answer[:300],
        "chunk_count":      len(chunks),
        "iterations":       iterations,
        "faithfulness":     faithfulness,
        "completeness":     completeness,
        "citation_accuracy": citation_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None,
                        help="只评估前 N 个 query（调试用）")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="直接指定要评估的问题字符串列表")
    args = parser.parse_args()

    if args.queries:
        queries = args.queries
    elif QUERY_SET.exists():
        with open(QUERY_SET, encoding="utf-8") as f:
            data = json.load(f)
        queries = [item["query"] for item in data]
        if args.limit:
            queries = queries[:args.limit]
    else:
        print(f"找不到 {QUERY_SET}，请先运行 step3a 或通过 --queries 指定问题")
        return

    results = []
    for q in queries:
        r = evaluate_one(q)
        results.append(r)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("所有 query 均评估失败，请检查服务连接")
        return

    avg_faith = sum(r["faithfulness"]  for r in valid) / len(valid)
    avg_comp  = sum(r["completeness"]  for r in valid) / len(valid)
    cit_scores = [r["citation_accuracy"] for r in valid if r["citation_accuracy"] is not None]
    avg_cit   = sum(cit_scores) / len(cit_scores) if cit_scores else None

    print(f"\n{'='*60}")
    print("答案质量评估报告")
    print(f"{'='*60}")
    print(f"有效评估数：{len(valid)} / {len(queries)}")
    print(f"  Faithfulness    (0-2)：{avg_faith:.2f}")
    print(f"  Completeness    (0-2)：{avg_comp:.2f}")
    if avg_cit is not None:
        print(f"  Citation Accuracy    ：{avg_cit:.2%}")
    else:
        print("  Citation Accuracy    ：答案中未检测到 pmcid 引用")

    print("\n── 各题详情 ──")
    for r in results:
        if "error" in r:
            print(f"  ✗ {r['query'][:50]} → 错误: {r['error']}")
        else:
            cit = f"{r['citation_accuracy']:.0%}" if r["citation_accuracy"] is not None else "N/A"
            print(
                f"  F={r['faithfulness']} C={r['completeness']} Cit={cit}"
                f"  chunks={r['chunk_count']} iter={r['iterations']}"
                f"  | {r['query'][:50]}"
            )

    print("\n── 建议 ──")
    if avg_faith < 1.5:
        print("  ⚠ 忠实度偏低：考虑开启 GROUNDING_CHECK=true 或进一步强化 QA 提示词")
    else:
        print("  ✓ 忠实度良好")
    if avg_comp < 1.5:
        print("  ⚠ 完整度偏低：考虑增大 MAX_ITERATIONS 或优化编排器重搜策略")
    else:
        print("  ✓ 完整度良好")
    if avg_cit is not None and avg_cit < 0.8:
        print("  ⚠ 引用准确率偏低：LLM 可能在引用不存在的 pmcid，检查 answer_question 提示词")
    elif avg_cit is not None:
        print("  ✓ 引用准确率良好")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "answer_quality_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "query_count": len(queries),
            "valid_count": len(valid),
            "averages": {
                "faithfulness":     round(avg_faith, 4),
                "completeness":     round(avg_comp,  4),
                "citation_accuracy": round(avg_cit, 4) if avg_cit is not None else None,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到 {out}")


if __name__ == "__main__":
    main()
