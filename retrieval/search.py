"""
retrieval/search.py

对外统一入口：Agent 工具调用这一个函数即可，
内部封装了 hybrid_search → rerank 的完整流程。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import QUERY_DECOMPOSE_MAX_SUBQUERIES, QUERY_OPT_MODE, QDRANT_COLLECTION_PC
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.query_optimizer import (
    build_hyde_query,
    classify_query_complexity,
    decompose_query,
    translate_query_to_english,
)

logger = logging.getLogger(__name__)
RRF_K = 60


def _expand_neighbors(hits: list[dict]) -> list[dict]:
    """
    为 rerank 后的每条 hit 查 Qdrant ±1 邻居段落（同 pmcid + section），
    拼成 context_text 写入 payload，供 LLM 阅读。
    首尾段落自动退化为单侧窗口。
    """
    from infra.clients import get_qdrant_client
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = get_qdrant_client()

    for hit in hits:
        p = hit["payload"]
        pmcid   = p.get("pmcid", "")
        section = p.get("section", "")
        idx     = p.get("chunk_index")
        if idx is None:
            p["context_text"] = p.get("text", "")
            continue

        neighbor_texts: dict[int, str] = {}
        for offset in (-1, 1):
            try:
                scroll_result, _ = client.scroll(
                    collection_name=QDRANT_COLLECTION_PC,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="pmcid",       match=MatchValue(value=pmcid)),
                        FieldCondition(key="section",     match=MatchValue(value=section)),
                        FieldCondition(key="chunk_index", match=MatchValue(value=idx + offset)),
                    ]),
                    with_payload=True,
                    limit=1,
                )
                if scroll_result:
                    t = scroll_result[0].payload.get("text", "")
                    if t:
                        neighbor_texts[offset] = t
            except Exception as e:
                logger.debug("邻居查询失败 pmcid=%s idx=%d offset=%d: %s", pmcid, idx, offset, e)

        parts = []
        if -1 in neighbor_texts:
            parts.append(neighbor_texts[-1])
        parts.append(p.get("text", ""))
        if 1 in neighbor_texts:
            parts.append(neighbor_texts[1])
        p["context_text"] = "\n\n".join(parts)

    return hits


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _fuse_results_with_rrf(
    result_lists: list[list[dict]], top_k: int, max_per_paper: int = 5
) -> list[dict]:
    """
    将多路检索结果做 RRF 融合（按 id 去重，累加 rrf_score）。
    选取时每篇论文最多保留 max_per_paper 条，保证候选池来源多样。
    """
    all_hits: dict[str, dict] = {}
    for bucket in result_lists:
        for rank, hit in enumerate(bucket, start=1):
            hit_id = hit["id"]
            all_hits.setdefault(
                hit_id,
                {"id": hit_id, "payload": hit.get("payload", {}), "rrf_score": 0.0},
            )
            all_hits[hit_id]["rrf_score"] += _rrf_score(rank)

    ranked = sorted(all_hits.values(), key=lambda x: x["rrf_score"], reverse=True)
    results: list[dict] = []
    paper_counts: dict[str, int] = {}
    for hit in ranked:
        pmcid = hit["payload"].get("pmcid", hit["id"])
        if paper_counts.get(pmcid, 0) < max_per_paper:
            paper_counts[pmcid] = paper_counts.get(pmcid, 0) + 1
            results.append(hit)
        if len(results) >= top_k:
            break
    return results


def _finalize(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    return _expand_neighbors(rerank(query, candidates, top_n=top_n))


def _baseline_search(query: str, top_k: int, top_n: int, filters: Optional[dict]) -> list[dict]:
    # PubMedBERT 是纯英文模型，Dense 和 Sparse 均需英文 query
    # translate_query_to_english 对英文原样返回，对中文调用 Qwen 翻译，零侵入
    translated = translate_query_to_english(query)
    candidates = hybrid_search(translated, top_k=top_k, filters=filters)
    return _finalize(query, candidates, top_n)  # rerank 仍用原始 query 对齐用户意图


def search(
    query: str,
    top_k: int = 40,
    top_n: int = 8,
    filters: Optional[dict] = None,
    query_opt_mode: Optional[str] = None,
) -> list[dict]:
    """
    完整检索流程：混合检索 → RRF 融合 → Reranking。

    Args:
        query:   用户查询文本（建议先经过 Query 改写）
        top_k:   混合检索召回条数，默认 40
        top_n:   Reranking 后保留条数，默认 8（送给 LLM）
        filters: 可选过滤条件，例如 {"pub_year": "2023"}
        query_opt_mode:
            off|hyde|decompose|auto（None 时使用 QUERY_OPT_MODE）

    Returns:
        list[dict]，Top-N 最相关结果，每条包含：
          - payload.pmcid
          - payload.text
          - payload.section
          - payload.source_type
          - payload.pub_year
          - payload.journal
          - rerank_score
    """
    if not (query or "").strip():
        return []
    mode = (query_opt_mode or QUERY_OPT_MODE or "off").strip().lower()

    try:
        if mode == "off":
            results = _baseline_search(query, top_k=top_k, top_n=top_n, filters=filters)
            logger.info("检索完成（route=baseline）：query='%s'，返回 %d 条", query[:50], len(results))
            return results

        if mode == "hyde":
            hyde_query = build_hyde_query(query)
            candidates = hybrid_search(hyde_query, top_k=top_k, filters=filters)
            results = _finalize(query, candidates, top_n)
            logger.info("检索完成（route=hyde）：query='%s'，返回 %d 条", query[:50], len(results))
            return results

        if mode == "decompose":
            subqueries = decompose_query(query, max_subqueries=QUERY_DECOMPOSE_MAX_SUBQUERIES)
            with ThreadPoolExecutor(max_workers=min(4, len(subqueries) or 1)) as executor:
                futures = [executor.submit(hybrid_search, subq, top_k, filters) for subq in subqueries]
                all_buckets = [fut.result() for fut in as_completed(futures)]
            candidates = _fuse_results_with_rrf(all_buckets, top_k=top_k)
            results = _finalize(query, candidates, top_n)
            logger.info(
                "检索完成（route=decompose）：query='%s'，子查询 %d 条，返回 %d 条",
                query[:50], len(subqueries), len(results)
            )
            return results

        if mode == "auto":
            complexity = classify_query_complexity(query)
            if complexity == "direct":
                route = "off"
            elif complexity == "complex":
                route = "decompose"
            else:
                route = "hyde"
            logger.info("auto route: complexity=%s → %s", complexity, route)
            return search(
                query=query,
                top_k=top_k,
                top_n=top_n,
                filters=filters,
                query_opt_mode=route,
            )

        # 未知模式回退
        logger.warning("未知 QUERY_OPT_MODE='%s'，回退 baseline", mode)
        return _baseline_search(query, top_k=top_k, top_n=top_n, filters=filters)

    except Exception as e:
        logger.warning("query 优化流程失败，回退 baseline: %s", e)
        return _baseline_search(query, top_k=top_k, top_n=top_n, filters=filters)