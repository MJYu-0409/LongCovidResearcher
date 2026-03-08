# Long COVID Researcher

A RAG + Agent system for academic literature analysis on Long COVID, built on ~4,900 PMC open-access papers (2020–2026).

## Overview

**Target users:** Researchers, policymakers, graduate students

**Core capabilities:**
- Hybrid semantic + keyword retrieval across full-text and abstracts
- Multi-strategy reranking for precision
- Agent-driven literature synthesis and Q&A
- On-demand sentiment analysis via external API

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│    Agent    │  GPT-4o orchestrator (ReAct, max 5 iterations)
│  (LangGraph)│  Qwen for QA · GPT-4o for synthesis
└──────┬──────┘
       │ tools
       ▼
┌─────────────────────────────────────────────┐
│              Retrieval Layer                │
│  dense_search  ──┐                          │
│                  ├── RRF hybrid ── rerank   │
│  sparse_search ──┘                          │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│    Qdrant    │    │  PostgreSQL  │
│ dense+sparse │    │  metadata    │
│   vectors    │    │              │
└──────────────┘    └──────────────┘
```

**Embedding:** `text-embedding-3-small` (OpenAI)  
**Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**Vector DB:** Qdrant (dense + sparse vectors per chunk)  
**Metadata DB:** PostgreSQL

## Project Structure

```
├── config.py                     # 配置集中管理（环境变量 / .env）
├── main.py                       # 入口：configure_logging + pipeline.run()
├── .env.example                  # 环境变量示例（复制为 .env 后填写）
│
├── infra/
│   ├── clients.py               # get_openai_client, get_qdrant_client, get_pg_engine
│   └── logging_config.py         # configure_logging（入口处调用）
│
├── data_pipeline/
│   ├── processor/
│   │   ├── xml_parser.py         # PMC JATS XML → 结构化段落
│   │   ├── chunker.py            # 段落 → 按 section 的 token 边界 chunk
│   │   ├── embedder.py           # Dense + sparse 向量化
│   │   └── metadata_parser.py   # 元数据解析
│   ├── storage/
│   │   ├── qdrant/db.py          # Qdrant upsert / 集合管理
│   │   ├── postgres/db.py        # PostgreSQL papers 表读写
│   │   └── raw/                  # progress.json, metadata/, fulltext/
│   │       └── progress.py       # 断点续跑进度
│   ├── pipeline.py               # 三阶段：fetch_raw → process_meta → process_fulltext
│   └── scripts/
│       ├── reprocess_failed_fulltext.py  # 对 scan 报告 FAIL 的全文重新向量化
│       ├── remediate_no_pmid.py          # 补救缺失 PMID
│       └── backfill_qdrant_metadata.py   # 回填 Qdrant payload 元数据
│
├── retrieval/
│   ├── search.py                 # 对外入口：hybrid_search → rerank
│   ├── hybrid.py                 # RRF 融合 dense + sparse
│   ├── dense.py                  # 语义检索（OpenAI embedding + Qdrant query_points）
│   ├── sparse.py                 # 稀疏检索（fastembed + Qdrant query_points）
│   └── reranker.py               # Cross-encoder 精排（config: RERANK_MODEL）
│
├── agent/
│   ├── __init__.py               # from agent import run
│   ├── state.py                  # AgentState (messages, retrieved_chunks, iteration_count)
│   ├── graph.py                  # LangGraph StateGraph 构建与编译
│   ├── nodes.py                  # orchestrator_node, tools_node_with_state_update, should_continue
│   ├── runner.py                 # 对外 API: run(user_input, history, retrieved_chunks)
│   └── tools/
│       ├── __init__.py           # ALL_TOOLS
│       ├── search.py             # search_literature — 封装 retrieval.search，chunks 写 State
│       ├── paper.py               # get_paper_detail — PostgreSQL 按 pmcid 查元数据
│       ├── sentiment.py          # analyze_sentiment — 外部情感分析 API（需配置 URL）
│       ├── qa.py                 # answer_question — Qwen 基于检索片段作答
│       └── synthesis.py          # synthesize_review — GPT-4o 文献综述
│
└── eval/
    ├── scan_parser_quality.py    # 批量 XML 解析质量扫描 → scan_report.json
    ├── step1_health_check.py     # 检索健康检查（source/year 分布、空结果）
    ├── step2_ablation.py        # 消融：dense / sparse / hybrid / hybrid+rerank
    ├── step3a_generate_queries.py # 生成评测 query 集（需 GPT-4o）
    ├── step3b_evaluate.py       # NDCG@5 / Recall@5 / MRR（ranx，需 pip install ranx）
    └── output/                   # 评测输出（gitignored）
```

## Setup

**Requirements:** Python 3.10+, Qdrant, PostgreSQL

```bash
pip install -r requirements.txt
```

**可选依赖（按需安装）：**  
- 检索 / Pipeline：`openai`, `qdrant-client`, `fastembed`, `sentence-transformers`  
- Agent：`langgraph`, `langchain`, `langchain-openai`  
- 评测 step3b：`ranx`

配置通过环境变量或 `.env`（项目根下复制 `.env.example` 为 `.env` 后填写）。与 `config.py` 对应关系：

| 用途 | 变量名 | 说明 |
|------|--------|------|
| OpenAI（embedding / orchestrator / synthesis） | `OPENAI_API_KEY` | 必填 |
| 千问（answer_question） | `QWEN_API_KEY` | Agent 用 Qwen 时必填 |
| Qdrant | `QDRANT_URL`, `QDRANT_API_KEY` | 默认 `http://localhost:6333`，本地可空 |
| 向量集合名 | — | `config.py` 内 `QDRANT_COLLECTION = "longcovid_papers"` |
| PostgreSQL | `DATABASE_URL` | 连接串，如 `postgresql://user:pass@host/dbname` |
| 数据路径 | — | `config.py` 内 `METADATA_DIR`, `FULLTEXT_DIR`, `PROGRESS_FILE` |

## Data Pipeline

三阶段需按顺序执行。`main.py` 仅调用 `pipeline.run()`；当前 `pipeline.run()` 默认只执行 Stage 3，如需跑全流程或单阶段，可在 `pipeline.run()` 中取消注释相应阶段，或直接：

```bash
# Stage 1：拉取 PMCID 列表并 EFetch 摘要与全文到 raw/
python -c "from data_pipeline.pipeline import run_fetch_raw; run_fetch_raw()"

# Stage 2：解析 metadata，写入 PostgreSQL，摘要向量化写入 Qdrant
python -c "from data_pipeline.pipeline import run_process_meta; run_process_meta()"

# Stage 3：全文解析 → chunk → 向量化写入 Qdrant（依赖 Stage 2）
python -c "from data_pipeline.pipeline import run_process_fulltext; run_process_fulltext()"
```

Stage 3 依赖 Stage 2（PostgreSQL `papers` 表需已存在）。各阶段支持断点续跑，进度在 `config.PROGRESS_FILE`。

## Retrieval

```python
from retrieval.search import search

results = search(
    query="autonomic dysfunction mechanism in long covid",
    top_k=20,   # hybrid retrieval pool size
    top_n=5,    # final results after reranking
    filters={"pub_year": "2024"},  # optional
)

# Each result:
# {
#   "payload": {
#     "pmcid": "PMC...",
#     "text": "...",
#     "section": "Introduction",
#     "source_type": "fulltext",
#     "pub_year": "2024",
#     "journal": "..."
#   },
#   "rrf_score": 0.031,
#   "rerank_score": 8.24
# }
```

Retrieval performance (ablation, 8 queries):
- Dense / Sparse average Jaccard overlap: **0.08** — two paths are highly complementary
- Reranker changes top-1 result in **62%** of queries — reranking is substantive

## Agent

The Agent is built on LangGraph with a ReAct loop (max 5 iterations). GPT-4o acts as the orchestrator; Qwen handles factual Q&A and GPT-4o handles synthesis.

**Single-turn:**

```python
from agent import run

result = run("long covid 的自主神经功能障碍有哪些治疗方案？")
print(result["answer"])
print(f"迭代 {result['iterations']} 次，检索 {len(result['retrieved_chunks'])} 个 chunk")
```

**Multi-turn (pass history to maintain context):**

```python
r1 = run("long covid 的发病机制是什么？")

r2 = run(
    "其中免疫失调的具体证据有哪些？",
    history=r1["messages"],
    retrieved_chunks=r1["retrieved_chunks"],
)
```

**Five tools available to the orchestrator:**

| Tool | Model | Purpose |
|------|-------|---------|
| `search_literature` | — | Hybrid retrieval, returns summary view to orchestrator; full chunks stored in State |
| `get_paper_detail` | — | PostgreSQL lookup by pmcid |
| `analyze_sentiment` | External API | On-demand sentiment analysis for retrieved papers |
| `answer_question` | Qwen | Factual Q&A grounded in retrieved chunks |
| `synthesize_review` | GPT-4o | Structured literature review from accumulated chunks |

**Additional config:**  
- Qwen：在 `.env` 或环境中设置 `QWEN_API_KEY`；`QWEN_API_BASE`、`QWEN_MODEL` 在 `config.py` 中已写。  
- 情感分析：在 `agent/tools/sentiment.py` 中修改 `SENTIMENT_API_URL` 为实际接口地址。

**Install agent dependencies:**

```bash
pip install langgraph langchain langchain-openai
```

## Evaluation

建议在**项目根目录**下执行：

```bash
# 1. XML 解析质量扫描
python eval/scan_parser_quality.py

# 2. 检索健康检查
python eval/step1_health_check.py

# 3. 消融实验
python eval/step2_ablation.py

# 4. 生成评测 query 集（需 GPT-4o）
python eval/step3a_generate_queries.py --sample 50

# 5. 定量评测（需先安装 ranx）
pip install ranx
python eval/step3b_evaluate.py --fast   # 小规模
python eval/step3b_evaluate.py          # 完整评测
```

Metrics computed by [ranx](https://github.com/AmenRa/ranx): NDCG@5, Recall@5, Precision@5, MRR.  
Relevance labels (0/1/2) are generated by GPT-4o-mini per (query, chunk) pair.

**Results (150 queries, LLM-scored):**

| Strategy | NDCG@5 | Recall@5 | Precision@5 | MRR |
|----------|--------|----------|-------------|-----|
| A — Dense only | 0.736 | 0.471 | 0.723 | 0.983 |
| B — Sparse only | 0.773 | 0.534 | 0.805 | 0.985 |
| C — Hybrid (RRF) | 0.769 | 0.492 | 0.753 | 0.983 |
| D — Hybrid + Rerank | **0.782** | **0.522** | **0.799** | 0.980 |

## Corpus

- **Source:** PMC Open Access
- **Size:** ~4,900 papers, 2020–2026
- **Coverage:** Long COVID / PASC — mechanisms, symptoms, treatment, epidemiology
- **Parse quality:** 91% OK, 4.7% no extractable full-text (abstract-only or correction notices)

## Notes

- `eval/output/` 为 gitignored，需本地运行评测脚本生成
- 连接与日志：统一使用 `infra/clients.py`（OpenAI、Qdrant、PostgreSQL）、`infra/logging_config.py` 的 `configure_logging()` 在入口调用
- 全文 chunk 在 pipeline 阶段从 PostgreSQL 注入 `pub_year`、`journal`
- XML 解析支持标准 JATS（`body → sec → p`）与无 section 短文（`body → p` 直接），覆盖社论、通讯等
- 评测与部分脚本需在**项目根目录**下运行，以保证 `eval/output`、`config` 等路径正确
