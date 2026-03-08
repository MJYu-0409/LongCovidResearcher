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
├── config.py                     # API keys, paths, constants
├── main.py                       # Entry point
│
├── data_pipeline/
│   ├── processor/
│   │   ├── xml_parser.py         # PMC JATS XML → structured paragraphs
│   │   ├── chunker.py            # Paragraphs → token-bounded chunks
│   │   ├── embedder.py           # Dense + sparse embedding
│   │   └── metadata_parser.py   # Extract paper metadata
│   ├── storage/
│   │   ├── qdrant/db.py          # Qdrant upsert / collection management
│   │   ├── postgres/db.py        # PostgreSQL read/write
│   │   └── raw/progress.py       # Pipeline progress tracking
│   └── pipeline.py               # Orchestrates all pipeline stages
│
├── retrieval/
│   ├── search.py                 # Public API: hybrid_search → rerank
│   ├── hybrid.py                 # RRF fusion of dense + sparse
│   ├── dense.py                  # Semantic search (OpenAI embeddings)
│   ├── sparse.py                 # BM25 keyword search
│   └── reranker.py               # Cross-encoder reranking
│
├── agent/
│   ├── __init__.py               # from agent import run
│   ├── state.py                  # AgentState (messages, retrieved_chunks, iteration_count)
│   ├── graph.py                  # Build and compile LangGraph StateGraph
│   ├── nodes.py                  # orchestrator_node, tools_node, should_continue
│   ├── runner.py                 # Public API: run(user_input, history, retrieved_chunks)
│   └── tools/
│       ├── __init__.py           # ALL_TOOLS list
│       ├── search.py             # search_literature — wraps retrieval/search.py
│       ├── paper.py              # get_paper_detail — PostgreSQL metadata lookup
│       ├── sentiment.py          # analyze_sentiment — on-demand sentiment API call
│       ├── synthesis.py          # synthesize_review — GPT-4o literature review
│       └── qa.py                 # answer_question — Qwen factual Q&A
│
└── eval/
    ├── scan_parser_quality.py    # Batch XML parse quality audit
    ├── step1_health_check.py     # Retrieval system sanity check
    ├── step2_ablation.py         # Dense vs sparse vs hybrid vs rerank
    ├── step3a_generate_queries.py # LLM-generated eval query set
    ├── step3b_evaluate.py        # NDCG@5 / Recall@5 / MRR via ranx
    └── output/                   # Eval results (gitignored)
```

## Setup

**Requirements:** Python 3.10+, Qdrant, PostgreSQL

```bash
pip install -r requirements.txt
```

Configure `config.py` (or environment variables):

```python
OPENAI_API_KEY    = "..."
QDRANT_URL        = "http://localhost:6333"
QDRANT_API_KEY    = ""          # leave empty if local
QDRANT_COLLECTION = "longcovid"
POSTGRES_DSN      = "postgresql://user:pass@localhost/longcovid"
FULLTEXT_DIR      = Path("data/raw/fulltext")
```

## Data Pipeline

Run the three stages in order:

```bash
# Stage 1: Parse and store paper metadata
python -c "from data_pipeline.pipeline import run_process_meta; run_process_meta()"

# Stage 2: Process and embed abstracts
python -c "from data_pipeline.pipeline import run_process_abstracts; run_process_abstracts()"

# Stage 3: Process and embed full-text XMLs
python -c "from data_pipeline.pipeline import run_process_fulltext; run_process_fulltext()"
```

Stage 3 depends on Stage 2 (PostgreSQL `papers` table must exist first).  
Each stage is resumable — progress is tracked in a local file and failed papers are skipped without affecting others.

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

**Additional config required in `config.py`:**

```python
# Qwen (used by answer_question)
QWEN_API_KEY  = "..."
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL    = "qwen-plus"

# Sentiment API (used by analyze_sentiment — replace with your endpoint)
# Edit SENTIMENT_API_URL in agent/tools/sentiment.py
```

**Install agent dependencies:**

```bash
pip install langgraph langchain langchain-openai
```

## Evaluation

```bash
# 1. Audit XML parse quality across all papers (~10 min, free)
python eval/scan_parser_quality.py

# 2. Sanity check: source type distribution, year distribution, diversity
python eval/step1_health_check.py

# 3. Ablation: dense vs sparse vs hybrid vs hybrid+rerank
python eval/step2_ablation.py

# 4. Generate LLM eval query set (~$5 GPT-4o, run once)
python eval/step3a_generate_queries.py --sample 50

# 5. Quantitative evaluation with ranx (NDCG@5, Recall@5, MRR)
pip install ranx
python eval/step3b_evaluate.py --fast   # dry run with known labels only
python eval/step3b_evaluate.py          # full eval (~$2-3 GPT-4o-mini)
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

- `eval/output/` is gitignored — regenerate locally by running the eval scripts
- Fulltext chunks include `pub_year` and `journal` metadata injected at pipeline time from PostgreSQL
- The XML parser handles both standard JATS (`body → sec → p`) and unsectioned articles (`body → p` directly), which covers short-form papers like editorials and correspondence
