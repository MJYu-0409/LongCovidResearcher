# LongCovidResearcher
A RAG + Agent system for academic literature analysis on Long COVID, built on ~4,900 PMC open-access papers (2020–2026).

## Overview

**Target users:** Researchers, policymakers, graduate students

**Core capabilities:**
- Hybrid semantic + keyword retrieval across full-text and abstracts
- Multi-strategy reranking for precision
- Agent-driven literature synthesis and Q&A
- Sentiment analysis across the corpus

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
│ dense+sparse │    │  metadata +  │
│   vectors    │    │  sentiment   │
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
├── agent/                        # (in development)
│   ├── state.py
│   ├── graph.py
│   ├── nodes.py
│   ├── runner.py
│   └── tools/
│       ├── search.py             # Wraps retrieval/search.py
│       ├── paper.py              # Paper metadata lookup
│       ├── sentiment.py          # Corpus sentiment overview
│       ├── synthesis.py          # GPT-4o literature review
│       └── qa.py                 # Qwen factual Q&A
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

## Corpus

- **Source:** PMC Open Access
- **Size:** ~4,900 papers, 2020–2026
- **Coverage:** Long COVID / PASC — mechanisms, symptoms, treatment, epidemiology
- **Parse quality:** 91% OK, 4.7% no extractable full-text (abstract-only or correction notices)

## Notes

- `eval/output/` is gitignored — regenerate locally by running the eval scripts
- Fulltext chunks include `pub_year` and `journal` metadata injected at pipeline time from PostgreSQL
- The XML parser handles both standard JATS (`body → sec → p`) and unsectioned articles (`body → p` directly), which covers short-form papers like editorials and correspondence
