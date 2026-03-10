# Long COVID Researcher

> 基于 RAG + Agent 的 Long COVID 学术文献智能分析系统

---

## 项目背景

新冠后遗症（Long COVID / PASC）是一个持续影响全球数百万患者的公共卫生问题。自 2020 年以来，相关学术论文呈爆发式增长，研究议题横跨免疫机制、神经系统损伤、心血管影响、康复治疗等多个方向。面对如此体量的文献，研究人员往往难以在有限时间内系统掌握领域进展。

本项目基于约 4,900 篇 PMC 开放获取论文（2020–2026），构建了一套面向研究者的智能文献分析系统。用户可以用自然语言提问，系统会自动检索最相关的文献片段，并由 AI 给出有据可查、注明来源的回答。

---

## 能做什么

**面向研究人员：**
- 快速了解某一议题（如自主神经功能障碍、肠道菌群失调）的研究现状
- 跨文献综合分析，自动生成结构化文献综述
- 追问具体论文的研究方法、结论和数据

**面向政策制定者：**
- 了解不同治疗方案的循证支持力度
- 掌握流行病学数据的最新进展（如 Omicron 变体后的患病率变化）
- 查询学界对特定议题的整体态度与趋势

**面向学生和科研入门者：**
- 以问答形式快速了解领域基础概念
- 获取关键文献推荐，省去文献综述的初步筛选工作

**示例问题：**
```
long covid 的自主神经功能障碍有哪些治疗方案？
2022 年后 Omicron 变体的 long covid 患病率有何变化？
学界对 Paxlovid 治疗 long covid 的评价如何？
肠道菌群在 long covid 发病中扮演什么角色？
```

---

## 系统架构

```
用户提问
    │
    ▼
┌─────────────────────────┐
│   Agent（LangGraph）     │  Qwen 编排 · 问答 · 综述（最多 5 轮工具调用）
└────────────┬────────────┘
             │ 5 个工具
             ▼
┌────────────────────────────────────────────┐
│                  检索层                     │
│  语义检索（dense）──┐                       │
│                     ├── RRF 融合 ── rerank  │
│  关键词检索（sparse）──┘                    │
└────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐     ┌──────────┐
│ Qdrant  │     │PostgreSQL│
│ 向量索引 │     │ 论文元数据 │
└─────────┘     └──────────┘
```

| 组件 | 技术选型 |
|------|---------|
| Embedding | `text-embedding-3-small`（OpenAI） |
| 向量数据库 | Qdrant（dense + sparse 双向量） |
| 重排模型 | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 元数据库 | PostgreSQL |
| Agent 框架 | LangGraph |
| Agent LLM | Qwen（编排 + 问答 + 综述统一） |

---

## 项目结构

```
├── config.py                        # 配置集中管理（从环境变量 / .env 读取）
├── .env.example                     # 环境变量示例（复制为 .env 后填写）
├── main.py                          # 入口：默认 Agent 交互；--pipeline 流水线；--api 启动 FastAPI
│
├── api/                             # FastAPI 服务（Agent 对话、检索、健康检查）
│   ├── app.py                       # /chat、/search、/health 路由
│   └── __init__.py
│
├── infra/                           # 连接与模型单例统一入口
│   ├── clients.py                   # get_openai_client / get_qdrant_client / get_pg_engine /
│   │                                 # get_sparse_embedding_model / get_rerank_model / get_qwen_chat_model
│   └── logging_config.py            # configure_logging（入口处调用一次）
│
├── data_pipeline/                   # 数据处理流水线
│   ├── processor/
│   │   ├── xml_parser.py            # PMC JATS XML → 结构化段落
│   │   ├── chunker.py               # 段落 → token 限制的 chunk
│   │   ├── embedder.py              # 生成 dense + sparse 向量
│   │   └── metadata_parser.py       # 提取论文元数据
│   ├── storage/
│   │   ├── qdrant/db.py             # Qdrant 写入 / 集合管理
│   │   ├── postgres/db.py           # PostgreSQL 读写
│   │   └── raw/progress.py          # 流水线断点续跑
│   ├── pipeline.py                  # 三阶段流水线编排
│   └── scripts/                     # 常用维护脚本（不详细列出）
│
├── retrieval/                       # 检索模块
│   ├── search.py                    # 对外统一接口：hybrid → rerank
│   ├── hybrid.py                    # RRF 融合算法
│   ├── dense.py                     # 语义检索
│   ├── sparse.py                    # BM25 关键词检索
│   └── reranker.py                  # Cross-encoder 重排
│
├── agent/                           # Agent 模块
│   ├── __init__.py                  # from agent import run
│   ├── state.py                     # AgentState（含可选 summary 字段）
│   ├── graph.py                     # LangGraph 图构建与编译
│   ├── nodes.py                     # orchestrator / tools / 路由节点
│   ├── runner.py                    # 对外接口：单轮 & 多轮对话
│   ├── summarizer.py                # 会话内摘要（每轮压摘要 + 保留最近 N 条）
│   ├── session_store.py             # 跨会话记忆：PostgreSQL 持久化（每轮写库）
│   └── tools/
│       ├── __init__.py              # ALL_TOOLS 汇总
│       ├── search.py                # search_literature：文献检索
│       ├── paper.py                 # get_paper_detail：论文元数据查询
│       ├── sentiment.py             # analyze_sentiment：按需情感分析
│       ├── synthesis.py             # synthesize_review：Qwen 文献综述
│       └── qa.py                    # answer_question：Qwen 事实问答
│
└── eval/                            # 评估模块
    ├── scan_parser_quality.py       # XML 解析质量批量扫描
    ├── step1_health_check.py        # 检索系统健康检查
    ├── step2_ablation.py            # 四路策略消融实验
    ├── step3a_generate_queries.py   # GPT-4o 生成评估 query 集
    ├── step3b_evaluate.py           # NDCG@5 / Recall@5 / MRR 量化评估
    └── output/                      # 评估结果输出（已加入 .gitignore）
```

---

## 环境要求

- Python 3.10+
- Qdrant（本地或云端）
- PostgreSQL

```bash
pip install -r requirements.txt
# requirements.txt 已含 fastapi、uvicorn；Agent 需额外：langgraph、langchain、langchain-openai 等（见项目依赖）
```

---

## 配置

推荐使用 `.env` / 环境变量配置（`config.py` 会通过 `os.getenv(...)` 读取），避免把 key 写进代码。

```python
# OpenAI
OPENAI_API_KEY    = "..."

# Qdrant
QDRANT_URL        = "http://localhost:6333"
QDRANT_API_KEY    = ""                    # 本地部署留空
QDRANT_COLLECTION = "longcovid_papers"

# PostgreSQL
DATABASE_URL      = "postgresql://user:pass@localhost/longcovid"

# Qwen（Agent 编排 + 问答 + 综述统一使用，默认 qwen3.5-plus）
QWEN_API_KEY      = "..."
QWEN_API_BASE     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL        = "qwen3.5-plus"

# 情感分析 API（Agent 工具 analyze_sentiment，在 config 中统一配置）
# SENTIMENT_API_SINGLE = "http://localhost:8000/predict"
# SENTIMENT_API_BATCH  = "http://localhost:8000/predict/batch"
# SENTIMENT_API_TIMEOUT = 30

# 可选：跨会话记忆 session_id、FastAPI 端口
# AGENT_SESSION_ID = "default"
# API_PORT = 8001
```

---

## 数据流水线

按顺序执行三个阶段：

```bash
# 阶段一：ESearch 获取 PMCID 列表 + EFetch 拉取摘要/全文落盘到 raw/
python -c "from data_pipeline.pipeline import run_fetch_raw; run_fetch_raw()"

# 阶段二：解析 metadata → 写入 PostgreSQL + 摘要向量化写入 Qdrant
python -c "from data_pipeline.pipeline import run_process_meta; run_process_meta()"

# 阶段三：处理全文 XML → chunk → 向量化写入 Qdrant（依赖阶段二已建好 papers 表）
python -c "from data_pipeline.pipeline import run_process_fulltext; run_process_fulltext()"
```

每个阶段均支持断点续跑，失败的单篇论文会被跳过，不影响其他论文的处理。

**入口说明**：

| 命令 | 说明 |
|------|------|
| `python main.py` | 默认 **Agent 交互**（CLI），支持会话内摘要 + 跨会话持久化（每轮写 PostgreSQL） |
| `python main.py --pipeline` | 跑数据流水线（当前默认只执行 Stage 3，全流程需在 `pipeline.run()` 中取消注释 Stage 1/2） |
| `python main.py --api [--port 8001]` | 启动 **FastAPI 服务**（Agent 对话、检索、健康检查），默认端口 8001 |

---

## 使用方式

**单轮问答：**

```python
from agent import run

result = run("long covid 的自主神经功能障碍有哪些治疗方案？")
print(result["answer"])
print(f"迭代 {result['iterations']} 次，检索 {len(result['retrieved_chunks'])} 个片段")
```

**多轮对话（传入历史保持上下文）：**

```python
r1 = run("long covid 的发病机制是什么？")

r2 = run(
    "其中免疫失调的具体证据有哪些？",
    history=r1["messages"],
    retrieved_chunks=r1["retrieved_chunks"],
)
```

**直接调用检索层：**

```python
from retrieval.search import search

results = search(
    query="autonomic dysfunction mechanism in long covid",
    top_k=20,   # 混合检索候选池大小
    top_n=5,    # rerank 后保留数量
    filters={"pub_year": "2024"},  # 可选过滤
)
# 每条结果包含 payload.pmcid / text / section / source_type / pub_year / journal
# 以及 rrf_score 和 rerank_score
```

**通过 FastAPI 调用（适合前端 / 其他服务）：**

```bash
# 启动服务（默认 http://0.0.0.0:8001）
python main.py --api

# Agent 对话（支持 session_id 多轮记忆）
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" \
  -d '{"user_input": "long covid 自主神经功能障碍有哪些治疗？", "session_id": "user1"}'

# 文献检索
curl -X POST http://localhost:8001/search -H "Content-Type: application/json" \
  -d '{"query": "microclots long covid", "top_n": 5}'

# 运维健康检查（PostgreSQL、Qdrant）
curl http://localhost:8001/health
```

交互式 API 文档：`http://localhost:8001/docs`（Swagger）、`http://localhost:8001/redoc`（ReDoc）。

---

## 记忆与持久化

- **会话内**：每轮对话结束后用 LLM 将较早消息压成摘要，只保留最近 3 条完整消息，避免上下文过长顶破 token 上限。
- **跨会话**：每轮结束后将当前摘要 + 最近 3 条 + `retrieved_chunks` 写入 PostgreSQL 表 `agent_sessions`（按 `session_id` 覆盖）；下次启动或同一 `session_id` 请求时自动加载，实现断线可恢复、多端共享同一会话。

---

## Agent 工具说明

| 工具 | 调用模型 | 用途 |
|------|---------|------|
| `search_literature` | — | 混合检索，向 Orchestrator 返回摘要视图，完整 chunk 存入 State |
| `get_paper_detail` | — | 根据 pmcid 查询 PostgreSQL 获取论文完整元数据 |
| `analyze_sentiment` | 外部 API | 对检索到的论文按需进行情感分析 |
| `answer_question` | Qwen | 基于已检索片段进行事实性问答 |
| `synthesize_review` | Qwen | 综合多篇文献生成结构化文献综述 |

---

## 检索系统评估结果

评估数据集：150 个 GPT-4o 生成的 query，相关性标签由 GPT-4o-mini 打分（0/1/2），指标计算使用 [ranx](https://github.com/AmenRa/ranx)。

| 策略 | NDCG@5 | Recall@5 | Precision@5 | MRR |
|------|--------|----------|-------------|-----|
| A — 仅语义检索 | 0.736 | 0.471 | 0.723 | 0.983 |
| B — 仅关键词检索 | 0.773 | 0.534 | 0.805 | 0.985 |
| C — Hybrid（RRF 融合） | 0.769 | 0.492 | 0.753 | 0.983 |
| **D — Hybrid + Rerank** | **0.782** | **0.522** | **0.799** | 0.980 |

**关键发现：**
- Dense / Sparse 平均 Jaccard 重叠率仅 **0.08**，两路检索高度互补，hybrid 融合有实质价值
- Reranker 在 **62%** 的 query 中改变了第一名结果，重排效果真实有效
- 在学术医学文献场景下，关键词检索（B）表现略优于纯语义检索（A），原因是专业术语高度精确

---

## 语料库说明

| 项目 | 内容 |
|------|------|
| 来源 | PMC Open Access |
| 规模 | ~4,900 篇，2020–2026 年 |
| 覆盖主题 | 发病机制、症状、治疗、流行病学、免疫、神经、心血管等 |
| 解析质量 | 91% 正常解析，4.7% 无可提取全文（仅摘要或更正通知） |

---

## 工程说明

- `eval/output/` 已加入 `.gitignore`，本地运行评估脚本重新生成
- XML 解析器同时支持标准 JATS 结构（`body → sec → p`）和无分节结构（`body → p`），后者覆盖编辑、通讯、病例报告等短文体裁
- Fulltext chunk 的 `pub_year` 和 `journal` 字段在流水线阶段从 PostgreSQL 注入，不依赖 XML 元数据
- Orchestrator 只接收检索结果的摘要视图（text 截断至 150 字符），完整 chunk 仅在 `answer_question` 和 `synthesize_review` 中消费，控制 token 成本
- 情感分析 API 地址在 `config.py` 中通过 `SENTIMENT_API_SINGLE`、`SENTIMENT_API_BATCH` 配置（单条/批量摘要情绪分析）
