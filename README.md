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
用户提问（Web UI / CLI / API）
    │
    ▼
┌──────────────────────────────────┐
│        Agent（LangGraph）         │
│   Qwen 编排 · 最多 5 轮工具调用   │
│   ┌──────────┬──────────────┐    │
│   │ 问答工具  │  综述工具    │    │
│   └──────────┴──────────────┘    │
└────────────────┬─────────────────┘
                 │ 检索请求
                 ▼
┌────────────────────────────────────────────┐
│               检索层                        │
│                                             │
│  Query 优化（HyDE / 分解 / 翻译）           │
│       ↓                                     │
│  PubMedBERT Dense ──┐                       │
│                      ├── RRF 融合           │
│  SPLADE Sparse ──────┘   (per-paper cap=5)  │
│       ↓                                     │
│  Cross-Encoder Rerank (per-paper cap=2)     │
│       ↓                                     │
│  ±1 邻居段落窗口扩展                         │
└───────────────────┬────────────────────────┘
         ┌──────────┴───────────┐
         ▼                      ▼
   ┌──────────┐          ┌──────────────┐
   │  Qdrant  │          │  PostgreSQL  │
   │ 向量索引  │          │  论文元数据   │
   └──────────┘          └──────────────┘
                              ↕
                         ┌─────────┐
                         │  Redis  │
                         │ 会话缓存 │
                         └─────────┘
```

### 技术选型

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| Dense Embedding | `NeuML/pubmedbert-base-embeddings` | 本地推理，768 维，针对生物医学文献优化 |
| Sparse Embedding | `prithivida/Splade_PP_en_v1` | 关键词级稀疏向量，补充语义检索 |
| 重排模型 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder 精排，本地推理 |
| 向量数据库 | Qdrant（dense + sparse 双向量） | 支持 payload 过滤 |
| 元数据库 | PostgreSQL | 论文元数据存储 |
| 会话缓存 | Redis | 活跃会话快速读写，TTL 24h |
| Agent 框架 | LangGraph | 有状态工具调用图 |
| Agent LLM | Qwen（qwen3.5-plus） | 编排 + 问答 + 综述统一 |
| 答案评测 | GPT-4o-mini | 第三方独立打分，避免 self-bias |

---

## 项目结构

```
├── config.py                        # 配置集中管理（从环境变量 / .env 读取）
├── .env.example                     # 环境变量示例（复制为 .env 后填写）
├── main.py                          # 入口：默认 Agent CLI；--pipeline 流水线；--api FastAPI
│
├── static/
│   └── index.html                   # Web 聊天界面（单文件 SPA，SSE 流式输出）
│
├── api/
│   └── app.py                       # FastAPI：/chat、/chat/stream、/search、/health
│
├── infra/
│   ├── clients.py                   # 所有连接/模型单例统一入口
│   └── logging_config.py
│
├── storage/
│   ├── postgres/
│   │   ├── papers.py                # papers 表：建表、插入、fetch_meta
│   │   └── session_store.py         # CLI 会话持久化（PostgreSQL）
│   ├── redis/
│   │   └── session_store.py         # API 会话缓存（Redis，TTL 24h）
│   └── qdrant/
│       ├── chunks.py                # 向量集合管理与 chunk 写入
│       └── memory_store.py          # 跨会话用户记忆（Qdrant）
│
├── data_pipeline/                   # 数据处理流水线
│   ├── processor/
│   │   ├── xml_parser.py            # PMC JATS XML → 结构化段落
│   │   ├── chunker.py               # 段落 → chunk（MAX_CHILD_CHARS=1200）
│   │   ├── embedder.py              # Dense + Sparse 批量向量化
│   │   └── metadata_parser.py       # 论文元数据提取
│   ├── pipeline.py                  # 三阶段流水线（批量处理 + 断点续传）
│   └── scripts/                     # 维护脚本（失败重处理等）
│
├── retrieval/
│   ├── search.py                    # 对外统一接口（含 Query 优化路由）
│   ├── hybrid.py                    # RRF 融合（并发双路）
│   ├── dense.py                     # PubMedBERT 语义检索
│   ├── sparse.py                    # SPLADE 关键词检索
│   ├── reranker.py                  # Cross-Encoder 精排
│   └── query_optimizer.py           # HyDE / 分解 / 中文翻译
│
├── agent/
│   ├── graph.py                     # LangGraph 图定义
│   ├── nodes.py                     # Orchestrator / Tools / 路由节点
│   ├── state.py                     # AgentState（messages / chunks / summary）
│   ├── runner.py                    # 同步入口（CLI 使用）
│   ├── runner_async.py              # 异步流式入口（API 使用，SSE）
│   ├── summarizer.py                # 会话内摘要压缩
│   ├── memory.py                    # 跨会话记忆读写
│   └── tools/
│       ├── search.py                # search_literature
│       ├── paper.py                 # get_paper_detail
│       ├── sentiment.py             # analyze_sentiment
│       ├── synthesis.py             # synthesize_review
│       └── qa.py                    # answer_question
│
└── eval/
    ├── step1_health_check.py        # 检索系统健康检查
    ├── step2_ablation.py            # 多策略消融实验
    ├── step3a_generate_queries.py   # GPT-4o 生成评估 query 集
    ├── step3b_evaluate.py           # 检索层评测（NDCG / Recall / MRR）
    ├── step4_answer_quality.py      # 端到端答案质量评测（Faithfulness / Completeness / Citation）
    └── output/                      # 评估结果（已加入 .gitignore）
```

---

## 环境要求

- Python 3.10+
- Qdrant（本地 Docker 或云端）
- PostgreSQL
- Redis

```bash
pip install -r requirements.txt
```

---

## 配置

复制 `.env.example` 为 `.env`，填写以下配置：

```bash
# NCBI（数据拉取，必填）
NCBI_API_KEY=your_ncbi_api_key
NCBI_EMAIL=your@email.com

# Qwen（Agent 全部使用：编排 + 问答 + 综述）
QWEN_API_KEY=sk-...
QWEN_MODEL=qwen3.5-plus           # 可选，默认 qwen3.5-plus

# OpenAI（仅用于评测打分，Agent 本身不依赖）
OPENAI_API_KEY=sk-...

# 数据库
DATABASE_URL=postgresql://user:pass@localhost/longcovid
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_TTL=86400           # 会话 TTL，单位秒

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                   # 本地部署留空

# 代理（如需科学上网访问 Qwen / OpenAI）
HTTPS_PROXY=http://127.0.0.1:7890

# 检索优化（off | hyde | decompose | auto，默认 auto）
QUERY_OPT_MODE=auto

# 答案 Grounding Check（生成后逐条核查断言依据，默认开启）
GROUNDING_CHECK=true

# 复杂问题最大子查询数（默认 4）
QUERY_DECOMPOSE_MAX_SUBQUERIES=4

# 情感分析 API（analyze_sentiment 工具使用）
SENTIMENT_API_SINGLE=http://localhost:8000/predict
SENTIMENT_API_BATCH=http://localhost:8000/predict/batch
```

---

## Web 界面

启动 API 后直接在浏览器打开 `http://localhost:8001`，无需额外部署前端。

```bash
python main.py --api
```

![Web 界面预览](static/frontend_preview.jpg)

**界面功能：**

| 功能 | 说明 |
|------|------|
| 流式输出 | 回答逐 token 实时推送（SSE），无需等待完整响应 |
| 工具状态指示 | 实时显示当前正在执行的工具（检索中 / 分析证据 / 生成综述...） |
| 引用展示 | 自动从回答中提取所有 PMC ID，以标签形式展示在回答底部 |
| Markdown 渲染 | 支持粗体、斜体、列表、行内代码的轻量渲染 |
| 会话管理 | Session ID 自动生成并持久化到 localStorage；点击 "New Session" 开启新对话 |
| 快捷提问 | 欢迎页提供 4 条示例问题，点击直接发送 |
| 键盘快捷键 | `Enter` 发送，`Shift+Enter` 换行 |
| 错误提示 | 检索系统不可用时在气泡内直接显示错误原因 |

工具状态 label 对照：

| 工具名 | 显示文字 |
|--------|---------|
| `search_literature` | Searching literature |
| `answer_question` | Analyzing evidence |
| `synthesize_review` | Synthesizing review |
| `analyze_sentiment` | Analyzing sentiment |
| `get_paper_detail` | Fetching paper details |

---

## 运行方式

| 命令 | 说明 |
|------|------|
| `python main.py` | **CLI 交互**，支持多轮对话 + 跨会话记忆（PostgreSQL） |
| `python main.py --api` | **Web 服务**，含 Web UI + SSE 流式 API，默认端口 8001 |
| `python main.py --api --port 9000` | 指定端口 |
| `python main.py --pipeline` | 运行数据流水线 |

---

## 数据流水线

按顺序执行三个阶段：

```bash
# Stage 1：ESearch 获取 PMCID + EFetch 拉取全文落盘 raw/
python -c "from data_pipeline.pipeline import run_fetch_raw; run_fetch_raw()"

# Stage 2：metadata → PostgreSQL papers 表 + 摘要向量化 → Qdrant
python -c "from data_pipeline.pipeline import run_process_meta; run_process_meta()"

# Stage 3：全文 XML → chunk → 向量化 → Qdrant（依赖 Stage 2）
python -c "from data_pipeline.pipeline import run_process_fulltext; run_process_fulltext()"

# 建立 Qdrant payload 索引（Stage 3 完成后执行一次）
python -c "from storage.qdrant.chunks import ensure_payload_indexes; ensure_payload_indexes()"
```

**设计特点：**
- 每阶段独立可重试，失败单篇不影响其他
- Stage 3 按 1000 篇/批次批量 embed，比逐篇处理快约 5×
- 启动时滚动扫描 Qdrant 获取已完成 pmcid，自动跳过（断点续传）

---

## API 说明

### POST /chat/stream（推荐）

SSE 流式对话，逐 token 推送。

```bash
curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_input": "long covid 自主神经功能障碍的治疗？", "session_id": "user1"}'
```

SSE 事件格式：

```
data: {"type": "tool_start", "tool": "search_literature", "query": "autonomic dysfunction treatment"}
data: {"type": "tool_end",   "tool": "search_literature"}
data: {"type": "token",      "content": "根据检索到的文献..."}
data: {"type": "done",       "iterations": 2}
data: {"type": "error",      "message": "..."}
```

### POST /chat（同步，兼容保留）

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "...", "session_id": "user1"}'
# → {"answer": "...", "iterations": 2, "session_id": "user1"}
```

### POST /search（直接检索）

```bash
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "microclots long covid", "top_k": 20, "top_n": 5}'
```

### GET /health

```bash
curl http://localhost:8001/health
# → {"status": "ok", "postgres": "ok", "qdrant": "ok", "redis": "ok"}
```

交互式文档：`http://localhost:8001/docs`（Swagger）、`http://localhost:8001/redoc`

---

## 代码调用

```python
# 单轮问答
from agent import run
result = run("long covid 的自主神经功能障碍有哪些治疗方案？")
print(result["answer"])

# 多轮对话
r1 = run("long covid 的发病机制是什么？")
r2 = run(
    "其中免疫失调的具体证据有哪些？",
    history=r1["messages"],
    retrieved_chunks=r1["retrieved_chunks"],
)

# 直接检索
from retrieval.search import search
results = search(
    query="autonomic dysfunction long covid",
    top_k=40,
    top_n=8,
    filters={"pub_year": "2024"},
)
```

---

## 优化策略

### 检索层

#### 1. 混合检索 + RRF 融合
Dense（PubMedBERT 语义）和 Sparse（SPLADE 关键词）并发双路检索，用 RRF 算法（k=60）融合排名。两路 Jaccard 重叠率约 0.08，互补性强。

#### 2. 级联多样性保证（per-paper cap）
解决单篇论文垄断检索结果的问题：

| 阶段 | cap | 来源论文下限 |
|------|-----|------------|
| RRF → top_k=40 | 每篇最多 5 条 | ≥ 8 篇 |
| Rerank → top_n=8 | 每篇最多 2 条 | ≥ 4 篇 |

贪心选取策略：按分数遍历，跳过超额论文继续向下选，不截断候选池。

#### 3. ±1 邻居段落窗口扩展
每个被 rerank 选中的 chunk，在返回前从 Qdrant 实时查询同篇论文 / 同 section 的相邻段落（chunk_index ±1），拼成完整上下文供 LLM 阅读。首尾段落自动退化为单侧窗口。

#### 4. Query 优化路由（`QUERY_OPT_MODE`）

| 模式 | 策略 | 适用场景 |
|------|------|---------|
| `off` | 直接混合检索 | 已优化查询 |
| `hyde` | 生成假设答案文档，增强语义召回 | 单意图问题 |
| `decompose` | 拆分为 2–4 个子查询并行检索，RRF 二次融合 | 多意图 / 比较类问题 |
| `auto` | 按复杂度分类自动路由 | 默认模式 |

`auto` 模式的分类逻辑：
- token 数 ≤ 5 → `direct`（plain hybrid）
- 含比较词（compare / 对比 / 差异等）或多问号 → `complex`（decompose）
- 其余 → `simple`（hyde）

> CJK 字符逐个计数（一个汉字 = 一个 token），避免中文查询因无空格被误判为 "direct"。

#### 5. 中文查询自动翻译
PubMedBERT 和 SPLADE 均为英文模型，中文输入会产生大量 `[UNK]` token 导致向量退化。检测到中文后调用 Qwen 翻译为 PubMed 风格英文检索式；rerank 仍使用原始问题以对齐用户意图。

### Agent 层

#### 6. Full-context 注入
LLM 在调用 `answer_question` / `synthesize_review` 时仅能看到 150 字符的文本预览（token 成本控制）。`tools_node_with_state_update` 在执行这两个工具前，自动将 State 中的完整 `context_text`（含邻居扩展）替换 LLM 传入的参数。LLM 感知不到这个替换。

#### 7. Grounding Check（`GROUNDING_CHECK=true`）
答案生成后，同一 Qwen 实例作为审核员二次阅读答案与参考原文，对无文献依据的断言追加 `[⚠ 文献中无直接依据]` 标注，抑制幻觉传播。

#### 8. 系统提示反幻觉约束
Orchestrator 提示词强制要求：
- 检索失败时禁止用模型自身知识代答
- 文献不足（count < 3）时强制重试换词，不允许强行作答
- 综述前须累积 ≥5 个 chunk
- 答案末尾必须附证据充分度标注

### 记忆与持久化

#### 9. 会话内摘要压缩
每轮对话结束后，用 Qwen 将较早的消息（含工具调用）压缩为摘要，仅保留最近 6 条 Human/AI 消息完整内容，避免上下文超出 token 上限。

#### 10. 跨会话记忆（Redis + Qdrant）
- 每轮结束后将摘要 + 最近消息写入 Redis（TTL 24h），下次同一 `session_id` 请求自动恢复
- 摘要异步双写到 Qdrant `user_memories` 集合，供后续会话按语义相似度检索历史研究兴趣，注入 Orchestrator 上下文

#### 11. 并发安全（ContextVar）
`search_literature` 的完整 chunks 通过 `contextvars.ContextVar` 传递（而非全局变量），确保多用户并发请求下各自的检索结果不会交叉污染。

---

## Agent 工具说明

| 工具 | 模型 | 用途 |
|------|------|------|
| `search_literature` | — | 混合检索（优先第一步），向 Orchestrator 返回摘要视图 |
| `get_paper_detail` | — | 根据 pmcid 查询 PostgreSQL 获取论文完整元数据 |
| `analyze_sentiment` | 外部 API | 对检索到的论文摘要进行情感分析（支持单条/批量） |
| `answer_question` | Qwen | 基于完整 chunk 文本做事实性问答；可选 Grounding Check |
| `synthesize_review` | Qwen | 综合多篇文献生成结构化综述（背景 / 发现 / 争议 / 局限性） |

---

## 检索评估结果

评估数据集：150 个 GPT-4o 生成的英文 query，相关性由 GPT-4o-mini 打 0/1/2 分，使用 [ranx](https://github.com/AmenRa/ranx) 计算指标。

| 策略 | NDCG@5 | Recall@5 | Precision@5 | MRR |
|------|--------|----------|-------------|-----|
| A — 仅语义（PubMedBERT） | 0.736 | 0.471 | 0.723 | 0.983 |
| B — 仅关键词（SPLADE） | 0.773 | 0.534 | 0.805 | 0.985 |
| C — Hybrid（RRF 融合） | 0.769 | 0.492 | 0.753 | 0.983 |
| **D — Hybrid + Rerank** | **0.782** | **0.522** | **0.799** | 0.980 |

**关键发现：**
- Dense / Sparse 平均 Jaccard 重叠率仅 **0.08**，两路检索高度互补
- Reranker 在 **62%** 的 query 中改变了第一名，重排效果真实有效
- 学术医学文献场景下，关键词检索（B）略优于语义检索（A）——专业术语高度精确，倾向于精确匹配

**指标关注优先级：** NDCG@5（排名质量，影响 LLM 阅读效果最直接）> Recall@5（覆盖率，决定回答的文献宽度）> MRR（首条命中速度）> Precision@5

---

## 端到端答案质量评测

`eval/step4_answer_quality.py` 对完整 Agent 链路做端到端评测：

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| Faithfulness（0–2） | GPT-4o-mini 对照原文核查 | 答案断言是否均有文献直接支持 |
| Completeness（0–2） | GPT-4o-mini 判断覆盖度 | 是否覆盖问题的所有方面 |
| Citation Accuracy | 程序正则匹配 | 答案引用的 PMC ID 在检索结果中的命中率 |

> 使用 GPT-4o-mini 而非 Qwen 打分，避免同系模型 self-evaluation bias。

```bash
# 调试：前 10 题
python eval/step4_answer_quality.py --limit 10

# 指定问题
python eval/step4_answer_quality.py --queries "brain fog mechanisms" "fatigue in PASC"

# 全量（读 eval/output/query_set.json）
python eval/step4_answer_quality.py
```

---

## 语料库说明

| 项目 | 内容 |
|------|------|
| 来源 | PMC Open Access |
| 规模 | ~4,900 篇，2020–2026 年 |
| 覆盖主题 | 发病机制、症状、治疗、流行病学、免疫、神经、心血管等 |
| 解析质量 | 91% 正常解析，4.7% 无可提取全文（仅摘要或更正通知） |
| 向量集合 | `longcovid_papers_pc`（段落级 chunk，PubMedBERT） |
| Chunk 策略 | MAX_CHILD_CHARS=1200，超长按句切分；含 chunk_index 字段 |

---

## 工程说明

- `eval/output/` 已加入 `.gitignore`，本地运行评估脚本重新生成
- XML 解析器同时支持标准 JATS（`body → sec → p`）和无分节结构（`body → p`），覆盖编辑、通讯、病例报告等短文体裁
- Fulltext chunk 的 `pub_year` 和 `journal` 字段在 pipeline 阶段从 PostgreSQL 注入，不依赖 XML 元数据
- Orchestrator 只接收 150 字符文本预览（控制 token 成本），完整 `context_text` 在执行 QA/综述工具前由 `tools_node_with_state_update` 自动注入
- `chunk_index` 字段已建 Qdrant payload 索引，`_expand_neighbors()` 依赖它做 ±1 邻居查询
- 代理配置：`HTTPS_PROXY` 控制 Qwen / OpenAI 的出站请求；`NO_PROXY` 自动注入 localhost 条目，防止本地服务（Qdrant / Redis / PostgreSQL）被代理拦截
