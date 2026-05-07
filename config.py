"""
config.py - 所有配置集中管理，不允许在其他文件 hardcode 任何参数
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure localhost traffic bypasses any system proxy (e.g. Clash/FlClash).
# Force-merge localhost entries into NO_PROXY even if it was already set by the system.
_required_bypass = {"localhost", "127.0.0.1", "::1"}
_existing_no_proxy = (os.environ.get("NO_PROXY", "") + "," + os.environ.get("no_proxy", ""))
_existing_entries = {e.strip() for e in _existing_no_proxy.split(",") if e.strip()}
_merged_no_proxy = ",".join(_required_bypass | _existing_entries)
os.environ["NO_PROXY"] = _merged_no_proxy
os.environ["no_proxy"] = _merged_no_proxy

# Explicit proxy for outbound API calls (OpenAI, Qwen).
# Set HTTPS_PROXY in .env to e.g. http://127.0.0.1:7890 if behind a firewall.
# Leave empty to rely on system env (or no proxy).
HTTPS_PROXY: str = os.getenv("HTTPS_PROXY", "")

# ── 项目路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_pipeline" / "raw"
METADATA_DIR = RAW_DIR / "metadata"
FULLTEXT_DIR = RAW_DIR / "fulltext"
PROGRESS_FILE = RAW_DIR / "progress.json"

# 确保目录存在
for d in [METADATA_DIR, FULLTEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── NCBI ──────────────────────────────────────────────────────
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "your@email.com")
NCBI_TOOL: str = "longcovid_rag"

# API 基础 URL
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

# 搜索条件（和你在 PMC 网页上确认过的一致）
PMC_SEARCH_QUERY = (
    '("long covid"[Title/Abstract] OR "long-covid"[Title/Abstract])'
    ' AND (published_article[Filter])'
    ' AND (pmc_public[Filter])'
    ' AND (2020/1:2026/2[pdat])'
    ' AND (open_access[Filter])'
)

PMC_DATE_MIN = "2020/01/01"
PMC_DATE_MAX = "2026/03/01"

# 每批拉取数量（NCBI 建议 ≤500）
BATCH_SIZE = 200

# 速率限制：有 API key 可以 10 req/s，保守设为 0.15s 间隔
REQUEST_INTERVAL = 0.15   # seconds between requests

# 测试模式：True 时只拉取前 N 篇，验证完毕后改为 False 跑全量
TEST_MODE = False
TEST_LIMIT = 10

# 数据库配置
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_TTL: int = int(os.getenv("REDIS_SESSION_TTL", "86400"))  # 24h
FIELD_ENCRYPTION_KEY: str = os.getenv("FIELD_ENCRYPTION_KEY", "")
USER_MEMORIES_COLLECTION: str = "user_memories"

# 排除的论文类型(可扩充)
EXCLUDED_ARTICLE_TYPES = {"Erratum", "Published Erratum", "Retraction of Publication"}

# OpenAI：仅用于向量化（pipeline / retrieval 的 DENSE_MODEL），Agent 不用
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Qwen：Agent 全部使用（编排 + 问答 + 综述），模型 qwen3.5-plus
QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
QWEN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen3.5-plus")

# Qdrant（向量化写入与检索共用，必须一致）
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION: str = "longcovid_papers"          # 旧表，保留备份
QDRANT_COLLECTION_PC: str = "longcovid_papers_pc"    # 新表：段落级切片 + PubMedBERT

# 向量模型（写入 pipeline 与检索 retrieval 共用，必须一致）
DENSE_MODEL: str = "NeuML/pubmedbert-base-embeddings"  # 原 text-embedding-3-small
DENSE_DIM: int = 768                                   # PubMedBERT-base 输出维度
SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"

# Rerank 模型（Cross-Encoder，用于检索结果精排）
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Query 优化策略（off|hyde|decompose|auto）
QUERY_OPT_MODE: str = os.getenv("QUERY_OPT_MODE", "auto")
# 答案 grounding check：生成答案后用同一 Qwen 对断言逐一核查，标注无依据内容
GROUNDING_CHECK: bool = os.getenv("GROUNDING_CHECK", "true").lower() in ("1", "true", "yes")
# 复杂问题分解最大子查询数（建议 2-4）
QUERY_DECOMPOSE_MAX_SUBQUERIES: int = int(os.getenv("QUERY_DECOMPOSE_MAX_SUBQUERIES", "4"))

# 情感分析 API（摘要情绪分析，Agent 工具 analyze_sentiment 使用）
# 单条：POST SENTIMENT_API_SINGLE  body: {"pmcid": str, "text": str}
# 批量：POST SENTIMENT_API_BATCH   body: [{"pmcid": str, "text": str}, ...]
SENTIMENT_API_SINGLE: str = os.getenv("SENTIMENT_API_SINGLE", "http://localhost:8000/predict")
SENTIMENT_API_BATCH: str = os.getenv("SENTIMENT_API_BATCH", "http://localhost:8000/predict/batch")
SENTIMENT_API_TIMEOUT: int = int(os.getenv("SENTIMENT_API_TIMEOUT", "30"))
