"""
config.py - 所有配置集中管理，不允许在其他文件 hardcode 任何参数
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── 项目路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_pipeline" / "storage" / "raw"
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
QDRANT_COLLECTION: str = "longcovid_papers"

# 向量模型（写入 pipeline 与检索 retrieval 共用，必须一致）
DENSE_MODEL: str = "text-embedding-3-small"
SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"

# Rerank 模型（Cross-Encoder，用于检索结果精排）
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 情感分析 API（摘要情绪分析，Agent 工具 analyze_sentiment 使用）
# 单条：POST SENTIMENT_API_SINGLE  body: {"pmcid": str, "text": str}
# 批量：POST SENTIMENT_API_BATCH   body: [{"pmcid": str, "text": str}, ...]
SENTIMENT_API_SINGLE: str = os.getenv("SENTIMENT_API_SINGLE", "http://localhost:8000/predict")
SENTIMENT_API_BATCH: str = os.getenv("SENTIMENT_API_BATCH", "http://localhost:8000/predict/batch")
SENTIMENT_API_TIMEOUT: int = int(os.getenv("SENTIMENT_API_TIMEOUT", "30"))