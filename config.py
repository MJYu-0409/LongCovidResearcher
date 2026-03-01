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
