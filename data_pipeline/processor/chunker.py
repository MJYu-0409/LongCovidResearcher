"""
data_pipeline/processor/chunker.py

将 xml_parser 输出的段落列表切分为适合向量化的 chunk。

策略：
  - 摘要（abstract）整体不切，直接作为一个 chunk
  - 正文段落按 section 边界优先，在 section 内按 token 数合并
  - 单段落超过 MAX_TOKENS 时按句子边界切分
  - 每个 chunk 携带 section 标题和 chunk 在该 section 内的序号

输出格式：list[dict]
  {
    "pmcid":       "PMC9387111",
    "source_type": "abstract" | "fulltext",
    "section":     "Introduction",
    "chunk_index": 0,
    "text":        "chunk 文本内容",
  }
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# chunk token 上下限（用字符数粗估，1 token ≈ 4 字符）
MAX_TOKENS = 800
MIN_TOKENS = 20
MAX_CHARS = MAX_TOKENS * 4   # 3200
MIN_CHARS = MIN_TOKENS * 4   # 80


def _split_by_sentences(text: str, max_chars: int) -> list[str]:
    """按句子边界切分超长文本。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def _merge_paragraphs(paragraphs: list[str], max_chars: int, min_chars: int) -> list[str]:
    """
    把同一 section 的段落列表合并成若干 chunk。
    相邻段落合并，超过 max_chars 时开新 chunk。
    """
    chunks = []
    current = ""

    for para in paragraphs:
        # 单段落本身超长，先按句子切
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_sentences(para, max_chars))
            continue

        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    # 过滤过短的 chunk（可能是噪音）
    return [c for c in chunks if len(c) >= min_chars]


# def chunk_abstract(pmcid: str, abstract_text: str) -> list[dict]:
#     """
#     摘要整体作为一个 chunk，不切分。
#     如果摘要超长则按句子切，实际上摘要很少超过 MAX_CHARS。
#     """
#     if not abstract_text or not abstract_text.strip():
#         return []

#     text = abstract_text.strip()

#     if len(text) <= MAX_CHARS:
#         return [{
#             "pmcid":       pmcid,
#             "source_type": "abstract",
#             "section":     "abstract",
#             "chunk_index": 0,
#             "text":        text,
#         }]
#     else:
#         # 超长摘要（极少见）按句子切
#         parts = _split_by_sentences(text, MAX_CHARS)
#         return [
#             {
#                 "pmcid":       pmcid,
#                 "source_type": "abstract",
#                 "section":     "abstract",
#                 "chunk_index": i,
#                 "text":        part,
#             }
#             for i, part in enumerate(parts)
#         ]


def chunk_fulltext(pmcid: str, paragraphs: list[dict]) -> list[dict]:
    """
    将 xml_parser 返回的段落列表切分为正文 chunk。
    按 section 分组，每组内合并段落，超长时按句子切。

    处理原则：
      - paragraph 类型：同 section 内合并，尽量填满 MAX_CHARS
      - table_caption / figure_caption 类型：每条独立成一个 chunk，不合并
        （caption 彼此独立，合并会导致不相关标题混在一起干扰检索）

    Args:
        pmcid: 论文 PMC ID
        paragraphs: xml_parser.parse_fulltext_xml() 的返回值

    Returns:
        list[dict]，每条是一个 chunk
    """
    if not paragraphs:
        return []

    # 按 section 分组，保持顺序，同时区分 paragraph 和 caption
    # caption 单独处理，不参与 paragraph 的合并逻辑
    section_paragraphs: dict[str, list[str]] = {}  # section -> paragraph texts
    caption_items: list[dict] = []                  # caption 条目保持原始顺序

    for para in paragraphs:
        section = para.get("section", "unknown")
        text = para.get("text", "").strip()
        ptype = para.get("type", "paragraph")
        if not text:
            continue

        if ptype in ("table_caption", "figure_caption"):
            caption_items.append({"section": section, "text": text, "type": ptype})
        else:
            section_paragraphs.setdefault(section, []).append(text)

    results = []

    # 1. 处理正文段落：section 内合并，chunk_index 从 0 开始
    # 用 dict 记录每个 section 已分配到的最大 index，caption 继续往后排
    section_counters: dict[str, int] = {}

    for section, texts in section_paragraphs.items():
        merged = _merge_paragraphs(texts, MAX_CHARS, MIN_CHARS)
        for chunk_text in merged:
            idx = section_counters.get(section, 0)
            results.append({
                "pmcid":       pmcid,
                "source_type": "fulltext",
                "section":     section,
                "chunk_index": idx,
                "text":        chunk_text,
            })
            section_counters[section] = idx + 1

    # 2. 处理 caption：每条独立成 chunk，chunk_index 接续同 section 的 paragraph 之后
    #    避免与 paragraph chunk 产生相同 (section, chunk_index) 组合，导致 UUID 冲突
    for item in caption_items:
        if len(item["text"]) < MIN_CHARS:
            continue
        section = item["section"]
        idx = section_counters.get(section, 0)
        results.append({
            "pmcid":       pmcid,
            "source_type": "fulltext",
            "section":     section,
            "chunk_index": idx,
            "text":        item["text"],
        })
        section_counters[section] = idx + 1

    logger.debug(
        "%s 正文切分完成：%d 个 section，%d 个 chunk",
        pmcid, len(section_counters), len(results)
    )
    return results
