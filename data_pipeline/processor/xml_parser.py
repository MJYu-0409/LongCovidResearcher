"""
data_pipeline/processor/xml_parser.py

解析 PMC JATS 格式全文 XML，提取干净的结构化文本。
输出格式：list[dict]，每条代表一个段落或 caption，包含：
  {
    "section": "Introduction",   # section 标题
    "text": "干净的段落文本",
    "type": "paragraph" | "table_caption" | "figure_caption"
  }

处理原则：
  - 保留 <sec> 标题和 <p> 段落文本
  - 用 itertext() 递归提取文本，自动去除 <italic>/<bold>/<xref> 等格式标签
  - 跳过 <ref-list>（参考文献，噪音）
  - 表格只保留 caption，图片只保留 caption
  - 过滤空文本和过短文本（< 20 字符）
  - <list> 节点（bullet/ordered）每条 list-item 独立提取，支持嵌套多级列表
  - <p> 内嵌 <list> 时，分离前导文本与列表项，避免 itertext() 拼接乱码
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

# JATS XML 命名空间前缀（有些文件带 namespace，有些不带）
_NS = "{http://www.w3.org/1999/xlink}"

# 跳过的顶级 section 关键词（参考文献、致谢等噪音）
_SKIP_SECTIONS = {
    "references", "reference", "acknowledgements", "acknowledgments",
    "funding", "conflict of interest", "competing interests",
    "supplementary", "appendix", "author contributions",
}

# 最短有效文本长度
_MIN_TEXT_LEN = 20


def _clean_text(node) -> str:
    """用 itertext() 提取节点下所有文本，合并空白。"""
    return " ".join("".join(node.itertext()).split()).strip()


def _should_skip_section(title: str) -> bool:
    """判断该 section 是否应该跳过。"""
    return title.lower().strip() in _SKIP_SECTIONS


def _extract_list(list_node, section_title: str) -> list[dict]:
    """
    递归提取 <list> 节点下所有 <list-item> 的文本。
    支持嵌套多级列表（list-item 内还有 list）。
    每个 list-item 的所有 <p> 合并为一条 paragraph 记录。
    """
    results = []
    for item in list_node:
        item_tag = item.tag.split("}")[-1] if "}" in item.tag else item.tag
        if item_tag != "list-item":
            continue

        texts = []
        sub_results = []
        for child in item:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag == "p":
                extracted = _extract_paragraph(child, section_title)
                for e in extracted:
                    if e["type"] == "paragraph":
                        texts.append(e["text"])
                    else:
                        sub_results.append(e)
            elif child_tag == "list":
                sub_results.extend(_extract_list(child, section_title))

        combined = " ".join(texts)
        if len(combined) >= _MIN_TEXT_LEN:
            results.append({
                "section": section_title,
                "text":    combined,
                "type":    "paragraph",
            })
        results.extend(sub_results)

    return results


def _extract_paragraph(p_node, section_title: str) -> list[dict]:
    """
    处理单个 <p> 节点，兼容两种情况：
      1. 普通 <p>：直接提取全部文本。
      2. <p> 内嵌套 <list>（Cochrane 等文献常见）：
         前导文本单独保留，内嵌 <list> 交给 _extract_list 处理。
    避免 itertext() 把 list-item 文本无分隔地拼成乱码。
    """
    has_inner_list = any(
        (c.tag.split("}")[-1] if "}" in c.tag else c.tag) == "list"
        for c in p_node
    )

    if not has_inner_list:
        text = _clean_text(p_node)
        if len(text) >= _MIN_TEXT_LEN:
            return [{"section": section_title, "text": text, "type": "paragraph"}]
        return []

    # <p> 内嵌 <list>：先取前导文字，再递归处理 list
    results = []
    leading = (p_node.text or "").strip()
    if len(leading) >= _MIN_TEXT_LEN:
        results.append({"section": section_title, "text": leading, "type": "paragraph"})

    for child in p_node:
        child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if child_tag == "list":
            results.extend(_extract_list(child, section_title))

    return results


def _extract_section(sec_node, parent_title: str = "") -> list[dict]:
    """
    递归提取一个 <sec> 节点下的所有段落、表格 caption、图片 caption。
    parent_title 用于子 section 没有标题时继承父标题。
    """
    results = []

    title_el = sec_node.find("title")
    section_title = _clean_text(title_el) if title_el is not None else parent_title
    if not section_title:
        section_title = parent_title

    if _should_skip_section(section_title):
        return []

    for child in sec_node:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "title":
            continue  # 已处理

        elif tag == "p":
            results.extend(_extract_paragraph(child, section_title))

        elif tag == "sec":
            results.extend(_extract_section(child, parent_title=section_title))

        elif tag == "list":
            # <sec> 下直接挂的 <list>（bullet/ordered list 作为正文结构）
            results.extend(_extract_list(child, section_title))

        elif tag == "table-wrap":
            caption_el = child.find("caption")
            if caption_el is not None:
                text = _clean_text(caption_el)
                if len(text) >= _MIN_TEXT_LEN:
                    results.append({
                        "section": section_title,
                        "text":    text,
                        "type":    "table_caption",
                    })

        elif tag == "fig":
            caption_el = child.find("caption")
            if caption_el is not None:
                text = _clean_text(caption_el)
                if len(text) >= _MIN_TEXT_LEN:
                    results.append({
                        "section": section_title,
                        "text":    text,
                        "type":    "figure_caption",
                    })

    return results


def parse_fulltext_xml(xml_path: Path) -> list[dict]:
    """
    解析单篇论文的 JATS XML 全文，返回结构化段落列表。
    解析失败返回空列表，不抛出异常（由调用方决定如何处理）。

    Args:
        xml_path: fulltext/{PMCID}.xml 文件路径

    Returns:
        list[dict]，每条包含 section / text / type 三个字段
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.warning("XML 解析失败 %s: %s", xml_path.name, e)
        return []
    except OSError as e:
        logger.warning("文件读取失败 %s: %s", xml_path.name, e)
        return []

    body = root.find(".//body")
    if body is None:
        logger.debug("未找到 <body> 节点: %s", xml_path.name)
        return []

    results = []
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "sec":
            # 标准结构：body → sec → p
            results.extend(_extract_section(child))

        elif tag == "p":
            # 无 sec 结构（编辑信、评论等短篇）：body 直接挂 p
            results.extend(_extract_paragraph(child, "body"))

        elif tag == "list":
            # 无 sec 结构：body 直接挂 list
            results.extend(_extract_list(child, "body"))

        elif tag == "fig":
            # 无 sec 结构：body 直接挂 fig
            caption_el = child.find("caption")
            if caption_el is not None:
                text = _clean_text(caption_el)
                if len(text) >= _MIN_TEXT_LEN:
                    results.append({
                        "section": "body",
                        "text":    text,
                        "type":    "figure_caption",
                    })

        elif tag == "table-wrap":
            # 无 sec 结构：body 直接挂 table-wrap
            caption_el = child.find("caption")
            if caption_el is not None:
                text = _clean_text(caption_el)
                if len(text) >= _MIN_TEXT_LEN:
                    results.append({
                        "section": "body",
                        "text":    text,
                        "type":    "table_caption",
                    })

    if not results:
        logger.debug("未提取到任何段落: %s", xml_path.name)

    return results
