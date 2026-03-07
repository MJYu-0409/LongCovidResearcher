"""
eval/scan_parser_quality.py

批量扫描全文解析质量，发现系统性问题。
不需要人工抽查，统计手段覆盖全部 4500+ 篇。

检测项：
  1. 解析完全失败（0个段落）
  2. 段落数异常少（< 5条，可能漏解析）
  3. 平均 chunk 长度分布（极短说明内容被截断）
  4. 缺少核心 section（Introduction/Results/Discussion/Methods）
  5. list 提取异常（可能遗留拼接乱码）

运行：
  python eval/scan_parser_quality.py
  python eval/scan_parser_quality.py --fulltext-dir /path/to/xml/dir
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 保证从项目根可找到 config（支持 python eval/scan_parser_quality.py 或 -m eval.scan_parser_quality）
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.WARNING)

from config import FULLTEXT_DIR
from data_pipeline.processor.xml_parser import parse_fulltext_xml
from data_pipeline.processor.chunker import chunk_fulltext

# 核心 section 关键词（论文里大概率存在的）
CORE_SECTIONS = {"introduction", "methods", "results", "discussion", "conclusion"}

# 异常阈值
THRESHOLD_EMPTY      = 0      # 段落数 = 0 → 完全失败
THRESHOLD_SPARSE     = 5      # 段落数 < 5 → 异常稀少
THRESHOLD_SHORT_CHAR = 100    # chunk 平均字符 < 100 → 内容极短
THRESHOLD_CONCAT     = 3      # 同一 text 里分号数 > N 且长度 < 300 → 疑似拼接乱码


def _detect_concat_garbage(text: str) -> bool:
    """检测疑似 list 拼接乱码：短文本里出现大量分号或无空格连续大写词。"""
    if len(text) > 400:
        return False
    # 多个分号分隔的短片段
    if text.count(";") > THRESHOLD_CONCAT:
        return True
    # 无空格连接的连续词（如 "CoronaVacWIBP‐CorV"）
    if re.search(r'[a-z][A-Z][a-z]', text) and len(text.split()) < 5:
        return True
    return False


def scan_file(xml_path: Path) -> dict:
    """扫描单篇文件，返回质量报告。"""
    pmcid = xml_path.stem

    paragraphs = parse_fulltext_xml(xml_path)
    chunks = chunk_fulltext(pmcid, paragraphs) if paragraphs else []

    # section 名称归一化（小写，去掉数字编号）
    sections_found = set()
    for p in paragraphs:
        sec = re.sub(r'^\d+[\.\s]*', '', p.get("section", "")).lower().strip()
        for core in CORE_SECTIONS:
            if core in sec:
                sections_found.add(core)

    # chunk 平均长度
    avg_chunk_len = (
        sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0
    )

    # 乱码检测
    garbage_chunks = [
        c for c in chunks if _detect_concat_garbage(c["text"])
    ]

    return {
        "pmcid":             pmcid,
        "para_count":        len(paragraphs),
        "chunk_count":       len(chunks),
        "avg_chunk_len":     round(avg_chunk_len),
        "sections_found":    sorted(sections_found),
        "missing_core":      sorted(CORE_SECTIONS - sections_found),
        "garbage_count":     len(garbage_chunks),
        "garbage_examples":  [c["text"][:100] for c in garbage_chunks[:2]],
        # 状态分级
        "status": (
            "FAIL"    if len(paragraphs) == 0 else
            "SPARSE"  if len(paragraphs) < THRESHOLD_SPARSE else
            "GARBAGE" if garbage_chunks else
            "OK"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulltext-dir", type=Path, default=FULLTEXT_DIR)
    parser.add_argument("--output",       type=Path,
                        default=Path("eval/output/scan_report.json"))
    parser.add_argument("--limit",        type=int, default=None,
                        help="限制扫描篇数（调试用）")
    args = parser.parse_args()

    xml_files = sorted(args.fulltext_dir.glob("*.xml"))
    if args.limit:
        xml_files = xml_files[:args.limit]
    total = len(xml_files)

    print(f"开始扫描 {total} 篇全文 XML…")

    reports     = []
    status_cnt  = Counter()
    missing_cnt = Counter()  # 缺失 core section 的统计
    chunk_lens  = []

    for i, xml_path in enumerate(xml_files, 1):
        r = scan_file(xml_path)
        reports.append(r)
        status_cnt[r["status"]] += 1
        for sec in r["missing_core"]:
            missing_cnt[sec] += 1
        if r["avg_chunk_len"] > 0:
            chunk_lens.append(r["avg_chunk_len"])

        if i % 500 == 0:
            print(f"  进度：{i}/{total}")

    # ── 统计报告 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"解析质量扫描报告（共 {total} 篇）")
    print(f"{'='*60}")

    print(f"\n状态分布：")
    for status in ["OK", "SPARSE", "GARBAGE", "FAIL"]:
        cnt = status_cnt[status]
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {status:8s}: {cnt:5d} 篇 ({pct:5.1f}%)  {bar}")

    if chunk_lens:
        chunk_lens.sort()
        n = len(chunk_lens)
        print(f"\nChunk 平均长度分布：")
        print(f"  中位数: {chunk_lens[n//2]} 字符")
        print(f"  P10:    {chunk_lens[n//10]} 字符  （最短10%的文章）")
        print(f"  P90:    {chunk_lens[n*9//10]} 字符  （最长10%的文章）")
        short = sum(1 for l in chunk_lens if l < THRESHOLD_SHORT_CHAR)
        print(f"  平均长度 < {THRESHOLD_SHORT_CHAR} 字符的文章: {short} 篇")

    print(f"\n核心 section 缺失情况（可能正常，部分文章确实没有该 section）：")
    for sec, cnt in sorted(missing_cnt.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f"  缺少 {sec:15s}: {cnt:5d} 篇 ({pct:5.1f}%)")

    # 列出 FAIL 和 GARBAGE 的前20个
    fail_list    = [r for r in reports if r["status"] == "FAIL"]
    garbage_list = [r for r in reports if r["status"] == "GARBAGE"]

    if fail_list:
        print(f"\n完全失败的文章（前20）：")
        for r in fail_list[:20]:
            print(f"  {r['pmcid']}")

    if garbage_list:
        print(f"\n疑似乱码的文章（前10）：")
        for r in garbage_list[:10]:
            print(f"  {r['pmcid']}  示例: {r['garbage_examples'][0][:80]}")

    # ── 保存 ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "total":        total,
            "status_count": dict(status_cnt),
            "missing_core": dict(missing_cnt),
            "reports":      reports,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n完整报告已保存到 {args.output}")
    print(f"\n建议：")
    fail_pct    = status_cnt["FAIL"]    / total * 100
    garbage_pct = status_cnt["GARBAGE"] / total * 100
    if fail_pct > 5:
        print(f"  ⚠ {fail_pct:.1f}% 完全失败，需修复 parser 再重建向量")
    if garbage_pct > 3:
        print(f"  ⚠ {garbage_pct:.1f}% 疑似乱码，parser 的 list 处理可能仍有漏网之鱼")
    if fail_pct <= 5 and garbage_pct <= 3:
        print(f"  ✓ 整体质量可接受，可以继续进行检索评估")


if __name__ == "__main__":
    main()
