
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import json

# 读取扫描报告
with open("eval/output/scan_report.json") as f:
    report = json.load(f)

fail_pmcids = [r["pmcid"] for r in report["reports"] if r["status"] == "FAIL"]
print(f"FAIL总数: {len(fail_pmcids)}")

from config import FULLTEXT_DIR

reasons = Counter()
samples = {}  # reason -> [pmcid示例]

for pmcid in fail_pmcids:
    xml_path = Path(FULLTEXT_DIR) / f"{pmcid}.xml"
    if not xml_path.exists():
        reasons["文件不存在"] += 1
        samples.setdefault("文件不存在", []).append(pmcid)
        continue
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        reasons["XML解析错误"] += 1
        samples.setdefault("XML解析错误", []).append(pmcid)
        continue

    body = root.find(".//body")
    if body is None:
        # 进一步区分：有没有body标签，还是结构完全不同
        all_tags = {el.tag.split("}")[-1] if "}" in el.tag else el.tag 
                    for el in root.iter()}
        if "body" not in all_tags:
            reasons["无body标签"] += 1
            samples.setdefault("无body标签", []).append(pmcid)
        else:
            reasons["body为空"] += 1
            samples.setdefault("body为空", []).append(pmcid)
        continue

    # body存在但解析出0段落
    secs = [el for el in body.iter() 
            if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "sec"]
    ps   = [el for el in body.iter()
            if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) == "p"]
    
    if not secs and not ps:
        reasons["body下无sec也无p"] += 1
        samples.setdefault("body下无sec也无p", []).append(pmcid)
    elif not secs:
        reasons["有p但无sec（段落不在sec内）"] += 1
        samples.setdefault("有p但无sec（段落不在sec内）", []).append(pmcid)
    else:
        reasons["sec存在但p为空或全被过滤"] += 1
        samples.setdefault("sec存在但p为空或全被过滤", []).append(pmcid)

print("\nFAIL原因分布：")
for reason, cnt in reasons.most_common():
    pct = cnt / len(fail_pmcids) * 100
    print(f"  {reason}: {cnt}篇 ({pct:.1f}%)")
    for pmcid in samples.get(reason, [])[:3]:
        print(f"    示例: {pmcid}")

# 对"有p但无sec"的情况，看一下实际XML结构
no_sec_examples = samples.get("有p但无sec（段落不在sec内）", [])[:2]
for pmcid in no_sec_examples:
    xml_path = Path(FULLTEXT_DIR) / f"{pmcid}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    body = root.find(".//body")
    print(f"\n{pmcid} body直接子节点：")
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        print(f"  <{tag}>")
        for grandchild in child[:3]:
            gtag = grandchild.tag.split("}")[-1] if "}" in grandchild.tag else grandchild.tag
            print(f"    <{gtag}>")
