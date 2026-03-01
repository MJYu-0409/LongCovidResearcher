"""
初始化临时执行文件
补救脚本：找出无 PMID 的 metadata，从 fetched 中移除并删除文件，便于重新拉取。
运行前请备份 progress.json 和 metadata 目录。
"""
from pathlib import Path
import json
from data_pipeline.fetcher.pmc_fetcher import fulltext_exists, metadata_exists


RAW_DIR = Path(__file__).parent.parent / "storage" / "raw"
METADATA_DIR = RAW_DIR / "metadata"
FULLTEXT_DIR = RAW_DIR / "fulltext"
PROGRESS_FILE = RAW_DIR / "progress.json"

def main():
    # 1. 扫描 metadata，找出无 PMID 的
    no_pmid_list = []
    for f in METADATA_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            if meta.get("no_pubmed_record") or meta.get("pmid") is None:
                no_pmid_list.append(f.stem)  # PMCID
        except Exception as e:
            print(f"读取失败 {f}: {e}")

    if not no_pmid_list:
        print("未发现无 PMID 的 metadata")
    else:
        print(f"发现 {len(no_pmid_list)} 篇无 PMID 的 metadata")


    # 2. 更新 progress.json - 无 PMID 的 PMCID 从 fetched 移除并加入 no_pmid
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        progress = json.load(f)

    fetched = set(progress.get("fetched_pmcids", []))
    failed = set(progress.get("failed_pmcids", []))
    no_pmid = set(progress.get("no_pmid_pmcids", []))

    for pmcid in no_pmid_list:
        fetched.discard(pmcid)
        no_pmid.add(pmcid)


    # 3.扫描 全文或者元数据拉取失败的
    no_fulltext_or_metadata_list = []
    for pmcid in list(fetched):
        if not fulltext_exists(pmcid) or not metadata_exists(pmcid):
            no_fulltext_or_metadata_list.append(pmcid)

    for pmcid in no_fulltext_or_metadata_list:
        fetched.discard(pmcid)
        failed.add(pmcid)

    if not no_fulltext_or_metadata_list:
        print("未发现 全文或元数据拉取失败的")
    else:
        print(f"发现 {len(no_fulltext_or_metadata_list)} 篇全文或元数据拉取失败的")

    progress["fetched_pmcids"] = list(fetched)
    progress["failed_pmcids"] = list(failed)
    progress["no_pmid_pmcids"] = list(no_pmid)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


    print("\n补救完成。")

if __name__ == "__main__":
    main()