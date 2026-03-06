"""
补救 Qdrant：把 abstract 点的 pub_year、journal 填到同 pmcid 的 fulltext 点里。
在项目根目录执行：python scripts/backfill_qdrant_metadata.py
"""
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import sys
from pathlib import Path

import sys
from pathlib import Path

# 把项目根目录加入 path，便于从任意位置运行脚本
_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

COLLECTION = QDRANT_COLLECTION
BATCH = 500


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    # 1) 扫 abstract，建 pmcid -> (pub_year, journal)
    meta_by_pmcid = {}
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="abstract"))]),
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            pmcid = p.payload.get("pmcid")
            if pmcid:
                meta_by_pmcid[pmcid] = {
                    "pub_year": p.payload.get("pub_year") or "",
                    "journal": p.payload.get("journal") or "",
                }
        if offset is None:
            break

    print(f"从 abstract 得到 {len(meta_by_pmcid)} 个 pmcid 的 pub_year/journal")

    # 2) 扫 fulltext，收集 (point_id, pub_year, journal)
    to_update = []  # (point_id, pub_year, journal)
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="fulltext"))]),
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            pmcid = p.payload.get("pmcid")
            if not pmcid:
                continue
            meta = meta_by_pmcid.get(pmcid)
            if meta:
                to_update.append((p.id, meta["pub_year"], meta["journal"]))
        if offset is None:
            break

    print(f"待更新 payload 的 fulltext 点数: {len(to_update)}")

    # 3) 按 (pub_year, journal) 分组，批量 set_payload
    group = defaultdict(list)  # (pub_year, journal) -> [point_id, ...]
    for point_id, py, j in to_update:
        group[(py, j)].append(point_id)

    for (pub_year, journal), ids in group.items():
        client.set_payload(COLLECTION, {"pub_year": pub_year, "journal": journal}, points=ids)

    print(f"已按 {len(group)} 组更新 pub_year/journal，完成")
    return 0


if __name__ == "__main__":
    exit(main())