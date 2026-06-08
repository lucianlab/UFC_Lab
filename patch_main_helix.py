"""
patch_main_helix.py — 在 main.py 末尾加入 HELIX 端點
用法：python3 ~/UFC/patch_main_helix.py ~/UFC/main.py
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("用法: python3 patch_main_helix.py ~/UFC/main.py")
    sys.exit(1)

target = Path(sys.argv[1])
if not target.exists():
    print(f"ERROR: 找不到 {target}")
    sys.exit(1)

# 備份
backup = target.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
shutil.copy2(target, backup)
print(f"備份: {backup}")

# 要加入的程式碼
HELIX_CODE = '''

# ══════════════════════════════════════════════════════
#  HELIX — Graph 資料端點
# ══════════════════════════════════════════════════════
HELIX_PATH = Path(__file__).parent / "data" / "helix_graph.json"
_HELIX = None

def _load_helix():
    global _HELIX
    if not HELIX_PATH.exists():
        print("WARNING: helix_graph.json 不存在，跳過")
        return
    with open(HELIX_PATH, encoding="utf-8") as f:
        _HELIX = json.load(f)
    print(f"HELIX loaded: {_HELIX['meta']['nodes']} nodes, {_HELIX['meta']['edges']} edges")

try:
    _load_helix()
except Exception as e:
    print(f"WARNING: HELIX not loaded: {e}")

@app.get("/api/helix/meta")
def get_helix_meta():
    """Graph 的 meta 資訊（nodes/edges 數量等）"""
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")
    return _HELIX["meta"]

@app.get("/api/helix/graph")
def get_helix_graph(ranked_only: bool = True):
    """
    取得 graph 資料
    ranked_only=true  → 只回傳曾排名的 node（預設，前端常駐）
    ranked_only=false → 全部 node
    edge 只回傳兩端 node 都在回傳集合內的
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    nodes = _HELIX["nodes"]
    edges = _HELIX["edges"]

    if ranked_only:
        nodes = [n for n in nodes if n.get("is_ranked")]

    node_ids = {n["id"] for n in nodes}
    edges = [
        e for e in edges
        if e["source"] in node_ids and e["target"] in node_ids
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }

@app.get("/api/helix/fighter/{name}")
def get_helix_fighter(name: str, depth: int = 1):
    """
    取得某選手的局部 graph（鄰居展開）
    depth=1 → 直接對手（預設）
    depth=2 → 對手的對手
    回傳包含 unranked 的完整鄰居
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    # 建立快速查詢 dict
    node_map = {n["id"]: n for n in _HELIX["nodes"]}

    if name not in node_map:
        raise HTTPException(status_code=404, detail=f"Fighter '{name}' not found in graph")

    # BFS 展開到指定 depth
    visited = {name}
    frontier = {name}

    all_edges = _HELIX["edges"]
    # 建立adjacency（雙向方便BFS）
    adj = {}
    for e in all_edges:
        adj.setdefault(e["source"], []).append(e["target"])
        adj.setdefault(e["target"], []).append(e["source"])

    for _ in range(min(depth, 2)):  # 最多 depth=2 避免爆量
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        frontier = next_frontier

    # 收集相關 nodes 和 edges
    result_nodes = [node_map[n] for n in visited if n in node_map]
    result_edges = [
        e for e in all_edges
        if e["source"] in visited and e["target"] in visited
    ]

    # beat chain：從 name 出發，只取出邊（他打贏的人）
    beat_chain = []
    for e in all_edges:
        if e["source"] == name:
            beat_chain.append({
                "opponent": e["target"],
                "bouts":    e["bouts"],
                "count":    e["count"],
                "weight":   e["weight"],
            })
    beat_chain.sort(key=lambda x: -x["weight"])

    # loss chain：輸給誰
    loss_chain = []
    for e in all_edges:
        if e["target"] == name:
            loss_chain.append({
                "opponent": e["source"],
                "bouts":    e["bouts"],
                "count":    e["count"],
            })

    return {
        "fighter":    node_map[name],
        "nodes":      result_nodes,
        "edges":      result_edges,
        "beat_chain": beat_chain,
        "loss_chain": loss_chain,
    }

@app.get("/api/helix/top")
def get_helix_top(n: int = 50, wc: str = None):
    """
    PageRank 前 n 名的選手 + 彼此之間的 edge
    wc: 按量級過濾（可選）
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    nodes = sorted(_HELIX["nodes"], key=lambda x: -x["pr"])

    if wc and wc.lower() != "all":
        # 用 beat_chain 的 wc 資訊過濾 edge，node 本身沒有固定量級
        # 這裡先回傳全量級的 top n，wc filter 留給前端
        pass

    top_nodes = nodes[:n]
    top_ids   = {nd["id"] for nd in top_nodes}

    top_edges = [
        e for e in _HELIX["edges"]
        if e["source"] in top_ids and e["target"] in top_ids
    ]

    return {
        "nodes":        top_nodes,
        "edges":        top_edges,
        "community_leaders": _HELIX.get("community_leaders", []),
    }
'''

# 讀原檔
content = target.read_text(encoding="utf-8")

# 確認沒有重複加入
if "HELIX" in content and "/api/helix" in content:
    print("WARNING: HELIX 端點已存在，不重複加入")
    sys.exit(0)

# 加在末尾
new_content = content.rstrip() + "\n" + HELIX_CODE + "\n"
target.write_text(new_content, encoding="utf-8")

print("✅ HELIX 端點加入成功")
print("\n新增端點：")
print("  GET /api/helix/meta              → graph 統計資訊")
print("  GET /api/helix/graph             → 全 graph（ranked_only 參數）")
print("  GET /api/helix/fighter/{name}    → 單一選手局部展開")
print("  GET /api/helix/top?n=50          → PageRank 前 N 名")
print("\n下一步：")
print("  1. 本地先跑 build_helix.py")
print("  2. 確認 data/helix_graph.json 產出")
print("  3. git add -f data/helix_graph.json")
print("  4. git add main.py && git commit && git push")
