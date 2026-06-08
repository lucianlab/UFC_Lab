"""
build_helix.py — HELIX Graph 資料建構腳本
用法：python3 ~/UFC/build_helix.py
輸出：~/UFC/data/helix_graph.json
"""

import json
import math
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

# ── 路徑 ──────────────────────────────────────────────────────
BASE         = Path.home() / "UFC"
FIGHTS_CSV   = BASE / "data" / "clean" / "fights_all_rounds.csv"
RANK_CSV     = BASE / "data" / "clean" / "rankings_history.csv"   # 修正路徑
FIGHTERS_CSV = BASE / "data" / "clean" / "fighters_all.csv"
OUT_PATH     = BASE / "data" / "helix_graph.json"

# ── 參數 ──────────────────────────────────────────────────────
MIN_YEAR    = 2013
EDGE_WEIGHT = {
    "KO/TKO":     1.5,
    "Submission": 1.5,
    "Decision":   1.0,
    "Other":      1.0,
}

print("=" * 60)
print("HELIX build_helix.py 開始")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. 讀 fights，aggregate 成每場一筆
# ══════════════════════════════════════════════════════════════
print("\n[1] 讀取 fights_all_rounds.csv ...")
fights_raw = pd.read_csv(FIGHTS_CSV)
print(f"    原始 rows: {len(fights_raw):,}")

fights_raw = fights_raw[fights_raw["won"].notna()]
fights_raw = fights_raw[fights_raw["winner"].notna()]

fights_raw["date"] = pd.to_datetime(fights_raw["date"], errors="coerce")
fights_raw = fights_raw[fights_raw["date"].dt.year >= MIN_YEAR]
print(f"    {MIN_YEAR} 後 rows: {len(fights_raw):,}")

# 每場只取勝者那筆
fights = (
    fights_raw[fights_raw["won"] == 1.0]
    .drop_duplicates(subset=["bout", "date"])
    .copy()
)
print(f"    Aggregate 後 fights: {len(fights):,}")

def classify_method(m):
    if pd.isna(m): return "Other"
    m = str(m)
    if "KO" in m or "TKO" in m: return "KO/TKO"
    if "Sub" in m: return "Submission"
    if "Decision" in m: return "Decision"
    return "Other"

fights["method_clean"] = fights["method"].apply(classify_method)
fights["wc_clean"] = (
    fights["weightclass"]
    .fillna("Unknown")
    .str.replace(r"\s*Bout$", "", regex=True)
    .str.strip()
)

# ══════════════════════════════════════════════════════════════
# 2. 讀 rankings_history → 標記曾排名選手
# ══════════════════════════════════════════════════════════════
print("\n[2] 讀取 rankings_history.csv ...")
try:
    rank_df = pd.read_csv(RANK_CSV)
    ranked_fighters = set(rank_df["fighter"].dropna().unique())
    print(f"    曾排名選手: {len(ranked_fighters):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    ranked_fighters = set()

# ══════════════════════════════════════════════════════════════
# 3. 讀 fighters_all → node 屬性
# ══════════════════════════════════════════════════════════════
print("\n[3] 讀取 fighters_all.csv ...")
try:
    fa_df  = pd.read_csv(FIGHTERS_CSV)
    fa_dict = {row["name"]: row.to_dict() for _, row in fa_df.iterrows()}
    print(f"    選手屬性: {len(fa_dict):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    fa_dict = {}

# ══════════════════════════════════════════════════════════════
# 4. 建立 Directed Graph
#
# 方向修正：loser → winner
# 意義：「聲望從輸家流向贏家」，跟 Google PageRank 的連結邏輯一致
# Khabib 打贏 Conor → Conor 的節點指向 Khabib → Khabib 分數高
# ══════════════════════════════════════════════════════════════
print("\n[4] 建立 Directed Graph（loser → winner）...")
G = nx.DiGraph()

for _, row in fights.iterrows():
    winner = row["fighter"]
    loser  = row["opponent"]
    method = row["method_clean"]
    wc     = row["wc_clean"]
    date   = str(row["date"])[:10]
    weight = EDGE_WEIGHT.get(method, 1.0)

    # loser → winner（聲望流向）
    if G.has_edge(loser, winner):
        G[loser][winner]["weight"] += weight
        G[loser][winner]["count"]  += 1
        G[loser][winner]["bouts"].append({"date": date, "method": method, "wc": wc})
    else:
        G.add_edge(loser, winner, weight=weight, count=1, bouts=[
            {"date": date, "method": method, "wc": wc}
        ])
    # 確保兩個 node 都存在
    if winner not in G: G.add_node(winner)
    if loser  not in G: G.add_node(loser)

print(f"    Nodes: {G.number_of_nodes():,}")
print(f"    Edges: {G.number_of_edges():,}")

# ══════════════════════════════════════════════════════════════
# 5. PageRank
# ══════════════════════════════════════════════════════════════
print("\n[5] 計算 PageRank ...")
pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

pr_vals = np.array(list(pagerank.values()))
pr_min, pr_max = pr_vals.min(), pr_vals.max()

def normalize_pr(v):
    if pr_max == pr_min: return 50.0
    return round((v - pr_min) / (pr_max - pr_min) * 100, 2)

pagerank_norm = {n: normalize_pr(v) for n, v in pagerank.items()}
pr_ranked     = sorted(pagerank.items(), key=lambda x: -x[1])
pr_rank_map   = {name: i+1 for i, (name, _) in enumerate(pr_ranked)}

print(f"    Top 10 PageRank:")
for name, score in pr_ranked[:10]:
    is_r = "★" if name in ranked_fighters else " "
    print(f"      {is_r} #{pr_rank_map[name]:4d}  {name}  ({score:.6f})")

# ══════════════════════════════════════════════════════════════
# 6. Community Detection
# ══════════════════════════════════════════════════════════════
print("\n[6] Community Detection ...")
try:
    from networkx.algorithms import community as nx_comm
    G_undirected = G.to_undirected()
    communities  = nx_comm.louvain_communities(G_undirected, seed=42)
    community_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_map[node] = i
    print(f"    發現 {len(communities)} 個社群")
    comm_leaders = {}
    for i, comm in enumerate(communities):
        leader = max(comm, key=lambda n: pagerank.get(n, 0))
        comm_leaders[i] = leader
        print(f"    社群 {i:2d} ({len(comm):4d} 人) 代表: {leader}")
except Exception as e:
    print(f"    WARNING: Community detection 失敗: {e}")
    community_map = {}
    comm_leaders  = {}

# ══════════════════════════════════════════════════════════════
# 7. 風格標籤
# ══════════════════════════════════════════════════════════════
def get_style_label(name):
    fa = fa_dict.get(name, {})
    td  = float(fa.get("td_avg",  0) or 0)
    sub = float(fa.get("sub_avg", 0) or 0)
    spl = float(fa.get("splm",    0) or 0)
    acc = float(fa.get("str_acc", 0) or 0)
    if sub > 1.5:              return "Submission"
    if td  > 3.0:              return "Wrestler"
    if td  > 1.5 and sub > 0.5: return "Grappler"
    if spl > 5.0 and acc > 50: return "Striker"
    if spl > 4.0:              return "Brawler"
    return "Balanced"

# ══════════════════════════════════════════════════════════════
# 8. 組裝 nodes
# ══════════════════════════════════════════════════════════════
print("\n[7] 組裝 nodes ...")
nodes = []
for name in G.nodes():
    fa        = fa_dict.get(name, {})
    pr_norm   = pagerank_norm.get(name, 0)
    pr_rank   = pr_rank_map.get(name, 9999)
    comm_id   = community_map.get(name, -1)
    is_ranked = name in ranked_fighters

    # 出邊 = 「被他打敗的人指向他」的出邊（在 loser→winner graph 裡，winner 是被指向的）
    # 實際勝場：在原始 fights 裡 fighter==name and won==1
    wins_count   = len([e for u, v in G.in_edges(name) for e in [1]])  # 入邊 = 贏
    losses_count = len(list(G.out_edges(name)))                         # 出邊 = 輸

    def safe_float_fa(key):
        v = fa.get(key)
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    nodes.append({
        "id":           name,
        "pr":           pr_norm,
        "pr_rank":      pr_rank,
        "community":    comm_id,
        "is_ranked":    is_ranked,
        "style":        get_style_label(name),
        "wins_graph":   wins_count,
        "losses_graph": losses_count,
        "stance":       str(fa.get("stance", "") or ""),
        "reach_cm":     safe_float_fa("reach_cm"),
        "height_cm":    safe_float_fa("height_cm"),
    })

print(f"    總 nodes: {len(nodes):,}")
print(f"    曾排名:   {sum(1 for n in nodes if n['is_ranked']):,}")
ranked_nodes = sorted([n for n in nodes if n["is_ranked"]], key=lambda x: x["pr_rank"])
print(f"    排名選手 Top 10 PageRank:")
for n in ranked_nodes[:10]:
    print(f"      #{n['pr_rank']:4d}  {n['id']}  (pr={n['pr']:.1f})")

# ══════════════════════════════════════════════════════════════
# 9. 組裝 edges（保留原始語意：winner 打贏 loser）
# 注意：JSON 裡 source/target 用業務語意，不是 graph 方向
# source = winner, target = loser（方便前端顯示 beat chain）
# ══════════════════════════════════════════════════════════════
print("\n[8] 組裝 edges ...")
edges = []
for u, v, data in G.edges(data=True):
    # graph 裡 u=loser, v=winner → 前端用 winner/loser 更直觀
    edges.append({
        "winner": v,
        "loser":  u,
        "weight": round(data["weight"], 2),
        "count":  data["count"],
        "bouts":  data["bouts"],
    })
print(f"    總 edges: {len(edges):,}")

# ══════════════════════════════════════════════════════════════
# 10. 輸出 JSON
# ══════════════════════════════════════════════════════════════
print("\n[9] 輸出 helix_graph.json ...")

output = {
    "meta": {
        "nodes":       len(nodes),
        "edges":       len(edges),
        "communities": len(comm_leaders),
        "min_year":    MIN_YEAR,
        "generated":   str(pd.Timestamp.now())[:19],
    },
    "community_leaders": [
        {"community": k, "leader": v} for k, v in comm_leaders.items()
    ],
    "nodes": nodes,
    "edges": edges,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"    輸出完成: {OUT_PATH}")
print(f"    檔案大小: {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("完成！下一步：")
print("  git add -f data/helix_graph.json")
print("  git commit -m 'Add HELIX graph data'")
print("  git push origin master")
print("=" * 60)
