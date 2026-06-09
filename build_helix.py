"""
build_helix.py — HELIX Graph 資料建構腳本
用法：python3 ~/UFC/build_helix.py
輸出：~/UFC/data/helix_graph.json
"""

import json, math
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

# ── 路徑 ──────────────────────────────────────────────────────
BASE         = Path.home() / "UFC"
FIGHTS_CSV   = BASE / "data" / "clean" / "fights_all_rounds.csv"
RANK_CSV     = BASE / "data" / "clean" / "rankings_history.csv"
FIGHTERS_CSV = BASE / "data" / "clean" / "fighters_all.csv"
CHAMP_CSV    = BASE / "data" / "raw"   / "ufc_championship_history.csv"
OUT_PATH     = BASE / "data" / "helix_graph.json"

MIN_YEAR    = 1994   # 完整資料
EDGE_WEIGHT = {"KO/TKO":1.5, "Submission":1.5, "Decision":1.0, "Other":1.0}

print("=" * 60)
print("HELIX build_helix.py 開始")
print("=" * 60)

# ── 1. Fights ────────────────────────────────────────────────
print("\n[1] 讀取 fights_all_rounds.csv ...")
fights_raw = pd.read_csv(FIGHTS_CSV)
print(f"    原始 rows: {len(fights_raw):,}")
fights_raw = fights_raw[fights_raw["won"].notna() & fights_raw["winner"].notna()]
fights_raw["date"] = pd.to_datetime(fights_raw["date"], errors="coerce")
fights_raw = fights_raw[fights_raw["date"].dt.year >= MIN_YEAR]
print(f"    {MIN_YEAR} 後 rows: {len(fights_raw):,}")

fights = fights_raw[fights_raw["won"]==1.0].drop_duplicates(subset=["bout","date"]).copy()
print(f"    Aggregate 後 fights: {len(fights):,}")

def classify_method(m):
    if pd.isna(m): return "Other"
    m=str(m)
    if "KO" in m or "TKO" in m: return "KO/TKO"
    if "Sub" in m: return "Submission"
    if "Decision" in m: return "Decision"
    return "Other"

fights["method_clean"] = fights["method"].apply(classify_method)
fights["wc_clean"] = fights["weightclass"].fillna("Unknown").str.replace(r"\s*Bout$","",regex=True).str.strip()

# ── 2. Rankings ──────────────────────────────────────────────
print("\n[2] 讀取 rankings_history.csv ...")
try:
    rank_df = pd.read_csv(RANK_CSV)
    ever_ranked = set(rank_df["fighter"].dropna().unique())
    print(f"    曾排名: {len(ever_ranked):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    ever_ranked = set()

# ── 3. Championship history ───────────────────────────────────
print("\n[3] 讀取 ufc_championship_history.csv ...")
current_champs  = set()
former_champs   = set()
try:
    champ_df = pd.read_csv(CHAMP_CSV)
    for _, row in champ_df.iterrows():
        name = str(row.get("champion","")).strip()
        if not name or name=="nan": continue
        incumbent = str(row.get("incumbent","")).strip().upper()
        if incumbent == "TRUE":
            current_champs.add(name)
        else:
            former_champs.add(name)
    # current champ 不重複出現在 former
    former_champs -= current_champs
    print(f"    現任冠軍: {len(current_champs):,} 人")
    print(f"    前冠軍:   {len(former_champs):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")

# ── 4. fighters_all ──────────────────────────────────────────
print("\n[4] 讀取 fighters_all.csv ...")
try:
    fa_df   = pd.read_csv(FIGHTERS_CSV)
    fa_dict = {row["name"]: row.to_dict() for _, row in fa_df.iterrows()}
    print(f"    選手屬性: {len(fa_dict):,} 人")
except Exception as e:
    print(f"    WARNING: {e}")
    fa_dict = {}

# 讀 fighters.json 取得 archetype（PCA 八象限）
import json as _json
_fj_path = BASE / "data" / "fighters.json"
archetype_dict = {}
try:
    _fj = _json.load(open(_fj_path, encoding='utf-8'))
    archetype_dict = {f['name']: f.get('archetype','') for f in _fj}
    print(f"    archetype: {sum(1 for v in archetype_dict.values() if v):,} 人有資料")
except Exception as e:
    print(f"    WARNING archetype: {e}")

# ── 5. Build Directed Graph (loser → winner) ─────────────────
print("\n[5] 建立 Directed Graph ...")
G = nx.DiGraph()
for _, row in fights.iterrows():
    winner = row["fighter"]
    loser  = row["opponent"]
    method = row["method_clean"]
    wc     = row["wc_clean"]
    date   = str(row["date"])[:10]
    weight = EDGE_WEIGHT.get(method, 1.0)
    if G.has_edge(loser, winner):
        G[loser][winner]["weight"] += weight
        G[loser][winner]["count"]  += 1
        G[loser][winner]["bouts"].append({"date":date,"method":method,"wc":wc})
    else:
        G.add_edge(loser, winner, weight=weight, count=1, bouts=[{"date":date,"method":method,"wc":wc}])
    if winner not in G: G.add_node(winner)
    if loser  not in G: G.add_node(loser)
print(f"    Nodes: {G.number_of_nodes():,}")
print(f"    Edges: {G.number_of_edges():,}")

# ── 6. PageRank ──────────────────────────────────────────────
print("\n[6] 計算 PageRank ...")
pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
pr_vals  = np.array(list(pagerank.values()))
pr_min, pr_max = pr_vals.min(), pr_vals.max()
def norm_pr(v):
    if pr_max==pr_min: return 50.0
    return round((v-pr_min)/(pr_max-pr_min)*100, 2)
pagerank_norm = {n: norm_pr(v) for n,v in pagerank.items()}
pr_ranked   = sorted(pagerank.items(), key=lambda x: -x[1])
pr_rank_map = {name:i+1 for i,(name,_) in enumerate(pr_ranked)}
print("    Top 10 PageRank:")
for name,score in pr_ranked[:10]:
    tier = "CHAMP" if name in current_champs else ("FORMER" if name in former_champs else ("RANKED" if name in ever_ranked else ""))
    print(f"      #{pr_rank_map[name]:4d}  [{tier:6s}]  {name}  ({score:.6f})")

# ── 7. Community Detection ───────────────────────────────────
print("\n[7] Community Detection ...")
community_map = {}
comm_leaders  = {}
try:
    from networkx.algorithms import community as nx_comm
    G_u = G.to_undirected()
    communities = nx_comm.louvain_communities(G_u, seed=42)
    for i,comm in enumerate(communities):
        for node in comm: community_map[node]=i
    for i,comm in enumerate(communities):
        comm_leaders[i] = max(comm, key=lambda n: pagerank.get(n,0))
    print(f"    發現 {len(communities)} 個社群")
    for i,comm in enumerate(communities):
        if len(comm)>10:
            print(f"    社群 {i:2d} ({len(comm):4d} 人) 代表: {comm_leaders[i]}")
except Exception as e:
    print(f"    WARNING: {e}")

# ── 8. Style label ───────────────────────────────────────────
def get_style(name):
    arch = archetype_dict.get(name, '')
    if arch: return arch
    # fallback if not in fighters.json
    fa  = fa_dict.get(name,{})
    td  = float(fa.get("td_avg",0) or 0)
    sub = float(fa.get("sub_avg",0) or 0)
    spl = float(fa.get("splm",0) or 0)
    acc = float(fa.get("str_acc",0) or 0)
    if sub>1.5:              return "Submission Hunter"
    if td>3.0:               return "Chain Controller"
    if td>1.5 and sub>0.5:   return "Submission Hunter"
    if spl>5.0 and acc>50:   return "Pace Setter"
    if spl>4.0:              return "Pressure Swarm"
    return "Range Technician" 

# ── 9. Node tier（4層）───────────────────────────────────────
def get_tier(name):
    if name in current_champs: return "champion"
    if name in former_champs:  return "former_champion"
    if name in ever_ranked:    return "ever_ranked"
    return "unranked"

# ── 10. Assemble nodes ───────────────────────────────────────
print("\n[8] 組裝 nodes ...")
nodes = []
for name in G.nodes():
    fa       = fa_dict.get(name,{})
    pr_norm  = pagerank_norm.get(name,0)
    pr_rank  = pr_rank_map.get(name,9999)
    comm_id  = community_map.get(name,-1)
    tier     = get_tier(name)

    def sf(key):
        v=fa.get(key)
        try:
            f=float(v); return None if math.isnan(f) else f
        except: return None

    nodes.append({
        "id":           name,
        "pr":           pr_norm,
        "pr_rank":      pr_rank,
        "community":    comm_id,
        "tier":         tier,
        "archetype":    get_style(name),
        "style":        get_style(name),
        "wins_graph":   len(list(G.in_edges(name))),
        "losses_graph": len(list(G.out_edges(name))),
        "stance":       str(fa.get("stance","") or ""),
        "reach_cm":     sf("reach_cm"),
        "height_cm":    sf("height_cm"),
    })

tier_counts = {}
for n in nodes:
    tier_counts[n["tier"]] = tier_counts.get(n["tier"],0)+1
print(f"    總 nodes: {len(nodes):,}")
for t,cnt in tier_counts.items():
    print(f"    {t}: {cnt}")

# ── 11. Assemble edges ───────────────────────────────────────
print("\n[9] 組裝 edges ...")
edges = []
for u,v,data in G.edges(data=True):
    edges.append({"winner":v,"loser":u,"weight":round(data["weight"],2),"count":data["count"],"bouts":data["bouts"]})
print(f"    總 edges: {len(edges):,}")

# ── 12. Output ───────────────────────────────────────────────
print("\n[10] 輸出 helix_graph.json ...")
output = {
    "meta": {
        "nodes":len(nodes),"edges":len(edges),
        "communities":len(comm_leaders),"min_year":MIN_YEAR,
        "generated":str(pd.Timestamp.now())[:19],
    },
    "community_leaders":[{"community":k,"leader":v} for k,v in comm_leaders.items()],
    "nodes":nodes,"edges":edges,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH,"w",encoding="utf-8") as f:
    json.dump(output,f,ensure_ascii=False,separators=(",",":"))
size_mb = OUT_PATH.stat().st_size/1024/1024
print(f"    輸出完成: {OUT_PATH}")
print(f"    檔案大小: {size_mb:.2f} MB")
print("\n" + "="*60)
print("完成！")
print("  git add -f data/helix_graph.json")
print("  git commit -m 'helix: full history 1994, 4-tier nodes'")
print("  git push origin master")
print("="*60)
