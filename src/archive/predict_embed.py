"""
UFC Matchup Prediction — Style Encoder v3
覆蓋: src/predict_embed.py

執行: python3 src/predict_embed.py
"""

import pandas as pd
import numpy as np
import json, pickle
import torch
import torch.nn as nn
from difflib import get_close_matches

DATA_PATH   = "data/processed/fights_clean.csv"
MODEL_PATH  = "models/model_embed.pt"
SCALER_PATH = "models/embed_scaler.pkl"
FEAT_PATH   = "models/embed_feature_cols.json"
EMBED_PATH  = "models/fighter_embeddings.npy"
ID_MAP_PATH = "models/fighter_id_map.json"

STANCE_MAP    = {"Orthodox":0,"Southpaw":1,"Switch":2,"Open Stance":2}
METHOD_LABELS = {0:"KO/TKO", 1:"Submission", 2:"Decision"}
STYLE_DIM     = 16

# ──────────────────────────────────────────────
# 模型定義（必須和 train_embed.py 完全一致）
# ──────────────────────────────────────────────
class StyleEncoder(nn.Module):
    def __init__(self, n_stats, style_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, style_dim), nn.Tanh(),
        )
    def forward(self, x): return self.net(x)

class UFCStyleModel(nn.Module):
    def __init__(self, n_stats, style_dim, n_delta):
        super().__init__()
        self.encoder = StyleEncoder(n_stats, style_dim)
        self.predictor = nn.Sequential(
            nn.Linear(style_dim*2+n_delta, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, r, b, d):
        return self.predictor(
            torch.cat([self.encoder(r), self.encoder(b), d], dim=1)
        ).squeeze(1)

# ──────────────────────────────────────────────
# 載入資源
# ──────────────────────────────────────────────
def load_resources():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    with open(FEAT_PATH) as f:
        feat_cols = json.load(f)
    FIGHTER_STATS  = feat_cols["fighter_stats"]
    DELTA_FEATURES = feat_cols["delta_features"]

    N_STATS = len(FIGHTER_STATS)
    N_DELTA = len(DELTA_FEATURES)

    model = UFCStyleModel(N_STATS, STYLE_DIM, N_DELTA)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scalers = pickle.load(f)

    embeddings     = np.load(EMBED_PATH)
    with open(ID_MAP_PATH) as f:
        fighter_to_idx = json.load(f)

    all_names = sorted(set(df["r_name"].tolist() + df["b_name"].tolist()))
    return df, model, scalers, embeddings, fighter_to_idx, all_names, FIGHTER_STATS, DELTA_FEATURES

# ──────────────────────────────────────────────
# 選手查詢
# ──────────────────────────────────────────────
def get_fighter_stats(name, df, FIGHTER_STATS):
    mask   = (df["r_name"]==name)|(df["b_name"]==name)
    fights = df[mask].sort_values("date")
    if len(fights) == 0: return None
    last = fights.iloc[-1]
    p    = "r_" if last["r_name"]==name else "b_"

    stats = {s: float(last[p+s]) if p+s in last.index else 0.0
             for s in FIGHTER_STATS}
    stats["stance"]     = last.get(p+"stance", "Orthodox")
    stats["stance_enc"] = STANCE_MAP.get(str(stats["stance"]), 0)
    stats["division"]   = last["division"]
    stats["n_fights"]   = len(fights)
    stats["last_fight"] = str(last["date"])[:10]
    stats["wins"]       = float(last[p+"wins"])
    stats["losses"]     = float(last[p+"losses"])
    stats["height"]     = float(last[p+"height"]) if p+"height" in last.index else 0.0

    # 標準化顯示用的原始數據
    stats["raw_splm"]    = float(last[p+"splm"])
    stats["raw_sapm"]    = float(last[p+"sapm"])
    stats["raw_str_def"] = float(last[p+"str_def"])
    stats["raw_td_avg"]  = float(last[p+"td_avg"])
    stats["raw_td_def"]  = float(last[p+"td_def"])
    stats["raw_sub_avg"] = float(last[p+"sub_avg"])
    stats["raw_momentum"] = float(last[p+"momentum"]) if p+"momentum" in last.index else 0.0
    stats["momentum"]     = stats["raw_momentum"]
    stats["splm_trend"]   = float(last[p+"splm_trend"]) if p+"splm_trend" in last.index else 0.0
    stats["last3_win_rate"] = float(last[p+"last3_win_rate"]) if p+"last3_win_rate" in last.index else 0.5
    stats["last3_splm"]   = float(last[p+"last3_splm"]) if p+"last3_splm" in last.index else 0.0
    stats["last5_win_rate"] = float(last[p+"last5_win_rate"]) if p+"last5_win_rate" in last.index else 0.5
    stats["last3_finish_rate"] = float(last[p+"last3_finish_rate"]) if p+"last3_finish_rate" in last.index else 0.3
    stats["last5_finish_rate"] = float(last[p+"last5_finish_rate"]) if p+"last5_finish_rate" in last.index else 0.3
    stats["last3_sapm"]   = float(last[p+"last3_sapm"]) if p+"last3_sapm" in last.index else 0.0
    stats["tko_rate"]     = float(last[p+"tko_rate"]) if p+"tko_rate" in last.index else 0.0
    stats["finish_rate"]  = float(last[p+"finish_rate"]) if p+"finish_rate" in last.index else 0.3
    stats["ko_vulnerability"] = float(last[p+"ko_vulnerability"]) if p+"ko_vulnerability" in last.index else 0.3
    stats["pure_ko_rate"] = float(last[p+"pure_ko_rate"]) if p+"pure_ko_rate" in last.index else 0.0
    return stats

def fuzzy_find(query, all_names):
    q = query.lower()
    for n in all_names:
        if n.lower() == q: return n
    partial = [n for n in all_names if q in n.lower()]
    if len(partial) == 1: return partial[0]
    if len(partial) > 1:
        print(f"  找到多個含 '{query}':")
        for i, n in enumerate(partial[:20]): print(f"    {i+1}. {n}")
        c = input("  請輸入編號: ").strip()
        try: return partial[int(c)-1]
        except: return None
    close = get_close_matches(query, all_names, n=5, cutoff=0.5)
    if close:
        print(f"  找不到 '{query}'，你是指：")
        for i, n in enumerate(close): print(f"    {i+1}. {n}")
        c = input("  請輸入編號 (或 Enter 取消): ").strip()
        if c:
            try: return close[int(c)-1]
            except: return None
    return None

# ──────────────────────────────────────────────
# 建立 feature vector
# ──────────────────────────────────────────────
def build_features(sa, sb, title_fight, scalers, FIGHTER_STATS, DELTA_FEATURES):
    r_raw = np.array([sa[s] for s in FIGHTER_STATS], dtype=np.float32).reshape(1,-1)
    b_raw = np.array([sb[s] for s in FIGHTER_STATS], dtype=np.float32).reshape(1,-1)
    r_sc  = scalers["stats"].transform(r_raw)
    b_sc  = scalers["stats"].transform(b_raw)

    # 計算所有可能用到的 delta / 交互值
    def eff_td(atk_td, def_td):   return atk_td * (1 - def_td/100)
    def eff_sub(atk_sub, def_td): return atk_sub * (1 - def_td/100)

    td_r   = sa.get("td_avg",  0); td_d_b  = sb.get("td_def",  50)
    td_b   = sb.get("td_avg",  0); td_d_r  = sa.get("td_def",  50)
    sub_r  = sa.get("sub_avg", 0); sub_b   = sb.get("sub_avg", 0)
    ko_r   = sa.get("pure_ko_rate", 0); sapm_b = sb.get("sapm", 3)
    ko_b   = sb.get("pure_ko_rate", 0); sapm_r = sa.get("sapm", 3)
    kov_r  = sa.get("ko_vulnerability", 0.3)
    kov_b  = sb.get("ko_vulnerability", 0.3)
    st_r   = sa.get("splm", 3) / (sa.get("td_avg", 0) + 0.5)
    st_b   = sb.get("splm", 3) / (sb.get("td_avg", 0) + 0.5)
    total_sub = sub_r + sub_b + 0.1

    feat_map = {
        "title_fight":           title_fight,
        "delta_wins":            sa["wins"]      - sb["wins"],
        "delta_losses":          sa["losses"]     - sb["losses"],
        "delta_win_rate":        sa.get("win_rate",0.5) - sb.get("win_rate",0.5),
        "delta_experience":      sa.get("experience",0) - sb.get("experience",0),
        "delta_splm":            sa.get("splm",0)       - sb.get("splm",0),
        "delta_sapm":            sa.get("sapm",0)       - sb.get("sapm",0),
        "delta_str_acc":         sa.get("str_acc",0)    - sb.get("str_acc",0),
        "delta_str_def":         sa.get("str_def",0)    - sb.get("str_def",0),
        "delta_td_avg":          td_r  - td_b,
        "delta_td_avg_acc":      sa.get("td_avg_acc",0) - sb.get("td_avg_acc",0),
        "delta_td_def":          sa.get("td_def",50)    - sb.get("td_def",50),
        "delta_sub_avg":         sub_r  - sub_b,
        "delta_height":          sa.get("height",0)     - sb.get("height",0),
        "delta_reach":           sa.get("reach",0)      - sb.get("reach",0),
        "southpaw_matchup":      int((sa["stance_enc"]==0 and sb["stance_enc"]==1) or
                                     (sa["stance_enc"]==1 and sb["stance_enc"]==0)),
        "same_stance":           int(sa["stance_enc"] == sb["stance_enc"]),
        "r_stance_enc":          sa["stance_enc"],
        "b_stance_enc":          sb["stance_enc"],
        "delta_pure_ko_rate":    ko_r  - ko_b,
        "delta_tko_rate":        sa.get("tko_rate",0)   - sb.get("tko_rate",0),
        "delta_finish_rate":     sa.get("finish_rate",0.3) - sb.get("finish_rate",0.3),
        "delta_ko_vulnerability":kov_r  - kov_b,
        "delta_momentum":        sa.get("momentum",0)   - sb.get("momentum",0),
        "delta_splm_trend":      sa.get("splm_trend",0) - sb.get("splm_trend",0),
        "delta_last3_win_rate":  sa.get("last3_win_rate",0.5) - sb.get("last3_win_rate",0.5),
        "delta_last3_splm":      sa.get("last3_splm",0) - sb.get("last3_splm",0),
        "delta_effective_td":    eff_td(td_r,td_d_b) - eff_td(td_b,td_d_r),
        "delta_effective_sub":   eff_sub(sub_r,td_d_b) - eff_sub(sub_b,td_d_r),
        "delta_ko_power":        (ko_r*sapm_b) - (ko_b*sapm_r),
        "delta_ctrl_dominance":  (td_r*(1-sub_b/total_sub)) - (td_b*(1-sub_r/total_sub)),
    }

    delta_raw = np.array([feat_map.get(k, 0.0) for k in DELTA_FEATURES],
                         dtype=np.float32).reshape(1,-1)
    delta_sc  = scalers["delta"].transform(delta_raw)

    return (torch.tensor(r_sc,    dtype=torch.float32),
            torch.tensor(b_sc,    dtype=torch.float32),
            torch.tensor(delta_sc, dtype=torch.float32))

# ──────────────────────────────────────────────
# 美式賠率換算
# ──────────────────────────────────────────────
def to_american_odds(prob):
    if prob <= 0 or prob >= 1: return "N/A"
    if prob >= 0.5:
        return f"-{int(round(prob/(1-prob)*100))}"
    else:
        return f"+{int(round((1-prob)/prob*100))}"

# ──────────────────────────────────────────────
# 標準化顯示（量級內z-score）
# ──────────────────────────────────────────────
def get_division_stats(df):
    """計算各量級的mean/std，用於顯示z-score"""
    stats_cols = ["splm","sapm","str_def","td_avg","td_def","sub_avg"]
    div_stats = {}
    for div in df["division"].unique():
        div_df = df[df["division"]==div]
        div_stats[div] = {}
        for s in stats_cols:
            vals = pd.concat([div_df[f"r_{s}"], div_df[f"b_{s}"]]).dropna()
            div_stats[div][s] = {"mean": vals.mean(), "std": vals.std()+1e-6}
    return div_stats

def zscore_label(val, mean, std):
    z = (val - mean) / std
    if   z >  1.5: return f"({z:+.1f}σ ↑↑)"
    elif z >  0.5: return f"({z:+.1f}σ ↑)"
    elif z < -1.5: return f"({z:+.1f}σ ↓↓)"
    elif z < -0.5: return f"({z:+.1f}σ ↓)"
    else:          return f"({z:+.1f}σ)"

# ──────────────────────────────────────────────
# 同量級相似選手
# ──────────────────────────────────────────────
def find_similar_fighters(name, embeddings, fighter_to_idx, df, top_n=4):
    if name not in fighter_to_idx: return []
    idx  = fighter_to_idx[name]
    vec  = embeddings[idx]
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    sims = (embeddings / norm) @ (vec / (np.linalg.norm(vec)+1e-8))
    idx_names = list(fighter_to_idx.keys())

    mask = (df["r_name"]==name)|(df["b_name"]==name)
    if mask.sum() == 0: return []
    own_div = df[mask].iloc[-1]["division"]

    results = []
    for i in np.argsort(-sims):
        if i == idx: continue
        fname = idx_names[i]
        fmask = (df["r_name"]==fname)|(df["b_name"]==fname)
        if fmask.sum() < 3: continue
        fdiv = df[fmask].iloc[-1]["division"]
        if fdiv != own_div: continue
        results.append((fname, float(sims[i]), fmask.sum()))
        if len(results) >= top_n: break
    return results

# ──────────────────────────────────────────────
# 模型信心評估
# ──────────────────────────────────────────────
def confidence_label(prob, n_fights_a, n_fights_b):
    warnings = []
    if n_fights_a < 5: warnings.append(f"數據不足（A僅{n_fights_a}場）")
    if n_fights_b < 5: warnings.append(f"數據不足（B僅{n_fights_b}場）")
    diff = abs(prob - 0.5)
    if diff < 0.05:   conf = "⚠️  極低信心（差距<5%）"
    elif diff < 0.10: conf = "⚠️  低信心（差距<10%）"
    elif diff < 0.20: conf = "中等信心"
    else:             conf = "較高信心"
    if warnings: conf += f"  |  {'、'.join(warnings)}"
    return conf

# ──────────────────────────────────────────────
# 主預測函數
# ──────────────────────────────────────────────
def predict_matchup(name_a, name_b, title_fight=False):
    (df, model, scalers, embeddings,
     fighter_to_idx, all_names,
     FIGHTER_STATS, DELTA_FEATURES) = load_resources()

    div_stats = get_division_stats(df)

    fa = fuzzy_find(name_a, all_names)
    fb = fuzzy_find(name_b, all_names)
    if not fa: print(f"❌ 找不到: {name_a}"); return
    if not fb: print(f"❌ 找不到: {name_b}"); return

    sa = get_fighter_stats(fa, df, FIGHTER_STATS)
    sb = get_fighter_stats(fb, df, FIGHTER_STATS)

    r_t, b_t, d_t = build_features(sa, sb, int(title_fight),
                                    scalers, FIGHTER_STATS, DELTA_FEATURES)

    with torch.no_grad():
        win_prob_a = float(model(r_t, b_t, d_t).item())
    win_prob_b = 1.0 - win_prob_a

    similar_a = find_similar_fighters(fa, embeddings, fighter_to_idx, df)
    similar_b = find_similar_fighters(fb, embeddings, fighter_to_idx, df)

    div = sa["division"]
    ds  = div_stats.get(div, {})

    # ── 輸出 ──
    SEP = "─" * 58
    print(f"\n{SEP}")
    print(f"  {fa}  vs  {fb}")
    if title_fight: print("  🏆 冠軍戰")
    print(f"  {div}  |  "
          f"{fa.split()[0]}: {int(sa['wins'])}勝{int(sa['losses'])}敗  "
          f"{fb.split()[0]}: {int(sb['wins'])}勝{int(sb['losses'])}敗")
    print(SEP)

    # 勝率 + 賠率
    odds_a = to_american_odds(win_prob_a)
    odds_b = to_american_odds(win_prob_b)
    bar_a  = "█" * int(win_prob_a * 32)
    bar_b  = "█" * int(win_prob_b * 32)
    print(f"\n  勝率 / 隱含賠率")
    print(f"  {fa:<26} {bar_a} {win_prob_a*100:.1f}%  {odds_a}")
    print(f"  {fb:<26} {bar_b} {win_prob_b*100:.1f}%  {odds_b}")

    # 信心
    conf = confidence_label(win_prob_a, sa["n_fights"], sb["n_fights"])
    print(f"\n  信心: {conf}")

    # Momentum 顯示
    mom_a = sa.get("momentum", 0)
    mom_b = sb.get("momentum", 0)
    mom_label = lambda m: ("🔥上升" if m > 0.15 else "📉下滑" if m < -0.15 else "→持平")
    print(f"\n  近期狀態")
    print(f"  {fa.split()[0]:<20} momentum={mom_a:+.2f}  {mom_label(mom_a)}")
    print(f"  {fb.split()[0]:<20} momentum={mom_b:+.2f}  {mom_label(mom_b)}")

    # 關鍵數據（含z-score）
    print(f"\n  關鍵數據對比（括號為量級內排名）")
    KEY = [
        ("raw_splm",    "splm",    "打擊輸出"),
        ("raw_sapm",    "sapm",    "被打頻率"),
        ("raw_str_def", "str_def", "打擊防禦"),
        ("raw_td_avg",  "td_avg",  "摔技頻率"),
        ("raw_td_def",  "td_def",  "防摔"),
        ("raw_sub_avg", "sub_avg", "降伏威脅"),
    ]
    for raw_key, stat_key, label in KEY:
        va, vb = sa.get(raw_key, 0), sb.get(raw_key, 0)
        diff   = va - vb
        winner = f"← {fa.split()[0]}" if diff > 0.3 else \
                 f"← {fb.split()[0]}" if diff < -0.3 else "≈ 相近"
        za = zscore_label(va, ds.get(stat_key,{}).get("mean",va),
                              ds.get(stat_key,{}).get("std",1)) if ds else ""
        zb = zscore_label(vb, ds.get(stat_key,{}).get("mean",vb),
                              ds.get(stat_key,{}).get("std",1)) if ds else ""
        print(f"  {label:<8} {va:>5.1f}{za:<14} vs  {vb:<5.1f}{zb:<14} {winner}")

    # 相似選手
    if similar_a:
        s = ", ".join(f"{n} ({sc:.2f})" for n,sc,_ in similar_a)
        print(f"\n  {fa.split()[0]} 風格近似: {s}")
    if similar_b:
        s = ", ".join(f"{n} ({sc:.2f})" for n,sc,_ in similar_b)
        print(f"  {fb.split()[0]} 風格近似: {s}")

    print(f"\n  資料時間: {fa} {sa['last_fight']} / {fb} {sb['last_fight']}")
    print(SEP)

# ──────────────────────────────────────────────
# 互動模式
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("UFC Style Encoder Predictor v3")
    print("輸入 q 離開\n")
    while True:
        a = input("選手 A: ").strip()
        if a.lower() == "q": break
        b = input("選手 B: ").strip()
        if b.lower() == "q": break
        t = input("冠軍戰? (y/n): ").strip().lower()
        predict_matchup(a, b, title_fight=(t=="y"))
        print()