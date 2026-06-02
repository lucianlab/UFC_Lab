"""
UFC Matchup Prediction — Step 1: Data Preparation v3
取代: src/data_prep.py

新增（相較v2）:
  Rolling features (最近3/5場):
    r/b_last3_win_rate, last3_finish_rate, last3_splm, last3_sapm
    r/b_last5_win_rate, last5_finish_rate
    r/b_momentum         最近3場勝率 - 生涯勝率（正=上升期）
    r/b_splm_trend       最近3場splm - 生涯splm

  改進的交互特徵:
    reach_striking_edge   臂展差 × 雙方站立傾向乘積
    ko_power_vs_chin_r/b  KO率 × 對手sapm
    ctrl_dominance_r/b    摔技頻率 × 對手地面技術劣勢

  修正:
    移除重複欄位 delta_*.1

執行: python3 src/data_prep.py
輸出: data/processed/fights_clean.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/UFC.csv"
OUT_PATH = "data/processed/fights_clean.csv"
os.makedirs("data/processed", exist_ok=True)

# ──────────────────────────────────────────────
# 1. 載入與基本過濾
# ──────────────────────────────────────────────
df_raw = pd.read_csv(RAW_PATH)
df_raw["date"] = pd.to_datetime(df_raw["date"])

MAIN_DIVISIONS = [
    "heavyweight", "light heavyweight", "middleweight",
    "welterweight", "lightweight", "featherweight",
    "bantamweight", "flyweight",
    "women's strawweight", "women's flyweight",
    "women's bantamweight", "women's featherweight",
]

df_raw = df_raw[
    (df_raw["date"] >= "2005-01-01") &
    (df_raw["division"].isin(MAIN_DIVISIONS))
].copy().sort_values("date").reset_index(drop=True)

print(f"原始資料 (2005+, 正規量級): {len(df_raw)} 場")

# ──────────────────────────────────────────────
# 2. 結束方式分類
# ──────────────────────────────────────────────
def map_method(m):
    if pd.isna(m): return np.nan
    m = str(m).lower()
    if "ko/tko" in m:     return "KO"
    if "submission" in m: return "SUB"
    if "decision" in m:   return "DEC"
    return np.nan

df_raw["method_clean"] = df_raw["method"].apply(map_method)
df_raw = df_raw[df_raw["method_clean"].notna()].copy()

df_raw["match_time_sec"] = pd.to_numeric(df_raw["match_time_sec"], errors="coerce")
df_raw["finish_round"]   = pd.to_numeric(df_raw["finish_round"],   errors="coerce")

df_raw["is_pure_ko"] = (
    (df_raw["method_clean"] == "KO") &
    (df_raw["finish_round"] == 1) &
    (df_raw["match_time_sec"] <= 90)
).astype(int)
df_raw["is_tko"]    = (df_raw["method_clean"] == "KO").astype(int)
df_raw["is_sub"]    = (df_raw["method_clean"] == "SUB").astype(int)
df_raw["is_finish"] = (df_raw["method_clean"].isin(["KO","SUB"])).astype(int)

print(f"清理後: {len(df_raw)} 場")

# ──────────────────────────────────────────────
# 3. 歷史特徵計算（賽前累積 + Rolling）
# ──────────────────────────────────────────────
print("計算選手歷史特徵（需要一點時間）...")

records = []
for _, row in df_raw.iterrows():
    for corner in ["r", "b"]:
        fighter = row[f"{corner}_name"]
        won     = (row["winner"] == fighter)
        splm_val = pd.to_numeric(row.get(f"{corner}_splm", np.nan), errors="coerce")
        sapm_val = pd.to_numeric(row.get(f"{corner}_sapm", np.nan), errors="coerce")
        records.append({
            "fight_id":   row["fight_id"],
            "date":       row["date"],
            "fighter":    fighter,
            "won":        int(won),
            "is_pure_ko": row["is_pure_ko"] if won else 0,
            "is_tko":     row["is_tko"]     if won else 0,
            "is_sub":     row["is_sub"]     if won else 0,
            "is_finish":  row["is_finish"]  if won else 0,
            "got_ko":     row["is_tko"]     if not won else 0,
            "got_sub":    row["is_sub"]     if not won else 0,
            "title_fight":row["title_fight"],
            "title_won":  int(row["title_fight"] == 1 and won),
            "splm":       splm_val,
            "sapm":       sapm_val,
        })

hist = pd.DataFrame(records).sort_values(["fighter","date"]).reset_index(drop=True)

def cumshift(s):
    return s.cumsum().shift(1).fillna(0)

def calc_rate(num, den, default=0.5):
    return np.where(den > 0, num / den, default)

def rolling_shift(s, w):
    return s.shift(1).rolling(w, min_periods=1).mean()

g = hist.groupby("fighter")

# 累積特徵
hist["cum_wins"]         = g["won"].transform(cumshift)
hist["cum_fights"]       = g["won"].transform(lambda x: (x*0+1).cumsum().shift(1).fillna(0))
hist["cum_pure_ko"]      = g["is_pure_ko"].transform(cumshift)
hist["cum_tko"]          = g["is_tko"].transform(cumshift)
hist["cum_sub"]          = g["is_sub"].transform(cumshift)
hist["cum_finish"]       = g["is_finish"].transform(cumshift)
hist["cum_got_ko"]       = g["got_ko"].transform(cumshift)
hist["cum_got_sub"]      = g["got_sub"].transform(cumshift)
hist["cum_title"]        = g["title_fight"].transform(cumshift)
hist["cum_title_won"]    = g["title_won"].transform(cumshift)

hist["pure_ko_rate"]     = calc_rate(hist["cum_pure_ko"],  hist["cum_wins"],   0.0)
hist["tko_rate"]         = calc_rate(hist["cum_tko"],      hist["cum_wins"],   0.0)
hist["finish_rate"]      = calc_rate(hist["cum_finish"],   hist["cum_wins"],   0.3)
hist["ko_vulnerability"] = calc_rate(hist["cum_got_ko"],   hist["cum_fights"], 0.3)
hist["sub_vulnerability"]= calc_rate(hist["cum_got_sub"],  hist["cum_fights"], 0.1)
hist["title_exp"]        = hist["cum_title"]
hist["title_win_rate"]   = calc_rate(hist["cum_title_won"],hist["cum_title"],  0.5)

# Rolling 特徵
hist["last3_win_rate"]    = g["won"].transform(lambda x: rolling_shift(x, 3))
hist["last5_win_rate"]    = g["won"].transform(lambda x: rolling_shift(x, 5))
hist["last3_finish_rate"] = g["is_finish"].transform(lambda x: rolling_shift(x, 3))
hist["last5_finish_rate"] = g["is_finish"].transform(lambda x: rolling_shift(x, 5))
hist["last3_splm"]        = g["splm"].transform(lambda x: rolling_shift(x, 3))
hist["last3_sapm"]        = g["sapm"].transform(lambda x: rolling_shift(x, 3))

# Momentum & trend
career_wr   = calc_rate(hist["cum_wins"], hist["cum_fights"], 0.5)
career_splm = g["splm"].transform(lambda x: x.shift(1).expanding().mean())
hist["momentum"]   = hist["last3_win_rate"] - career_wr
hist["splm_trend"] = hist["last3_splm"] - career_splm

# NaN 補值
rolling_defaults = {
    "last3_win_rate":0.5, "last5_win_rate":0.5,
    "last3_finish_rate":0.3, "last5_finish_rate":0.3,
    "last3_splm":0.0, "last3_sapm":0.0,
    "momentum":0.0, "splm_trend":0.0,
}
for col, val in rolling_defaults.items():
    hist[col] = hist[col].fillna(val)

HIST_FEAT_COLS = [
    "pure_ko_rate","tko_rate","finish_rate",
    "ko_vulnerability","sub_vulnerability",
    "title_exp","title_win_rate",
    "last3_win_rate","last5_win_rate",
    "last3_finish_rate","last5_finish_rate",
    "last3_splm","last3_sapm",
    "momentum","splm_trend",
]
hist_feats = hist[["fight_id","fighter"] + HIST_FEAT_COLS].copy()
print(f"歷史特徵計算完成，{len(hist_feats)} 筆記錄")

# ──────────────────────────────────────────────
# 4. Merge
# ──────────────────────────────────────────────
def merge_corner(df, feats, corner):
    rename = {"fighter": f"{corner}_name"}
    rename.update({c: f"{corner}_{c}" for c in HIST_FEAT_COLS})
    return df.merge(feats.rename(columns=rename),
                    on=["fight_id", f"{corner}_name"], how="left")

df = merge_corner(df_raw, hist_feats, "r")
df = merge_corner(df, hist_feats, "b")

new_feat_defaults = {
    "pure_ko_rate":0.0,"tko_rate":0.0,"finish_rate":0.3,
    "ko_vulnerability":0.3,"sub_vulnerability":0.1,
    "title_exp":0.0,"title_win_rate":0.5,
    "last3_win_rate":0.5,"last5_win_rate":0.5,
    "last3_finish_rate":0.3,"last5_finish_rate":0.3,
    "last3_splm":0.0,"last3_sapm":0.0,
    "momentum":0.0,"splm_trend":0.0,
}
for prefix in ["r_","b_"]:
    for feat, default in new_feat_defaults.items():
        col = prefix + feat
        if col in df.columns:
            df[col] = df[col].fillna(default)

print(f"Merge 後: {len(df)} 場")

# ──────────────────────────────────────────────
# 5. 目標變數
# ──────────────────────────────────────────────
df["winner_is_red"] = (df["winner"] == df["r_name"]).astype(int)

def map_finish_method(m):
    if pd.isna(m): return np.nan
    m = str(m).lower()
    if "ko/tko" in m or "doctor" in m or "could not" in m: return 0
    if "sub" in m:      return 1
    if "decision" in m: return 2
    return np.nan

df["finish_method"] = df["method"].apply(map_finish_method)
df = df[df["finish_method"].notna()].copy()
df["finish_round"]  = pd.to_numeric(df["finish_round"], errors="coerce")

# ──────────────────────────────────────────────
# 6. Career stats
# ──────────────────────────────────────────────
PREFIGHT_STATS = [
    "wins","losses","draws","height","weight","reach",
    "splm","sapm","str_acc","str_def",
    "td_avg","td_avg_acc","td_def","sub_avg",
]
STANCE_MAP = {"Orthodox":0,"Southpaw":1,"Switch":2,"Open Stance":2}

for prefix in ["r_","b_"]:
    for col in PREFIGHT_STATS:
        full = prefix + col
        if full in df.columns:
            df[full] = pd.to_numeric(df[full], errors="coerce")
    for stat in ["height","weight","reach"]:
        col = prefix + stat
        if col in df.columns:
            medians = df.groupby("division")[col].transform("median")
            df[col] = df[col].fillna(medians).fillna(df[col].median())
    for stat in [s for s in PREFIGHT_STATS if s not in ["height","weight","reach"]]:
        col = prefix + stat
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if prefix+"stance" in df.columns:
        df[prefix+"stance"] = df[prefix+"stance"].fillna("Orthodox")

for p in ["r_","b_"]:
    total = df[p+"wins"] + df[p+"losses"]
    df[p+"win_rate"]   = np.where(total > 0, df[p+"wins"] / total, 0.5)
    df[p+"experience"] = df[p+"wins"] + df[p+"losses"] + df[p+"draws"]
    df[p+"stance_enc"] = df[p+"stance"].map(STANCE_MAP).fillna(0).astype(int)

df["southpaw_matchup"] = (
    ((df["r_stance_enc"]==0)&(df["b_stance_enc"]==1)) |
    ((df["r_stance_enc"]==1)&(df["b_stance_enc"]==0))
).astype(int)
df["same_stance"]  = (df["r_stance_enc"] == df["b_stance_enc"]).astype(int)
df["title_fight"]  = pd.to_numeric(df["title_fight"], errors="coerce").fillna(0).astype(int)

# ──────────────────────────────────────────────
# 7. 交互特徵
# ──────────────────────────────────────────────
# 有效摔技威脅
df["effective_td_threat_r"] = df["r_td_avg"] * (1 - df["b_td_def"] / 100)
df["effective_td_threat_b"] = df["b_td_avg"] * (1 - df["r_td_def"] / 100)
df["delta_effective_td"]    = df["effective_td_threat_r"] - df["effective_td_threat_b"]

# 有效降伏威脅
df["effective_sub_threat_r"] = df["r_sub_avg"] * (1 - df["b_td_def"] / 100)
df["effective_sub_threat_b"] = df["b_sub_avg"] * (1 - df["r_td_def"] / 100)
df["delta_effective_sub"]    = df["effective_sub_threat_r"] - df["effective_sub_threat_b"]

# 站立傾向
df["r_striking_tendency"] = df["r_splm"] / (df["r_td_avg"] + 0.5)
df["b_striking_tendency"] = df["b_splm"] / (df["b_td_avg"] + 0.5)

# 臂展 × 雙方站立傾向乘積（v3改進）
df["reach_striking_edge"] = (
    (df["r_reach"] - df["b_reach"]) *
    df["r_striking_tendency"] * df["b_striking_tendency"]
)

# 重拳威脅 × 對手sapm（v3新增）
df["ko_power_vs_chin_r"] = df["r_pure_ko_rate"] * df["b_sapm"]
df["ko_power_vs_chin_b"] = df["b_pure_ko_rate"] * df["r_sapm"]
df["delta_ko_power"]     = df["ko_power_vs_chin_r"] - df["ko_power_vs_chin_b"]

# 地面控制主導（v3新增）
total_sub = df["r_sub_avg"] + df["b_sub_avg"] + 0.1
df["ctrl_dominance_r"]     = df["r_td_avg"] * (1 - df["b_sub_avg"] / total_sub)
df["ctrl_dominance_b"]     = df["b_td_avg"] * (1 - df["r_sub_avg"] / total_sub)
df["delta_ctrl_dominance"] = df["ctrl_dominance_r"] - df["ctrl_dominance_b"]

# KO 威脅（原版保留）
df["ko_threat_r"]     = df["r_pure_ko_rate"] * df["b_ko_vulnerability"]
df["ko_threat_b"]     = df["b_pure_ko_rate"] * df["r_ko_vulnerability"]
df["delta_ko_threat"] = df["ko_threat_r"] - df["ko_threat_b"]

df["delta_title_exp"]      = df["r_title_exp"]      - df["b_title_exp"]
df["delta_title_win_rate"] = df["r_title_win_rate"] - df["b_title_win_rate"]

# ──────────────────────────────────────────────
# 8. Delta features
# ──────────────────────────────────────────────
DELTA_STATS = [
    "wins","losses","win_rate","experience",
    "splm","sapm","str_acc","str_def",
    "td_avg","td_avg_acc","td_def","sub_avg",
    "height","reach",
    "pure_ko_rate","tko_rate","finish_rate",
    "ko_vulnerability","sub_vulnerability",
    "last3_win_rate","last5_win_rate",
    "last3_finish_rate","last5_finish_rate",
    "last3_splm","last3_sapm",
    "momentum","splm_trend",
]
for stat in DELTA_STATS:
    r_col, b_col = f"r_{stat}", f"b_{stat}"
    if r_col in df.columns and b_col in df.columns:
        df[f"delta_{stat}"] = df[r_col] - df[b_col]

# ──────────────────────────────────────────────
# 9. 整理輸出（去重複欄位）
# ──────────────────────────────────────────────
META = [
    "fight_id","date","division","title_fight",
    "r_name","r_id","r_stance","r_stance_enc",
    "b_name","b_id","b_stance","b_stance_enc",
]
R_STATS = [f"r_{s}" for s in PREFIGHT_STATS] + [
    "r_win_rate","r_experience",
    "r_pure_ko_rate","r_tko_rate","r_finish_rate",
    "r_ko_vulnerability","r_sub_vulnerability",
    "r_title_exp","r_title_win_rate","r_striking_tendency",
    "r_last3_win_rate","r_last5_win_rate",
    "r_last3_finish_rate","r_last5_finish_rate",
    "r_last3_splm","r_last3_sapm",
    "r_momentum","r_splm_trend",
]
B_STATS = [c.replace("r_","b_",1) for c in R_STATS]
INTERACT = [
    "effective_td_threat_r","effective_td_threat_b","delta_effective_td",
    "effective_sub_threat_r","effective_sub_threat_b","delta_effective_sub",
    "reach_striking_edge",
    "ko_power_vs_chin_r","ko_power_vs_chin_b","delta_ko_power",
    "ctrl_dominance_r","ctrl_dominance_b","delta_ctrl_dominance",
    "ko_threat_r","ko_threat_b","delta_ko_threat",
    "southpaw_matchup","same_stance",
]
DELTA_COLS = [c for c in df.columns if c.startswith("delta_")]
TARGET     = ["winner_is_red","finish_method","finish_round","total_rounds"]

keep = META + R_STATS + B_STATS + INTERACT + DELTA_COLS + TARGET
seen, keep_dedup = set(), []
for c in keep:
    if c in df.columns and c not in seen:
        keep_dedup.append(c)
        seen.add(c)

df_out = df[keep_dedup].copy()

# ──────────────────────────────────────────────
# 10. 輸出
# ──────────────────────────────────────────────
nan_counts = df_out.isnull().sum()
if nan_counts.sum() > 0:
    print(f"\n⚠️  剩餘 NaN:")
    print(nan_counts[nan_counts > 0])
else:
    print("\n✅ 無 NaN")

df_out.to_csv(OUT_PATH, index=False)
print(f"✅ 儲存至 {OUT_PATH}")
print(f"最終資料: {len(df_out)} 場, {df_out.shape[1]} 欄位")

delta_cols = [c for c in df_out.columns if c.startswith("delta_")]
print(f"\nDelta features ({len(delta_cols)}個):")
for c in sorted(delta_cols): print(f"  {c}")

# 驗證 rolling：Strickland momentum
print("\n驗證 rolling（Sean Strickland 最近5場）:")
mask = (df_out["r_name"]=="Sean Strickland")|(df_out["b_name"]=="Sean Strickland")
for _, row in df_out[mask].sort_values("date").tail(5).iterrows():
    p = "r_" if row["r_name"]=="Sean Strickland" else "b_"
    print(f"  {str(row['date'])[:10]}  "
          f"momentum={row[p+'momentum']:+.3f}  "
          f"last3_wr={row[p+'last3_win_rate']:.2f}  "
          f"splm_trend={row[p+'splm_trend']:+.2f}") 