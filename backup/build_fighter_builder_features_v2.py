"""
build_fighter_builder_features_v2.py
=====================================
主表: fighter_vectors.csv (1239 選手)
補充:
  fighters_all.csv         → wins/losses → win_rate
  ufc_championship_history → defenses, dual_weight_champ
  rankings_history.csv     → best_hist_rank (2013+)
  ufc_active_top15_rankings→ 現役排名補充

Tier 系統 (10級):
  S   真神          95-100   JJ, GSP, DJ, Silva
  A+  半神          85-94    Israel, Amanda, Valentina
  A   統治力冠軍    75-84    Khabib, Usman, Volk, Islam
  B+  正常冠軍有防衛 67-74   Max, Aldo
  B   一次性冠軍    60-66    Strickland, Conor, Chimaev
  C+  無冕之王      54-59    TKZ (曾#1-3,未奪冠)
  C   歷史最高#4-5  47-53
  D+  歷史最高#6-10 40-46
  D   歷史最高#11-15 33-39
  E   從未入top15   0-32

Tier 內部加權:
  冠軍:   def_score(防衛/tier天花板) * 0.55 + wr_bonus(非線性勝率) * 0.45
  非冠軍: rank_pct(排名位置) * 0.65 + wr_bonus * 0.35
  
  wr_bonus 非線性:
    >= 0.99 → 1.00  (不敗)
    >= 0.90 → 0.70
    >= 0.80 → 0.40
    else    → wr * 0.30
"""

import pandas as pd
import numpy as np
import os

# ── 路徑 ──────────────────────────────────────────────
BASE       = os.path.expanduser('~/UFC')
CLEAN      = os.path.join(BASE, 'data', 'clean')
RAW        = os.path.join(BASE, 'data', 'raw')
OUT_PATH   = os.path.join(CLEAN, 'fighter_builder_features.csv')

FV_CSV     = os.path.join(CLEAN, 'fighter_vectors.csv')
FA_CSV     = os.path.join(CLEAN, 'fighters_all.csv')
CHAMP_CSV  = os.path.join(RAW,   'ufc_championship_history.csv')
RANK_H_CSV = os.path.join(CLEAN, 'rankings_history.csv')
RANK_A_CSV = os.path.join(RAW,   'ufc_active_top15_rankings.csv')

# ── 讀檔 ──────────────────────────────────────────────
print("載入資料...")
fv     = pd.read_csv(FV_CSV)
fa     = pd.read_csv(FA_CSV)
champ  = pd.read_csv(CHAMP_CSV)
rank_h = pd.read_csv(RANK_H_CSV)
rank_a = pd.read_csv(RANK_A_CSV)
print(f"  fighter_vectors : {len(fv)}")

# ═══════════════════════════════════════════════════════
# BLOCK 1 — win_rate from fighters_all
# ═══════════════════════════════════════════════════════
print("Block 1: win_rate...")
fa = fa.copy()
fa['n_fights_fa'] = fa['wins'] + fa['losses']
fa['win_rate']    = np.where(fa['n_fights_fa'] > 0,
                              fa['wins'] / fa['n_fights_fa'], np.nan)
fv = fv.merge(fa[['name','win_rate','height_cm','reach_cm']], on='name', how='left')

# ═══════════════════════════════════════════════════════
# BLOCK 2 — championship_history
# ═══════════════════════════════════════════════════════
print("Block 2: championship history...")
champ_real = champ[champ['title_type'].isin(['Undisputed', 'Interim'])].copy()
champ_agg  = champ_real.groupby('champion').agg(
    max_defenses     = ('title_defenses_count', 'max'),
    n_weight_classes = ('weight_class', 'nunique'),
    is_incumbent     = ('incumbent', 'max'),
).reset_index()
champ_agg.rename(columns={'champion': 'name'}, inplace=True)
champ_agg['ever_champion']     = True
champ_agg['dual_weight_champ'] = champ_agg['n_weight_classes'] >= 2

fv = fv.merge(
    champ_agg[['name','ever_champion','max_defenses','dual_weight_champ','is_incumbent']],
    on='name', how='left'
)
fv['ever_champion']     = fv['ever_champion'].fillna(False)
fv['dual_weight_champ'] = fv['dual_weight_champ'].fillna(False)
fv['max_defenses']      = fv['max_defenses'].fillna(0)
fv['is_incumbent']      = fv['is_incumbent'].fillna(False)

# ═══════════════════════════════════════════════════════
# BLOCK 3 — rankings_history (2013+), 排除 rank=0
# ═══════════════════════════════════════════════════════
print("Block 3: historical rankings...")
rank_div  = rank_h[
    ~rank_h['weightclass'].str.contains('Pound-for-Pound', na=False) &
    (rank_h['rank'] > 0)
].copy()
best_hist = rank_div.groupby('fighter')['rank'].min().reset_index()
best_hist.rename(columns={'rank': 'best_hist_rank'}, inplace=True)
fv = fv.merge(best_hist, left_on='name', right_on='fighter', how='left').drop(
    columns=['fighter'], errors='ignore')

# ═══════════════════════════════════════════════════════
# BLOCK 4 — active_top15_rankings
# ═══════════════════════════════════════════════════════
print("Block 4: active rankings...")
rank_a_div = rank_a[~rank_a['division'].str.contains('Pound-for-Pound', na=False)].copy()

def parse_rank(r):
    try:    return int(r)
    except: return 1   # 'Champion' → 1

rank_a_div['rank_num'] = rank_a_div['rank'].apply(parse_rank)
active_best = rank_a_div.groupby('fighter')['rank_num'].min().reset_index()
active_best.rename(columns={'fighter':'name','rank_num':'active_best_rank'}, inplace=True)
fv = fv.merge(active_best, on='name', how='left')
fv['best_rank'] = fv[['best_hist_rank','active_best_rank']].min(axis=1)

# ═══════════════════════════════════════════════════════
# TIER 計算
# ═══════════════════════════════════════════════════════
print("Computing tiers...")

def wr_bonus(wr):
    """非線性勝率轉換,放大不敗與高勝率的差距"""
    if pd.isna(wr): return 0.3
    if wr >= 0.99:  return 1.00   # 不敗
    if wr >= 0.90:  return 0.70
    if wr >= 0.80:  return 0.40
    return wr * 0.30

def rank_pct(best, r_min, r_max):
    """排名越小 → 越高,線性映射到 0-1"""
    if pd.isna(best): return 0.0
    best = max(r_min, min(r_max, best))
    return 1.0 - (best - r_min) / (r_max - r_min + 1e-9)

def compute_tier(row):
    defenses = float(row['max_defenses'] or 0)
    dual     = bool(row['dual_weight_champ'])
    wr       = row['win_rate']
    best     = row['best_rank']
    wb       = wr_bonus(wr)

    # ── 冠軍 tier ────────────────────────────────────
    if row['ever_champion']:

        def champ_score(base, ceil_, def_ceil):
            ds = min(defenses / def_ceil, 1.0)
            return round(base + (ceil_ - base) * (ds * 0.55 + wb * 0.45))

        # S: 雙量級3+防衛 or 8+防衛
        if (dual and defenses >= 3) or defenses >= 8:
            return 'S', champ_score(95, 100, 11)

        # A+: 5+防衛 or 雙量級2+防衛
        if defenses >= 5 or (dual and defenses >= 2):
            return 'A+', champ_score(85, 94, 7)

        # A: 3-4防衛 or 雙量級1+防衛
        if defenses >= 3 or (dual and defenses >= 1):
            return 'A', champ_score(75, 84, 4)

        # B+: 1-2防衛
        if defenses >= 1:
            return 'B+', champ_score(67, 74, 2)

        # B: 0防衛,雙冠加成
        bonus = 3 if dual else 0
        score = round(60 + bonus + (66 - 60) * wb * 0.5)
        return 'B', min(66, score)

    # ── 非冠軍 tier ──────────────────────────────────
    def nc_score(base, ceil_, r_min, r_max):
        rp = rank_pct(best, r_min, r_max)
        return round(base + (ceil_ - base) * (rp * 0.65 + wb * 0.35))

    # C+: 曾 #1-3,未奪冠
    if pd.notna(best) and best <= 3:
        return 'C+', nc_score(54, 59, 1, 3)

    # C: 歷史最高 #4-5
    if pd.notna(best) and best <= 5:
        return 'C',  nc_score(47, 53, 4, 5)

    # D+: 歷史最高 #6-10
    if pd.notna(best) and best <= 10:
        return 'D+', nc_score(40, 46, 6, 10)

    # D: 歷史最高 #11-15
    if pd.notna(best) and best <= 15:
        return 'D',  nc_score(33, 39, 11, 15)

    # E: 從未入 top15
    nf    = min(row['n_fights'], 30)
    score = wb * 20 + np.log1p(nf) * 3
    return 'E', round(min(32, score))

results          = fv.apply(compute_tier, axis=1)
fv['tier_label'] = results.apply(lambda x: x[0])
fv['tier_score'] = results.apply(lambda x: x[1])

# ═══════════════════════════════════════════════════════
# PERCENTILE RANKS — 14 builder features → 1-10 分
# ═══════════════════════════════════════════════════════
print("Computing percentile ranks (1-10)...")

pct_map = {
    'height_cm'  : 'pct_height',
    'reach_cm'   : 'pct_reach',
    'ko_rate'    : 'pct_striking_power',
    'sig_per_r'  : 'pct_striking_volume',
    'sig_acc'    : 'pct_striking_accuracy',
    'str_def'    : 'pct_striking_defense',
    'leg_pct'    : 'pct_leg_kicks',
    'clinch_pct' : 'pct_clinch',
    'td_per_r'   : 'pct_takedown_offense',
    'td_def'     : 'pct_takedown_defense',
    'ctrl_per_r' : 'pct_ground_control',
    'sub_rate'   : 'pct_submission',
    'gnp_per_r'  : 'pct_ground_pound',
    'gas_tank'   : 'pct_cardio',
}

for raw_col, pct_col in pct_map.items():
    if raw_col in fv.columns:
        fv[pct_col] = (
            fv[raw_col].rank(pct=True, na_option='bottom') * 9 + 1
        ).clip(1, 10).round(2)
    else:
        print(f"  WARNING: {raw_col} not found")

# ═══════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════
keep = (
    ['name', 'wc', 'n_fights', 'tier_label', 'tier_score',
     'best_rank', 'ever_champion', 'max_defenses', 'dual_weight_champ',
     'is_incumbent', 'win_rate',
     # raw stats (KNN 用)
     'height_cm', 'reach_cm', 'ko_rate', 'sig_per_r', 'sig_acc',
     'str_def', 'leg_pct', 'clinch_pct', 'td_per_r', 'td_def',
     'ctrl_per_r', 'sub_rate', 'gnp_per_r', 'gas_tank',
     # pct features (UI 顯示用)
     'pct_height', 'pct_reach', 'pct_striking_power', 'pct_striking_volume',
     'pct_striking_accuracy', 'pct_striking_defense', 'pct_leg_kicks',
     'pct_clinch', 'pct_takedown_offense', 'pct_takedown_defense',
     'pct_ground_control', 'pct_submission', 'pct_ground_pound', 'pct_cardio',
     # PCA coords
     'x', 'y', 'z']
)
keep = [c for c in keep if c in fv.columns]
out  = fv[keep].copy()

os.makedirs(CLEAN, exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"\n完成 → {OUT_PATH}")
print(f"選手數: {len(out)}")

print(f"\nTier 分佈:")
print(out['tier_label'].value_counts().sort_index())

print(f"\nS tier:")
print(out[out['tier_label']=='S'][
    ['name','max_defenses','dual_weight_champ','win_rate','tier_score']
].to_string())

print(f"\n重點選手確認:")
names = ['Khabib Nurmagomedov','Jon Jones','Georges St-Pierre',
         'Sean Strickland','Conor McGregor','Islam Makhachev',
         'Israel Adesanya','Max Holloway','Kamaru Usman',
         'Alexander Volkanovski']
print(out[out['name'].isin(names)][
    ['name','tier_label','tier_score','max_defenses','dual_weight_champ','win_rate','best_rank']
].sort_values('tier_score', ascending=False).to_string())
