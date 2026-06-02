"""
build_fighter_builder_features_v3.py
======================================
13 features, 5 categories:

  Physical  : Reach (量級內 percentile)
  Boxing    : KO Power, Volume, Accuracy, Striking Defense
  Muay Thai : Leg Kicks, Clinch, Body Strikes
  Wrestling : TD Frequency, TD Accuracy, TD Defense
  Jiu-Jitsu : Submission, Control, Ground & Pound

變更:
  - 移除 Height
  - Reach 改為量級內 percentile
  - 新增 Body Strikes (body_per_r from rounds)
  - TD Accuracy (td_avg_acc from fighters_all)
  - Control 移到 Jiu-Jitsu
  - 分類改名 Muay Thai
  - Tier 系統保持不變 (來自 v2)
"""

import pandas as pd
import numpy as np
import os

BASE   = os.path.expanduser('~/UFC')
CLEAN  = os.path.join(BASE, 'data', 'clean')
RAW    = os.path.join(BASE, 'data', 'raw')
OUT    = os.path.join(CLEAN, 'fighter_builder_features.csv')

FV_CSV     = os.path.join(CLEAN, 'fighter_vectors.csv')
FA_CSV     = os.path.join(CLEAN, 'fighters_all.csv')
ROUNDS_CSV = os.path.join(CLEAN, 'fights_all_rounds.csv')
CHAMP_CSV  = os.path.join(RAW,   'ufc_championship_history.csv')
RANK_H_CSV = os.path.join(CLEAN, 'rankings_history.csv')
RANK_A_CSV = os.path.join(RAW,   'ufc_active_top15_rankings.csv')

print("載入資料...")
fv     = pd.read_csv(FV_CSV)
fa     = pd.read_csv(FA_CSV)
rounds = pd.read_csv(ROUNDS_CSV)
champ  = pd.read_csv(CHAMP_CSV)
rank_h = pd.read_csv(RANK_H_CSV)
rank_a = pd.read_csv(RANK_A_CSV)
print(f"  fighter_vectors: {len(fv)}")

# ══════════════════════════════════════════════════════
# BLOCK 1 — fighters_all: win_rate, reach_cm, td_avg_acc
# ══════════════════════════════════════════════════════
print("Block 1: fighters_all...")
fa = fa.copy()
fa['n_fights_fa'] = fa['wins'] + fa['losses']
fa['win_rate']    = np.where(fa['n_fights_fa'] > 0,
                              fa['wins'] / fa['n_fights_fa'], np.nan)

fv = fv.merge(
    fa[['name','win_rate','reach_cm','td_avg_acc']],
    on='name', how='left'
)

# ══════════════════════════════════════════════════════
# BLOCK 2 — rounds: body_per_r
# ══════════════════════════════════════════════════════
print("Block 2: rounds -> body_per_r...")
body_agg = rounds.groupby('fighter').agg(
    total_rounds = ('round', 'count'),
    body_total   = ('body_landed', 'sum'),
).reset_index()
body_agg['body_per_r'] = body_agg['body_total'] / body_agg['total_rounds']

fv = fv.merge(
    body_agg[['fighter','body_per_r']],
    left_on='name', right_on='fighter', how='left'
).drop(columns=['fighter'], errors='ignore')

# ══════════════════════════════════════════════════════
# BLOCK 3 — championship_history
# ══════════════════════════════════════════════════════
print("Block 3: championship history...")

champ_real = champ[champ['title_type'].isin(['Undisputed','Interim'])].copy()
champ_agg  = champ_real.groupby('champion').agg(
    max_defenses     = ('title_defenses_count', 'max'),
    n_weight_classes = ('weight_class', 'nunique'),
    is_incumbent     = ('incumbent', 'max'),
).reset_index()
champ_agg.rename(columns={'champion':'name'}, inplace=True)
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

# ══════════════════════════════════════════════════════
# BLOCK 4 — rankings_history (2013+)
# ══════════════════════════════════════════════════════
print("Block 4: historical rankings...")
rank_div  = rank_h[
    ~rank_h['weightclass'].str.contains('Pound-for-Pound', na=False) &
    (rank_h['rank'] > 0)
].copy()
best_hist = rank_div.groupby('fighter')['rank'].min().reset_index()
best_hist.rename(columns={'rank':'best_hist_rank'}, inplace=True)
fv = fv.merge(best_hist, left_on='name', right_on='fighter', how='left').drop(
    columns=['fighter'], errors='ignore')

# ══════════════════════════════════════════════════════
# BLOCK 5 — active rankings
# ══════════════════════════════════════════════════════
print("Block 5: active rankings...")
rank_a_div = rank_a[~rank_a['division'].str.contains('Pound-for-Pound', na=False)].copy()
def parse_rank(r):
    try:    return int(r)
    except: return 1
rank_a_div['rank_num'] = rank_a_div['rank'].apply(parse_rank)
active_best = rank_a_div.groupby('fighter')['rank_num'].min().reset_index()
active_best.rename(columns={'fighter':'name','rank_num':'active_best_rank'}, inplace=True)
fv = fv.merge(active_best, on='name', how='left')
fv['best_rank'] = fv[['best_hist_rank','active_best_rank']].min(axis=1)

# ══════════════════════════════════════════════════════
# TIER 計算 (同 v2 邏輯)
# ══════════════════════════════════════════════════════
print("Computing tiers...")

def wr_bonus(wr):
    if pd.isna(wr): return 0.3
    if wr >= 0.99:  return 1.00
    if wr >= 0.90:  return 0.70
    if wr >= 0.80:  return 0.40
    return wr * 0.30

def rank_pct(best, r_min, r_max):
    if pd.isna(best): return 0.0
    best = max(r_min, min(r_max, best))
    return 1.0 - (best - r_min) / (r_max - r_min + 1e-9)

def compute_tier(row):
    defenses = float(row['max_defenses'] or 0)
    dual     = bool(row['dual_weight_champ'])
    wr       = row['win_rate']
    best     = row['best_rank']
    wb       = wr_bonus(wr)

    def champ_score(base, ceil_, def_ceil):
        ds = min(defenses / def_ceil, 1.0)
        return round(base + (ceil_ - base) * (ds * 0.55 + wb * 0.45))

    if row['ever_champion']:
        if (dual and defenses >= 3) or defenses >= 8:
            return 'S',  champ_score(95, 100, 11)
        if defenses >= 5 or (dual and defenses >= 2):
            return 'A+', champ_score(85, 94, 7)
        if defenses >= 3 or (dual and defenses >= 1):
            return 'A',  champ_score(75, 84, 4)
        if defenses >= 1:
            return 'B+', champ_score(67, 74, 2)
        bonus = 3 if dual else 0
        return 'B', min(66, round(60 + bonus + (66-60) * wb * 0.5))

    def nc(base, ceil_, r_min, r_max):
        rp = rank_pct(best, r_min, r_max)
        return round(base + (ceil_ - base) * (rp * 0.65 + wb * 0.35))

    if pd.notna(best) and best <= 3:  return 'C+', nc(54, 59, 1, 3)
    if pd.notna(best) and best <= 5:  return 'C',  nc(47, 53, 4, 5)
    if pd.notna(best) and best <= 10: return 'D+', nc(40, 46, 6, 10)
    if pd.notna(best) and best <= 15: return 'D',  nc(33, 39, 11, 15)

    nf    = min(row['n_fights'], 30)
    score = wb * 20 + np.log1p(nf) * 3
    return 'E', round(min(32, score))

results          = fv.apply(compute_tier, axis=1)
fv['tier_label'] = results.apply(lambda x: x[0])
fv['tier_score'] = results.apply(lambda x: x[1])

# ══════════════════════════════════════════════════════
# PERCENTILE RANKS
# 所有 pct 都是 0-1 之間的真實百分比 (rank pct)
# 乘以 9 + 1 → 1-10 分
# Reach 特別處理：量級內 percentile
# ══════════════════════════════════════════════════════
print("Computing percentile ranks...")

def make_pct(series):
    """全選手 percentile → 1-10"""
    return (series.rank(pct=True, na_option='bottom') * 9 + 1).clip(1, 10).round(2)

# Reach: 量級內 percentile
def reach_pct_by_wc(df):
    result = pd.Series(index=df.index, dtype=float)
    for wc, grp in df.groupby('wc'):
        pct = grp['reach_cm'].rank(pct=True, na_option='bottom') * 9 + 1
        result.loc[grp.index] = pct.clip(1, 10).round(2)
    return result

fv['pct_reach']             = reach_pct_by_wc(fv)
fv['pct_striking_power']    = make_pct(fv['ko_rate'])
fv['pct_striking_volume']   = make_pct(fv['sig_per_r'])
fv['pct_striking_accuracy'] = make_pct(fv['sig_acc'])
fv['pct_striking_defense']  = make_pct(fv['str_def'])
fv['pct_leg_kicks']         = make_pct(fv['leg_pct'])
fv['pct_clinch']            = make_pct(fv['clinch_pct'])
fv['pct_body']              = make_pct(fv['body_per_r'])
fv['pct_td_frequency']      = make_pct(fv['td_per_r'])
fv['pct_td_accuracy']       = make_pct(fv['td_avg_acc'])
fv['pct_td_defense']        = make_pct(fv['td_def'])
fv['pct_submission']        = make_pct(fv['sub_rate'])
fv['pct_control']           = make_pct(fv['ctrl_per_r'])
fv['pct_ground_pound']      = make_pct(fv['gnp_per_r'])

# ══════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════
PCT_COLS = [
    'pct_reach',
    'pct_striking_power', 'pct_striking_volume',
    'pct_striking_accuracy', 'pct_striking_defense',
    'pct_leg_kicks', 'pct_clinch', 'pct_body',
    'pct_td_frequency', 'pct_td_accuracy', 'pct_td_defense',
    'pct_submission', 'pct_control', 'pct_ground_pound',
]

RAW_COLS = [
    'reach_cm',
    'ko_rate', 'sig_per_r', 'sig_acc', 'str_def',
    'leg_pct', 'clinch_pct', 'body_per_r',
    'td_per_r', 'td_avg_acc', 'td_def',
    'sub_rate', 'ctrl_per_r', 'gnp_per_r',
]

keep = (
    ['name','wc','n_fights','tier_label','tier_score',
     'best_rank','ever_champion','max_defenses','dual_weight_champ',
     'is_incumbent','win_rate'] +
    RAW_COLS + PCT_COLS +
    ['x','y','z']
)
keep = [c for c in keep if c in fv.columns]
out  = fv[keep].copy()

os.makedirs(CLEAN, exist_ok=True)
out.to_csv(OUT, index=False)

print(f"\n完成 → {OUT}")
print(f"選手數: {len(out)}")
print(f"PCT columns: {len(PCT_COLS)}")

print(f"\nTier 分佈:")
print(out['tier_label'].value_counts().sort_index())

print(f"\n重點選手:")
names = ['Khabib Nurmagomedov','Jon Jones','Islam Makhachev',
         'Sean Strickland','Max Holloway','TJ Dillashaw']
check = out[out['name'].isin(names)][
    ['name','tier_label','tier_score'] + PCT_COLS
].sort_values('tier_score', ascending=False)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
print(check.to_string())
