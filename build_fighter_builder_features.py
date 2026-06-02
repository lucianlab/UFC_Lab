"""
build_fighter_builder_features.py
==================================
產出 fighter_builder_features.csv，供 Fighter Builder 的 KNN 和 Tier Score 使用。

四個資料源各司其職，不混用：
  fighters_all.csv          → 基礎 stats (striking, grappling, physical)
  fights_all_rounds.csv     → aggregate 細項 (leg kicks, clinch, ground, ctrl)
  ufc_championship_history  → 冠軍防衛次數
  rankings_history.csv      → 歷史最佳排名 (2013+)
  ufc_active_top15_rankings → 現役排名補充

執行：
  python3 build_fighter_builder_features.py

輸出：
  data/clean/fighter_builder_features.csv
"""

import pandas as pd
import numpy as np
import os

# ── 路徑設定 ──────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_CLEAN = os.path.join(BASE, 'data', 'clean')
OUT_PATH   = os.path.join(DATA_CLEAN, 'fighter_builder_features.csv')

FIGHTERS_CSV      = os.path.join(DATA_CLEAN, 'fighters_all.csv')
ROUNDS_CSV        = os.path.join(DATA_CLEAN, 'fights_all_rounds.csv')   # 調整成你實際路徑
CHAMP_CSV         = os.path.join(BASE, 'data', 'raw', 'ufc_championship_history.csv')
RANKINGS_HIST_CSV = os.path.join(DATA_CLEAN, 'rankings_history.csv')
ACTIVE_RANK_CSV   = os.path.join(os.path.join(BASE, 'data', 'raw', 'ufc_active_top15_rankings.csv'))

# ── 讀檔 ──────────────────────────────────────────────
print("載入資料 / Loading data...")
fighters  = pd.read_csv(FIGHTERS_CSV)
rounds    = pd.read_csv(ROUNDS_CSV)
champ     = pd.read_csv(CHAMP_CSV)
rank_hist = pd.read_csv(RANKINGS_HIST_CSV)
active    = pd.read_csv(ACTIVE_RANK_CSV)

# ═══════════════════════════════════════════════════════
# BLOCK 1 — fighters_all.csv
# 直接用的欄位：splm, str_acc, str_def, td_avg, td_def, sub_avg, reach_cm
# 衍生：win_rate, ko_rate (用 championship_history 沒有，這裡先用 wins/losses 估)
# ═══════════════════════════════════════════════════════
print("Block 1: fighters_all stats...")

f = fighters[['name','height_cm','reach_cm','splm','str_acc','str_def',
              'td_avg','td_def','sub_avg','wins','losses']].copy()

f['n_fights']    = f['wins'] + f['losses']
f['win_rate']    = np.where(f['n_fights'] > 0, f['wins'] / f['n_fights'], np.nan)

# 只保留有足夠比賽紀錄的選手 (至少 3 場)
f = f[f['n_fights'] >= 3].reset_index(drop=True)

# ═══════════════════════════════════════════════════════
# BLOCK 2 — fights_all_rounds.csv
# aggregate per fighter：leg_per_r, clinch_per_r, gnd_per_r, ctrl_per_r
# ko_rate 也從這裡算（method == KO/TKO）
# ═══════════════════════════════════════════════════════
print("Block 2: rounds aggregate...")

# 每個 fighter 的總數據（跨所有 round）
r = rounds.groupby('fighter').agg(
    total_rounds      = ('round',         'count'),
    leg_landed_total  = ('leg_landed',    'sum'),
    clinch_landed_total=('clinch_landed', 'sum'),
    gnd_landed_total  = ('ground_landed', 'sum'),
    ctrl_sec_total    = ('ctrl_sec',      'sum'),
).reset_index()

r['leg_per_r']   = r['leg_landed_total']   / r['total_rounds']
r['clinch_per_r']= r['clinch_landed_total']/ r['total_rounds']
r['gnd_per_r']   = r['gnd_landed_total']   / r['total_rounds']
r['ctrl_per_r']  = r['ctrl_sec_total']     / r['total_rounds']

# KO rate: 在 fighter 視角下，贏的場次中有多少是 KO/TKO
# 每場比賽只取最後一 round（finish_round）避免重複計算
fights_dedup = rounds.drop_duplicates(subset=['event','bout','fighter'])
ko_df = fights_dedup.groupby('fighter').agg(
    total_fights = ('won', 'count'),
    ko_wins      = ('method', lambda x: ((x.str.contains('KO|TKO', na=False)) &
                                          (fights_dedup.loc[x.index,'won'] == 1)).sum())
).reset_index()
ko_df['ko_rate'] = np.where(ko_df['total_fights'] > 0,
                             ko_df['ko_wins'] / ko_df['total_fights'], np.nan)

r = r.merge(ko_df[['fighter','ko_rate']], on='fighter', how='left')

# ═══════════════════════════════════════════════════════
# BLOCK 3 — championship_history.csv
# 每個選手的：是否曾拿冠軍、最高防衛次數、雙量級冠軍
# ═══════════════════════════════════════════════════════
print("Block 3: championship history...")

# 只計算 Undisputed / Interim（排除 symbolic）
champ_real = champ[champ['title_type'].isin(['Undisputed','Interim'])].copy()

champ_agg = champ_real.groupby('champion').agg(
    max_defenses      = ('title_defenses_count', 'max'),
    total_reigns      = ('reign_number_for_fighter', 'max'),
    n_weight_classes  = ('weight_class', 'nunique'),
    is_incumbent      = ('incumbent', 'max'),
).reset_index()
champ_agg.rename(columns={'champion': 'name'}, inplace=True)
champ_agg['ever_champion']      = True
champ_agg['dual_weight_champ']  = champ_agg['n_weight_classes'] >= 2

# ═══════════════════════════════════════════════════════
# BLOCK 4 — rankings_history.csv  (2013+)
# 每個選手的歷史最佳排名（排除 P4P，只看量級排名）
# ═══════════════════════════════════════════════════════
print("Block 4: historical rankings...")

# 排除 P4P 排名（只看量級排名）
rank_div = rank_hist[~rank_hist['weightclass'].str.contains(
    'Pound-for-Pound', na=False)].copy()

# rank 0 = champion in some datasets, 1 = #1 contender etc.
# 我們用數字最小 = 最好 (champion/rank 0 視為最高)
best_rank = rank_div.groupby('fighter')['rank'].min().reset_index()
best_rank.rename(columns={'rank': 'best_hist_rank'}, inplace=True)

# ═══════════════════════════════════════════════════════
# BLOCK 5 — active_top15_rankings.csv
# 現役排名（補充 rankings_history 可能還沒更新到的最新名次）
# ═══════════════════════════════════════════════════════
print("Block 5: active rankings...")

active_div = active[~active['division'].str.contains(
    'Pound-for-Pound', na=False)].copy()

# rank 欄位可能是字串 "1".."15" 或 "C"
def parse_rank(r):
    try:
        return int(r)
    except:
        return 0  # Champion = 0

active_div['rank_num'] = active_div['rank'].apply(parse_rank)

active_best = active_div.groupby('fighter')['rank_num'].min().reset_index()
active_best.rename(columns={'fighter':'name','rank_num':'active_best_rank'}, inplace=True)

# ═══════════════════════════════════════════════════════
# MERGE — 合併所有 blocks
# ═══════════════════════════════════════════════════════
print("Merging all blocks...")

df = f.copy()

# Block 2: rounds stats
df = df.merge(
    r[['fighter','leg_per_r','clinch_per_r','gnd_per_r','ctrl_per_r','ko_rate']],
    left_on='name', right_on='fighter', how='left'
).drop(columns=['fighter'])

# Block 3: championship
df = df.merge(champ_agg[['name','ever_champion','max_defenses',
                          'dual_weight_champ','is_incumbent']],
              on='name', how='left')
df['ever_champion']   = df['ever_champion'].fillna(False)
df['dual_weight_champ']= df['dual_weight_champ'].fillna(False)
df['max_defenses']    = df['max_defenses'].fillna(0)
df['is_incumbent']    = df['is_incumbent'].fillna(False)

# Block 4: historical best rank
df = df.merge(best_rank, left_on='name', right_on='fighter', how='left').drop(
    columns=['fighter'], errors='ignore')

# Block 5: active best rank
df = df.merge(active_best, on='name', how='left')

# 取兩個 rank source 的最佳值
df['best_rank'] = df[['best_hist_rank','active_best_rank']].min(axis=1)

# ═══════════════════════════════════════════════════════
# TIER SCORE — 各資料源獨立計算，最後合成
# ═══════════════════════════════════════════════════════
print("Computing tier scores...")

def compute_tier(row):
    # 冠軍 → championship_history 決定
    if row['ever_champion']:
        defenses = row['max_defenses'] or 0
        if row['dual_weight_champ'] or defenses >= 5:
            return 'S', 97
        elif defenses >= 2:
            return 'A+', 89
        else:
            return 'A', 78

    # 非冠軍 → 歷史最佳排名決定
    best = row['best_rank']
    if pd.notna(best):
        if best <= 3:
            return 'B+', 70
        elif best <= 10:
            return 'B', 60
        elif best <= 15:
            return 'C', 50

    # 從未入 top 15 → win rate + ko_rate 估 D tier
    wr  = row['win_rate']  if pd.notna(row['win_rate'])  else 0.5
    ko  = row['ko_rate']   if pd.notna(row['ko_rate'])   else 0.1
    nf  = min(row['n_fights'], 30)
    score = wr * 25 + ko * 10 + np.log1p(nf) * 2
    return 'D', round(min(44, score))

tiers  = df.apply(compute_tier, axis=1)
df['tier']        = tiers.apply(lambda x: x[0])
df['tier_score']  = tiers.apply(lambda x: x[1])

# ═══════════════════════════════════════════════════════
# PERCENTILE RANKS — 14 個 builder features 的百分位
# 這才是玩家看到的 1-10 分的基礎
# ═══════════════════════════════════════════════════════
print("Computing percentile ranks...")

BUILDER_FEATURES = {
    # Physical
    'height_cm'   : 'pct_height',
    'reach_cm'    : 'pct_reach',
    # Boxing
    'ko_rate'     : 'pct_striking_power',
    'splm'        : 'pct_striking_volume',
    'str_acc'     : 'pct_striking_accuracy',
    'str_def'     : 'pct_striking_defense',
    # Kickboxing
    'leg_per_r'   : 'pct_leg_kicks',
    'clinch_per_r': 'pct_clinch',
    # Wrestling
    'td_avg'      : 'pct_takedown_offense',
    'td_def'      : 'pct_takedown_defense',
    'ctrl_per_r'  : 'pct_ground_control',
    # BJJ
    'sub_avg'     : 'pct_submission',
    'gnd_per_r'   : 'pct_ground_pound',
    # Cardio (win_rate as proxy for now)
    'win_rate'    : 'pct_cardio',
}

for raw_col, pct_col in BUILDER_FEATURES.items():
    if raw_col in df.columns:
        df[pct_col] = df[raw_col].rank(pct=True, na_option='bottom') * 9 + 1
        df[pct_col] = df[pct_col].clip(1, 10).round(2)
    else:
        print(f"  WARNING: {raw_col} not found, skipping {pct_col}")

# ═══════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════
keep_cols = (
    ['name', 'n_fights', 'tier', 'tier_score', 'best_rank',
     'ever_champion', 'max_defenses', 'dual_weight_champ', 'is_incumbent'] +
    list(BUILDER_FEATURES.keys()) +
    list(BUILDER_FEATURES.values())
)
keep_cols = [c for c in keep_cols if c in df.columns]
out = df[keep_cols].copy()

os.makedirs(DATA_CLEAN, exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"\n完成 / Done → {OUT_PATH}")
print(f"選手數 / Fighters: {len(out)}")
print(f"\nTier 分佈 / Distribution:")
print(out['tier'].value_counts().sort_index())
print(f"\nS tier 選手:")
print(out[out['tier']=='S'][['name','max_defenses','dual_weight_champ','tier_score']].to_string())
