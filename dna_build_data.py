"""
dna_build_data.py — DNA Mode 資料建構 v2
=========================================
修正：
- 冠軍取 title_fights_won = max(defense_count) + 1
- 同一選手多段統治累計
- 半徑 r = 4 * sqrt(title_fights_won) + 6
- 只顯示 2007 年後（樣本 > 100）
- 2026 標記為進行中

輸出：
    dna_fighters.csv   — 每個選手的風格分數
    dna_yearly.csv     — 年度 meta 趨勢
    dna_champions.csv  — 冠軍錨點
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH  = "data/clean/fights_all_rounds.csv"
BUILDER_PATH = "data/clean/fighter_builder_features.csv"
CHAMP_PATH   = "data/raw/ufc_championship_history.csv"
MIN_ROUNDS   = 15
TIER_WEIGHTS = {'S':10,'A+':8,'A':6,'B+':4,'B':3,'C+':2,'D+':1,'D':1,'C':1,'E':0.5}

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

print("=" * 65)
print("DNA Build Data v2")
print("=" * 65)

# ══════════════════════════════════════════════════════
#  STEP 1：計算每個選手的風格特徵與 meta_score
# ══════════════════════════════════════════════════════
print("\n[1/4] 計算選手風格特徵...")

fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)
champs  = pd.read_csv(CHAMP_PATH)

g   = fights.groupby('fighter')
agg = pd.DataFrame()
agg['n_rounds']         = g['round'].count()
agg['total_sig_att']    = g['sig_str_attempted'].sum()
agg['total_sig_landed'] = g['sig_str_landed'].sum()
agg['dist_land']        = g['dist_landed'].sum()
agg['ground_land']      = g['ground_landed'].sum()
agg['td_att']           = g['td_attempted'].sum()
agg['td_land']          = g['td_landed'].sum()
agg['ctrl_sec_total']   = g['ctrl_sec'].sum()
agg['sub_att_total']    = g['sub_att'].sum()
agg['kd_total']         = g['kd'].sum()
agg['n_fights']         = g['bout'].nunique()

wins_df = fights[fights['won']==1].groupby('fighter')['bout'].nunique().rename('wins')
agg     = agg.join(wins_df)
agg['wins']     = agg['wins'].fillna(0)
agg['win_rate'] = safe_div(agg['wins'].values, agg['n_fights'].values)
agg             = agg.reset_index()
agg             = agg[agg['n_rounds'] >= MIN_ROUNDS].copy()

agg['distance_ratio']  = safe_div(agg['dist_land'].values,       agg['total_sig_landed'].values)
agg['ground_ratio']    = safe_div(agg['ground_land'].values,      agg['total_sig_landed'].values)
agg['td_per_round']    = safe_div(agg['td_land'].values,          agg['n_rounds'].values)
agg['ctrl_per_round']  = np.clip(safe_div(agg['ctrl_sec_total'].values, agg['n_rounds'].values), 0, 300)
agg['sub_per_round']   = safe_div(agg['sub_att_total'].values,    agg['n_rounds'].values)
agg['kd_per_round']    = safe_div(agg['kd_total'].values,         agg['n_rounds'].values)
agg['output_per_round']= safe_div(agg['total_sig_att'].values,    agg['n_rounds'].values)

dist_min, dist_max = agg['distance_ratio'].min(), agg['distance_ratio'].max()
ctrl_min, ctrl_max = agg['ctrl_per_round'].min(), agg['ctrl_per_round'].max()
td_min,   td_max   = agg['td_per_round'].min(),   agg['td_per_round'].max()

agg['dist_norm'] = (agg['distance_ratio'] - dist_min) / (dist_max - dist_min)
agg['ctrl_norm'] = (agg['ctrl_per_round'] - ctrl_min) / (ctrl_max - ctrl_min)
agg['td_norm']   = (agg['td_per_round']   - td_min)   / (td_max   - td_min)

agg['striking_signal']  = agg['dist_norm']
agg['grappling_signal'] = (agg['ctrl_norm'] + agg['td_norm']) / 2
agg['meta_score']       = agg['striking_signal'] - agg['grappling_signal']

m_min, m_max          = agg['meta_score'].min(), agg['meta_score'].max()
agg['meta_score_pct'] = (agg['meta_score'] - m_min) / (m_max - m_min) * 100

tier_info      = builder[['name', 'tier_label', 'ever_champion', 'wc']].copy()
agg            = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)
agg['tier_weight']   = agg['tier_label'].map(TIER_WEIGHTS).fillna(0.5)

print(f"      {len(agg)} 選手")

fighter_cols = ['fighter', 'wc', 'tier_label', 'ever_champion',
                'win_rate', 'n_rounds', 'meta_score_pct',
                'striking_signal', 'grappling_signal']
agg[fighter_cols].to_csv('dna_fighters.csv', index=False)
print(f"      → dna_fighters.csv")

# ══════════════════════════════════════════════════════
#  STEP 2：年度 meta 趨勢
# ══════════════════════════════════════════════════════
print("\n[2/4] 計算年度 meta 趨勢...")

fights['year'] = pd.to_datetime(fights['date'], errors='coerce').dt.year
fights_scored  = fights.merge(
    agg[['fighter', 'meta_score_pct', 'tier_label', 'tier_weight']],
    on='fighter', how='left'
)
fights_scored = fights_scored.dropna(subset=['year', 'meta_score_pct']).copy()
fights_scored['year'] = fights_scored['year'].astype(int)

fights_scored['is_striking']  = fights_scored['meta_score_pct'] > 58
fights_scored['is_grappling'] = fights_scored['meta_score_pct'] < 42

yearly_rows = []
for year, grp in fights_scored.groupby('year'):
    if len(grp) < 20:
        continue

    fighters_this_year = grp.drop_duplicates('fighter')
    n_total     = len(fighters_this_year)
    n_striking  = fighters_this_year['is_striking'].sum()
    n_grappling = fighters_this_year['is_grappling'].sum()
    n_balanced  = n_total - n_striking - n_grappling

    grp_wins    = grp.dropna(subset=['won'])
    str_grp     = grp_wins[grp_wins['is_striking']]
    grp_grp     = grp_wins[grp_wins['is_grappling']]

    str_winrate = str_grp['won'].mean() if len(str_grp) >= 10 else np.nan
    grp_winrate = grp_grp['won'].mean() if len(grp_grp) >= 10 else np.nan
    winrate_diff = (str_winrate - grp_winrate) * 100 \
                   if (not np.isnan(str_winrate) and not np.isnan(grp_winrate)) else np.nan

    yearly_rows.append({
        'year':          year,
        'n_fighters':    n_total,
        'pct_striking':  round(n_striking  / n_total * 100, 1),
        'pct_grappling': round(n_grappling / n_total * 100, 1),
        'pct_balanced':  round(n_balanced  / n_total * 100, 1),
        'str_winrate':   round(str_winrate  * 100, 1) if not np.isnan(str_winrate)  else None,
        'grp_winrate':   round(grp_winrate  * 100, 1) if not np.isnan(grp_winrate)  else None,
        'winrate_diff':  round(winrate_diff, 1)        if not np.isnan(winrate_diff) else None,
        'is_partial':    year >= 2026,
    })

yearly_df = pd.DataFrame(yearly_rows)
yearly_df = yearly_df[
    (yearly_df['year'] >= 2007) &
    (yearly_df['n_fighters'] >= 50)
].copy()

print(f"\n  {'年份':>6} | {'打擊%':>7} | {'摔角%':>7} | {'打擊勝率':>9} | {'摔角勝率':>9} | {'勝率差':>7} | {'n':>5}")
print(f"  {'-'*65}")
for _, row in yearly_df.iterrows():
    diff_str = f"{row['winrate_diff']:>+7.1f}" if row['winrate_diff'] is not None else f"{'N/A':>7}"
    partial  = '*' if row['is_partial'] else ' '
    print(f"  {int(row['year']):>6}{partial}| {row['pct_striking']:>7.1f} | {row['pct_grappling']:>7.1f} | "
          f"{str(row['str_winrate']):>9} | {str(row['grp_winrate']):>9} | {diff_str} | {int(row['n_fighters']):>5}")

yearly_df.to_csv('dna_yearly.csv', index=False)
print(f"\n      → dna_yearly.csv")

# ══════════════════════════════════════════════════════
#  STEP 3：冠軍錨點
#  修正：同一選手取最大防衛次數那筆，多段統治累計
#  title_fights_won = sum of (defense_count + 1) across all reigns
# ══════════════════════════════════════════════════════
print("\n[3/4] 計算冠軍錨點...")

champs_clean = champs[
    champs['title_type'].isin(['Undisputed', 'Interim']) |
    champs['title_type'].isna()
].copy()

champs_clean['date_won']          = pd.to_datetime(champs_clean['date_won'], errors='coerce')
champs_clean['year_won']          = champs_clean['date_won'].dt.year
champs_clean['reign_days']        = pd.to_numeric(champs_clean['reign_days'],          errors='coerce').fillna(180)
champs_clean['title_defenses_count'] = pd.to_numeric(champs_clean['title_defenses_count'], errors='coerce').fillna(0)

# 每段統治的 title fights won = defense_count + 1（奪冠那場）
champs_clean['reign_title_fights'] = champs_clean['title_defenses_count'] + 1

# 統治期中點
champs_clean['title_year'] = (
    champs_clean['date_won'] +
    pd.to_timedelta(champs_clean['reign_days'] / 2, unit='d')
).dt.year

# 每個選手：用最長那段統治的中點作為 X 軸位置
# title_fights_won = 累計所有段數
champ_summary = champs_clean.groupby('champion').apply(lambda x: pd.Series({
    'title_fights_won': x['reign_title_fights'].sum(),
    'title_year':       x.loc[x['reign_days'].idxmax(), 'title_year'],   # 最長統治期的中點
    'best_defense':     x['title_defenses_count'].max(),
    'n_reigns':         len(x),
})).reset_index()

# 合併 meta_score
champ_summary = champ_summary.merge(
    agg[['fighter', 'meta_score_pct', 'wc', 'tier_label']],
    left_on='champion', right_on='fighter', how='left'
)

# 半徑
champ_summary['radius'] = (4 * np.sqrt(champ_summary['title_fights_won']) + 6).round(1)

champs_out = champ_summary.dropna(subset=['meta_score_pct', 'title_year']).copy()
champs_out = champs_out[champs_out['title_year'] >= 2007].copy()
champs_out = champs_out.sort_values('title_year')

print(f"\n  {'選手':<28} {'量級':<18} {'中點年':>6} | {'meta':>6} | {'title_W':>7} | {'r':>5}")
print(f"  {'-'*72}")
for _, row in champs_out.iterrows():
    print(f"  {row['champion']:<28} {str(row['wc']):<18} "
          f"{int(row['title_year']):>6} | "
          f"{row['meta_score_pct']:>6.1f} | {int(row['title_fights_won']):>7} | {row['radius']:>5.1f}")

print(f"\n  關鍵選手驗證：")
for name in ['Khabib Nurmagomedov','Anderson Silva','Jon Jones',
             'Georges St-Pierre','Demetrious Johnson','Alex Pereira']:
    row = champs_out[champs_out['champion'] == name]
    if len(row):
        r = row.iloc[0]
        print(f"  {name:<28} meta={r['meta_score_pct']:>5.1f}  "
              f"year={int(r['title_year'])}  "
              f"title_W={int(r['title_fights_won'])}  r={r['radius']}")
    else:
        print(f"  {name:<28} NOT FOUND")

out_cols = ['champion', 'wc', 'tier_label', 'title_year',
            'meta_score_pct', 'title_fights_won', 'radius', 'n_reigns']
champs_out[out_cols].to_csv('dna_champions.csv', index=False)
print(f"\n      → dna_champions.csv")

# ══════════════════════════════════════════════════════
print(f"\n[4/4] 摘要")
print(f"  dna_fighters.csv  : {len(agg)} 選手")
print(f"  dna_yearly.csv    : {len(yearly_df)} 年（{int(yearly_df['year'].min())}-{int(yearly_df['year'].max())}）")
print(f"  dna_champions.csv : {len(champs_out)} 冠軍")
print("=" * 65)
