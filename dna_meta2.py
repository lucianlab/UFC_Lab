"""
dna_meta2.py — MMA Meta-Game Evolution Analysis v2
===================================================
修正版：用 distance_ratio vs ctrl_per_round+td_per_round 作為核心二元軸
這兩個特徵是 anchor 分析裡最能區分兩群的信號

輸出：
    dna_meta_fighters.csv — 每個選手的二元分數
    dna_meta_yearly.csv   — 每年的 meta 趨勢
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH  = "data/clean/fights_all_rounds.csv"
BUILDER_PATH = "data/clean/fighter_builder_features.csv"
MIN_ROUNDS   = 15

TIER_WEIGHTS = {'S':10,'A+':8,'A':6,'B+':4,'B':3,'C+':2,'D+':1,'D':1,'C':1,'E':0.5}

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

print("=" * 65)
print("MMA Meta-Game Evolution v2")
print("=" * 65)

# ── 讀資料 ──
print("\n[1/4] 讀取資料...")
fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)

# ── 計算特徵 ──
print("[2/4] 計算特徵...")
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

# 核心特徵
agg['distance_ratio']  = safe_div(agg['dist_land'].values, agg['total_sig_landed'].values)
agg['ground_ratio']    = safe_div(agg['ground_land'].values, agg['total_sig_landed'].values)
agg['td_per_round']    = safe_div(agg['td_land'].values, agg['n_rounds'].values)
agg['ctrl_per_round']  = np.clip(safe_div(agg['ctrl_sec_total'].values, agg['n_rounds'].values), 0, 300)
agg['sub_per_round']   = safe_div(agg['sub_att_total'].values, agg['n_rounds'].values)
agg['kd_per_round']    = safe_div(agg['kd_total'].values, agg['n_rounds'].values)
agg['output_per_round']= safe_div(agg['total_sig_att'].values, agg['n_rounds'].values)

# ── 二元分數計算 ──
print("[3/4] 計算二元分數...")

# 打擊信號：distance_ratio（最純粹的打擊意圖指標）
# 摔角信號：ctrl_per_round（最純粹的摔角意圖指標）
# 兩者都正規化到 0-1 再相減

dist_min, dist_max = agg['distance_ratio'].min(), agg['distance_ratio'].max()
ctrl_min, ctrl_max = agg['ctrl_per_round'].min(), agg['ctrl_per_round'].max()

agg['dist_norm'] = (agg['distance_ratio'] - dist_min) / (dist_max - dist_min)
agg['ctrl_norm'] = (agg['ctrl_per_round'] - ctrl_min) / (ctrl_max - ctrl_min)

# meta_score: +1 = 純打擊, -1 = 純摔角
# 加入 td_per_round 作為摔角信號的補充（避免只靠 ctrl）
td_min, td_max = agg['td_per_round'].min(), agg['td_per_round'].max()
agg['td_norm'] = (agg['td_per_round'] - td_min) / (td_max - td_min)

agg['striking_signal']  = agg['dist_norm']
agg['grappling_signal'] = (agg['ctrl_norm'] + agg['td_norm']) / 2

agg['meta_score'] = agg['striking_signal'] - agg['grappling_signal']

# 正規化到 0-100
m_min, m_max = agg['meta_score'].min(), agg['meta_score'].max()
agg['meta_score_pct'] = (agg['meta_score'] - m_min) / (m_max - m_min) * 100

# 標籤
agg['style_label'] = pd.cut(
    agg['meta_score_pct'],
    bins=[0, 25, 42, 58, 75, 100],
    labels=['純摔角系', '摔角傾向', '均衡', '打擊傾向', '純打擊系'],
    include_lowest=True
)

# 合併 tier
tier_info      = builder[['name', 'tier_label', 'ever_champion', 'wc']].copy()
agg            = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)
agg['tier_weight']   = agg['tier_label'].map(TIER_WEIGHTS).fillna(0.5)

# 印出分布
print(f"\n      風格分布:")
print(agg['style_label'].value_counts().sort_index().to_string())

# 驗證關鍵選手
check = ['Khabib Nurmagomedov', 'Islam Makhachev', 'Georges St-Pierre',
         'Conor McGregor', 'Alex Pereira', 'Anderson Silva',
         'Alexander Volkanovski', 'Ciryl Gane', 'Jon Jones',
         'Charles Oliveira', 'Amanda Nunes', 'Carla Esparza']

print(f"\n      關鍵選手驗證:")
print(f"      {'選手':<30} {'meta_score':>10} {'風格':>10}")
print(f"      {'-'*55}")
for name in check:
    row = agg[agg['fighter'] == name]
    if len(row):
        score = row['meta_score_pct'].values[0]
        label = str(row['style_label'].values[0])
        print(f"      {name:<30} {score:>10.1f} {label:>10}")

# ── 年度趨勢 ──
print(f"\n[4/4] 計算年度 meta 趨勢...")

fights['year'] = pd.to_datetime(fights['date'], errors='coerce').dt.year
fights_scored  = fights.merge(
    agg[['fighter', 'meta_score_pct', 'style_label', 'tier_label', 'tier_weight', 'ever_champion']],
    on='fighter', how='left'
)

yearly = fights_scored.dropna(subset=['year', 'meta_score_pct']).copy()
yearly['year'] = yearly['year'].astype(int)
yearly_dedup   = yearly.drop_duplicates(subset=['fighter', 'year'])

yearly_agg = yearly_dedup.groupby('year').apply(lambda x: pd.Series({
    'n_fighters':      len(x),
    'meta_avg':        np.average(x['meta_score_pct'], weights=x['tier_weight'] + 0.1),
    'meta_median':     x['meta_score_pct'].median(),
    'pct_striking':    (x['meta_score_pct'] > 58).mean() * 100,
    'pct_grappling':   (x['meta_score_pct'] < 42).mean() * 100,
    'pct_balanced':    ((x['meta_score_pct'] >= 42) & (x['meta_score_pct'] <= 58)).mean() * 100,
})).reset_index()

yearly_agg = yearly_agg[
    (yearly_agg['n_fighters'] >= 10) &
    (yearly_agg['year'] >= 1993)
].copy()

print(f"\n  年度趨勢（50=均衡, >50=打擊主流, <50=摔角主流）：")
print(f"  {'年份':>6} | {'加權均值':>8} | {'打擊%':>7} | {'摔角%':>7} | {'均衡%':>7} | {'n':>5}")
print(f"  {'-'*52}")
for _, row in yearly_agg.iterrows():
    arrow = '▲' if row['meta_avg'] > 55 else ('▼' if row['meta_avg'] < 45 else '—')
    print(f"  {int(row['year']):>6} | {row['meta_avg']:>8.1f} {arrow} | "
          f"{row['pct_striking']:>7.1f} | {row['pct_grappling']:>7.1f} | "
          f"{row['pct_balanced']:>7.1f} | {int(row['n_fighters']):>5}")

# 冠軍分布
print(f"\n  冠軍風格分布：")
champ_df = agg[agg['ever_champion']==True].copy()
for label in ['純打擊系', '打擊傾向', '均衡', '摔角傾向', '純摔角系']:
    names = champ_df[champ_df['style_label']==label].sort_values(
        'meta_score_pct', ascending=('摔角' in label)
    )['fighter'].tolist()
    print(f"  {label}: {names[:6]}")

# ── 輸出 ──
out_cols = ['fighter', 'wc', 'tier_label', 'ever_champion', 'win_rate',
            'n_rounds', 'meta_score_pct', 'striking_signal',
            'grappling_signal', 'style_label']
agg[out_cols].to_csv('dna_meta_fighters.csv', index=False)
yearly_agg.to_csv('dna_meta_yearly.csv', index=False)

print(f"\n輸出：dna_meta_fighters.csv / dna_meta_yearly.csv")
print("=" * 65)
