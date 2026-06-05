"""
dna_meta.py — MMA Meta-Game Evolution Analysis
===============================================
計算每個選手的打擊/摔角二元分數
然後按年份聚合，看 meta 的演化趨勢

輸出：
    dna_meta_fighters.csv  — 每個選手的二元分數
    dna_meta_yearly.csv    — 每年的 meta 趨勢
    dna_meta_champions.csv — 每個冠軍的二元分數 + 奪冠年份
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH      = "data/clean/fights_all_rounds.csv"
BUILDER_PATH     = "data/clean/fighter_builder_features.csv"
CHAMP_PATH       = "data/raw/ufc_championship_history.csv"
MIN_ROUNDS       = 15

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

# ══════════════════════════════════════════════════════
print("=" * 65)
print("MMA Meta-Game Evolution Analysis")
print("=" * 65)

# ── 讀資料 ──
print("\n[1/4] 讀取資料...")
fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)

try:
    champs = pd.read_csv(CHAMP_PATH)
    has_champ = True
    print(f"      championship history: {len(champs)} rows")
except:
    has_champ = False
    print("      championship history: 找不到，跳過冠軍分析")

print(f"      fights: {len(fights):,} rows")

# ── 計算風格特徵 ──
print("\n[2/4] 計算每個選手的風格特徵...")

g   = fights.groupby('fighter')
agg = pd.DataFrame()
agg['n_rounds']         = g['round'].count()
agg['total_sig_att']    = g['sig_str_attempted'].sum()
agg['total_sig_landed'] = g['sig_str_landed'].sum()
agg['dist_land']        = g['dist_landed'].sum()
agg['clinch_land']      = g['clinch_landed'].sum()
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

agg['distance_ratio']   = safe_div(agg['dist_land'].values,       agg['total_sig_landed'].values)
agg['ground_ratio']     = safe_div(agg['ground_land'].values,      agg['total_sig_landed'].values)
agg['td_per_round']     = safe_div(agg['td_land'].values,          agg['n_rounds'].values)
agg['ctrl_per_round']   = np.clip(safe_div(agg['ctrl_sec_total'].values, agg['n_rounds'].values), 0, 300)
agg['sub_per_round']    = safe_div(agg['sub_att_total'].values,    agg['n_rounds'].values)
agg['kd_per_round']     = safe_div(agg['kd_total'].values,         agg['n_rounds'].values)
agg['output_per_round'] = safe_div(agg['total_sig_att'].values,    agg['n_rounds'].values)

# ── 計算二元分數 ──
# 打擊分數：距離打擊佔比 + 高輸出 + KO傾向
# 摔角分數：摔抱頻率 + 控制時間 + 柔術傾向
# 最終分數：striking_score - grappling_score，正數=打擊系，負數=摔角系

print("\n[3/4] 計算打擊/摔角二元分數...")

# 先標準化每個特徵（z-score），讓不同量綱的特徵可比
from sklearn.preprocessing import StandardScaler
features_for_scoring = ['distance_ratio', 'ground_ratio', 'td_per_round',
                        'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round']
X = agg[features_for_scoring].fillna(0).values
scaler = StandardScaler()
X_z = scaler.fit_transform(X)
agg_z = pd.DataFrame(X_z, columns=features_for_scoring)

# 打擊分數 = 距離打擊佔比(z) + KO傾向(z) + 輸出量(z)
# 摔角分數 = 地面打擊(z) + 摔抱(z) + 控制時間(z) + 柔術(z)
agg['striking_score']   = (agg_z['distance_ratio'] + agg_z['kd_per_round'] + agg_z['output_per_round']) / 3
agg['grappling_score']  = (agg_z['ground_ratio'] + agg_z['td_per_round'] + agg_z['ctrl_per_round'] + agg_z['sub_per_round']) / 4

# meta_score: +1 = 純打擊系, -1 = 純摔角系
agg['meta_score'] = agg['striking_score'] - agg['grappling_score']

# 正規化到 0-100，50 = 中性
meta_min = agg['meta_score'].min()
meta_max = agg['meta_score'].max()
agg['meta_score_pct'] = (agg['meta_score'] - meta_min) / (meta_max - meta_min) * 100

# 標籤
agg['style_label'] = pd.cut(
    agg['meta_score_pct'],
    bins=[0, 30, 45, 55, 70, 100],
    labels=['純摔角系', '摔角傾向', '均衡', '打擊傾向', '純打擊系']
)

# 合併 tier
tier_info      = builder[['name', 'tier_label', 'ever_champion', 'wc']].copy()
agg            = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)

print(f"      選手數: {len(agg)}")
print(f"\n      風格分布:")
print(agg['style_label'].value_counts().to_string())

# 印出各風格代表選手
print(f"\n      各風格前5名（依 meta_score_pct）：")
for label in ['純打擊系', '打擊傾向', '均衡', '摔角傾向', '純摔角系']:
    subset = agg[agg['style_label'] == label].nlargest(5, 'meta_score_pct') if '打擊' in str(label) \
             else agg[agg['style_label'] == label].nsmallest(5, 'meta_score_pct')
    names  = subset['fighter'].tolist()
    print(f"      {label}: {', '.join(names)}")

# ── 按年份聚合，計算 meta 趨勢 ──
print("\n[4/4] 計算年度 meta 趨勢...")

fights['year'] = pd.to_datetime(fights['date'], errors='coerce').dt.year
fights_with_score = fights.merge(
    agg[['fighter', 'meta_score_pct', 'style_label', 'tier_label', 'ever_champion']],
    on='fighter', how='left'
)

# 每年：所有出賽選手的平均 meta_score（代表這一年的主流風格）
# 用 tier 加權，高端選手影響力更大
TIER_WEIGHTS = {'S':10,'A+':8,'A':6,'B+':4,'B':3,'C+':2,'D+':1,'D':1,'C':1,'E':0.5}
fights_with_score['tier_weight'] = fights_with_score['tier_label'].map(TIER_WEIGHTS).fillna(0.5)

yearly = fights_with_score.dropna(subset=['year', 'meta_score_pct']).copy()
yearly['year'] = yearly['year'].astype(int)

# 去重（每個選手每年只算一次）
yearly_dedup = yearly.drop_duplicates(subset=['fighter', 'year'])

yearly_agg = yearly_dedup.groupby('year').apply(lambda x: pd.Series({
    'n_fighters':        len(x),
    'meta_avg':          np.average(x['meta_score_pct'], weights=x['tier_weight']+0.1),
    'meta_unweighted':   x['meta_score_pct'].mean(),
    'pct_striking':      (x['meta_score_pct'] > 55).mean() * 100,
    'pct_grappling':     (x['meta_score_pct'] < 45).mean() * 100,
    'pct_balanced':      ((x['meta_score_pct'] >= 45) & (x['meta_score_pct'] <= 55)).mean() * 100,
})).reset_index()

# 只看有足夠資料的年份
yearly_agg = yearly_agg[yearly_agg['n_fighters'] >= 10].copy()

print(f"\n  年度 meta 趨勢（50 = 均衡，>50 = 打擊主流，<50 = 摔角主流）：")
print(f"  {'年份':>6} | {'加權均值':>8} | {'打擊%':>7} | {'摔角%':>7} | {'均衡%':>7} | {'選手數':>6}")
print(f"  {'-'*55}")

for _, row in yearly_agg.iterrows():
    bar_pos = '█' * int((row['meta_avg'] - 40) / 3) if row['meta_avg'] > 40 else ''
    bar_neg = '░' * int((50 - row['meta_avg']) / 3) if row['meta_avg'] < 50 else ''
    print(f"  {int(row['year']):>6} | {row['meta_avg']:>8.1f} | "
          f"{row['pct_striking']:>7.1f} | {row['pct_grappling']:>7.1f} | "
          f"{row['pct_balanced']:>7.1f} | {int(row['n_fighters']):>6}")

# ── 冠軍分析 ──
if has_champ:
    print(f"\n  冠軍的風格分布：")
    champ_scores = agg[agg['ever_champion'] == True][['fighter', 'meta_score_pct', 'style_label', 'wc']].copy()
    champ_scores = champ_scores.sort_values('meta_score_pct', ascending=False)
    print(f"  純打擊系冠軍: { champ_scores[champ_scores['style_label']=='純打擊系']['fighter'].tolist()[:8] }")
    print(f"  打擊傾向冠軍: { champ_scores[champ_scores['style_label']=='打擊傾向']['fighter'].tolist()[:8] }")
    print(f"  均衡型冠軍:   { champ_scores[champ_scores['style_label']=='均衡']['fighter'].tolist()[:8] }")
    print(f"  摔角傾向冠軍: { champ_scores[champ_scores['style_label']=='摔角傾向']['fighter'].tolist()[:8] }")
    print(f"  純摔角系冠軍: { champ_scores[champ_scores['style_label']=='純摔角系']['fighter'].tolist()[:8] }")

# ── 輸出 ──
output_cols = ['fighter', 'wc', 'tier_label', 'ever_champion', 'win_rate',
               'n_rounds', 'meta_score_pct', 'striking_score', 'grappling_score', 'style_label']
agg[output_cols].to_csv('dna_meta_fighters.csv', index=False)
yearly_agg.to_csv('dna_meta_yearly.csv', index=False)

print(f"\n\n輸出：")
print(f"  dna_meta_fighters.csv — 每個選手的二元分數")
print(f"  dna_meta_yearly.csv   — 每年的 meta 趨勢")
print("=" * 65)
