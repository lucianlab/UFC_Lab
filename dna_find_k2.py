"""
dna_find_k2.py — 用 GMM + BIC/AIC 找最佳 k
=============================================
GMM 允許軟邊界（每個選手有多個原型的混合比例）
BIC/AIC 是 GMM 的原生模型選擇指標，比 silhouette 更適合

使用方式：
    python3 dna_find_k2.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH  = "data/clean/fights_all_rounds.csv"
BUILDER_PATH = "data/clean/fighter_builder_features.csv"
MIN_ROUNDS   = 15
K_RANGE      = range(2, 13)

STYLE_FEATURES = [
    'distance_ratio', 'clinch_ratio', 'ground_ratio',
    'head_ratio', 'body_ratio', 'leg_ratio',
    'str_accuracy', 'td_per_round', 'td_accuracy',
    'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round',
]

# ── 讀資料（同前）──
print("讀取資料...")
fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

g   = fights.groupby('fighter')
agg = pd.DataFrame()
agg['n_rounds']         = g['round'].count()
agg['total_sig_att']    = g['sig_str_attempted'].sum()
agg['total_sig_landed'] = g['sig_str_landed'].sum()
agg['dist_land']        = g['dist_landed'].sum()
agg['clinch_land']      = g['clinch_landed'].sum()
agg['ground_land']      = g['ground_landed'].sum()
agg['head_land']        = g['head_landed'].sum()
agg['body_land']        = g['body_landed'].sum()
agg['leg_land']         = g['leg_landed'].sum()
agg['td_att']           = g['td_attempted'].sum()
agg['td_land']          = g['td_landed'].sum()
agg['ctrl_sec_total']   = g['ctrl_sec'].sum()
agg['sub_att_total']    = g['sub_att'].sum()
agg['kd_total']         = g['kd'].sum()
agg['n_fights']         = g['bout'].nunique()
wins_df = fights[fights['won']==1].groupby('fighter')['bout'].nunique().rename('wins')
agg     = agg.join(wins_df)
agg['wins'] = agg['wins'].fillna(0)
agg         = agg.reset_index()
agg         = agg[agg['n_rounds'] >= MIN_ROUNDS].copy()

agg['distance_ratio']   = safe_div(agg['dist_land'].values,       agg['total_sig_landed'].values)
agg['clinch_ratio']     = safe_div(agg['clinch_land'].values,      agg['total_sig_landed'].values)
agg['ground_ratio']     = safe_div(agg['ground_land'].values,      agg['total_sig_landed'].values)
agg['head_ratio']       = safe_div(agg['head_land'].values,        agg['total_sig_landed'].values)
agg['body_ratio']       = safe_div(agg['body_land'].values,        agg['total_sig_landed'].values)
agg['leg_ratio']        = safe_div(agg['leg_land'].values,         agg['total_sig_landed'].values)
agg['str_accuracy']     = safe_div(agg['total_sig_landed'].values, agg['total_sig_att'].values)
agg['td_per_round']     = safe_div(agg['td_land'].values,          agg['n_rounds'].values)
agg['td_accuracy']      = safe_div(agg['td_land'].values,          agg['td_att'].values, fill=0.5)
agg['ctrl_per_round']   = np.clip(safe_div(agg['ctrl_sec_total'].values, agg['n_rounds'].values), 0, 300)
agg['sub_per_round']    = safe_div(agg['sub_att_total'].values,    agg['n_rounds'].values)
agg['kd_per_round']     = safe_div(agg['kd_total'].values,         agg['n_rounds'].values)
agg['output_per_round'] = safe_div(agg['total_sig_att'].values,    agg['n_rounds'].values)

tier_info      = builder[['name', 'tier_label', 'ever_champion']].copy()
agg            = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)

X        = agg[STYLE_FEATURES].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"選手數: {len(agg)}")
print(f"\n跑 GMM k={K_RANGE.start} 到 k={K_RANGE.stop - 1}...\n")

TIER_WEIGHTS = {'S':10,'A+':8,'A':6,'B+':4,'B':3,'C+':2,'D':1,'D+':1,'C':1,'E':0}
agg['tier_weight'] = agg['tier_label'].map(TIER_WEIGHTS).fillna(0)
is_champ     = agg['ever_champion'].values
tier_weights = agg['tier_weight'].values

print(f"{'k':>4} | {'BIC':>12} | {'AIC':>12} | {'silhouette':>12} | {'champ_var':>10} | {'tier_var':>10}")
print("-" * 70)

rows = []

for k in K_RANGE:
    gmm    = GaussianMixture(n_components=k, random_state=42, n_init=5,
                              covariance_type='full', max_iter=200)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    # 用 tier 評價分群品質
    densities  = []
    tier_means = []
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            densities.append(is_champ[mask].sum() / mask.sum())
            tier_means.append(tier_weights[mask].mean())

    champ_var = np.var(densities)
    tier_var  = np.var(tier_means)

    rows.append({
        'k': k, 'bic': round(bic, 1), 'aic': round(aic, 1),
        'silhouette': round(sil, 4),
        'champ_var': round(champ_var, 4),
        'tier_var': round(tier_var, 4),
    })

    print(f"{k:>4} | {bic:>12.1f} | {aic:>12.1f} | {sil:>12.4f} | {champ_var:>10.4f} | {tier_var:>10.4f}")

df_r = pd.DataFrame(rows)

best_bic = df_r.loc[df_r['bic'].idxmin(), 'k']
best_aic = df_r.loc[df_r['aic'].idxmin(), 'k']
best_sil = df_r.loc[df_r['silhouette'].idxmax(), 'k']
best_tv  = df_r.loc[df_r['tier_var'].idxmax(), 'k']

print(f"\n{'='*70}")
print(f"指標建議：")
print(f"  BIC 最低        → k = {best_bic}  （模型複雜度 vs 擬合品質，最嚴格）")
print(f"  AIC 最低        → k = {best_aic}  （比 BIC 更傾向多一點 cluster）")
print(f"  Silhouette 最高 → k = {best_sil}  （cluster 分離程度）")
print(f"  Tier Var 最高   → k = {best_tv}   （分群最能區分高低端選手）")

from collections import Counter
votes      = [best_bic, best_aic, best_sil]
vote_count = Counter(votes)
consensus  = vote_count.most_common(1)[0][0]
print(f"\n  BIC/AIC/Silhouette 投票 → k = {consensus}")
print(f"  加入 Tier Var 考量     → k = {best_tv}")
print(f"{'='*70}")

df_r.to_csv('dna_k_search_gmm.csv', index=False)
print("\n輸出：dna_k_search_gmm.csv")
