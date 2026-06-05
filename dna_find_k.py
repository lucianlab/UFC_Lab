"""
dna_find_k.py — 找最佳 k 值
=============================
用三個標準 DS 指標評估 k=2 到 k=12
讓資料說話，不靠 composite score 猜測

使用方式：
    python3 dna_find_k.py

輸出：
    dna_k_search.csv — 所有 k 的三個指標數值
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH  = "data/clean/fights_all_rounds.csv"
BUILDER_PATH = "data/clean/fighter_builder_features.csv"
MIN_ROUNDS   = 15
K_RANGE      = range(2, 13)   # k=2 到 k=12

STYLE_FEATURES = [
    'distance_ratio', 'clinch_ratio', 'ground_ratio',
    'head_ratio', 'body_ratio', 'leg_ratio',
    'str_accuracy', 'td_per_round', 'td_accuracy',
    'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round',
]

# ── 讀資料 ──
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
agg['wins']     = agg['wins'].fillna(0)
agg             = agg.reset_index()
agg             = agg[agg['n_rounds'] >= MIN_ROUNDS].copy()

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

X        = agg[STYLE_FEATURES].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"選手數: {len(agg)}")
print(f"\n跑 k={K_RANGE.start} 到 k={K_RANGE.stop - 1}...\n")

# ── 三個指標 ──
print(f"{'k':>4} | {'inertia':>12} | {'silhouette':>12} | {'davies_bouldin':>15} | {'elbow_drop':>12}")
print("-" * 65)

rows      = []
inertias  = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)

    inertia = km.inertia_
    sil     = silhouette_score(X_scaled, labels)
    db      = davies_bouldin_score(X_scaled, labels)

    inertias.append(inertia)

    # Elbow drop：inertia 從上一個 k 的下降量（越小代表 elbow 之後）
    if len(inertias) >= 2:
        drop = inertias[-2] - inertias[-1]
    else:
        drop = float('nan')

    rows.append({
        'k': k, 'inertia': round(inertia, 1),
        'silhouette': round(sil, 4),
        'davies_bouldin': round(db, 4),
        'elbow_drop': round(drop, 1) if not np.isnan(drop) else None
    })

    drop_str = f"{drop:>12.1f}" if not np.isnan(drop) else f"{'—':>12}"
    print(f"{k:>4} | {inertia:>12.1f} | {sil:>12.4f} | {db:>15.4f} | {drop_str}")

# ── 自動建議 ──
df_r = pd.DataFrame(rows)

# Silhouette 最高的 k
best_sil = df_r.loc[df_r['silhouette'].idxmax(), 'k']
# Davies-Bouldin 最低的 k
best_db  = df_r.loc[df_r['davies_bouldin'].idxmin(), 'k']
# Elbow：elbow_drop 開始明顯變小的點（二階差分最大）
drops = df_r['elbow_drop'].dropna().values
if len(drops) >= 2:
    second_diff = np.diff(drops)
    elbow_idx   = np.argmin(second_diff) + 2   # +2 因為 drop 從 k=3 開始
    best_elbow  = K_RANGE.start + elbow_idx
else:
    best_elbow  = None

print(f"\n{'='*65}")
print(f"三個指標的建議：")
print(f"  Silhouette 最高  → k = {best_sil}  （cluster 內緊密、cluster 間分離）")
print(f"  Davies-Bouldin 最低 → k = {best_db}  （cluster 間距離 vs 內距離最佳）")
print(f"  Elbow 拐點       → k = {best_elbow}  （inertia 下降開始趨緩）")

votes = [best_sil, best_db, best_elbow]
from collections import Counter
vote_count = Counter(votes)
consensus  = vote_count.most_common(1)[0][0]
print(f"\n  三個指標投票結果: {votes}")
print(f"  建議 k = {consensus}")
print(f"{'='*65}")

df_r.to_csv('dna_k_search.csv', index=False)
print("\n輸出：dna_k_search.csv")
