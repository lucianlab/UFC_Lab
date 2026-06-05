"""
dna_anchors.py — 極端值錨點分析
=================================
對每個風格特徵找前 N% 的極端值選手
看這些錨點自然形成幾個群
測試 N = 3, 5, 8, 10, 13, 15%

使用方式：
    python3 dna_anchors.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH  = "data/clean/fights_all_rounds.csv"
BUILDER_PATH = "data/clean/fighter_builder_features.csv"
MIN_ROUNDS   = 15
PERCENTILES  = [3, 5, 8, 10, 13, 15]

STYLE_FEATURES = [
    'distance_ratio', 'clinch_ratio', 'ground_ratio',
    'head_ratio', 'body_ratio', 'leg_ratio',
    'str_accuracy', 'td_per_round', 'td_accuracy',
    'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round',
]

FEATURE_LABELS = {
    'distance_ratio':   '距離打擊佔比',
    'clinch_ratio':     '貼身打擊佔比',
    'ground_ratio':     '地面打擊佔比',
    'head_ratio':       '頭部攻擊佔比',
    'body_ratio':       '身體攻擊佔比',
    'leg_ratio':        '腿擊佔比',
    'str_accuracy':     '打擊精準度',
    'td_per_round':     '每回合摔抱',
    'td_accuracy':      '摔抱成功率',
    'ctrl_per_round':   '每回合控制(秒)',
    'sub_per_round':    '每回合柔術嘗試',
    'kd_per_round':     '每回合擊倒',
    'output_per_round': '每回合出拳量',
}

TIER_WEIGHTS = {'S':10,'A+':8,'A':6,'B+':4,'B':3,'C+':2,'D+':1,'D':1,'C':1,'E':0}

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
agg['tier_weight']   = agg['tier_label'].map(TIER_WEIGHTS).fillna(0)

X        = agg[STYLE_FEATURES].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"選手數: {len(agg)}\n")
print("=" * 70)

all_results = []

for pct in PERCENTILES:
    # ── 找每個特徵的前 pct% 極端值選手 ──
    anchor_indices = set()
    feature_extremes = {}

    for feat in STYLE_FEATURES:
        threshold = np.percentile(agg[feat].values, 100 - pct)
        extreme_mask = agg[feat].values >= threshold
        idxs = np.where(extreme_mask)[0]
        anchor_indices.update(idxs)
        feature_extremes[feat] = agg.iloc[idxs]['fighter'].tolist()

    anchor_idx = sorted(anchor_indices)
    X_anchors  = X_scaled[anchor_idx]
    n_anchors  = len(anchor_idx)
    anchor_df  = agg.iloc[anchor_idx].copy()

    # ── 在錨點集合內跑 KMeans，找自然群數 ──
    k_results = []
    max_k = min(12, n_anchors // 5)  # 每群至少 5 人

    for k in range(2, max_k + 1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_anchors)
        sil    = silhouette_score(X_anchors, labels)

        # tier variance
        tier_means = [anchor_df.iloc[labels==i]['tier_weight'].mean()
                      for i in range(k) if (labels==i).sum() > 0]
        tier_var   = np.var(tier_means)

        # champ density variance
        is_champ   = anchor_df['ever_champion'].values
        densities  = [is_champ[labels==i].sum() / (labels==i).sum()
                      for i in range(k) if (labels==i).sum() > 0]
        champ_var  = np.var(densities)

        k_results.append({
            'k': k, 'sil': sil, 'tier_var': tier_var, 'champ_var': champ_var
        })

    best_k_row = max(k_results, key=lambda x: x['sil'])
    best_k     = best_k_row['k']
    best_sil   = best_k_row['sil']

    # ── 用最佳 k 跑一次，看每群的代表選手 ──
    km_final   = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels_final = km_final.fit_predict(X_anchors)
    anchor_df  = anchor_df.copy()
    anchor_df['anchor_cluster'] = labels_final

    print(f"N = {pct:>2}%  |  錨點數 = {n_anchors:>4}  |  最佳 k = {best_k}  |  silhouette = {best_sil:.4f}")
    print(f"  k值掃描: { {r['k']: round(r['sil'],4) for r in k_results} }")

    tier_order = {'S':0,'A+':1,'A':2,'B+':3,'B':4,'C+':5,'D+':6,'D':7,'C':8,'E':9}
    for i in range(best_k):
        cluster_df  = anchor_df[anchor_df['anchor_cluster'] == i].copy()
        cluster_df['t'] = cluster_df['tier_label'].map(tier_order).fillna(9)
        top5 = cluster_df.sort_values(['t', 'tier_weight'], ascending=[True, False])['fighter'].head(5).tolist()
        champs = cluster_df[cluster_df['ever_champion']==True]['fighter'].tolist()
        n = len(cluster_df)

        # 這群的主導特徵（質心最高的3個）
        center = scaler.inverse_transform(km_final.cluster_centers_[i:i+1])[0]
        top3_feat = sorted(zip(STYLE_FEATURES, center), key=lambda x: x[1], reverse=True)[:3]

        print(f"  群 {i} (n={n:>3}): {', '.join(top5)}")
        print(f"         主導: { {FEATURE_LABELS[f]: round(v,3) for f,v in top3_feat} }")
        print(f"         冠軍: {', '.join(champs[:4])}")

    all_results.append({
        'pct': pct, 'n_anchors': n_anchors,
        'best_k': best_k, 'best_sil': best_sil,
        'k_scan': {r['k']: round(r['sil'],4) for r in k_results}
    })
    print("-" * 70)

print("\n總結：")
print(f"  {'N%':>4} | {'錨點數':>6} | {'建議k':>6} | {'silhouette':>12}")
print(f"  {'-'*35}")
for r in all_results:
    print(f"  {r['pct']:>4} | {r['n_anchors']:>6} | {r['best_k']:>6} | {r['best_sil']:>12.4f}")
