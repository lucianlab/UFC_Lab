"""
build_dna.py — UFC Fighter DNA Clustering v2
=============================================
階段一：用所有回合數達標選手找打法原型（純風格，不帶價值判斷）
階段二：用 tier / 冠軍身份評價每個 cluster 的「勝率驗證程度」

使用方式：
    python3 build_dna.py

輸出：
    dna_results.csv     — 每個選手的風格特徵 + cluster 標籤
    dna_centroids.csv   — 每個 cluster 的質心
    dna_summary.txt     — 完整分析報告
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════

FIGHTS_PATH   = "data/clean/fights_all_rounds.csv"
BUILDER_PATH  = "data/clean/fighter_builder_features.csv"

MIN_ROUNDS    = 15
K_RANGE       = range(4, 10)
FINAL_K       = None     # None = 自動選最佳，或填數字強制指定

TIER_WEIGHTS = {
    'S':  10,
    'A+':  8,
    'A':   6,
    'B+':  4,
    'B':   3,
    'C+':  2,
    'D':   1,
    'E':   0,
}

STYLE_FEATURES = [
    'distance_ratio', 'clinch_ratio', 'ground_ratio',
    'head_ratio', 'body_ratio', 'leg_ratio',
    'str_accuracy', 'td_per_round', 'td_accuracy',
    'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round',
]

# ══════════════════════════════════════════════════════
#  STEP 1：計算風格本質特徵
# ══════════════════════════════════════════════════════

print("=" * 65)
print("UFC Fighter DNA v2")
print("=" * 65)

print("\n[1/4] 讀取資料...")
fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)
print(f"      fights: {len(fights):,} rows | builder: {len(builder):,} fighters")

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

print(f"\n[2/4] 計算風格特徵（min {MIN_ROUNDS} rounds）...")

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
agg['win_rate'] = safe_div(agg['wins'].values, agg['n_fights'].values)
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

tier_info      = builder[['name', 'tier_label', 'tier_score', 'ever_champion', 'wc']].copy()
agg            = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)
agg['tier_weight']   = agg['tier_label'].map(TIER_WEIGHTS).fillna(0)

print(f"      {len(agg)} 選手通過篩選")
print(f"      Tier 分布: { dict(agg['tier_label'].value_counts()) }")

# ══════════════════════════════════════════════════════
#  STEP 2：找最佳 k（階段一，純幾何 + 品質評分）
# ══════════════════════════════════════════════════════

print(f"\n[3/4] 階段一：找最佳分群數...")

X        = agg[STYLE_FEATURES].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

is_champ     = agg['ever_champion'].values
tier_weights = agg['tier_weight'].values

print(f"\n  {'k':>3} | {'silhouette':>10} | {'champ_var':>10} | {'tier_var':>10} | {'min_size':>9} | {'score':>8}")
print(f"  {'-'*58}")

search_rows = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)

    densities  = []
    tier_means = []
    sizes      = []

    for i in range(k):
        mask = labels == i
        size = mask.sum()
        sizes.append(size)
        if size > 0:
            densities.append(is_champ[mask].sum() / size)
            tier_means.append(tier_weights[mask].mean())

    champ_var = np.var(densities)
    tier_var  = np.var(tier_means)
    min_size  = min(sizes)

    sil_norm   = (sil + 1) / 2
    size_score = min(1.0, min_size / 20)
    score      = sil_norm * 0.25 + champ_var * 5 * 0.35 + tier_var * 0.1 * 0.15 + size_score * 0.25

    search_rows.append({
        'k': k, 'silhouette': sil, 'champ_var': champ_var,
        'tier_var': tier_var, 'min_size': min_size, 'score': score,
        'labels': labels, 'model': km
    })

    print(f"  {k:>3} | {sil:>10.4f} | {champ_var:>10.4f} | {tier_var:>10.4f} | {min_size:>9} | {score:>8.4f}")

if FINAL_K is not None:
    best = next(r for r in search_rows if r['k'] == FINAL_K)
else:
    best = max(search_rows, key=lambda x: x['score'])

best_k      = best['k']
best_labels = best['labels']
best_model  = best['model']
print(f"\n  → 最佳 k = {best_k}（score={best['score']:.4f}）")

# ══════════════════════════════════════════════════════
#  STEP 3：階段二 — 用 tier 評價每個 cluster
# ══════════════════════════════════════════════════════

print(f"\n[4/4] 階段二：用 tier 評價 k={best_k} 的分群...")

agg            = agg.copy()
agg['cluster'] = best_labels

centers_orig = scaler.inverse_transform(best_model.cluster_centers_)
centers_df   = pd.DataFrame(centers_orig, columns=STYLE_FEATURES)

feature_labels = {
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

summary_lines = []
summary_lines.append(f"UFC Fighter DNA v2 — k={best_k} 分群結果")
summary_lines.append(f"資料：{len(agg)} 名選手（全部回合數達標，不篩 tier）")
summary_lines.append(f"特徵：{len(STYLE_FEATURES)} 個風格本質比例特徵")
summary_lines.append("=" * 70)

cluster_scores = []

for i in range(best_k):
    cluster_df     = agg[agg['cluster'] == i].copy()
    n              = len(cluster_df)
    champs         = cluster_df[cluster_df['ever_champion'] == True]['fighter'].tolist()
    champ_rate     = len(champs) / n
    tier_score_mean= cluster_df['tier_weight'].mean()
    tier_score_std = cluster_df['tier_weight'].std()
    tier_dist      = cluster_df['tier_label'].value_counts().to_dict()

    tier_order     = {'S':0,'A+':1,'A':2,'B+':3,'B':4,'C+':5,'D':6,'E':7}
    sorted_df      = cluster_df.copy()
    sorted_df['t'] = sorted_df['tier_label'].map(tier_order).fillna(9)
    top_fighters   = sorted_df.sort_values(['t','win_rate'], ascending=[True,False])['fighter'].head(8).tolist()

    row  = centers_df.iloc[i]
    top3 = row.nlargest(3)
    low3 = row.nsmallest(3)

    style_tags = []
    if row['td_per_round'] > 0.5 and row['ctrl_per_round'] > 50:
        style_tags.append('wrestling-dominant')
    if row['distance_ratio'] > 0.75:
        style_tags.append('distance-striker')
    elif row['distance_ratio'] > 0.60:
        style_tags.append('distance-oriented')
    if row['ground_ratio'] > 0.18:
        style_tags.append('ground-and-pound')
    if row['sub_per_round'] > 0.18:
        style_tags.append('submission-hunter')
    if row['kd_per_round'] > 0.08:
        style_tags.append('KO-power')
    if row['leg_ratio'] > 0.22:
        style_tags.append('leg-kick-heavy')
    if row['clinch_ratio'] > 0.18:
        style_tags.append('clinch-fighter')
    if row['output_per_round'] > 45:
        style_tags.append('high-volume')
    if row['ctrl_per_round'] > 90:
        style_tags.append('heavy-control')

    cluster_scores.append({
        'cluster': i, 'n': n, 'champ_rate': champ_rate,
        'tier_score_mean': tier_score_mean, 'style_tags': style_tags,
        'top_fighters': top_fighters
    })

    summary_lines.append(f"\nCluster {i}  (n={n})")
    summary_lines.append(f"  ── 階段一：風格特徵 ──")
    summary_lines.append(f"  主導: { {feature_labels[f]: round(v,3) for f,v in top3.items()} }")
    summary_lines.append(f"  偏低: { {feature_labels[f]: round(v,3) for f,v in low3.items()} }")
    summary_lines.append(f"  風格: {' | '.join(style_tags) if style_tags else '(待命名)'}")
    summary_lines.append(f"")
    summary_lines.append(f"  ── 階段二：Tier 評價 ──")
    summary_lines.append(f"  Tier 加權均值: {tier_score_mean:.2f} ± {tier_score_std:.2f}")
    summary_lines.append(f"  冠軍密度:      {champ_rate*100:.1f}%  ({len(champs)}/{n})")
    summary_lines.append(f"  Tier 分布:     {tier_dist}")
    summary_lines.append(f"  代表選手:      {', '.join(top_fighters)}")
    summary_lines.append(f"  冠軍:          {', '.join(champs[:6])}")
    summary_lines.append("-" * 70)

summary_lines.append(f"\n\nCluster 排名（依 Tier 加權均值）：")
summary_lines.append(f"  {'rank':>4} | {'cluster':>7} | {'tier_mean':>10} | {'champ%':>8} | 風格")
summary_lines.append(f"  {'-'*60}")
ranked = sorted(cluster_scores, key=lambda x: x['tier_score_mean'], reverse=True)
for rank, c in enumerate(ranked, 1):
    summary_lines.append(
        f"  {rank:>4} | {c['cluster']:>7} | {c['tier_score_mean']:>10.2f} | "
        f"{c['champ_rate']*100:>7.1f}% | {' | '.join(c['style_tags'][:2])}"
    )

summary_lines.append(f"\n\n驗證：Ciryl Gane vs Alexander Volkanovski")
gane = agg[agg['fighter'] == 'Ciryl Gane']
volk = agg[agg['fighter'] == 'Alexander Volkanovski']
if len(gane) and len(volk):
    gc, vc = gane['cluster'].values[0], volk['cluster'].values[0]
    summary_lines.append(f"  Gane cluster: {gc}  |  Volk cluster: {vc}")
    summary_lines.append(f"  同一群: {'✓ YES' if gc == vc else '✗ NO'}")

text = "\n".join(summary_lines)
print("\n" + text)

output_cols = ['fighter', 'wc', 'tier_label', 'tier_weight', 'ever_champion',
               'win_rate', 'n_rounds', 'cluster'] + STYLE_FEATURES
agg[output_cols].to_csv('dna_results.csv', index=False)
centers_df['cluster'] = range(best_k)
centers_df.to_csv('dna_centroids.csv', index=False)
with open('dna_summary.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("\n\n輸出：dna_results.csv / dna_centroids.csv / dna_summary.txt")
print("=" * 65)
