"""
dna_search.py — DNA Clustering Parameter Search
=================================================
跑不同資料品質 x 不同 k 值的組合，找最有意義的分群設定。

評分標準：
  1. Silhouette Score      — 幾何緊密度（標準指標）
  2. Champion Density Var  — 冠軍密度在不同 cluster 的差異（越高越有意義）
  3. Min Cluster Size      — 最小 cluster 人數（太小不可靠）
  4. Champ Coverage        — 所有 cluster 都有冠軍（True/False）

使用方式：
    python3 dna_search.py

輸出：
    dna_search_results.csv — 所有組合的評分
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

FIGHTS_PATH   = "data/clean/fights_all_rounds.csv"
BUILDER_PATH  = "data/clean/fighter_builder_features.csv"

MIN_ROUNDS = 15

# ── 資料品質篩選條件 ──
TIER_CONFIGS = {
    'all_ranked':    ['S', 'A+', 'A', 'B+', 'B', 'C+'],  # 220人
    'champions_up':  ['S', 'A+', 'A', 'B+', 'B'],         # 104人，全是冠軍
    'elite_only':    ['S', 'A+', 'A', 'B+'],               # 59人，多次防衛以上
    'goat_tier':     ['S', 'A+', 'A'],                     # 32人，歷史頂端
}

K_RANGE = range(4, 10)

# ══════════════════════════════════════════════════════
print("讀取資料...")
fights  = pd.read_csv(FIGHTS_PATH)
builder = pd.read_csv(BUILDER_PATH)

# ── 計算風格本質特徵 ──
def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

g = fights.groupby('fighter')
agg = pd.DataFrame()
agg['n_rounds']         = g['round'].count()
agg['total_sig_att']    = g['sig_str_attempted'].sum()
agg['total_sig_landed'] = g['sig_str_landed'].sum()
agg['dist_att']         = g['dist_attempted'].sum()
agg['dist_land']        = g['dist_landed'].sum()
agg['clinch_att']       = g['clinch_attempted'].sum()
agg['clinch_land']      = g['clinch_landed'].sum()
agg['ground_att']       = g['ground_attempted'].sum()
agg['ground_land']      = g['ground_landed'].sum()
agg['head_att']         = g['head_attempted'].sum()
agg['head_land']        = g['head_landed'].sum()
agg['body_att']         = g['body_attempted'].sum()
agg['body_land']        = g['body_landed'].sum()
agg['leg_att']          = g['leg_attempted'].sum()
agg['leg_land']         = g['leg_landed'].sum()
agg['td_att']           = g['td_attempted'].sum()
agg['td_land']          = g['td_landed'].sum()
agg['ctrl_sec_total']   = g['ctrl_sec'].sum()
agg['sub_att_total']    = g['sub_att'].sum()
agg['kd_total']         = g['kd'].sum()
agg['n_fights']         = g['bout'].nunique()

wins_df = fights[fights['won']==1].groupby('fighter')['bout'].nunique().rename('wins')
agg = agg.join(wins_df)
agg['wins']     = agg['wins'].fillna(0)
agg['win_rate'] = safe_div(agg['wins'].values, agg['n_fights'].values)
agg = agg.reset_index()
agg = agg[agg['n_rounds'] >= MIN_ROUNDS].copy()

agg['distance_ratio']  = safe_div(agg['dist_land'].values,         agg['total_sig_landed'].values)
agg['clinch_ratio']    = safe_div(agg['clinch_land'].values,        agg['total_sig_landed'].values)
agg['ground_ratio']    = safe_div(agg['ground_land'].values,        agg['total_sig_landed'].values)
agg['head_ratio']      = safe_div(agg['head_land'].values,          agg['total_sig_landed'].values)
agg['body_ratio']      = safe_div(agg['body_land'].values,          agg['total_sig_landed'].values)
agg['leg_ratio']       = safe_div(agg['leg_land'].values,           agg['total_sig_landed'].values)
agg['str_accuracy']    = safe_div(agg['total_sig_landed'].values,   agg['total_sig_att'].values)
agg['td_per_round']    = safe_div(agg['td_land'].values,            agg['n_rounds'].values)
agg['td_accuracy']     = safe_div(agg['td_land'].values,            agg['td_att'].values, fill=0.5)
agg['ctrl_per_round']  = np.clip(safe_div(agg['ctrl_sec_total'].values, agg['n_rounds'].values), 0, 300)
agg['sub_per_round']   = safe_div(agg['sub_att_total'].values,      agg['n_rounds'].values)
agg['kd_per_round']    = safe_div(agg['kd_total'].values,           agg['n_rounds'].values)
agg['output_per_round']= safe_div(agg['total_sig_att'].values,      agg['n_rounds'].values)

STYLE_FEATURES = [
    'distance_ratio', 'clinch_ratio', 'ground_ratio',
    'head_ratio', 'body_ratio', 'leg_ratio',
    'str_accuracy', 'td_per_round', 'td_accuracy',
    'ctrl_per_round', 'sub_per_round', 'kd_per_round', 'output_per_round',
]

tier_info = builder[['name', 'tier_label', 'ever_champion']].copy()
agg = agg.merge(tier_info, left_on='fighter', right_on='name', how='left')
agg['tier_label']    = agg['tier_label'].fillna('E')
agg['ever_champion'] = agg['ever_champion'].fillna(False)

# ══════════════════════════════════════════════════════
print("跑所有組合...\n")

rows = []

header = f"{'config':<18} {'k':>3} | {'n':>4} | {'silhouette':>10} | {'champ_var':>10} | {'min_size':>9} | {'all_have_champ':>14} | {'composite':>10}"
print(header)
print("-" * len(header))

for config_name, tiers in TIER_CONFIGS.items():
    df = agg[agg['tier_label'].isin(tiers)].copy()
    n  = len(df)
    
    if n < 20:
        print(f"{config_name:<18}  — 樣本不足 ({n}人)")
        continue
    
    X       = df[STYLE_FEATURES].fillna(0).values
    scaler  = StandardScaler()
    X_scaled= scaler.fit_transform(X)
    is_champ= df['ever_champion'].values
    
    for k in K_RANGE:
        if n < k * 5:   # 每個 cluster 至少 5 人才有意義
            continue
        
        km     = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        
        # 1. Silhouette
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        
        # 2. Champion Density Variance
        #    每個 cluster 的冠軍比例，variance 越高代表分群抓到了有意義的結構
        densities = []
        sizes     = []
        for i in range(k):
            mask = labels == i
            size = mask.sum()
            sizes.append(size)
            if size > 0:
                d = is_champ[mask].sum() / size
                densities.append(d)
        champ_var = np.var(densities)
        
        # 3. Min cluster size
        min_size = min(sizes)
        
        # 4. 所有 cluster 都有冠軍
        champ_labels   = set(labels[is_champ])
        all_have_champ = len(champ_labels) == k
        
        # 5. Composite score
        #    normalized silhouette(0-1) * 0.3
        #    + normalized champ_var * 0.4
        #    + min_size penalty * 0.3
        #    bonus if all_have_champ
        sil_norm      = (sil + 1) / 2          # -1~1 → 0~1
        size_score    = min(1.0, min_size / 15) # 15人以上滿分
        composite     = (sil_norm * 0.3 + champ_var * 4 * 0.4 + size_score * 0.3)
        if all_have_champ:
            composite += 0.1
        
        rows.append({
            'config':         config_name,
            'tiers':          '+'.join(tiers),
            'n':              n,
            'k':              k,
            'silhouette':     round(sil, 4),
            'champ_var':      round(champ_var, 4),
            'min_size':       min_size,
            'all_have_champ': all_have_champ,
            'composite':      round(composite, 4),
        })
        
        flag = ' ◀' if all_have_champ else ''
        print(f"{config_name:<18} {k:>3} | {n:>4} | {sil:>10.4f} | {champ_var:>10.4f} | {min_size:>9} | {str(all_have_champ):>14} | {composite:>10.4f}{flag}")
    
    print()

# ── 輸出結果 ──
results_df = pd.DataFrame(rows)
results_df = results_df.sort_values('composite', ascending=False)
results_df.to_csv('dna_search_results.csv', index=False)

print("\n" + "=" * 60)
print("Top 5 組合（依 composite score）：")
print("=" * 60)
for _, row in results_df.head(5).iterrows():
    print(f"  {row['config']:<18} k={row['k']}  composite={row['composite']:.4f}  "
          f"sil={row['silhouette']:.4f}  champ_var={row['champ_var']:.4f}  "
          f"min_size={row['min_size']}  all_champ={row['all_have_champ']}")

print("\n輸出：dna_search_results.csv")
