"""
test_distribution.py
隨機生成各 10 組 budget 60/80/100 的配置
輸出: YOUR FIGHTER | NOTABLE 1 | NOTABLE 2
"""
import pandas as pd
import numpy as np

DF = pd.read_csv('/home/angu/UFC/data/clean/fighter_builder_features.csv')

PCT_COLS = [
    'pct_reach',
    'pct_striking_power', 'pct_striking_volume',
    'pct_striking_accuracy', 'pct_striking_defense',
    'pct_leg_kicks', 'pct_clinch', 'pct_body',
    'pct_td_frequency', 'pct_td_accuracy', 'pct_td_defense',
    'pct_submission', 'pct_control', 'pct_ground_pound',
]

_means = {c: float(DF[c].dropna().mean()) for c in PCT_COLS}
_stds  = {c: float(DF[c].dropna().std()) if DF[c].dropna().std()>0 else 1.0 for c in PCT_COLS}

def build_matrix():
    rows = []
    for _, row in DF.iterrows():
        vec = [(float(row[c] if pd.notna(row[c]) else _means[c]) - _means[c]) / _stds[c] for c in PCT_COLS]
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

MAT = build_matrix()
n = len(PCT_COLS)

def random_alloc(budget):
    vals = np.ones(n, dtype=int)
    remaining = budget - n
    for i in np.random.permutation(n):
        add = np.random.randint(0, min(9, remaining) + 1)
        vals[i] += add
        remaining -= add
        if remaining <= 0:
            break
    return vals

def find(vals):
    query = np.array([(vals[i] - _means[PCT_COLS[i]]) / _stds[PCT_COLS[i]] for i in range(n)], dtype=np.float32)
    query = np.nan_to_num(query)
    dists = np.sqrt(((MAT - query)**2).sum(axis=1))

    # your fighter: pure nearest
    your_idx = int(np.argmin(dists))
    your = DF.iloc[your_idx]
    your_str = f"{your['name']:28s} [{your['tier_label']:3s}·{int(your['tier_score']):3d}] d={dists[your_idx]:.2f}"

    # notable: top30, tier B+, sorted by tier_score
    top30 = np.argsort(dists)[:30]
    notables = []
    for idx in top30:
        if idx == your_idx: continue
        row = DF.iloc[idx]
        if row['tier_label'] in ('D+','D','E'): continue
        notables.append((row['name'], row['tier_label'], int(row['tier_score']), dists[idx]))
    notables.sort(key=lambda x: (-x[2], x[3]))

    notable_strs = []
    for nm, tl, ts, d in notables[:2]:
        notable_strs.append(f"{nm:28s} [{tl:3s}·{ts:3d}] d={d:.2f}")
    while len(notable_strs) < 2:
        notable_strs.append(f"{'—':28s}")

    return your_str, notable_strs[0], notable_strs[1]

print(f"{'BUDGET':8s}  {'YOUR FIGHTER':42s}  {'NOTABLE 1':42s}  {'NOTABLE 2'}")
print('─' * 145)

for budget, label in [(60,'Hard 60'), (80,'Medium 80'), (100,'Easy 100')]:
    print(f"\n  {label}")
    print('  ' + '─' * 141)
    for i in range(10):
        vals = random_alloc(budget)
        your, n1, n2 = find(vals)
        print(f"  {your}  {n1}  {n2}")
