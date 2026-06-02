"""
test_builder.py — 終端測試 Fighter Builder KNN
用法: python3 test_builder.py
"""
import pandas as pd
import numpy as np
import os, math

CLEAN = os.path.expanduser('~/UFC/data/clean')
DF    = pd.read_csv(os.path.join(CLEAN, 'fighter_builder_features.csv'))

FEATURES = [
    ('Physical',  'pct_reach',             'Reach (within division)'),
    ('Boxing',    'pct_striking_power',     'KO Power'),
    ('Boxing',    'pct_striking_volume',    'Striking Volume'),
    ('Boxing',    'pct_striking_accuracy',  'Striking Accuracy'),
    ('Boxing',    'pct_striking_defense',   'Striking Defense'),
    ('Muay Thai', 'pct_leg_kicks',          'Leg Kicks'),
    ('Muay Thai', 'pct_clinch',             'Clinch'),
    ('Muay Thai', 'pct_body',               'Body Strikes'),
    ('Wrestling', 'pct_td_frequency',       'TD Frequency'),
    ('Wrestling', 'pct_td_accuracy',        'TD Accuracy'),
    ('Wrestling', 'pct_td_defense',         'TD Defense'),
    ('JiuJitsu',  'pct_submission',         'Submission'),
    ('JiuJitsu',  'pct_control',            'Control'),
    ('JiuJitsu',  'pct_ground_pound',       'Ground & Pound'),
]

PCT_COLS = [f[1] for f in FEATURES]

# z-score 標準化
_means = {}
_stds  = {}
for col in PCT_COLS:
    vals = DF[col].dropna()
    _means[col] = float(vals.mean())
    _stds[col]  = float(vals.std()) if vals.std() > 0 else 1.0

# 預建 matrix
def build_matrix():
    rows = []
    for _, row in DF.iterrows():
        vec = []
        for col in PCT_COLS:
            val = row[col] if pd.notna(row[col]) else _means[col]
            vec.append((float(val) - _means[col]) / _stds[col])
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

MAT = build_matrix()

CAT_COLORS = {
    'Physical':  '\033[37m',
    'Boxing':    '\033[91m',
    'Muay Thai': '\033[93m',
    'Wrestling': '\033[94m',
    'JiuJitsu':  '\033[95m',
}
RESET = '\033[0m'
BOLD  = '\033[1m'

TIER_COLORS = {
    'S':  '\033[33m',   # gold
    'A+': '\033[33m',
    'A':  '\033[33m',
    'B+': '\033[37m',
    'B':  '\033[37m',
    'C+': '\033[90m',
    'C':  '\033[90m',
    'D+': '\033[90m',
    'D':  '\033[90m',
    'E':  '\033[90m',
}

def get_input():
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  FIGHTER BUILDER — KNN Test{RESET}")
    print(f"{'─'*60}")
    print("  Enter 1-10 for each attribute (1=UFC bottom, 10=UFC top)")
    print("  Press Enter to use default (5)\n")

    vals = {}
    cur_cat = None
    for cat, col, label in FEATURES:
        if cat != cur_cat:
            cur_cat = cat
            color = CAT_COLORS.get(cat, '')
            print(f"\n  {color}{BOLD}{cat.upper()}{RESET}")
        while True:
            try:
                raw = input(f"    {label:<28} [1-10]: ").strip()
                v   = float(raw) if raw else 5.0
                if 1 <= v <= 10:
                    vals[col] = v
                    break
                print("    → Enter a number between 1 and 10")
            except ValueError:
                print("    → Invalid input")
    return vals

def find_nearest(vals, k=5):
    query = []
    for col in PCT_COLS:
        v = vals.get(col, 5.0)
        query.append((v - _means[col]) / _stds[col])
    query = np.array(query, dtype=np.float32)
    query = np.nan_to_num(query, nan=0.0)

    diffs = MAT - query
    dists = np.sqrt((diffs**2).sum(axis=1))

    top_idx = np.argsort(dists)[:k]
    results = []
    for i in top_idx:
        row = DF.iloc[i]
        results.append({
            'name':       row['name'],
            'wc':         row.get('wc',''),
            'tier_label': row.get('tier_label','E'),
            'tier_score': int(row.get('tier_score',0)),
            'win_rate':   row.get('win_rate'),
            'best_rank':  row.get('best_rank'),
            'distance':   round(float(dists[i]),3),
            'pct':        {col: round(float(row[col]),1) for col in PCT_COLS if col in DF.columns},
        })
    return results

def show_results(matches, player_vals):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  RESULTS — Nearest Fighters{RESET}")
    print(f"{'─'*60}")

    # filter: show only B+ and above first; fallback to all
    notable = [m for m in matches if m['tier_label'] not in ('D+','D','E')]
    to_show = notable[:3] if notable else matches[:1]

    if not to_show:
        print("  No notable match found.")
        return

    for i, m in enumerate(to_show):
        tc = TIER_COLORS.get(m['tier_label'],'\033[90m')
        wr = f"{round(m['win_rate']*100)}%" if m['win_rate'] else '—'
        rk = f"#{int(m['best_rank'])}" if pd.notna(m.get('best_rank')) else 'unranked'
        print(f"\n  {BOLD}{i+1}. {m['name']}{RESET}  {tc}[{m['tier_label']} · {m['tier_score']}/100]{RESET}")
        print(f"     {m['wc']}  ·  Win {wr}  ·  {rk}  ·  dist {m['distance']}")
        print()

        # Show comparison bars
        for cat, col, label in FEATURES:
            pv = player_vals.get(col, 5.0)
            mv = m['pct'].get(col, 5.0)
            color = CAT_COLORS.get(cat, '')
            bar_p = '█' * int(pv)
            bar_m = '░' * int(mv)
            diff  = mv - pv
            sign  = '+' if diff > 0 else ''
            print(f"     {color}{label:<24}{RESET}  you:{pv:4.0f}  them:{mv:4.1f}  [{sign}{diff:.1f}]")

    print(f"\n{'─'*60}")

def main():
    print(f"\n{BOLD}Loading fighter data...{RESET}", end='')
    print(f" {len(DF)} fighters loaded")

    while True:
        vals = get_input()
        total = sum(vals.values())
        print(f"\n  Total points: {BOLD}{int(total)}{RESET} / 140 max")

        matches = find_nearest(vals, k=10)
        show_results(matches, vals)

        again = input("\n  Try another? [Y/n]: ").strip().lower()
        if again == 'n':
            break

if __name__ == '__main__':
    main()
