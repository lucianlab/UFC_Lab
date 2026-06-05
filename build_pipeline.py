#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UFC_Lab  build_pipeline.py
讀 round-by-round 比賽資料 + 冠軍/排名兩個檔，算出：
  軸一  Striker(-) <-> Grappler(+)        模態：你在哪裡打
  軸二  Decision(-) <-> Finisher(+)        結果：比賽怎麼結束（KO+Sub 合一，跟模態無關）
  軸三  Counter(-) <-> Pressure(+)         交火：你怎麼參與交火（用對手數據算防守姿態）
三條軸全部「設計 + Gram-Schmidt 正交化」，並輸出：
  data/clean/fighter_vectors.csv  完整向量（你自己分析用）
  data/fighters.json              前端/後端用（drop-in，欄位已對好合約）
跑法見檔尾，或直接：python3 build_pipeline.py
"""

import os, sys, json
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# CONFIG —— 如果你的檔案放在別的地方，改這裡就好
# ──────────────────────────────────────────────────────────────────────────
FIGHTS_CANDIDATES = [
    "data/processed/fights_all_rounds.csv",
    "data/clean/fights_all_rounds.csv",
    "data/fights_all_rounds.csv",
    "fights_all_rounds.csv",
]
CHAMP_CANDIDATES = [
    "data/raw/ufc_championship_history.csv",
    "data/ufc_championship_history.csv",
    "ufc_championship_history.csv",
]
RANK_CANDIDATES = [
    "data/raw/ufc_active_top15_rankings.csv",
    "data/ufc_active_top15_rankings.csv",
    "ufc_active_top15_rankings.csv",
]
OUT_VECTORS = "data/clean/fighter_vectors.csv"
OUT_JSON    = "data/fighters.json"
MIN_FIGHTS  = 5

MEN = ['Flyweight','Bantamweight','Featherweight','Lightweight','Welterweight',
       'Middleweight','Light Heavyweight','Heavyweight']
WOMEN = ["Women's Strawweight","Women's Flyweight","Women's Bantamweight","Women's Featherweight"]
VALID_WC = set(MEN + WOMEN)


def find_file(candidates, label):
    for p in candidates:
        if os.path.exists(p):
            print(f"  找到 {label}: {p}")
            return p
    sys.exit(f"[錯誤] 找不到 {label}。請把檔案放到下列任一路徑，或改 CONFIG：\n   " +
             "\n   ".join(candidates))


def norm_wc(v):
    if not isinstance(v, str):
        return None
    s = v.replace(' Bout', '').strip()
    return s if s in VALID_WC else None


def zscore_within(df, cols, group):
    """量級內 z-score；單人/零變異組安全處理（補 0）。"""
    out = df.copy()
    for c in cols:
        gmean = df.groupby(group)[c].transform('mean')
        gstd  = df.groupby(group)[c].transform('std').replace(0, np.nan)
        out[c] = ((df[c] - gmean) / gstd).fillna(0.0)
    return out


def residualize(target, predictors):
    """回傳 target 對 predictors 做線性回歸後的殘差（含截距）→ 與 predictors 正交。"""
    t = np.asarray(target, dtype=float)
    cols = [np.ones(len(t))] + [np.asarray(p, dtype=float) for p in predictors]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, t, rcond=None)
    return t - X @ beta


def std01(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / (s if s else 1.0)


# ══════════════════════════════════════════════════════════════════════════
print("=== 1. 讀資料 ===")
FIGHTS = find_file(FIGHTS_CANDIDATES, "比賽資料")
CHAMP  = find_file(CHAMP_CANDIDATES, "冠軍歷史")
RANK   = find_file(RANK_CANDIDATES, "現役排名")

df = pd.read_csv(FIGHTS, low_memory=False)
df['wc'] = df['weightclass'].map(norm_wc)

num_cols = ['kd','sig_str_landed','sig_str_attempted','td_landed','td_attempted',
            'ctrl_sec','sub_att','rev','head_landed','body_landed','leg_landed',
            'dist_landed','clinch_landed','ground_landed']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['round_i'] = pd.to_numeric(df['round'], errors='coerce')

# ══════════════════════════════════════════════════════════════════════════
print("=== 2. 對接對手同回合數據（算防守面）===")
opp_src = ['sig_str_landed','sig_str_attempted','kd','td_landed','td_attempted']
right = (df[['event','bout','round','fighter'] + opp_src]
         .rename(columns={'fighter':'opp_name', **{c:'opp_'+c for c in opp_src}})
         .drop_duplicates(subset=['event','bout','round','opp_name']))
merged = df.merge(right,
                  left_on=['event','bout','round','opponent'],
                  right_on=['event','bout','round','opp_name'],
                  how='left')

# ══════════════════════════════════════════════════════════════════════════
print("=== 3. 場次層級：終結/判定統計 ===")
fl = (merged.groupby(['event','bout','fighter'], as_index=False)
            .agg(winner=('winner','first'), method=('method','first')))
mstr = fl['method'].astype(str)
fl['is_win'] = (fl['winner'] == fl['fighter'])
fl['ko']  = fl['is_win'] & mstr.str.contains('KO', na=False)       # KO/TKO + TKO-Doctor
fl['sub'] = fl['is_win'] & mstr.str.fullmatch('Submission', na=False)
fl['finish'] = fl['ko'] | fl['sub']
fin = (fl.groupby('fighter')
         .agg(n_fights=('bout','count'),
              ko_rate=('ko','mean'),
              sub_rate=('sub','mean'),
              finish_rate=('finish','mean')))

# ══════════════════════════════════════════════════════════════════════════
print("=== 4. 選手層級特徵彙總 ===")
g = merged.groupby('fighter')

def S(col):  # 安全 sum（全 NaN -> NaN，避免 0/0）
    return g[col].sum(min_count=1)
def M(col):
    return g[col].mean()

feat = pd.DataFrame(index=sorted(merged['fighter'].dropna().unique()))
feat.index.name = 'fighter'

# 進攻 / 打擊
feat['sig_per_r']  = M('sig_str_landed')
feat['kd_per_r']   = M('kd')
feat['sig_acc']    = S('sig_str_landed') / S('sig_str_attempted')
feat['sig_att_per_r'] = M('sig_str_attempted')   # 出手量（pace 用，不進相似度）
hbl = S('head_landed') + S('body_landed') + S('leg_landed')
feat['head_pct']   = S('head_landed') / hbl
feat['body_pct']   = S('body_landed') / hbl
feat['leg_pct']    = S('leg_landed')  / hbl
pos = S('dist_landed') + S('clinch_landed') + S('ground_landed')
feat['dist_pct']   = S('dist_landed')   / pos
feat['clinch_pct'] = S('clinch_landed') / pos
feat['ground_pct'] = S('ground_landed') / pos
# 寢技（加厚）
feat['td_per_r']   = M('td_landed')
feat['td_att_per_r'] = M('td_attempted')          # 摔倒嘗試量（pace 用，不進相似度）
feat['td_acc']     = S('td_landed') / S('td_attempted')
feat['ctrl_per_r'] = M('ctrl_sec')
feat['sub_per_r']  = M('sub_att')
feat['rev_per_r']  = M('rev')
feat['gnp_per_r']  = M('ground_landed')          # 地面打擊量 = GnP
# 防守 / 交火（用對手數據）
feat['absorbed_per_r'] = M('opp_sig_str_landed')
feat['kd_absorbed_r']  = M('opp_kd')
feat['str_def']        = 1 - S('opp_sig_str_landed') / S('opp_sig_str_attempted')
feat['td_def']         = 1 - S('opp_td_landed') / S('opp_td_attempted')

# gas_tank：後段(>=R3) vs 前段(R1-2) 打擊輸出比；只當展示欄位，不是軸
early = merged[merged.round_i <= 2].groupby('fighter')['sig_str_landed'].mean()
late  = merged[merged.round_i >= 3].groupby('fighter')['sig_str_landed'].mean()
feat['gas_tank'] = (late / early)

feat = feat.replace([np.inf, -np.inf], np.nan)

# 量級：主量級(眾數) + 全部打過的量級
wcs = merged.dropna(subset=['wc']).groupby('fighter')['wc']
feat['wc']     = wcs.agg(lambda x: x.mode().iat[0] if len(x.mode()) else None)
all_wc_map     = wcs.agg(lambda x: sorted(x.unique().tolist())).to_dict()

# 合併終結統計、過濾
feat = feat.join(fin)
data = feat[(feat['n_fights'] >= MIN_FIGHTS) & (feat['wc'].notna())].copy()
data = data.reset_index()  # 'fighter' 變回欄位
data = data.rename(columns={'fighter': 'name'})
data['all_wc'] = data['name'].map(all_wc_map)
print(f"  符合條件選手數 (n_fights>={MIN_FIGHTS}): {len(data)}")

# ══════════════════════════════════════════════════════════════════════════
print("=== 5. 量級內標準化 + 三軸（設計 + 正交化）===")
FEATURES = ['sig_per_r','kd_per_r','sig_acc','head_pct','body_pct','leg_pct',
            'dist_pct','clinch_pct','ground_pct','td_per_r','td_acc','ctrl_per_r',
            'sub_per_r','rev_per_r','gnp_per_r','absorbed_per_r','kd_absorbed_r',
            'str_def','td_def']
# 補洞：剩餘 NaN 用同量級中位數，再不行用全體中位數
for c in FEATURES + ['finish_rate','sig_att_per_r','td_att_per_r']:
    data[c] = data.groupby('wc')[c].transform(lambda s: s.fillna(s.median()))
    data[c] = data[c].fillna(data[c].median())

Z  = zscore_within(data[['wc'] + FEATURES].copy(), FEATURES, 'wc')
Zf = zscore_within(data[['wc','finish_rate']].copy(), ['finish_rate'], 'wc')

def block(cols):
    return Z[cols].mean(axis=1).values

# 軸一  + = grappler / - = striker
GRP = ['ground_pct','ctrl_per_r','td_per_r','sub_per_r','rev_per_r','gnp_per_r']
STK = ['dist_pct','sig_per_r','kd_per_r']
axis1_raw = block(GRP) - block(STK)

# 軸二  + = finisher / - = decision（KO+Sub 合一，模態相消）
axis2_raw = Zf['finish_rate'].values

# 軸三  + = high-output / - = patient（工作量 work-rate；用「出手量」attempts，
#        正交化後 = 扣掉模態與終結後，你比同類忙還是惜打）
Zp = zscore_within(data[['wc','sig_att_per_r','td_att_per_r']].copy(),
                   ['sig_att_per_r','td_att_per_r'], 'wc')
axis3_raw = (Zp['sig_att_per_r'].values + Zp['td_att_per_r'].values + Z['sub_per_r'].values) / 3.0

# Gram-Schmidt 正交化
a1 = std01(axis1_raw)
a2 = std01(residualize(axis2_raw, [a1]))
a3 = std01(residualize(axis3_raw, [a1, a2]))
data['x'], data['y'], data['z'] = a1, a2, a3

# ══════════════════════════════════════════════════════════════════════════
print("=== 6. 相似選手（同量級、全特徵 KNN top5）===")
Zsim = Z[FEATURES].copy()
Zsim['name'] = data['name'].values
Zsim['wc']   = data['wc'].values
similar_map = {}
for wc, grp in Zsim.groupby('wc'):
    names = grp['name'].values
    mat = grp[FEATURES].values
    for i, nm in enumerate(names):
        d = np.sqrt(((mat - mat[i]) ** 2).sum(axis=1))
        order = np.argsort(d)
        similar_map[nm] = [names[j] for j in order if names[j] != nm][:5]
data['similar'] = data['name'].map(similar_map)

# ══════════════════════════════════════════════════════════════════════════
print("=== 7. Tier（冠軍 / 前冠軍 / 排名）===")
champ_hist = pd.read_csv(CHAMP, encoding='utf-8-sig')
rank = pd.read_csv(RANK, encoding='utf-8-sig')

import unicodedata as _ud
def _norm(s):
    return _ud.normalize('NFKD', str(s)).encode('ascii','ignore').decode().strip().lower()

current = set(_norm(n) for n in rank.loc[rank['is_champion'].astype(str).str.strip().str.lower() == 'yes', 'fighter'].dropna())
if 'incumbent' in champ_hist.columns:
    inc = champ_hist['incumbent'].astype(str).str.lower().isin(['true','1'])
    current |= set(_norm(n) for n in champ_hist.loc[inc, 'champion'].dropna())
ever    = set(_norm(n) for n in champ_hist['champion'].dropna())
ex      = ever - current
ranked  = set(_norm(n) for n in rank['fighter'].dropna()) - current - ex

def tier_of(name):
    n = _norm(name)
    if n in current: return 'champion'
    if n in ex:      return 'ex_champion'
    if n in ranked:  return 'ranked'
    return ''
data['tier'] = data['name'].map(tier_of)

# ══════════════════════════════════════════════════════════════════════════
print("=== 8. 輸出 ===")
os.makedirs(os.path.dirname(OUT_VECTORS), exist_ok=True)
os.makedirs(os.path.dirname(OUT_JSON) or '.', exist_ok=True)

vec_cols = ['name','wc','n_fights','tier','x','y','z'] + FEATURES + \
           ['finish_rate','ko_rate','sub_rate','gas_tank','sig_att_per_r','td_att_per_r']
data[vec_cols].round(4).to_csv(OUT_VECTORS, index=False)
print(f"  → {OUT_VECTORS}")

def num(v, nd=4):
    if pd.isna(v): return None
    return round(float(v), nd)

records = []
for _, r in data.iterrows():
    records.append({
        "name": r['name'],
        "wc": r['wc'],
        "all_wc": r['all_wc'],
        "x": num(r['x']), "y": num(r['y']), "z": num(r['z']),
        "n_fights": int(r['n_fights']),
        "similar": list(r['similar']),
        "sig_per_r": num(r['sig_per_r'], 2),
        "td_per_r": num(r['td_per_r'], 2),
        "ko_rate": num(r['ko_rate'], 3),
        "sub_rate": num(r['sub_rate'], 3),
        "ctrl_per_r": num(r['ctrl_per_r'], 1),
        "gas_tank": num(r['gas_tank'], 3) if pd.notna(r['gas_tank']) else 1.0,
        "tier": r['tier'],
        "str_def": num(r['str_def'], 3),
        "td_def": num(r['td_def'], 3),
    })
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False)
print(f"  → {OUT_JSON}  ({len(records)} 名選手)")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("驗收區（把以下整段貼回來給我）")
print("=" * 60)

print("\n[A] 三軸相關矩陣（對角外應接近 0 = 正交成功）")
C = np.corrcoef(np.vstack([data['x'], data['y'], data['z']]))
print("        x        y        z")
for i, ax in enumerate(['x','y','z']):
    print(f"  {ax}  " + "  ".join(f"{C[i,j]:+.3f}" for j in range(3)))

def show_loadings(axis_name):
    print(f"\n[B] {axis_name} 軸：各特徵與此軸的相關（軸到底在量什麼）")
    cors = {c: np.corrcoef(data[axis_name], Z[c])[0,1] for c in FEATURES}
    sc = pd.Series(cors).sort_values()
    for c in list(sc.index[:5]) + list(sc.index[-5:]):
        print(f"    {c:16s} {sc[c]:+.2f}")

for ax, lab in [('x','軸一 Striker(-)/Grappler(+)'),
                ('y','軸二 Decision(-)/Finisher(+)'),
                ('z','軸三 Patient(-)/High-output(+)')]:
    show_loadings(ax)

print("\n[C] 軸三兩端各 15 人（眼睛檢查：高輸出端該是 Holloway/Merab 那種爆量工作型，")
print("    低輸出端該是惜打/精算的 Adesanya/Ngannou 那種；兩端都要好壞混雜=風格不是品質）")
srt = data.sort_values('z')
print("  ── 低輸出/精算端 (z 最低) ──")
for _, r in srt.head(15).iterrows():
    print(f"    {r['z']:+.2f}  {r['name']:24s} ({r['wc']}, {int(r['n_fights'])}場)")
print("  ── 高輸出/工作量端 (z 最高) ──")
for _, r in srt.tail(15)[::-1].iterrows():
    print(f"    {r['z']:+.2f}  {r['name']:24s} ({r['wc']}, {int(r['n_fights'])}場)")

print("\n[D] tier 分布:",
      {k: int((data['tier'] == k).sum()) for k in ['champion','ex_champion','ranked','']})
print("\n完成。輸出檔在", OUT_VECTORS, "和", OUT_JSON)
