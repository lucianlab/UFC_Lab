"""
UFC Sequence Data Preparation v6
覆蓋: src/data_prep_seq.py

序列特徵 29 個:
  打法統計 (12): dist/clinch/ground/head/body/leg_per,
                 sig_str_acc, td_acc, kd/sub_att/ctrl/sig_str_per_min
  控制時間輪次 (3): ctrl_R1, ctrl_R2, ctrl_trend
  打擊輪次 (2): dist_R1, dist_last
  節奏變化 (2): sig_str_pace_change, finish_round
  比賽脈絡 (4): opponent_wr, title_fight, total_rounds, fight_duration
  結果 (2): won, finish_type
  量級戰績 (2): div_win_rate, div_experience
  新增 (2): opponent_wr_recent, opponent_wr_trend

執行: python3 src/data_prep_seq.py
"""

import pandas as pd
import numpy as np
import pickle, json, os, re
from collections import defaultdict
from datetime import datetime

os.makedirs("data/processed", exist_ok=True)

# ──────────────────────────────────────────────
# 1. 載入原始資料
# ──────────────────────────────────────────────
print("載入原始資料...")
fs  = pd.read_csv("data/raw/ufc_fight_stats.csv")
fr  = pd.read_csv("data/raw/ufc_fight_results.csv")
fed = pd.read_csv("data/raw/ufc_event_details.csv")
ftt = pd.read_csv("data/raw/ufc_fighter_tott.csv")

for df in [fs, fr, fed]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

fr = fr.merge(fed[["EVENT","DATE"]], on="EVENT", how="left")
fr["DATE"] = pd.to_datetime(fr["DATE"], format="%B %d, %Y", errors="coerce")

MAIN_DIVISIONS = [
    "Heavyweight","Light Heavyweight","Middleweight","Welterweight",
    "Lightweight","Featherweight","Bantamweight","Flyweight",
    "Women's Strawweight","Women's Flyweight",
    "Women's Bantamweight","Women's Featherweight",
]
DIVISION_ENC = {d: i/11.0 for i, d in enumerate(MAIN_DIVISIONS)}

fr["DIV_CLEAN"] = fr["WEIGHTCLASS"]\
    .str.replace("UFC ", "", regex=False)\
    .str.replace(" Title", "", regex=False)\
    .str.replace(" Bout", "", regex=False)\
    .str.strip()

fr = fr[
    (fr["DATE"] >= "2005-01-01") &
    (fr["DIV_CLEAN"].isin(MAIN_DIVISIONS))
].copy().sort_values("DATE").reset_index(drop=True)

# 資料集最後一場比賽日期（用於計算 months_ago）
REFERENCE_DATE = fr["DATE"].max()
print(f"比賽總數: {len(fr)}")
print(f"資料截止: {REFERENCE_DATE.date()}")

# ──────────────────────────────────────────────
# 2. 結束方式編碼（5類）
# ──────────────────────────────────────────────
def encode_finish_5(method, round_num):
    """
    0: Early KO/TKO (R1-R2)
    1: Early Sub    (R1-R2)
    2: Late KO/TKO  (R3-R5)
    3: Late Sub     (R3-R5)
    4: Decision
    """
    m = str(method).lower() if not pd.isna(method) else ""
    try: r = int(round_num)
    except: r = 3
    early = r <= 2

    if "ko" in m or "tko" in m:
        return 0 if early else 2
    if "sub" in m:
        return 1 if early else 3
    return 4

def encode_finish_binary(method):
    m = str(method).lower() if not pd.isna(method) else ""
    if "ko" in m or "tko" in m: return 0
    if "sub" in m: return 1
    return 2

fr["finish_5class"]  = [encode_finish_5(m, r)
                         for m, r in zip(fr["METHOD"], fr["ROUND"])]
fr["finish_binary"]  = fr["METHOD"].apply(encode_finish_binary)

# ──────────────────────────────────────────────
# 3. 選手身體數據
# ──────────────────────────────────────────────
print("處理選手身體數據...")

def parse_height(h):
    if pd.isna(h) or h == "--": return None
    m = re.match(r"(\d+)' (\d+)\"", str(h))
    if m: return int(m.group(1))*30.48 + int(m.group(2))*2.54
    return None

def parse_reach(r):
    if pd.isna(r) or r == "--": return None
    m = re.match(r"([\d.]+)\"", str(r))
    if m: return float(m.group(1)) * 2.54
    return None

def parse_weight(w):
    if pd.isna(w) or w == "--": return None
    m = re.match(r"(\d+) lbs", str(w))
    if m: return float(m.group(1)) * 0.453592
    return None

STANCE_MAP = {"Orthodox":0,"Southpaw":1,"Switch":2,"Open Stance":2}

fighter_physical = {}
for _, row in ftt.iterrows():
    name = str(row["FIGHTER"]).strip()
    fighter_physical[name] = {
        "height":     parse_height(row.get("HEIGHT")),
        "reach":      parse_reach(row.get("REACH")),
        "weight":     parse_weight(row.get("WEIGHT")),
        "stance_enc": STANCE_MAP.get(str(row.get("STANCE","")), 0),
    }

# ──────────────────────────────────────────────
# 4. 各量級身體數據分布
# ──────────────────────────────────────────────
print("計算各量級身體數據分布...")
fighter_divisions = defaultdict(set)
for _, row in fr.iterrows():
    bout = str(row["BOUT"])
    div  = row["DIV_CLEAN"]
    parts = [p.strip() for p in bout.split(" vs. ")]
    if len(parts) == 2:
        fighter_divisions[parts[0]].add(div)
        fighter_divisions[parts[1]].add(div)

div_physical_stats = {}
for div in MAIN_DIVISIONS:
    fighters_in_div = [n for n, divs in fighter_divisions.items() if div in divs]
    div_physical_stats[div] = {}
    for stat in ["reach", "height"]:
        vals = [fighter_physical[n][stat]
                for n in fighters_in_div
                if n in fighter_physical and fighter_physical[n][stat] is not None]
        if vals:
            div_physical_stats[div][stat] = {"sorted": sorted(vals)}

def get_reach_pct(name, division):
    p    = fighter_physical.get(name, {})
    v    = p.get("reach")
    info = div_physical_stats.get(division, {}).get("reach")
    if info is None or v is None: return 0.5
    sv = info["sorted"]
    return sum(1 for x in sv if x < v) / len(sv) if sv else 0.5

# ──────────────────────────────────────────────
# 5. 解析 fight_stats
# ──────────────────────────────────────────────
def parse_of(val, default=0.0):
    if pd.isna(val): return default, default
    val = str(val).strip()
    m = re.match(r"(\d+)\s+of\s+(\d+)", val)
    if m: return float(m.group(1)), float(m.group(2))
    try: return float(val), float(val)
    except: return default, default

def parse_ctrl(val):
    if pd.isna(val): return 0.0
    m = re.match(r"(\d+):(\d+)", str(val).strip())
    if m: return int(m.group(1))*60 + int(m.group(2))
    return 0.0

def parse_pct(val):
    if pd.isna(val): return 0.0
    try: return float(str(val).replace("%","").strip()) / 100
    except: return 0.0

print("解析 fight_stats...")
fs["kd_num"]      = pd.to_numeric(fs["KD"], errors="coerce").fillna(0)
fs["sub_att_num"] = pd.to_numeric(fs["SUB.ATT"], errors="coerce").fillna(0)
fs["ctrl_sec"]    = fs["CTRL"].apply(parse_ctrl)
fs["sig_acc"]     = fs["SIG.STR. %"].apply(parse_pct)
fs[["sig_l","sig_a"]]       = fs["SIG.STR."].apply(lambda x: pd.Series(parse_of(x)))
fs[["td_l","td_a"]]         = fs["TD"].apply(lambda x: pd.Series(parse_of(x)))
fs[["head_l","head_a"]]     = fs["HEAD"].apply(lambda x: pd.Series(parse_of(x)))
fs[["body_l","body_a"]]     = fs["BODY"].apply(lambda x: pd.Series(parse_of(x)))
fs[["leg_l","leg_a"]]       = fs["LEG"].apply(lambda x: pd.Series(parse_of(x)))
fs[["dist_l","dist_a"]]     = fs["DISTANCE"].apply(lambda x: pd.Series(parse_of(x)))
fs[["clinch_l","clinch_a"]] = fs["CLINCH"].apply(lambda x: pd.Series(parse_of(x)))
fs[["ground_l","ground_a"]] = fs["GROUND"].apply(lambda x: pd.Series(parse_of(x)))
fs["round_num"] = pd.to_numeric(
    fs["ROUND"].str.extract(r"(\d+)")[0], errors="coerce").fillna(1).astype(int)

# ──────────────────────────────────────────────
# 6. Round-by-round 索引
# ──────────────────────────────────────────────
print("建立 round-by-round 索引...")
round_data = defaultdict(lambda: defaultdict(dict))
for _, row in fs.iterrows():
    event   = row["EVENT"]
    bout    = row["BOUT"]
    fighter = str(row["FIGHTER"]).strip()
    rnd     = row["round_num"]
    round_data[(event, bout, fighter)][rnd] = {
        "kd":       row["kd_num"],
        "sig_l":    row["sig_l"],
        "sig_a":    row["sig_a"],
        "td_l":     row["td_l"],
        "td_a":     row["td_a"],
        "sub_att":  row["sub_att_num"],
        "ctrl":     row["ctrl_sec"],
        "head_l":   row["head_l"],
        "body_l":   row["body_l"],
        "leg_l":    row["leg_l"],
        "dist_l":   row["dist_l"],
        "clinch_l": row["clinch_l"],
        "ground_l": row["ground_l"],
    }

# ──────────────────────────────────────────────
# 7. 比賽時長解析
# ──────────────────────────────────────────────
def parse_finish_round(r):
    try: return int(r)
    except: return 3

def parse_time_to_sec(t):
    if pd.isna(t): return 300
    m = re.match(r"(\d+):(\d+)", str(t))
    if m: return int(m.group(1))*60 + int(m.group(2))
    return 300

fr["finish_round_num"] = fr["ROUND"].apply(parse_finish_round)
fr["finish_sec"]       = fr["TIME"].apply(parse_time_to_sec)
fr["total_rounds_num"] = fr["TIME FORMAT"].apply(
    lambda x: 5 if "5-5-5-5-5" in str(x) else 3)

def parse_bout(bout_str):
    parts = str(bout_str).split(" vs. ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None

# ──────────────────────────────────────────────
# 8. 選手累積勝率（整體 + 量級 + 最近5場）
# ──────────────────────────────────────────────
print("計算選手累積勝率...")

overall_wins   = defaultdict(int)
overall_fights = defaultdict(int)
div_wins       = defaultdict(lambda: defaultdict(int))
div_fights     = defaultdict(lambda: defaultdict(int))

# 用 deque 記錄最近5場勝負
from collections import deque
recent_results = defaultdict(lambda: deque(maxlen=5))

def get_overall_wr(name):
    t = overall_fights[name]
    return overall_wins[name] / t if t > 0 else 0.5

def get_div_wr(name, div):
    t = div_fights[name][div]
    return div_wins[name][div] / t if t > 0 else get_overall_wr(name)

def get_div_exp(name, div):
    return div_fights[name][div]

def get_recent_wr(name):
    results = list(recent_results[name])
    if not results: return 0.5
    return sum(results) / len(results)

# ──────────────────────────────────────────────
# 9. 序列特徵定義（29個）
# ──────────────────────────────────────────────
SEQ_FEATURES = [
    # 打法統計（12）
    "dist_per","clinch_per","ground_per",
    "head_per","body_per","leg_per",
    "sig_str_acc","td_acc",
    "kd_per_min","sub_att_per_min","ctrl_per_min","sig_str_per_min",
    # 控制時間輪次（3）
    "ctrl_R1","ctrl_R2","ctrl_trend",
    # 打擊輪次（2）
    "dist_R1","dist_last",
    # 節奏變化（2）
    "sig_str_pace_change","finish_round",
    # 比賽脈絡（4）
    "opponent_wr","title_fight","total_rounds","fight_duration",
    # 結果（2）
    "won","finish_type",
    # 量級戰績（2）
    "div_win_rate","div_experience",
    # 新增（2）
    "opponent_wr_recent","opponent_wr_trend",
]
N_FEATURES = len(SEQ_FEATURES)
SEQ_LEN    = 10

assert N_FEATURES == 29, f"特徵數應為29，實際為{N_FEATURES}"
print(f"序列特徵數: {N_FEATURES}")
print(f"序列長度: {SEQ_LEN}")

# ──────────────────────────────────────────────
# 10. 逐場建立選手序列
# ──────────────────────────────────────────────
print("建立選手序列...")
fighter_history  = defaultdict(list)
matchup_records  = []

for _, row in fr.iterrows():
    f_round  = row["finish_round_num"]
    f_sec    = row["finish_sec"]
    f5       = row["finish_5class"]
    f_bin    = row["finish_binary"]
    t_rounds = row["total_rounds_num"]
    div      = row["DIV_CLEAN"]
    event    = row["EVENT"]
    bout     = row["BOUT"]
    date     = row["DATE"]
    tf       = 1.0 if "title" in str(row.get("WEIGHTCLASS","")).lower() else 0.0

    fighter_a, fighter_b = parse_bout(bout)
    if fighter_a is None or fighter_b is None: continue

    outcome = str(row.get("OUTCOME","")).strip()
    if outcome == "W/L":
        winner, loser = fighter_a, fighter_b
    elif outcome == "L/W":
        winner, loser = fighter_b, fighter_a
    else:
        continue

    prev_sec     = max(0, f_round - 1) * 5 * 60
    total_sec    = prev_sec + f_sec
    duration_min = max(total_sec / 60, 0.1)

    # months_ago：這場距今幾個月（標準化到0-1，60個月=5年為上限）
    months_ago = (REFERENCE_DATE - date).days / 30.0
    months_ago_norm = min(months_ago / 60.0, 1.0)

    def get_rounds_for(fighter):
        return round_data.get((event, bout, fighter), {})

    def make_vec(fighter, won, opp_name):
        rounds = get_rounds_for(fighter)
        if not rounds: return None

        total_sig_l  = sum(r.get("sig_l",0)   for r in rounds.values())
        total_td_l   = sum(r.get("td_l",0)    for r in rounds.values())
        total_td_a   = sum(r.get("td_a",0)    for r in rounds.values())
        total_kd     = sum(r.get("kd",0)      for r in rounds.values())
        total_sub    = sum(r.get("sub_att",0)  for r in rounds.values())
        total_ctrl   = sum(r.get("ctrl",0)    for r in rounds.values())
        total_head   = sum(r.get("head_l",0)  for r in rounds.values())
        total_body   = sum(r.get("body_l",0)  for r in rounds.values())
        total_leg    = sum(r.get("leg_l",0)   for r in rounds.values())
        total_dist   = sum(r.get("dist_l",0)  for r in rounds.values())
        total_clinch = sum(r.get("clinch_l",0) for r in rounds.values())
        total_ground = sum(r.get("ground_l",0) for r in rounds.values())
        total_struck = total_head + total_body + total_leg + 1e-6

        dist_per    = total_dist   / total_struck
        clinch_per  = total_clinch / total_struck
        ground_per  = total_ground / total_struck
        head_per    = total_head   / total_struck
        body_per    = total_body   / total_struck
        leg_per     = total_leg    / total_struck
        sig_acc     = total_sig_l  / (sum(r.get("sig_a",0) for r in rounds.values()) + 1e-6)
        td_acc      = total_td_l   / (total_td_a + 1e-6)
        kd_pm       = total_kd    / duration_min
        sub_pm      = total_sub   / duration_min
        ctrl_pm     = total_ctrl  / duration_min
        sig_pm      = total_sig_l / duration_min

        sorted_rounds = sorted(rounds.keys())
        r1   = rounds.get(1, {})
        r2   = rounds.get(2, {})
        r_last = rounds.get(sorted_rounds[-1], {}) if sorted_rounds else {}

        ctrl_R1    = r1.get("ctrl", 0) / 300.0
        ctrl_R2    = r2.get("ctrl", 0) / 300.0
        ctrl_last  = r_last.get("ctrl", 0) / 300.0
        ctrl_trend = ctrl_last - ctrl_R1

        r1_struck  = r1.get("head_l",0)+r1.get("body_l",0)+r1.get("leg_l",0)+1e-6
        rl_struck  = r_last.get("head_l",0)+r_last.get("body_l",0)+r_last.get("leg_l",0)+1e-6
        dist_R1    = r1.get("dist_l",0)    / r1_struck
        dist_last  = r_last.get("dist_l",0) / rl_struck

        r1_sig = r1.get("sig_l", 0)
        rl_sig = r_last.get("sig_l", 0)
        pace_change = (rl_sig - r1_sig) / (r1_sig + 1e-6)

        opp_wr        = get_overall_wr(opp_name)
        opp_wr_recent = get_recent_wr(opp_name)
        opp_wr_trend  = opp_wr_recent - opp_wr

        rounds_f  = float(t_rounds) / 5.0
        dur_f     = duration_min / 25.0
        fr_f      = float(f_round) / 5.0
        dwr       = get_div_wr(fighter, div)
        dex       = float(get_div_exp(fighter, div)) / 20.0
        won_f     = float(won)
        finish_f  = float(f_bin)

        vec = [
            dist_per, clinch_per, ground_per,
            head_per, body_per, leg_per,
            sig_acc, td_acc,
            kd_pm, sub_pm, ctrl_pm, sig_pm,
            ctrl_R1, ctrl_R2, ctrl_trend,
            dist_R1, dist_last,
            pace_change, fr_f,
            opp_wr, tf, rounds_f, dur_f,
            won_f, finish_f,
            dwr, dex,
            opp_wr_recent, opp_wr_trend,
        ]
        assert len(vec) == N_FEATURES, f"向量長度錯誤: {len(vec)} != {N_FEATURES}"
        return vec

    vec_a = make_vec(fighter_a, fighter_a==winner, fighter_b)
    vec_b = make_vec(fighter_b, fighter_b==winner, fighter_a)

    if vec_a is None or vec_b is None:
        for name, won in [(fighter_a, fighter_a==winner), (fighter_b, fighter_b==winner)]:
            overall_fights[name] += 1
            div_fights[name][div] += 1
            if won:
                overall_wins[name] += 1
                div_wins[name][div] += 1
            recent_results[name].append(1 if won else 0)
        continue

    seq_a_before = list(fighter_history[fighter_a])
    seq_b_before = list(fighter_history[fighter_b])

    if len(seq_a_before) >= 1 and len(seq_b_before) >= 1:
        matchup_records.append({
            "date":          str(date)[:10],
            "division":      div,
            "title_fight":   int(tf),
            "fighter_a":     fighter_a,
            "fighter_b":     fighter_b,
            "a_seq_len":     len(seq_a_before),
            "b_seq_len":     len(seq_b_before),
            "winner_is_a":   int(fighter_a == winner),
            "finish_5class": int(f5),
            "finish_method": int(f_bin),
            "finish_round":  int(f_round),
            "total_rounds":  int(t_rounds),
        })

    fighter_history[fighter_a].append(vec_a)
    fighter_history[fighter_b].append(vec_b)

    for name, won in [(fighter_a, fighter_a==winner), (fighter_b, fighter_b==winner)]:
        overall_fights[name] += 1
        div_fights[name][div] += 1
        if won:
            overall_wins[name] += 1
            div_wins[name][div] += 1
        recent_results[name].append(1 if won else 0)

print(f"訓練樣本數: {len(matchup_records)}")
print(f"選手數: {len(fighter_history)}")

# ──────────────────────────────────────────────
# 11. 建立序列矩陣
# ──────────────────────────────────────────────
print("建立序列矩陣...")
fighter_seq_data = {}
for name, history in fighter_history.items():
    seq = history[-SEQ_LEN:]
    n   = len(seq)
    pad = SEQ_LEN - n
    mat = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    msk = np.zeros(SEQ_LEN, dtype=bool)
    if n > 0:
        mat[pad:] = np.array(seq, dtype=np.float32)
        msk[pad:] = True
    fighter_seq_data[name] = {"seq": mat, "mask": msk, "n_fights": n}

# ──────────────────────────────────────────────
# 12. 對戰配對 + finish_5class 分布
# ──────────────────────────────────────────────
matchups_df = pd.DataFrame(matchup_records)
print(f"\n訓練樣本分布:")
print(f"  總樣本: {len(matchups_df)}")
print(f"  A方勝率: {matchups_df['winner_is_a'].mean():.2%}")
print(f"  結束方式分布:")
labels = {0:"EarlyKO", 1:"EarlySub", 2:"LateKO", 3:"LateSub", 4:"Decision"}
for k, v in labels.items():
    n = (matchups_df["finish_5class"]==k).sum()
    print(f"    {v}: {n} ({n/len(matchups_df)*100:.1f}%)")

# ──────────────────────────────────────────────
# 13. 物理特徵
# ──────────────────────────────────────────────
physical = {}
for name in fighter_history:
    p = fighter_physical.get(name, {})
    physical[name] = {
        "reach":      p.get("reach")      or 180.0,
        "height":     p.get("height")     or 175.0,
        "stance_enc": p.get("stance_enc") or 0,
    }

# ──────────────────────────────────────────────
# 14. 儲存
# ──────────────────────────────────────────────
print("\n儲存...")
with open("data/processed/fighter_sequences.pkl", "wb") as f:
    pickle.dump({
        "sequences":        fighter_seq_data,
        "physical":         physical,
        "seq_features":     SEQ_FEATURES,
        "seq_len":          SEQ_LEN,
        "div_physical":     div_physical_stats,
        "fighter_physical": fighter_physical,
        "division_enc":     DIVISION_ENC,
        "reference_date":   str(REFERENCE_DATE.date()),
    }, f)

matchups_df.to_csv("data/processed/matchups_seq.csv", index=False)

with open("data/processed/seq_stats.json", "w") as f:
    json.dump({
        "n_fighters":   len(fighter_history),
        "n_matchups":   len(matchups_df),
        "n_features":   N_FEATURES,
        "seq_len":      SEQ_LEN,
        "seq_features": SEQ_FEATURES,
        "a_win_rate":   float(matchups_df["winner_is_a"].mean()),
    }, f, indent=2)

with open("data/processed/division_physical_stats.json", "w") as f:
    json.dump({
        div: {stat: {"median": float(np.median(info["sorted"]))}
              for stat, info in stats.items()}
        for div, stats in div_physical_stats.items()
    }, f, indent=2)

print("✅ 完成！")
print(f"  data/processed/fighter_sequences.pkl")
print(f"  data/processed/matchups_seq.csv ({len(matchups_df)} 筆)")

# ──────────────────────────────────────────────
# 15. 驗證
# ──────────────────────────────────────────────
IDX = {name: i for i, name in enumerate(SEQ_FEATURES)}
print("\n驗證 Carlos Prates:")
if "Carlos Prates" in fighter_seq_data:
    p = fighter_seq_data["Carlos Prates"]
    print(f"  場次: {p['n_fights']}")
    for i, (vec, valid) in enumerate(zip(p["seq"], p["mask"])):
        if valid:
            print(f"  場{i+1}: dist={vec[IDX['dist_per']]:.2f} "
                  f"opp_wr_recent={vec[IDX['opponent_wr_recent']]:.2f} "
                  f"opp_trend={vec[IDX['opponent_wr_trend']]:+.2f}")

print("\n驗證 Tatsuro Taira (ctrl_trend):")
if "Tatsuro Taira" in fighter_seq_data:
    p = fighter_seq_data["Tatsuro Taira"]
    for i, (vec, valid) in enumerate(zip(p["seq"], p["mask"])):
        if valid:
            print(f"  場{i+1}: ctrl_R1={vec[IDX['ctrl_R1']]:.2f} "
                  f"ctrl_trend={vec[IDX['ctrl_trend']]:+.2f} "
                  f"won={vec[IDX['won']]:.0f}")
