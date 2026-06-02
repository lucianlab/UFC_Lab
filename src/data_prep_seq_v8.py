"""
UFC Sequence Data Preparation v8
覆蓋: src/data_prep_seq.py

v8 新增（基於 UFC PI Cross-Sectional Analysis Vol.2）：
  - sig_str_defense   打擊防守率（對手打中率的反面）
  - td_defense        摔角防守率
  - absorption_rate   每分鐘被打中次數（normalized by duration）
  - sig_str_rel_diff  相對打擊效率（我 vs 對手，比例形式）
  - td_rel_diff       相對摔角效率

  make_vec 加入 opp_rounds 參數，同一場中同時取雙方數據
  特徵數 28 → 33

v7 繼承：
  - point-in-time snapshot（在 append 之前截取）
  - 輸出 matchup_samples.pkl（訓練用）+ fighter_sequences.pkl（predict用）

執行: python3 src/data_prep_seq.py
"""

import pandas as pd
import numpy as np
import pickle, json, os, re
from collections import defaultdict, deque

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
    (fr["DATE"] >= "2010-01-01") &
    (fr["DATE"] <  "2026-01-01") &
    (fr["DIV_CLEAN"].isin(MAIN_DIVISIONS))
].copy().sort_values("DATE").reset_index(drop=True)

print(f"比賽總數: {len(fr)}")
print(f"日期範圍: {fr['DATE'].min().date()} ~ {fr['DATE'].max().date()}")

# ──────────────────────────────────────────────
# 2. 選手身體數據
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

STANCE_MAP = {"Orthodox":0, "Southpaw":1, "Switch":2, "Open Stance":2}

fighter_physical = {}
for _, row in ftt.iterrows():
    name = str(row["FIGHTER"]).strip()
    fighter_physical[name] = {
        "height":     parse_height(row.get("HEIGHT")),
        "reach":      parse_reach(row.get("REACH")),
        "stance_enc": STANCE_MAP.get(str(row.get("STANCE","")), 0),
    }

# ──────────────────────────────────────────────
# 3. 各量級身體數據分布
# ──────────────────────────────────────────────
print("計算各量級身體數據分布...")
fighter_divisions = defaultdict(set)
for _, row in fr.iterrows():
    parts = [p.strip() for p in str(row["BOUT"]).split(" vs. ")]
    if len(parts) == 2:
        fighter_divisions[parts[0]].add(row["DIV_CLEAN"])
        fighter_divisions[parts[1]].add(row["DIV_CLEAN"])

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

# ──────────────────────────────────────────────
# 4. 解析 fight_stats
# ──────────────────────────────────────────────
def parse_of(val, default=0.0):
    if pd.isna(val): return default, default
    m = re.match(r"(\d+)\s+of\s+(\d+)", str(val).strip())
    if m: return float(m.group(1)), float(m.group(2))
    try: return float(val), float(val)
    except: return default, default

def parse_ctrl(val):
    if pd.isna(val): return 0.0
    m = re.match(r"(\d+):(\d+)", str(val).strip())
    if m: return int(m.group(1))*60 + int(m.group(2))
    return 0.0

print("解析 fight_stats...")
fs["kd_num"]      = pd.to_numeric(fs["KD"], errors="coerce").fillna(0)
fs["sub_att_num"] = pd.to_numeric(fs["SUB.ATT"], errors="coerce").fillna(0)
fs["ctrl_sec"]    = fs["CTRL"].apply(parse_ctrl)
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
# 5. Round-by-round 索引
# ──────────────────────────────────────────────
print("建立 round-by-round 索引...")
round_data = defaultdict(lambda: defaultdict(dict))
for _, row in fs.iterrows():
    key = (row["EVENT"], row["BOUT"], str(row["FIGHTER"]).strip())
    round_data[key][row["round_num"]] = {
        "kd":       row["kd_num"],
        "sig_l":    row["sig_l"],    "sig_a":    row["sig_a"],
        "td_l":     row["td_l"],     "td_a":     row["td_a"],
        "sub_att":  row["sub_att_num"],
        "ctrl":     row["ctrl_sec"],
        "head_l":   row["head_l"],   "body_l":   row["body_l"],
        "leg_l":    row["leg_l"],    "dist_l":   row["dist_l"],
        "clinch_l": row["clinch_l"], "ground_l": row["ground_l"],
    }

# ──────────────────────────────────────────────
# 6. 比賽結果解析
# ──────────────────────────────────────────────
def parse_finish_round(r):
    try: return int(r)
    except: return 3

def parse_time_to_sec(t):
    if pd.isna(t): return 300
    m = re.match(r"(\d+):(\d+)", str(t))
    if m: return int(m.group(1))*60 + int(m.group(2))
    return 300

def encode_method(m):
    if pd.isna(m): return 2
    m = str(m).lower()
    if "ko" in m or "tko" in m: return 0
    if "sub" in m: return 1
    return 2

fr["finish_round_num"] = fr["ROUND"].apply(parse_finish_round)
fr["finish_sec"]       = fr["TIME"].apply(parse_time_to_sec)
fr["finish_type"]      = fr["METHOD"].apply(encode_method)
fr["total_rounds_num"] = fr["TIME FORMAT"].apply(
    lambda x: 5 if "5-5-5-5-5" in str(x) else 3)

def parse_bout(bout_str):
    parts = str(bout_str).split(" vs. ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None

# ──────────────────────────────────────────────
# 7. 累積勝率 trackers
# ──────────────────────────────────────────────
print("初始化累積勝率追蹤...")
overall_wins   = defaultdict(int)
overall_fights = defaultdict(int)
div_wins       = defaultdict(lambda: defaultdict(int))
div_fights     = defaultdict(lambda: defaultdict(int))
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
    r = list(recent_results[name])
    return sum(r) / len(r) if r else 0.5

# ──────────────────────────────────────────────
# 8. 序列特徵定義（33個）
# ──────────────────────────────────────────────
SEQ_FEATURES = [
    # 打法統計—進攻（11）
    "dist_per", "clinch_per", "ground_per",
    "head_per", "body_per", "leg_per",
    "sig_str_acc", "td_acc",
    "kd_per_min", "sub_att_per_min", "sig_str_per_min",
    # 控制時間輪次（3）
    "ctrl_R1", "ctrl_R2", "ctrl_trend",
    # 打擊輪次（2）
    "dist_R1", "dist_last",
    # 節奏變化（2）
    "sig_str_pace_change", "finish_round",
    # 比賽脈絡（4）
    "opponent_wr", "title_fight", "total_rounds", "fight_duration",
    # 結果（2）
    "won", "finish_type",
    # 量級戰績（2）
    "div_win_rate", "div_experience",
    # 對手近期（2）
    "opponent_wr_recent", "opponent_wr_trend",
    # ── v8 新增：防守 + 相對效率（5）──
    "sig_str_defense",   # 打擊防守率 (1 - opp打中率)
    "td_defense",        # 摔角防守率 (1 - opp摔倒率)
    "absorption_rate",   # 每分鐘被打中次數 normalized
    "sig_str_rel_diff",  # 相對打擊效率 (我-對手)/(我+對手)
    "td_rel_diff",       # 相對摔角效率 (我-對手)/(我+對手)
]
N_FEATURES = len(SEQ_FEATURES)
SEQ_LEN    = 10

assert N_FEATURES == 33, f"特徵數應為33，實際為{N_FEATURES}"
print(f"序列特徵數: {N_FEATURES} (v7: 28 + v8新增: 5)")
print(f"序列長度: {SEQ_LEN}")

# ──────────────────────────────────────────────
# 9. make_vec（v8：加入 opp_rounds 參數）
# ──────────────────────────────────────────────
def make_vec(fighter, won, opp_name,
             my_rounds, opp_rounds,
             f_round, f_sec, f_type, t_rounds, div, tf):
    """
    my_rounds  : dict {round_num: stats}  自己這場的數據
    opp_rounds : dict {round_num: stats}  對手這場的數據（用於計算防守特徵）
    """
    if not my_rounds:
        return None

    prev_sec     = max(0, f_round - 1) * 5 * 60
    total_sec    = prev_sec + f_sec
    duration_min = max(total_sec / 60, 0.1)

    # ── 自己的進攻數據 ──
    total_sig_l  = sum(r.get("sig_l",0)    for r in my_rounds.values())
    total_sig_a  = sum(r.get("sig_a",0)    for r in my_rounds.values())
    total_td_l   = sum(r.get("td_l",0)     for r in my_rounds.values())
    total_td_a   = sum(r.get("td_a",0)     for r in my_rounds.values())
    total_kd     = sum(r.get("kd",0)       for r in my_rounds.values())
    total_sub    = sum(r.get("sub_att",0)  for r in my_rounds.values())
    total_ctrl   = sum(r.get("ctrl",0)     for r in my_rounds.values())
    total_head   = sum(r.get("head_l",0)   for r in my_rounds.values())
    total_body   = sum(r.get("body_l",0)   for r in my_rounds.values())
    total_leg    = sum(r.get("leg_l",0)    for r in my_rounds.values())
    total_dist   = sum(r.get("dist_l",0)   for r in my_rounds.values())
    total_clinch = sum(r.get("clinch_l",0) for r in my_rounds.values())
    total_ground = sum(r.get("ground_l",0) for r in my_rounds.values())
    total_struck = total_head + total_body + total_leg + 1e-6

    sig_acc  = total_sig_l / (total_sig_a + 1e-6)
    td_acc   = total_td_l  / (total_td_a  + 1e-6)
    sig_pm   = total_sig_l / duration_min
    td_pm    = total_td_l  / duration_min

    sorted_rounds = sorted(my_rounds.keys())
    r1     = my_rounds.get(1, {})
    r2     = my_rounds.get(2, {})
    r_last = my_rounds.get(sorted_rounds[-1], {}) if sorted_rounds else {}

    ctrl_R1    = r1.get("ctrl", 0) / 300.0
    ctrl_R2    = r2.get("ctrl", 0) / 300.0
    ctrl_trend = r_last.get("ctrl", 0) / 300.0 - ctrl_R1

    r1_struck  = r1.get("head_l",0)+r1.get("body_l",0)+r1.get("leg_l",0)+1e-6
    rl_struck  = r_last.get("head_l",0)+r_last.get("body_l",0)+r_last.get("leg_l",0)+1e-6
    dist_R1    = r1.get("dist_l",0)    / r1_struck
    dist_last  = r_last.get("dist_l",0) / rl_struck
    pace_change = (r_last.get("sig_l",0) - r1.get("sig_l",0)) / (r1.get("sig_l",0) + 1e-6)

    # ── 對手方數據（用於防守特徵）──
    # 注意：opp 的 sig_l = 對手打中「我」的次數
    if opp_rounds:
        opp_sig_l = sum(r.get("sig_l",0) for r in opp_rounds.values())
        opp_sig_a = sum(r.get("sig_a",0) for r in opp_rounds.values())
        opp_td_l  = sum(r.get("td_l",0)  for r in opp_rounds.values())
        opp_td_a  = sum(r.get("td_a",0)  for r in opp_rounds.values())
        opp_sig_pm = opp_sig_l / duration_min
        opp_td_pm  = opp_td_l  / duration_min

        sig_str_defense = 1.0 - (opp_sig_l / (opp_sig_a + 1e-6))
        td_defense      = 1.0 - (opp_td_l  / (opp_td_a  + 1e-6))
        # absorption_rate: normalize to [0,1] range，除以合理上限 10次/分鐘
        absorption_rate = min(opp_sig_pm / 10.0, 1.0)
        # 相對效率：(我 - 對手) / (我 + 對手 + eps)，結果在 (-1, 1)
        sig_str_rel_diff = (sig_pm  - opp_sig_pm) / (sig_pm  + opp_sig_pm + 1e-6)
        td_rel_diff      = (td_pm   - opp_td_pm)  / (td_pm   + opp_td_pm  + 1e-6)
    else:
        # 沒有對手數據時填中性值
        sig_str_defense  = 0.5
        td_defense       = 0.5
        absorption_rate  = 0.5
        sig_str_rel_diff = 0.0
        td_rel_diff      = 0.0

    # ── 對手的歷史勝率（用累積 tracker，這場之前的數據）──
    opp_wr        = get_overall_wr(opp_name)
    opp_wr_recent = get_recent_wr(opp_name)
    opp_wr_trend  = opp_wr_recent - opp_wr

    vec = [
        # 打法統計—進攻（11）
        total_dist   / total_struck,
        total_clinch / total_struck,
        total_ground / total_struck,
        total_head   / total_struck,
        total_body   / total_struck,
        total_leg    / total_struck,
        sig_acc, td_acc,
        total_kd  / duration_min,
        total_sub / duration_min,
        sig_pm,
        # 控制時間輪次（3）
        ctrl_R1, ctrl_R2, ctrl_trend,
        # 打擊輪次（2）
        dist_R1, dist_last,
        # 節奏變化（2）
        pace_change,
        float(f_round) / 5.0,
        # 比賽脈絡（4）
        opp_wr, float(tf),
        float(t_rounds) / 5.0,
        (prev_sec + f_sec) / 60.0 / 25.0,
        # 結果（2）
        float(won), float(f_type),
        # 量級戰績（2）
        get_div_wr(fighter, div),
        float(get_div_exp(fighter, div)) / 20.0,
        # 對手近期（2）
        opp_wr_recent, opp_wr_trend,
        # v8 防守 + 相對效率（5）
        sig_str_defense,
        td_defense,
        absorption_rate,
        sig_str_rel_diff,
        td_rel_diff,
    ]
    assert len(vec) == N_FEATURES, f"vec長度 {len(vec)} != {N_FEATURES}"
    return vec

# ──────────────────────────────────────────────
# 10. 主迴圈：逐場建立序列 + point-in-time snapshot
# ──────────────────────────────────────────────
print("建立選手序列（含 point-in-time snapshot）...")

fighter_history = defaultdict(list)
matchup_samples = []

def build_snapshot(history_list):
    seq  = history_list[-SEQ_LEN:]
    n    = len(seq)
    pad  = SEQ_LEN - n
    mat  = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    mask = np.zeros(SEQ_LEN, dtype=bool)
    if n > 0:
        mat[pad:]  = np.array(seq, dtype=np.float32)
        mask[pad:] = True
    return mat, mask

skipped = 0
for _, row in fr.iterrows():
    f_round  = row["finish_round_num"]
    f_sec    = row["finish_sec"]
    f_type   = row["finish_type"]
    t_rounds = row["total_rounds_num"]
    div      = row["DIV_CLEAN"]
    event    = row["EVENT"]
    bout     = row["BOUT"]
    date     = row["DATE"]
    tf       = 1.0 if "title" in str(row.get("WEIGHTCLASS","")).lower() else 0.0

    fighter_a, fighter_b = parse_bout(bout)
    if fighter_a is None or fighter_b is None:
        continue

    outcome = str(row.get("OUTCOME","")).strip()
    if outcome == "W/L":
        winner = fighter_a
    elif outcome == "L/W":
        winner = fighter_b
    else:
        continue

    # 取得雙方的 round_data（互傳給對方）
    rounds_a = round_data.get((event, bout, fighter_a), {})
    rounds_b = round_data.get((event, bout, fighter_b), {})

    vec_a = make_vec(
        fighter_a, fighter_a==winner, fighter_b,
        rounds_a, rounds_b,          # my_rounds=A, opp_rounds=B
        f_round, f_sec, f_type, t_rounds, div, tf
    )
    vec_b = make_vec(
        fighter_b, fighter_b==winner, fighter_a,
        rounds_b, rounds_a,          # my_rounds=B, opp_rounds=A
        f_round, f_sec, f_type, t_rounds, div, tf
    )

    if vec_a is None or vec_b is None:
        for name, won in [(fighter_a, fighter_a==winner),
                          (fighter_b, fighter_b==winner)]:
            overall_fights[name] += 1
            div_fights[name][div] += 1
            if won:
                overall_wins[name] += 1
                div_wins[name][div] += 1
            recent_results[name].append(1 if won else 0)
        skipped += 1
        continue

    # ── 在 append 之前截取 snapshot（point-in-time）──
    if len(fighter_history[fighter_a]) >= 1 and \
       len(fighter_history[fighter_b]) >= 1:

        snap_a_seq, snap_a_mask = build_snapshot(fighter_history[fighter_a])
        snap_b_seq, snap_b_mask = build_snapshot(fighter_history[fighter_b])

        matchup_samples.append({
            "date":          str(date)[:10],
            "division":      div,
            "title_fight":   int(tf),
            "fighter_a":     fighter_a,
            "fighter_b":     fighter_b,
            "n_hist_a":      len(fighter_history[fighter_a]),
            "n_hist_b":      len(fighter_history[fighter_b]),
            "winner_is_a":   int(fighter_a == winner),
            "finish_method": int(f_type),
            "finish_round":  int(f_round),
            "total_rounds":  int(t_rounds),
            "seq_a":         snap_a_seq,
            "mask_a":        snap_a_mask,
            "seq_b":         snap_b_seq,
            "mask_b":        snap_b_mask,
        })

    # append 在 snapshot 之後
    fighter_history[fighter_a].append(vec_a)
    fighter_history[fighter_b].append(vec_b)

    # 更新累積勝率（在 append 之後）
    for name, won in [(fighter_a, fighter_a==winner),
                      (fighter_b, fighter_b==winner)]:
        overall_fights[name] += 1
        div_fights[name][div] += 1
        if won:
            overall_wins[name] += 1
            div_wins[name][div] += 1
        recent_results[name].append(1 if won else 0)

print(f"訓練樣本數: {len(matchup_samples)}")
print(f"跳過（無打擊數據）: {skipped}")
print(f"選手數: {len(fighter_history)}")

# ──────────────────────────────────────────────
# 11. 全量選手序列（predict/backtest 用）
# ──────────────────────────────────────────────
print("建立全量選手序列（predict 用）...")
fighter_seq_data = {}
for name, history in fighter_history.items():
    seq  = history[-SEQ_LEN:]
    n    = len(seq)
    pad  = SEQ_LEN - n
    mat  = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    mask = np.zeros(SEQ_LEN, dtype=bool)
    if n > 0:
        mat[pad:]  = np.array(seq, dtype=np.float32)
        mask[pad:] = True
    fighter_seq_data[name] = {"seq": mat, "mask": mask, "n_fights": n}

# ──────────────────────────────────────────────
# 12. 物理特徵
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
# 13. 儲存
# ──────────────────────────────────────────────
print("\n儲存...")

with open("data/processed/matchup_samples.pkl", "wb") as f:
    pickle.dump(matchup_samples, f)

with open("data/processed/fighter_sequences.pkl", "wb") as f:
    pickle.dump({
        "sequences":        fighter_seq_data,
        "physical":         physical,
        "seq_features":     SEQ_FEATURES,
        "seq_len":          SEQ_LEN,
        "div_physical":     div_physical_stats,
        "fighter_physical": fighter_physical,
        "division_enc":     DIVISION_ENC,
        "cutoff_date":      "2025-12-31",
    }, f)

win_rate = sum(s["winner_is_a"] for s in matchup_samples) / max(len(matchup_samples),1)
with open("data/processed/seq_stats.json", "w") as f:
    json.dump({
        "n_fighters":   len(fighter_history),
        "n_samples":    len(matchup_samples),
        "n_features":   N_FEATURES,
        "seq_len":      SEQ_LEN,
        "seq_features": SEQ_FEATURES,
        "a_win_rate":   win_rate,
        "version":      "v8",
    }, f, indent=2)

size_mb = os.path.getsize("data/processed/matchup_samples.pkl") / 1e6
print(f"✅ 完成！")
print(f"  matchup_samples.pkl  ({size_mb:.1f} MB, {len(matchup_samples)} samples)")
print(f"  fighter_sequences.pkl ({len(fighter_seq_data)} 位選手, {N_FEATURES} 特徵)")

# ──────────────────────────────────────────────
# 14. 驗證：抽查特徵值範圍
# ──────────────────────────────────────────────
print("\n驗證特徵值範圍...")
if matchup_samples:
    all_vecs = []
    for s in matchup_samples[:500]:
        for row in s["seq_a"]:
            if row.sum() != 0:
                all_vecs.append(row)
    if all_vecs:
        arr = np.array(all_vecs)
        IDX = {name: i for i, name in enumerate(SEQ_FEATURES)}
        for feat in ["sig_str_defense","td_defense","absorption_rate",
                     "sig_str_rel_diff","td_rel_diff"]:
            i = IDX[feat]
            print(f"  {feat:<22} min={arr[:,i].min():.3f}  "
                  f"max={arr[:,i].max():.3f}  mean={arr[:,i].mean():.3f}")
