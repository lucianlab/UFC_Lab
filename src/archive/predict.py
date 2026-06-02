"""
UFC Matchup Prediction — Step 3: Predict
執行方式: python3 src/predict.py
或直接 import: from src.predict import predict_matchup
"""

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from difflib import get_close_matches

# ──────────────────────────────────────────────
# 設定路徑
# ──────────────────────────────────────────────
DATA_PATH    = "data/processed/fights_clean.csv"
FEATURE_PATH = "models/feature_cols.json"
MODEL_WIN    = "models/model_win.json"
MODEL_METHOD = "models/model_method.json"
MODEL_ROUND  = "models/model_round.json"

METHOD_LABELS = {0: "KO/TKO", 1: "Submission", 2: "Decision"}
STANCE_MAP    = {"Orthodox": 0, "Southpaw": 1, "Switch": 2, "Open Stance": 2}

# ──────────────────────────────────────────────
# 1. 載入模型（只載入一次）
# ──────────────────────────────────────────────
def load_models():
    with open(FEATURE_PATH) as f:
        feature_cols = json.load(f)

    m_win = xgb.XGBClassifier()
    m_win.load_model(MODEL_WIN)

    m_method = xgb.XGBClassifier()
    m_method.load_model(MODEL_METHOD)

    m_round = xgb.XGBClassifier()
    m_round.load_model(MODEL_ROUND)

    return m_win, m_method, m_round, feature_cols


# ──────────────────────────────────────────────
# 2. 選手查詢：從資料庫找出選手最新統計
#    用「最近一場比賽」的 career stats，
#    因為那是目前最新的累積數據
# ──────────────────────────────────────────────
def get_fighter_stats(name: str, df: pd.DataFrame) -> dict | None:
    mask   = (df["r_name"] == name) | (df["b_name"] == name)
    fights = df[mask].sort_values("date")

    if len(fights) == 0:
        return None

    last   = fights.iloc[-1]
    prefix = "r_" if last["r_name"] == name else "b_"

    stat_keys = [
        "wins", "losses", "draws", "height", "weight", "reach",
        "splm", "sapm", "str_acc", "str_def",
        "td_avg", "td_avg_acc", "td_def", "sub_avg",
        "win_rate", "experience",
    ]

    stats = {"name": name, "division": last["division"]}
    for k in stat_keys:
        col = prefix + k
        stats[k] = last[col] if col in last.index else np.nan

    stats["stance"]     = last.get(prefix + "stance", "Orthodox")
    stats["stance_enc"] = STANCE_MAP.get(str(stats["stance"]), 0)
    stats["last_fight"] = str(last["date"])[:10]
    stats["n_fights"]   = len(fights)

    return stats


# ──────────────────────────────────────────────
# 3. 模糊搜尋：輸入不完整的名字也能找到
#    例如輸入 "makhachev" → "Islam Makhachev"
# ──────────────────────────────────────────────
def fuzzy_find(query: str, all_names: list[str]) -> str | None:
    query_lower = query.lower()

    # 完全符合（不分大小寫）
    for name in all_names:
        if name.lower() == query_lower:
            return name

    # 部分符合：名字裡包含搜尋字串
    partial = [n for n in all_names if query_lower in n.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print(f"  找到多個選手包含 '{query}':")
        for i, n in enumerate(partial[:8]):
            print(f"    {i+1}. {n}")
        choice = input("  請輸入編號: ").strip()
        try:
            return partial[int(choice) - 1]
        except:
            return None

    # 模糊匹配
    close = get_close_matches(query, all_names, n=3, cutoff=0.5)
    if close:
        print(f"  找不到 '{query}'，你是指：")
        for i, n in enumerate(close):
            print(f"    {i+1}. {n}")
        choice = input("  請輸入編號 (或按 Enter 取消): ").strip()
        if choice:
            try:
                return close[int(choice) - 1]
            except:
                return None

    return None


# ──────────────────────────────────────────────
# 4. 建立 Feature Vector
#    把兩個選手的 stats 轉成 delta features
# ──────────────────────────────────────────────
def build_features(fighter_a: dict, fighter_b: dict,
                   title_fight: int, feature_cols: list) -> np.ndarray:

    delta_map = {
        "delta_wins":       fighter_a["wins"]       - fighter_b["wins"],
        "delta_losses":     fighter_a["losses"]      - fighter_b["losses"],
        "delta_win_rate":   fighter_a["win_rate"]    - fighter_b["win_rate"],
        "delta_experience": fighter_a["experience"]  - fighter_b["experience"],
        "delta_splm":       fighter_a["splm"]        - fighter_b["splm"],
        "delta_sapm":       fighter_a["sapm"]        - fighter_b["sapm"],
        "delta_str_acc":    fighter_a["str_acc"]     - fighter_b["str_acc"],
        "delta_str_def":    fighter_a["str_def"]     - fighter_b["str_def"],
        "delta_td_avg":     fighter_a["td_avg"]      - fighter_b["td_avg"],
        "delta_td_avg_acc": fighter_a["td_avg_acc"]  - fighter_b["td_avg_acc"],
        "delta_td_def":     fighter_a["td_def"]      - fighter_b["td_def"],
        "delta_sub_avg":    fighter_a["sub_avg"]     - fighter_b["sub_avg"],
        "delta_height":     fighter_a["height"]      - fighter_b["height"],
        "delta_reach":      fighter_a["reach"]       - fighter_b["reach"],
        "title_fight":      title_fight,
        "r_stance_enc":     fighter_a["stance_enc"],
        "b_stance_enc":     fighter_b["stance_enc"],
        "southpaw_matchup": int(
            (fighter_a["stance_enc"] == 0 and fighter_b["stance_enc"] == 1) or
            (fighter_a["stance_enc"] == 1 and fighter_b["stance_enc"] == 0)
        ),
        "same_stance": int(fighter_a["stance_enc"] == fighter_b["stance_enc"]),
    }

    row = [delta_map.get(col, 0.0) for col in feature_cols]
    return np.array(row, dtype=float).reshape(1, -1)


# ──────────────────────────────────────────────
# 5. 主預測函數
# ──────────────────────────────────────────────
def predict_matchup(name_a: str, name_b: str,
                    title_fight: bool = False,
                    verbose: bool = True):

    df              = pd.read_csv(DATA_PATH, parse_dates=["date"])
    all_names       = sorted(set(df["r_name"].tolist() + df["b_name"].tolist()))
    m_win, m_method, m_round, feature_cols = load_models()

    # 搜尋選手
    found_a = fuzzy_find(name_a, all_names)
    found_b = fuzzy_find(name_b, all_names)

    if not found_a:
        print(f"❌ 找不到選手: {name_a}")
        return None
    if not found_b:
        print(f"❌ 找不到選手: {name_b}")
        return None

    stats_a = get_fighter_stats(found_a, df)
    stats_b = get_fighter_stats(found_b, df)

    # 建立 feature vector（A 為紅角）
    X = build_features(stats_a, stats_b,
                       int(title_fight), feature_cols)

    # 預測
    win_prob_a  = float(m_win.predict_proba(X)[0][1])   # A 贏的機率
    win_prob_b  = 1.0 - win_prob_a

    method_prob = m_method.predict_proba(X)[0]          # [KO, SUB, DEC]
    round_prob  = m_round.predict_proba(X)[0]           # [R1, R2, R3, R4, R5]

    top_method  = METHOD_LABELS[int(np.argmax(method_prob))]
    top_round   = int(np.argmax(round_prob)) + 1

    # ── 解釋：找出影響最大的 delta features ──
    feat_vals   = X[0]
    feat_names  = feature_cols
    delta_idx   = [(i, n, v) for i, (n, v) in enumerate(zip(feat_names, feat_vals))
                   if n.startswith("delta_")]

    # 正值 = A 佔優，負值 = B 佔優，按絕對值排序
    delta_idx.sort(key=lambda x: abs(x[2]), reverse=True)
    top_factors = delta_idx[:3]

    FACTOR_LABEL = {
        "delta_win_rate":   "勝率",
        "delta_wins":       "勝場數",
        "delta_splm":       "打擊輸出",
        "delta_sapm":       "被打頻率",
        "delta_str_acc":    "打擊精準度",
        "delta_str_def":    "打擊防禦",
        "delta_td_avg":     "摔技平均",
        "delta_td_def":     "防摔能力",
        "delta_sub_avg":    "降伏嘗試",
        "delta_reach":      "臂展",
        "delta_height":     "身高",
        "delta_experience": "出賽經驗",
        "delta_losses":     "敗場差距",
    }

    # ── 輸出 ──
    if verbose:
        SEP = "─" * 52
        print(f"\n{SEP}")
        print(f"  {found_a}  vs  {found_b}")
        if title_fight:
            print(f"  🏆 冠軍戰")
        print(SEP)

        bar_a = "█" * int(win_prob_a * 30)
        bar_b = "█" * int(win_prob_b * 30)
        print(f"\n  勝率預測")
        print(f"  {found_a:<22} {bar_a} {win_prob_a*100:.1f}%")
        print(f"  {found_b:<22} {bar_b} {win_prob_b*100:.1f}%")

        print(f"\n  結束方式")
        for i, label in METHOD_LABELS.items():
            p = method_prob[i]
            bar = "░" * int(p * 30)
            print(f"  {label:<12} {bar} {p*100:.1f}%")

        print(f"\n  結束輪次")
        for i, p in enumerate(round_prob):
            bar = "░" * int(p * 30)
            mark = " ◀" if i + 1 == top_round else ""
            print(f"  第 {i+1} 輪    {bar} {p*100:.1f}%{mark}")

        print(f"\n  關鍵因素 (影響勝負的前3個數據差距)")
        for _, name, val in top_factors:
            label = FACTOR_LABEL.get(name, name)
            if val > 0:
                advantage = f"{found_a} 佔優 (+{val:.2f})"
            else:
                advantage = f"{found_b} 佔優 ({val:.2f})"
            print(f"  • {label:<10} {advantage}")

        print(f"\n  選手資料時間")
        print(f"  {found_a}: 最近一戰 {stats_a['last_fight']} ({stats_a['n_fights']} 場)")
        print(f"  {found_b}: 最近一戰 {stats_b['last_fight']} ({stats_b['n_fights']} 場)")
        print(SEP)

    return {
        "fighter_a": found_a,
        "fighter_b": found_b,
        "win_prob_a": round(win_prob_a, 4),
        "win_prob_b": round(win_prob_b, 4),
        "method_probs": {METHOD_LABELS[i]: round(float(p), 4)
                         for i, p in enumerate(method_prob)},
        "round_probs":  {f"R{i+1}": round(float(p), 4)
                         for i, p in enumerate(round_prob)},
        "top_method": top_method,
        "top_round":  top_round,
    }


# ──────────────────────────────────────────────
# 6. 互動模式（直接執行時啟動）
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("UFC Matchup Predictor")
    print("輸入選手名稱（部分名字也可以，例如 'makhachev'）")
    print("輸入 'q' 離開\n")

    while True:
        a = input("選手 A: ").strip()
        if a.lower() == "q":
            break
        b = input("選手 B: ").strip()
        if b.lower() == "q":
            break
        t = input("冠軍戰? (y/n, 預設 n): ").strip().lower()
        title = (t == "y")

        predict_matchup(a, b, title_fight=title)

        print()
