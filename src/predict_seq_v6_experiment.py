"""
UFC GRU Sequence Model — Prediction v6
覆蓋: src/predict_seq.py

新增功能：
  - MC Dropout 信心區間（推論跑100次）
  - Platt Scaling 校準
  - 結束方式機率輸出（5類）
  - 保守 Kelly（用信心區間下限）

執行: python3 src/predict_seq.py
"""

import pandas as pd
import numpy as np
import json, os, pickle
import torch
import torch.nn as nn
from difflib import get_close_matches

SEQ_DATA_PATH    = "data/processed/fighter_sequences.pkl"
MATCHUPS_PATH    = "data/processed/matchups_seq.csv"
MODEL_PATH       = "models/model_seq.pt"
CONFIG_PATH      = "models/seq_model_config.json"
EMBED_PATH       = "models/seq_embeddings.npy"
ID_MAP_PATH      = "models/seq_fighter_id_map.json"
CALIBRATOR_PATH  = "models/platt_calibrator.pkl"
RAW_FIGHT_PATH   = "data/raw/ufc_fight_stats.csv"
RAW_RESULT_PATH  = "data/raw/ufc_fight_results.csv"
RAW_EVENT_PATH   = "data/raw/ufc_event_details.csv"
RAW_FIGHTER_PATH = "data/raw/ufc_fighter_tott.csv"

FINISH_LABELS = {
    0: "Early KO/TKO (R1-2)",
    1: "Early Sub (R1-2)",
    2: "Late KO/TKO (R3-5)",
    3: "Late Sub (R3-5)",
    4: "Decision",
}

# ──────────────────────────────────────────────
# 模型定義
# ──────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, gru_out, mask):
        scores  = self.attn(gru_out).squeeze(-1)
        scores  = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = (gru_out * weights.unsqueeze(-1)).sum(dim=1)
        return context, weights

class FighterEncoder(nn.Module):
    def __init__(self, n_feat, hidden_dim):
        super().__init__()
        self.gru       = nn.GRU(n_feat, hidden_dim, num_layers=2,
                                batch_first=True, dropout=0.2)
        self.attention = Attention(hidden_dim)
        self.norm      = nn.LayerNorm(hidden_dim)
    def forward(self, seq, mask):
        out, _ = self.gru(seq)
        ctx, w = self.attention(out, mask)
        return self.norm(ctx), w

class UFCSeqModel(nn.Module):
    def __init__(self, n_feat, hidden_dim, matchup_dim, n_finish):
        super().__init__()
        self.encoder = FighterEncoder(n_feat, hidden_dim)
        in_dim = hidden_dim * 2 + matchup_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.win_head    = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.finish_head = nn.Linear(64, n_finish)

    def forward(self, a_seq, a_mask, b_seq, b_mask, matchup):
        style_a, _ = self.encoder(a_seq, a_mask)
        style_b, _ = self.encoder(b_seq, b_mask)
        x = torch.cat([style_a, style_b, matchup], dim=1)
        shared = self.shared(x)
        return self.win_head(shared).squeeze(1), self.finish_head(shared)

    def encode_with_attention(self, seq, mask):
        with torch.no_grad():
            style, weights = self.encoder(seq, mask)
            return style.numpy(), weights.numpy()

# ──────────────────────────────────────────────
# 載入資源
# ──────────────────────────────────────────────
def load_resources():
    with open(SEQ_DATA_PATH, "rb") as f:
        seq_data = pickle.load(f)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model = UFCSeqModel(
        config["n_feat"], config["hidden_dim"],
        config["matchup_dim"], config.get("n_finish", 5)
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    calibrator = None
    if os.path.exists(CALIBRATOR_PATH):
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)

    embeddings = np.load(EMBED_PATH)
    with open(ID_MAP_PATH) as f:
        fighter_to_idx = json.load(f)

    matchups = pd.read_csv(MATCHUPS_PATH, parse_dates=["date"])

    fs  = pd.read_csv(RAW_FIGHT_PATH)
    fr  = pd.read_csv(RAW_RESULT_PATH)
    fed = pd.read_csv(RAW_EVENT_PATH)
    for df in [fs, fr, fed]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
    fr = fr.merge(fed[["EVENT","DATE"]], on="EVENT", how="left")
    fr["DATE"] = pd.to_datetime(fr["DATE"], format="%B %d, %Y", errors="coerce")
    ftt = pd.read_csv(RAW_FIGHTER_PATH)

    all_names = sorted(set(
        matchups["fighter_a"].tolist() + matchups["fighter_b"].tolist()
    ))
    return (seq_data, config, model, calibrator, embeddings,
            fighter_to_idx, matchups, fs, fr, ftt, all_names)

# ──────────────────────────────────────────────
# 工具函數
# ──────────────────────────────────────────────
def fuzzy_find(query, all_names):
    q = query.lower()
    for n in all_names:
        if n.lower() == q: return n
    partial = [n for n in all_names if q in n.lower()]
    if len(partial) == 1: return partial[0]
    if len(partial) > 1:
        print(f"  找到多個含 '{query}':")
        for i, n in enumerate(partial[:20]): print(f"    {i+1}. {n}")
        c = input("  請輸入編號: ").strip()
        try: return partial[int(c)-1]
        except: return None
    close = get_close_matches(query, all_names, n=5, cutoff=0.5)
    if close:
        print(f"  找不到 '{query}'，你是指：")
        for i, n in enumerate(close): print(f"    {i+1}. {n}")
        c = input("  請輸入編號 (Enter 取消): ").strip()
        if c:
            try: return close[int(c)-1]
            except: return None
    return None

def get_main_division(name, matchups):
    m = matchups[(matchups["fighter_a"]==name)|(matchups["fighter_b"]==name)]
    if len(m) == 0: return "Middleweight"
    return m["division"].mode()[0]

def get_reach_pct(name, division, seq_data):
    ftr_physical = seq_data.get("fighter_physical", {})
    div_physical = seq_data.get("div_physical", {})
    p    = ftr_physical.get(name, {})
    v    = p.get("reach")
    info = div_physical.get(division, {}).get("reach")
    if info is None or v is None: return 0.5
    sv = info.get("sorted", [])
    return sum(1 for x in sv if x < v) / len(sv) if sv else 0.5

def get_matchup_feat(fa, fb, title_fight, division,
                     physical, config, seq_data):
    DIVISION_ENC = seq_data.get("division_enc", {})
    pa = physical.get(fa, {"reach":180.0,"height":175.0,"stance_enc":0})
    pb = physical.get(fb, {"reach":180.0,"height":175.0,"stance_enc":0})
    h_mean, h_std = config["h_mean"], config["h_std"]
    r_mean, r_std = config["r_mean"], config["r_std"]
    ar = (pa["reach"]  - r_mean) / r_std
    br = (pb["reach"]  - r_mean) / r_std
    ah = (pa["height"] - h_mean) / h_std
    bh = (pb["height"] - h_mean) / h_std
    ast = pa["stance_enc"] / 2.0
    bst = pb["stance_enc"] / 2.0
    arp = get_reach_pct(fa, division, seq_data)
    brp = get_reach_pct(fb, division, seq_data)
    div_enc = DIVISION_ENC.get(division, 0.5)
    return [
        float(title_fight), ar-br, ah-bh,
        float(ast != bst),
        float((ast==0 and bst==0.5) or (ast==0.5 and bst==0)),
        arp, brp, arp-brp, div_enc,
    ]

def to_american_odds(prob):
    if prob <= 0 or prob >= 1: return "N/A"
    if prob >= 0.5: return f"-{int(round(prob/(1-prob)*100))}"
    return f"+{int(round((1-prob)/prob*100))}"

def parse_american_odds(s):
    try:
        v = float(s.strip())
        return 100/(v+100) if v > 0 else abs(v)/(abs(v)+100)
    except: return None

def value_label(edge):
    if   edge >  0.15: return "🔥 強 Value"
    elif edge >  0.08: return "✅ 有 Value"
    elif edge >  0.03: return "⚠️  輕微 Value"
    elif edge < -0.15: return "❌ 強反向（避免）"
    elif edge < -0.08: return "❌ 反向（避免）"
    else:              return "—  無明顯 Value"

# ──────────────────────────────────────────────
# MC Dropout 預測（含信心區間）
# ──────────────────────────────────────────────
def mc_predict(model, calibrator,
               a_seq, a_mask, b_seq, b_mask,
               mf_ab, mf_ba,
               n_samples=100):
    """
    推論時保持 Dropout 開著，跑 n_samples 次取分布
    同時跑正反兩向取平均消除不對稱性
    校準器套用在平均值上
    """
    model.train()  # 保持 Dropout 開著

    win_preds    = []
    finish_probs_list = []

    with torch.no_grad():
        for _ in range(n_samples):
            wp_ab, fl_ab = model(a_seq, a_mask, b_seq, b_mask, mf_ab)
            wp_ba, _     = model(b_seq, b_mask, a_seq, a_mask, mf_ba)
            p = float(((wp_ab + (1 - wp_ba)) / 2).item())
            win_preds.append(p)

            fp = torch.softmax(fl_ab, dim=1).numpy()[0]
            finish_probs_list.append(fp)

    model.eval()

    win_arr = np.array(win_preds)
    raw_mean = win_arr.mean()

    # Platt Scaling 校準
    if calibrator is not None:
        cal_mean = calibrator.predict_proba([[raw_mean]])[0][1]
    else:
        cal_mean = raw_mean

    # 信心區間（用原始分布，不校準，因為 CI 反映的是不確定性）
    ci_low  = np.percentile(win_arr, 5)
    ci_high = np.percentile(win_arr, 95)

    # 結束方式平均機率
    finish_probs = np.array(finish_probs_list).mean(axis=0)

    return {
        "win_prob":    cal_mean,
        "raw_prob":    raw_mean,
        "ci_low":      ci_low,
        "ci_high":     ci_high,
        "std":         win_arr.std(),
        "finish_probs": finish_probs,
    }

# ──────────────────────────────────────────────
# 保守 Kelly
# ──────────────────────────────────────────────
def conservative_kelly(model_result, market_prob, dec_odds, max_kelly=0.20):
    """
    用信心區間的保守估計計算 Kelly
    如果信心區間跨越 50%，不押
    """
    ci_low  = model_result["ci_low"]
    ci_high = model_result["ci_high"]
    win_prob = model_result["win_prob"]

    # 信心區間跨越 50%：不確定性太高
    if ci_low < 0.5 < ci_high:
        return 0.0, "信心區間跨越50%，不建議押注"

    # 用保守估計：如果看好A，用ci_low；如果看好B，用(1-ci_high)
    if win_prob > 0.5:
        conservative_prob = ci_low
    else:
        conservative_prob = 1 - ci_high

    conservative_edge = conservative_prob - market_prob
    if conservative_edge <= 0:
        return 0.0, "保守估計無正 Edge"

    kelly = conservative_edge / (dec_odds - 1)
    kelly = min(kelly, max_kelly)
    return kelly, None

# ──────────────────────────────────────────────
# 相似選手、Attention、近期比賽
# ──────────────────────────────────────────────
def find_similar(name, embeddings, fighter_to_idx, matchups, sequences, top_n=4):
    if name not in fighter_to_idx: return []
    idx   = fighter_to_idx[name]
    vec   = embeddings[idx]
    norm  = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    sims  = (embeddings / norm) @ (vec / (np.linalg.norm(vec)+1e-8))
    names = list(fighter_to_idx.keys())
    own_div = get_main_division(name, matchups)
    results = []
    for i in np.argsort(-sims):
        if i == idx: continue
        fname = names[i]
        fdiv  = get_main_division(fname, matchups)
        if fdiv != own_div: continue
        if sequences[fname]["n_fights"] < 3: continue
        results.append((fname, float(sims[i])))
        if len(results) >= top_n: break
    return results

def show_attention(name, attn_weights, seq_mask, fr, top_n=3):
    mask_df = fr["BOUT"].str.contains(name, na=False)
    fights  = fr[mask_df].sort_values("DATE").tail(10)
    valid   = [i for i, m in enumerate(seq_mask) if m]
    if not valid: return
    ranked  = sorted([(i, attn_weights[i]) for i in valid], key=lambda x: -x[1])
    print(f"\n  {name} — Attention 最重視:")
    for pos, w in ranked[:top_n]:
        fidx = pos - (10 - len(valid))
        if 0 <= fidx < len(fights):
            row = fights.iloc[fidx]
            bout = row["BOUT"]
            parts = [p.strip() for p in bout.split(" vs. ")]
            opp   = parts[1] if parts[0]==name else parts[0] if len(parts)==2 else "?"
            won   = (parts[0]==name and row.get("OUTCOME","")=="W/L") or \
                    (parts[1]==name and row.get("OUTCOME","")=="L/W") if len(parts)==2 else False
            method = str(row.get("METHOD","?"))[:10]
            print(f"    {str(row['DATE'])[:10]}  {'✅' if won else '❌'}  "
                  f"vs {opp:<22} {method:<12} 權重:{w:.3f}")

def recent_form(name, fs, fr, n=4):
    mask   = fr["BOUT"].str.contains(name, na=False)
    fights = fr[mask].sort_values("DATE").tail(n)
    results = []
    for _, row in fights.iterrows():
        bout  = row["BOUT"]
        parts = [p.strip() for p in bout.split(" vs. ")]
        if len(parts) != 2: continue
        opp  = parts[1] if parts[0]==name else parts[0]
        won  = (parts[0]==name and row.get("OUTCOME","")=="W/L") or \
               (parts[1]==name and row.get("OUTCOME","")=="L/W")
        event = row["EVENT"]
        fs_mask = (fs["EVENT"]==event) & (fs["BOUT"]==bout) & \
                  (fs["FIGHTER"].str.strip()==name)
        fs_rows = fs[fs_mask]

        def sum_col(col):
            try:
                vals = []
                for v in fs_rows[col]:
                    try: vals.append(float(str(v).split(" of ")[0]))
                    except: pass
                return sum(vals)
            except: return 0.0

        import re
        def parse_ctrl(val):
            if pd.isna(val): return 0.0
            m = re.match(r"(\d+):(\d+)", str(val))
            return int(m.group(1))*60 + int(m.group(2)) if m else 0.0

        dist   = sum_col("DISTANCE")
        gnd    = sum_col("GROUND")
        total  = dist + gnd + sum_col("CLINCH") + 1e-6
        kd     = sum_col("KD")
        td     = sum_col("TD")
        ctrl   = sum(parse_ctrl(v) for v in fs_rows.get("CTRL", []))

        results.append({
            "date": str(row["DATE"])[:10],
            "won":  won, "opp": opp,
            "dist": dist/total*100, "td": td,
            "kd":   kd, "ctrl": ctrl,
        })
    return results

# ──────────────────────────────────────────────
# 主預測函數
# ──────────────────────────────────────────────
def predict_matchup(name_a, name_b, title_fight=False,
                    market_odds_a=None, market_odds_b=None,
                    bankroll=None, n_mc=100):

    (seq_data, config, model, calibrator, embeddings,
     fighter_to_idx, matchups, fs, fr, ftt, all_names) = load_resources()

    sequences = seq_data["sequences"]
    physical  = seq_data["physical"]

    fa = fuzzy_find(name_a, all_names)
    fb = fuzzy_find(name_b, all_names)
    if not fa: print(f"❌ 找不到: {name_a}"); return
    if not fb: print(f"❌ 找不到: {name_b}"); return

    info_a = sequences.get(fa)
    info_b = sequences.get(fb)
    if not info_a: print(f"❌ {fa} 無序列數據"); return
    if not info_b: print(f"❌ {fb} 無序列數據"); return

    division = get_main_division(fa, matchups)

    a_seq  = torch.tensor(info_a["seq"],  dtype=torch.float32).unsqueeze(0)
    a_mask = torch.tensor(info_a["mask"], dtype=torch.bool).unsqueeze(0)
    b_seq  = torch.tensor(info_b["seq"],  dtype=torch.float32).unsqueeze(0)
    b_mask = torch.tensor(info_b["mask"], dtype=torch.bool).unsqueeze(0)

    mf_ab = torch.tensor([get_matchup_feat(
        fa, fb, int(title_fight), division, physical, config, seq_data
    )], dtype=torch.float32)
    mf_ba = torch.tensor([get_matchup_feat(
        fb, fa, int(title_fight), division, physical, config, seq_data
    )], dtype=torch.float32)

    # MC Dropout 預測
    print(f"  運行 Monte Carlo 預測（{n_mc}次）...")
    result_a = mc_predict(model, calibrator,
                          a_seq, a_mask, b_seq, b_mask,
                          mf_ab, mf_ba, n_samples=n_mc)

    win_prob_a = result_a["win_prob"]
    win_prob_b = 1.0 - win_prob_a
    ci_low_a   = result_a["ci_low"]
    ci_high_a  = result_a["ci_high"]

    # Attention
    _, attn_a = model.encode_with_attention(a_seq, a_mask)
    _, attn_b = model.encode_with_attention(b_seq, b_mask)

    sim_a  = find_similar(fa, embeddings, fighter_to_idx, matchups, sequences)
    sim_b  = find_similar(fb, embeddings, fighter_to_idx, matchups, sequences)
    form_a = recent_form(fa, fs, fr)
    form_b = recent_form(fb, fs, fr)

    a_rpct = get_reach_pct(fa, division, seq_data)
    b_rpct = get_reach_pct(fb, division, seq_data)

    # ── 輸出 ──
    SEP = "─" * 65
    print(f"\n{SEP}")
    print(f"  {fa}  vs  {fb}")
    if title_fight: print("  🏆 冠軍戰")
    print(f"  {division}  |  序列: {fa.split()[0]} {info_a['n_fights']}場 / "
          f"{fb.split()[0]} {info_b['n_fights']}場")
    print(SEP)

    # 勝率 + 信心區間
    odds_a = to_american_odds(win_prob_a)
    odds_b = to_american_odds(win_prob_b)
    bar_a  = "█" * int(win_prob_a * 30)
    bar_b  = "█" * int(win_prob_b * 30)

    print(f"\n  勝率（校準後）/ 90% 信心區間")
    print(f"  {fa:<26} {bar_a} {win_prob_a*100:.1f}%  "
          f"[{ci_low_a*100:.1f}%-{ci_high_a*100:.1f}%]  {odds_a}")
    print(f"  {fb:<26} {bar_b} {win_prob_b*100:.1f}%  "
          f"[{(1-ci_high_a)*100:.1f}%-{(1-ci_low_a)*100:.1f}%]  {odds_b}")

    # 信心標示
    ci_width = ci_high_a - ci_low_a
    if ci_width < 0.08:
        conf_label = "🟢 高信心（區間窄）"
    elif ci_width < 0.15:
        conf_label = "🟡 中等信心"
    else:
        conf_label = "🔴 低信心（區間寬，謹慎押注）"

    crosses_50 = ci_low_a < 0.5 < ci_high_a
    if crosses_50:
        conf_label = "⚠️  信心區間跨越50%，不建議押注"

    print(f"\n  信心：{conf_label}")

    # 臂展百分位
    print(f"\n  臂展在量級內排名")
    print(f"  {fa.split()[0]:<22} {a_rpct*100:.0f}th percentile")
    print(f"  {fb.split()[0]:<22} {b_rpct*100:.0f}th percentile")

    # 結束方式預測
    finish_probs = result_a["finish_probs"]
    sorted_finish = sorted(enumerate(finish_probs), key=lambda x: -x[1])
    print(f"\n  結束方式預測")
    for rank, (idx, prob) in enumerate(sorted_finish):
        marker = " ←" if rank < 2 else ""
        print(f"  {'  ' if rank >= 2 else '→ '}{FINISH_LABELS[idx]:<25} {prob*100:.1f}%{marker}")

    # Value 分析
    if market_odds_a or market_odds_b:
        print(f"\n  Value 分析")
        bet_suggestions = []

        if market_odds_a:
            mkt_prob_a = parse_american_odds(market_odds_a)
            if mkt_prob_a:
                mkt_prob_a_clean = mkt_prob_a / (mkt_prob_a + parse_american_odds(market_odds_b or to_american_odds(win_prob_b))) \
                    if market_odds_b else mkt_prob_a
                edge_a = win_prob_a - mkt_prob_a_clean
                dec_a  = (1 + float(market_odds_a)/100) if float(market_odds_a) > 0 \
                         else (1 + 100/abs(float(market_odds_a)))
                kelly_a, reason = conservative_kelly(
                    result_a, mkt_prob_a_clean, dec_a)
                bet_size_a = (bankroll * kelly_a) if bankroll else None

                print(f"  {fa.split()[0]:<22} 市場:{market_odds_a:>6}  "
                      f"模型:{odds_a:>6}  Edge:{edge_a:+.1%}  {value_label(edge_a)}")
                print(f"  {'':22} 保守Kelly: {kelly_a*100:.1f}%"
                      + (f"  建議押注: ${bet_size_a:,.0f}" if bet_size_a else "")
                      + (f"  ({reason})" if reason else ""))

        if market_odds_b:
            mkt_prob_b = parse_american_odds(market_odds_b)
            if mkt_prob_b:
                mkt_prob_b_clean = mkt_prob_b / (mkt_prob_b + parse_american_odds(market_odds_a or to_american_odds(win_prob_a))) \
                    if market_odds_a else mkt_prob_b
                edge_b = win_prob_b - mkt_prob_b_clean

                result_b = {
                    "win_prob": win_prob_b,
                    "ci_low":   1 - ci_high_a,
                    "ci_high":  1 - ci_low_a,
                    "std":      result_a["std"],
                }
                dec_b  = (1 + float(market_odds_b)/100) if float(market_odds_b) > 0 \
                         else (1 + 100/abs(float(market_odds_b)))
                kelly_b, reason = conservative_kelly(
                    result_b, mkt_prob_b_clean, dec_b)
                bet_size_b = (bankroll * kelly_b) if bankroll else None

                print(f"  {fb.split()[0]:<22} 市場:{market_odds_b:>6}  "
                      f"模型:{odds_b:>6}  Edge:{edge_b:+.1%}  {value_label(edge_b)}")
                print(f"  {'':22} 保守Kelly: {kelly_b*100:.1f}%"
                      + (f"  建議押注: ${bet_size_b:,.0f}" if bet_size_b else "")
                      + (f"  ({reason})" if reason else ""))

    # 近期比賽
    print(f"\n  近期比賽")
    print(f"  {'─'*32} {fa.split()[0]}")
    for f in form_a:
        r = "✅" if f["won"] else "❌"
        print(f"  {f['date']}  {r}  vs {f['opp']:<22} "
              f"dist={f['dist']:.0f}%  td={f['td']:.0f}  "
              f"kd={f['kd']:.0f}  ctrl={f['ctrl']:.0f}s")
    print(f"  {'─'*32} {fb.split()[0]}")
    for f in form_b:
        r = "✅" if f["won"] else "❌"
        print(f"  {f['date']}  {r}  vs {f['opp']:<22} "
              f"dist={f['dist']:.0f}%  td={f['td']:.0f}  "
              f"kd={f['kd']:.0f}  ctrl={f['ctrl']:.0f}s")

    # Attention
    show_attention(fa, attn_a[0], info_a["mask"], fr)
    show_attention(fb, attn_b[0], info_b["mask"], fr)

    # 相似選手
    if sim_a:
        s = ", ".join(f"{n} ({sc:.2f})" for n,sc in sim_a)
        print(f"\n  {fa.split()[0]} 風格近似: {s}")
    if sim_b:
        s = ", ".join(f"{n} ({sc:.2f})" for n,sc in sim_b)
        print(f"  {fb.split()[0]} 風格近似: {s}")

    print(SEP)

# ──────────────────────────────────────────────
# 互動模式
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("UFC GRU Sequence Predictor v6")
    print("市場賠率格式：美式賠率，例如 -250 或 +180")
    print("輸入 q 離開\n")

    # 詢問資金（用於計算建議押注金額）
    bk_input = input("目前資金（選填，直接 Enter 跳過）: $").strip()
    bankroll = float(bk_input) if bk_input else None

    while True:
        a = input("\n選手 A: ").strip()
        if a.lower() == "q": break
        b = input("選手 B: ").strip()
        if b.lower() == "q": break
        t = input("冠軍戰? (y/n): ").strip().lower()
        print("市場賠率（選填）")
        oa = input(f"  {a} 的市場賠率: ").strip() or None
        ob = input(f"  {b} 的市場賠率: ").strip() or None

        predict_matchup(a, b, title_fight=(t=="y"),
                        market_odds_a=oa, market_odds_b=ob,
                        bankroll=bankroll)
