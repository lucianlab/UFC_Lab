"""
UFC EV 回測 v2
使用 bestfightodds.com 的真實開盤賠率

執行: python3 src/backtest_ev.py
需要: data/raw/bfo_odds_2026.csv
"""

import pandas as pd
import numpy as np
import pickle, json, torch
import torch.nn as nn
from difflib import get_close_matches

# ──────────────────────────────────────────────
# 1. 模型定義
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
    def __init__(self, n_feat, hidden_dim, matchup_dim):
        super().__init__()
        self.encoder = FighterEncoder(n_feat, hidden_dim)
        in_dim = hidden_dim * 2 + matchup_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, a_seq, a_mask, b_seq, b_mask, matchup):
        style_a, _ = self.encoder(a_seq, a_mask)
        style_b, _ = self.encoder(b_seq, b_mask)
        return self.predictor(
            torch.cat([style_a, style_b, matchup], dim=1)
        ).squeeze(1)

# ──────────────────────────────────────────────
# 2. 載入資源
# ──────────────────────────────────────────────
print("載入模型...")
with open("data/processed/fighter_sequences.pkl", "rb") as f:
    seq_data = pickle.load(f)
with open("models/seq_model_config.json") as f:
    config = json.load(f)

sequences    = seq_data["sequences"]
physical     = seq_data["physical"]
div_physical = seq_data["div_physical"]
ftr_physical = seq_data["fighter_physical"]
DIVISION_ENC = seq_data["division_enc"]

model = UFCSeqModel(config["n_feat"], config["hidden_dim"], config["matchup_dim"])
model.load_state_dict(torch.load("models/model_seq.pt", weights_only=True))
model.eval()

all_reaches = [v["reach"]  for v in physical.values() if v.get("reach")]
all_heights = [v["height"] for v in physical.values() if v.get("height")]
h_mean, h_std = np.mean(all_heights), np.std(all_heights) + 1e-6
r_mean, r_std = np.mean(all_reaches),  np.std(all_reaches)  + 1e-6

all_seq_names = list(sequences.keys())

# ──────────────────────────────────────────────
# 3. 輔助函數
# ──────────────────────────────────────────────
def fuzzy_match(name):
    if name in all_seq_names: return name
    clean = name.replace("'","").lower()
    for n in all_seq_names:
        if n.replace("'","").lower() == clean: return n
    matches = get_close_matches(name, all_seq_names, n=1, cutoff=0.75)
    return matches[0] if matches else None

def get_reach_pct(name, division):
    p    = ftr_physical.get(name, {})
    v    = p.get("reach")
    info = div_physical.get(division, {}).get("reach")
    if info is None or v is None: return 0.5
    sv = info.get("sorted", [])
    return sum(1 for x in sv if x < v) / len(sv) if sv else 0.5

def get_matchup_feat(fa, fb, title_fight, division):
    pa = physical.get(fa, {"reach":180.0,"height":175.0,"stance_enc":0})
    pb = physical.get(fb, {"reach":180.0,"height":175.0,"stance_enc":0})
    ar = (pa["reach"]  - r_mean) / r_std
    br = (pb["reach"]  - r_mean) / r_std
    ah = (pa["height"] - h_mean) / h_std
    bh = (pb["height"] - h_mean) / h_std
    ast = pa["stance_enc"] / 2.0
    bst = pb["stance_enc"] / 2.0
    arp = get_reach_pct(fa, division)
    brp = get_reach_pct(fb, division)
    return [
        float(title_fight), ar-br, ah-bh,
        float(ast != bst),
        float((ast==0 and bst==0.5) or (ast==0.5 and bst==0)),
        arp, brp, arp-brp, DIVISION_ENC.get(division, 0.5),
    ]

def get_main_division(name):
    mq = pd.read_csv("data/processed/matchups_seq.csv")
    mask = (mq["fighter_a"]==name)|(mq["fighter_b"]==name)
    if mask.sum() == 0: return "Middleweight"
    return mq[mask]["division"].mode()[0]

def predict_prob(fa, fb, division, title_fight=0):
    a = sequences[fa]; b = sequences[fb]
    a_seq  = torch.tensor(a["seq"],  dtype=torch.float32).unsqueeze(0)
    a_mask = torch.tensor(a["mask"], dtype=torch.bool).unsqueeze(0)
    b_seq  = torch.tensor(b["seq"],  dtype=torch.float32).unsqueeze(0)
    b_mask = torch.tensor(b["mask"], dtype=torch.bool).unsqueeze(0)
    mf_ab = torch.tensor([get_matchup_feat(fa, fb, title_fight, division)],
                         dtype=torch.float32)
    mf_ba = torch.tensor([get_matchup_feat(fb, fa, title_fight, division)],
                         dtype=torch.float32)
    with torch.no_grad():
        p_ab = float(model(a_seq, a_mask, b_seq, b_mask, mf_ab).item())
        p_ba = float(model(b_seq, b_mask, a_seq, a_mask, mf_ba).item())
    return (p_ab + (1 - p_ba)) / 2

def american_to_prob(odds):
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def remove_vig(p1, p2):
    total = p1 + p2
    return p1/total, p2/total

# ──────────────────────────────────────────────
# 4. 載入賠率和比賽結果
# ──────────────────────────────────────────────
print("載入賠率...")
odds_df = pd.read_csv("data/raw/bfo_odds_2026.csv")
print(f"賠率資料: {len(odds_df)} 筆")

mq = pd.read_csv("data/processed/matchups_seq.csv", parse_dates=["date"])
mq_2026 = mq[mq["date"].dt.year == 2026].copy()
print(f"2026年比賽: {len(mq_2026)} 場")

# ──────────────────────────────────────────────
# 5. 主回測
# ──────────────────────────────────────────────
print("\n開始回測...\n")

results = []
skipped_no_odds  = 0
skipped_no_seq   = 0
skipped_no_match = 0

for _, mrow in mq_2026.iterrows():
    fa_seq = mrow["fighter_a"]
    fb_seq = mrow["fighter_b"]
    actual_a_wins = bool(mrow["winner_is_a"])
    div = mrow["division"]
    tf  = mrow["title_fight"]

    # 在賠率資料裡找這場（雙向）
    fa_m = fuzzy_match(fa_seq)
    fb_m = fuzzy_match(fb_seq)
    if not fa_m or not fb_m:
        skipped_no_seq += 1
        continue

    # 找賠率（雙向）
    found = odds_df[
        (odds_df["fighter_1"]==fa_seq) & (odds_df["fighter_2"]==fb_seq)
    ]
    flipped = False
    if len(found) == 0:
        found = odds_df[
            (odds_df["fighter_1"]==fb_seq) & (odds_df["fighter_2"]==fa_seq)
        ]
        flipped = True
    # fuzzy match 賠率
    if len(found) == 0:
        found = odds_df[
            (odds_df["fighter_1"]==fa_m) & (odds_df["fighter_2"]==fb_m)
        ]
    if len(found) == 0:
        found = odds_df[
            (odds_df["fighter_1"]==fb_m) & (odds_df["fighter_2"]==fa_m)
        ]
        flipped = True
    if len(found) == 0:
        skipped_no_odds += 1
        continue

    o1 = float(found["odds_1"].mean())
    o2 = float(found["odds_2"].mean())
    if flipped: o1, o2 = o2, o1

    if abs(o1) < 100 or abs(o2) < 100:
        continue

    # 模型預測
    try:
        prob_a = predict_prob(fa_m, fb_m, div, tf)
    except:
        skipped_no_seq += 1
        continue

    prob_b = 1.0 - prob_a

    # 市場隱含勝率
    mkt_raw_a = american_to_prob(o1)
    mkt_raw_b = american_to_prob(o2)
    mkt_a, mkt_b = remove_vig(mkt_raw_a, mkt_raw_b)

    edge_a = prob_a - mkt_a
    edge_b = prob_b - mkt_b

    def to_us(o):
        if o > 0: return f"+{int(o)}"
        return str(int(o))

    results.append({
        "fighter_a":     fa_seq,
        "fighter_b":     fb_seq,
        "division":      div,
        "prob_a":        prob_a,
        "mkt_a":         mkt_a,
        "edge_a":        edge_a,
        "edge_b":        edge_b,
        "odds_a":        o1,
        "odds_b":        o2,
        "odds_a_us":     to_us(o1),
        "odds_b_us":     to_us(o2),
        "actual_a_wins": actual_a_wins,
        "n_books":       int(found["n_books"].mean()),
    })

res = pd.DataFrame(results)
print(f"成功分析: {len(res)} 場")
print(f"無賠率:   {skipped_no_odds} 場")
print(f"無序列:   {skipped_no_seq} 場")

# ──────────────────────────────────────────────
# 6. EV 分析
# ──────────────────────────────────────────────
def analyze(df, threshold, label):
    bets = []
    for _, r in df.iterrows():
        if abs(r["edge_a"]) >= abs(r["edge_b"]):
            edge     = r["edge_a"]
            bet_name = r["fighter_a"]
            bet_odds = r["odds_a"]
            bet_us   = r["odds_a_us"]
            bet_wins = r["actual_a_wins"]
            model_p  = r["prob_a"]
            mkt_p    = r["mkt_a"]
        else:
            edge     = r["edge_b"]
            bet_name = r["fighter_b"]
            bet_odds = r["odds_b"]
            bet_us   = r["odds_b_us"]
            bet_wins = not r["actual_a_wins"]
            model_p  = 1 - r["prob_a"]
            mkt_p    = 1 - r["mkt_a"]

        if edge < threshold: continue

        dec_odds = (1 + bet_odds/100) if bet_odds > 0 else (1 + 100/abs(bet_odds))
        kelly = min(max(0, edge / (dec_odds - 1)), 0.20)

        bets.append({
            "bout":      f"{r['fighter_a']} vs {r['fighter_b']}",
            "bet_on":    bet_name,
            "model_p":   model_p,
            "mkt_p":     mkt_p,
            "edge":      edge,
            "odds_us":   bet_us,
            "dec_odds":  dec_odds,
            "kelly":     kelly,
            "bet_wins":  bet_wins,
        })

    if not bets:
        print(f"\n{label}: 沒有符合條件的比賽")
        return

    bdf = pd.DataFrame(bets)
    wins  = bdf["bet_wins"].sum()
    total = len(bdf)

    flat_pnl = sum(
        (r["dec_odds"]-1) if r["bet_wins"] else -1
        for _, r in bdf.iterrows()
    )

    bankroll = 1.0
    kelly_pnl = 0
    for _, r in bdf.iterrows():
        bet_size = bankroll * r["kelly"]
        if r["bet_wins"]:
            bankroll  += bet_size * (r["dec_odds"]-1)
            kelly_pnl += bet_size * (r["dec_odds"]-1)
        else:
            bankroll  -= bet_size
            kelly_pnl -= bet_size

    print(f"\n{'='*65}")
    print(f"{label}  (共 {total} 場)")
    print(f"{'='*65}")
    print(f"獲勝: {wins}/{total} = {wins/total*100:.1f}%")
    print(f"平押損益:  {flat_pnl:+.2f} 單位  ROI: {flat_pnl/total*100:+.1f}%")
    print(f"Kelly損益: {kelly_pnl:+.4f}  最終資金: {bankroll:.4f}x")
    print()
    print(f"{'比賽':<38} {'押注方':<20} {'模型':>6} {'市場':>6} {'Edge':>6} {'賠率':>7} {'結果':>4}")
    print("-" * 90)
    for _, r in bdf.sort_values("edge", ascending=False).iterrows():
        result = "✅" if r["bet_wins"] else "❌"
        print(f"  {r['bout'][:36]:<36} {r['bet_on'].split()[-1]:<20} "
              f"{r['model_p']*100:>5.1f}% {r['mkt_p']*100:>5.1f}% "
              f"{r['edge']*100:>+5.1f}% {r['odds_us']:>6} {result}")

analyze(res, 0.05, "Edge > 5%")
analyze(res, 0.10, "Edge > 10%")
analyze(res, 0.15, "Edge > 15%")

# ──────────────────────────────────────────────
# 7. 整體統計
# ──────────────────────────────────────────────
print(f"\n{'='*65}")
print("整體統計")
print(f"{'='*65}")
correct = sum((r["prob_a"]>0.5)==r["actual_a_wins"] for _,r in res.iterrows())
print(f"模型勝負準確率: {correct}/{len(res)} = {correct/len(res)*100:.1f}%")

# 模型看好的 underdog
ud = res[res["edge_a"] > 0.05]
if len(ud) > 0:
    ud_wins = ud["actual_a_wins"].sum()
    print(f"模型看好的 underdog（A方 edge>5%）: {ud_wins}/{len(ud)} = {ud_wins/len(ud)*100:.1f}%")

ud_b = res[res["edge_b"] > 0.05]
if len(ud_b) > 0:
    ud_wins_b = (~ud_b["actual_a_wins"]).sum()
    print(f"模型看好的 underdog（B方 edge>5%）: {ud_wins_b}/{len(ud_b)} = {ud_wins_b/len(ud_b)*100:.1f}%")

res.to_csv("data/processed/backtest_2026.csv", index=False)
print(f"\n結果儲存至 data/processed/backtest_2026.csv")
