"""
即時賠率分析
新增檔案: src/analyze_odds.py

從 The Odds API 抓到的 mma_odds.json
用模型分析每場比賽的 Edge

執行: python3 src/analyze_odds.py
需要: data/raw/mma_odds.json
"""

import json, pickle, torch
import torch.nn as nn
import numpy as np
import pandas as pd
from difflib import get_close_matches

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
        return (gru_out * weights.unsqueeze(-1)).sum(dim=1), weights

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
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim*2+matchup_dim, 128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, a_seq, a_mask, b_seq, b_mask, matchup):
        sa, _ = self.encoder(a_seq, a_mask)
        sb, _ = self.encoder(b_seq, b_mask)
        return self.predictor(torch.cat([sa, sb, matchup], dim=1)).squeeze(1)

# ──────────────────────────────────────────────
# 載入資源
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

# ──────────────────────────────────────────────
# 輔助函數
# ──────────────────────────────────────────────
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

def predict_prob(fa, fb, division, title_fight=0):
    if fa not in sequences or fb not in sequences:
        return None
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

def get_main_division(name):
    mq = pd.read_csv("data/processed/matchups_seq.csv")
    mask = (mq["fighter_a"]==name)|(mq["fighter_b"]==name)
    if mask.sum() == 0: return "Middleweight"
    return mq[mask]["division"].mode()[0]

def fuzzy_match(name, all_names, cutoff=0.6):
    """模糊匹配選手名字"""
    if name in all_names: return name
    # 嘗試部分匹配
    for n in all_names:
        if name.lower() in n.lower() or n.lower() in name.lower():
            return n
    # difflib 模糊匹配
    matches = get_close_matches(name, all_names, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def american_to_prob(odds):
    """美式賠率轉隱含勝率"""
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def american_to_decimal(odds):
    if odds > 0: return 1 + odds/100
    return 1 + 100/abs(odds)

def value_label(edge):
    if   edge >  0.15: return "🔥 強 Value"
    elif edge >  0.08: return "✅ 有 Value"
    elif edge >  0.03: return "⚠️  輕微 Value"
    elif edge < -0.08: return "❌ 反向"
    else:              return "—"

# ──────────────────────────────────────────────
# 載入賠率
# ──────────────────────────────────────────────
print("載入賠率資料...")
with open("data/raw/mma_odds.json") as f:
    odds_data = json.load(f)

all_seq_names = list(sequences.keys())
print(f"賠率中有 {len(odds_data)} 場比賽")
print(f"模型有 {len(all_seq_names)} 位選手的序列")

# ──────────────────────────────────────────────
# 分析每場比賽
# ──────────────────────────────────────────────
results = []
skipped = []

for fight in odds_data:
    fa_raw = fight["home_team"]
    fb_raw = fight["away_team"]
    time   = fight["commence_time"][:10]

    # 從多個博彩公司取平均賠率
    odds_a_list, odds_b_list = [], []
    for book in fight.get("bookmakers", []):
        for market in book.get("markets", []):
            if market["key"] != "h2h": continue
            for outcome in market["outcomes"]:
                if outcome["name"] == fa_raw:
                    odds_a_list.append(outcome["price"])
                elif outcome["name"] == fb_raw:
                    odds_b_list.append(outcome["price"])

    if not odds_a_list or not odds_b_list:
        skipped.append(f"{fa_raw} vs {fb_raw} (無賠率)")
        continue

    mkt_odds_a = np.mean(odds_a_list)
    mkt_odds_b = np.mean(odds_b_list)

    # 模糊匹配選手名字
    fa = fuzzy_match(fa_raw, all_seq_names)
    fb = fuzzy_match(fb_raw, all_seq_names)

    if fa is None or fb is None:
        skipped.append(f"{fa_raw} vs {fb_raw} (找不到選手)")
        continue

    # 取得量級
    division = get_main_division(fa)

    # 模型預測
    prob_a = predict_prob(fa, fb, division)
    if prob_a is None:
        skipped.append(f"{fa_raw} vs {fb_raw} (無序列數據)")
        continue

    prob_b = 1.0 - prob_a

    # 市場隱含勝率（去 vig）
    mkt_raw_a = american_to_prob(mkt_odds_a)
    mkt_raw_b = american_to_prob(mkt_odds_b)
    total = mkt_raw_a + mkt_raw_b
    mkt_a = mkt_raw_a / total
    mkt_b = mkt_raw_b / total

    edge_a = prob_a - mkt_a
    edge_b = prob_b - mkt_b

    # 信心（場次）
    n_a = sequences[fa]["n_fights"]
    n_b = sequences[fb]["n_fights"]
    confidence = "⚠️ 低" if n_a < 5 or n_b < 5 else "OK"

    results.append({
        "date":       time,
        "fa_raw":     fa_raw,
        "fb_raw":     fb_raw,
        "fa":         fa,
        "fb":         fb,
        "division":   division,
        "prob_a":     prob_a,
        "prob_b":     prob_b,
        "mkt_a":      mkt_a,
        "mkt_b":      mkt_b,
        "edge_a":     edge_a,
        "edge_b":     edge_b,
        "odds_a":     mkt_odds_a,
        "odds_b":     mkt_odds_b,
        "n_a":        n_a,
        "n_b":        n_b,
        "confidence": confidence,
    })

# ──────────────────────────────────────────────
# 輸出結果
# ──────────────────────────────────────────────
res = pd.DataFrame(results)
print(f"\n成功分析: {len(res)} 場")
print(f"跳過: {len(skipped)} 場")

# 找出最大 edge 的方向
res["best_edge"] = res[["edge_a","edge_b"]].abs().max(axis=1)
res["bet_side"]  = res.apply(
    lambda r: r["fa"] if abs(r["edge_a"]) >= abs(r["edge_b"]) else r["fb"], axis=1)
res["bet_edge"]  = res.apply(
    lambda r: r["edge_a"] if abs(r["edge_a"]) >= abs(r["edge_b"]) else r["edge_b"], axis=1)
res["bet_odds"]  = res.apply(
    lambda r: r["odds_a"] if abs(r["edge_a"]) >= abs(r["edge_b"]) else r["odds_b"], axis=1)

res = res.sort_values("best_edge", ascending=False)

print(f"\n{'='*70}")
print("所有比賽 Edge 分析（按 Edge 排序）")
print(f"{'='*70}")
print(f"{'比賽':<42} {'押注方':<18} {'模型':>6} {'市場':>6} {'Edge':>6} {'賠率':>6} {'信心':>4}")
print("-" * 90)

for _, r in res.iterrows():
    bout = f"{r['fa_raw']} vs {r['fb_raw']}"
    odds_str = f"{int(r['bet_odds']):+d}" if r['bet_odds'] > 0 else f"{int(r['bet_odds'])}"
    model_pct = r['prob_a']*100 if r['bet_side']==r['fa'] else r['prob_b']*100
    mkt_pct   = r['mkt_a']*100  if r['bet_side']==r['fa'] else r['mkt_b']*100
    vl = value_label(r['bet_edge'])
    print(f"  {bout[:40]:<40} {r['bet_side'].split()[-1]:<18} "
          f"{model_pct:>5.1f}% {mkt_pct:>5.1f}% {r['bet_edge']*100:>+5.1f}% "
          f"{odds_str:>6} {r['confidence']:>4}  {vl}")

# Value bet 總結
value_bets = res[res["bet_edge"] > 0.08]
print(f"\n{'='*70}")
print(f"Value Bet 總結（Edge > 8%）: {len(value_bets)} 場")
print(f"{'='*70}")
for _, r in value_bets.iterrows():
    bout = f"{r['fa_raw']} vs {r['fb_raw']}"
    odds_str = f"{int(r['bet_odds']):+d}" if r['bet_odds'] > 0 else f"{int(r['bet_odds'])}"
    model_pct = r['prob_a']*100 if r['bet_side']==r['fa'] else r['prob_b']*100
    mkt_pct   = r['mkt_a']*100  if r['bet_side']==r['fa'] else r['mkt_b']*100
    print(f"\n  {bout}")
    print(f"  押注: {r['bet_side']}  賠率: {odds_str}")
    print(f"  模型勝率: {model_pct:.1f}%  市場隱含: {mkt_pct:.1f}%  Edge: {r['bet_edge']*100:+.1f}%")
    print(f"  {value_label(r['bet_edge'])}")
    if r['confidence'] == '⚠️ 低':
        print(f"  ⚠️  注意：{r['fa']}({r['n_a']}場) / {r['fb']}({r['n_b']}場) 場次偏少")

print(f"\n注意：此分析基於歷史序列數據，不含傷病/備戰等賽前資訊")
print(f"建議只在 Edge > 10% 且信心 OK 的比賽考慮下注")
