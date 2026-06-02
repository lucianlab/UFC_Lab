"""
UFC 2026 年完整回測
src/backtest_2026.py

兩大目標：
  1. 模型表現：2026 年所有比賽的勝負準確率
  2. 賺錢表現：結合 bestfightodds 開盤賠率做 EV 分析

需要：
  models/model_seq.pt
  models/platt_calibrator.pkl
  data/processed/fighter_sequences.pkl（截斷至 2025 底）
  data/raw/ufc_fight_results.csv（含 2026 年比賽）
  data/raw/bfo_odds_2026.csv（bestfightodds 開盤賠率）

執行: python3 src/backtest_2026.py
"""

import pandas as pd, numpy as np
import pickle, json, torch, torch.nn as nn
from difflib import get_close_matches

# ─────────────────────────────────────────────
# 1. 模型定義
# ─────────────────────────────────────────────
class Attn(nn.Module):
    def __init__(self,h):
        super().__init__(); self.attn=nn.Linear(h,1)
    def forward(self,x,mask):
        s=self.attn(x).squeeze(-1).masked_fill(~mask,-1e9)
        w=torch.softmax(s,dim=1)
        return (x*w.unsqueeze(-1)).sum(dim=1),w

class Enc(nn.Module):
    def __init__(self,n,h):
        super().__init__()
        self.gru=nn.GRU(n,h,num_layers=2,batch_first=True,dropout=0.2)
        self.attention=Attn(h); self.norm=nn.LayerNorm(h)
    def forward(self,x,mask):
        o,_=self.gru(x); c,w=self.attention(o,mask)
        return self.norm(c),w

class Model(nn.Module):
    def __init__(self,n,h,m):
        super().__init__()
        self.encoder=Enc(n,h)
        self.predictor=nn.Sequential(
            nn.Linear(h*2+m,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,1),nn.Sigmoid())
    def forward(self,as_,am,bs,bm,mf):
        sa,_=self.encoder(as_,am); sb,_=self.encoder(bs,bm)
        return self.predictor(torch.cat([sa,sb,mf],dim=1)).squeeze(1)

# ─────────────────────────────────────────────
# 2. 載入
# ─────────────────────────────────────────────
print("載入模型和資料...")
with open("data/processed/fighter_sequences.pkl","rb") as f:
    sd = pickle.load(f)
with open("models/seq_model_config.json") as f:
    cfg = json.load(f)
with open("models/platt_calibrator.pkl","rb") as f:
    cal = pickle.load(f)

sequences = sd["sequences"]; physical = sd["physical"]
div_phy   = sd["div_physical"]; ftr_phy = sd["fighter_physical"]
DIV_ENC   = sd["division_enc"]
SEQ_LEN   = sd["seq_len"]; N_FEAT = len(sd["seq_features"])

model = Model(cfg["n_feat"],cfg["hidden_dim"],cfg["matchup_dim"])
model.load_state_dict(torch.load("models/model_seq.pt",weights_only=True))
model.eval()

print(f"序列截斷日期: {sd.get('cutoff_date','2025-12-31')}")
print(f"序列選手數: {len(sequences)}")

# ─────────────────────────────────────────────
# 3. 輔助函數
# ─────────────────────────────────────────────
rm,rs = cfg["r_mean"],cfg["r_std"]
hm,hs = cfg["h_mean"],cfg["h_std"]

all_seq_names = list(sequences.keys())

def fuzzy(name):
    if name in all_seq_names: return name
    clean = name.replace("'","").lower()
    for n in all_seq_names:
        if n.replace("'","").lower()==clean: return n
    m = get_close_matches(name,all_seq_names,n=1,cutoff=0.75)
    return m[0] if m else None

def rpct(name,div):
    p=ftr_phy.get(name,{}); v=p.get("reach")
    info=div_phy.get(div,{}).get("reach")
    if not info or not v: return 0.5
    sv=info["sorted"]
    return sum(1 for x in sv if x<v)/len(sv)

def get_mfeat(a,b,tf,div):
    pa=physical.get(a,{"reach":180.,"height":175.,"stance_enc":0})
    pb=physical.get(b,{"reach":180.,"height":175.,"stance_enc":0})
    ar=(pa["reach"]-rm)/rs; br=(pb["reach"]-rm)/rs
    ah=(pa["height"]-hm)/hs; bh=(pb["height"]-hm)/hs
    ast=pa["stance_enc"]/2.; bst=pb["stance_enc"]/2.
    arp=rpct(a,div); brp=rpct(b,div)
    return [float(tf),ar-br,ah-bh,float(ast!=bst),
            float((ast==0 and bst==.5)or(ast==.5 and bst==0)),
            arp,brp,arp-brp,DIV_ENC.get(div,.5)]

def predict(fa,fb,tf,div):
    a=sequences[fa]; b=sequences[fb]
    as_=torch.tensor(a["seq"],dtype=torch.float32).unsqueeze(0)
    am =torch.tensor(a["mask"],dtype=torch.bool).unsqueeze(0)
    bs =torch.tensor(b["seq"],dtype=torch.float32).unsqueeze(0)
    bm =torch.tensor(b["mask"],dtype=torch.bool).unsqueeze(0)
    mf_ab=torch.tensor([get_mfeat(fa,fb,tf,div)],dtype=torch.float32)
    mf_ba=torch.tensor([get_mfeat(fb,fa,tf,div)],dtype=torch.float32)
    with torch.no_grad():
        p_ab=float(model(as_,am,bs,bm,mf_ab).item())
        p_ba=float(model(bs,bm,as_,am,mf_ba).item())
    raw = (p_ab+(1-p_ba))/2
    cal_p = raw  # Platt Scaling 暫時停用
    return cal_p

def us_to_prob(odds):
    v=float(odds)
    return 100/(v+100) if v>0 else abs(v)/(abs(v)+100)

def us_to_dec(odds):
    v=float(odds)
    return 1+v/100 if v>0 else 1+100/abs(v)

def remove_vig(p1,p2):
    t=p1+p2; return p1/t, p2/t

# ─────────────────────────────────────────────
# 4. 載入 2026 年比賽結果
# ─────────────────────────────────────────────
print("\n載入 2026 年比賽資料...")
fr = pd.read_csv("data/raw/ufc_fight_results.csv")
fed = pd.read_csv("data/raw/ufc_event_details.csv")
for df in [fr,fed]:
    for col in df.columns:
        if df[col].dtype==object: df[col]=df[col].str.strip()

fr = fr.merge(fed[["EVENT","DATE"]],on="EVENT",how="left")
fr["DATE"] = pd.to_datetime(fr["DATE"],format="%B %d, %Y",errors="coerce")

MAIN_DIV = [
    "Heavyweight","Light Heavyweight","Middleweight","Welterweight",
    "Lightweight","Featherweight","Bantamweight","Flyweight",
    "Women's Strawweight","Women's Flyweight",
    "Women's Bantamweight","Women's Featherweight",
]
fr["DIV"] = fr["WEIGHTCLASS"]\
    .str.replace("UFC ","",regex=False)\
    .str.replace(" Title","",regex=False)\
    .str.replace(" Bout","",regex=False)\
    .str.strip()

fr_2026 = fr[
    (fr["DATE"] >= "2026-01-01") &
    (fr["DIV"].isin(MAIN_DIV))
].copy().sort_values("DATE").reset_index(drop=True)

print(f"2026 年比賽: {len(fr_2026)} 場")

# ─────────────────────────────────────────────
# 5. 載入賠率
# ─────────────────────────────────────────────
has_odds = False
try:
    odds_df = pd.read_csv("data/raw/bfo_odds_2026.csv")
    print(f"賠率資料: {len(odds_df)} 筆（bestfightodds 開盤賠率）")
    has_odds = True
except:
    print("⚠️  找不到 data/raw/bfo_odds_2026.csv，跳過 EV 分析")

# ─────────────────────────────────────────────
# 6. 主回測
# ─────────────────────────────────────────────
print("\n開始回測...")

results = []
skipped_no_seq  = 0
skipped_no_odds = 0

for _, row in fr_2026.iterrows():
    bout = str(row["BOUT"])
    parts = [p.strip() for p in bout.split(" vs. ")]
    if len(parts)!=2: continue

    fa_raw, fb_raw = parts
    oc = str(row.get("OUTCOME","")).strip()
    if   oc=="W/L": actual_a_wins=True
    elif oc=="L/W": actual_a_wins=False
    else: continue

    div = row["DIV"]
    tf  = 1 if "title" in str(row.get("WEIGHTCLASS","")).lower() else 0

    # 模糊匹配序列
    fa = fuzzy(fa_raw); fb = fuzzy(fb_raw)
    if not fa or not fb:
        skipped_no_seq+=1; continue

    # 預測
    try:
        prob_a = predict(fa,fb,tf,div)
    except:
        skipped_no_seq+=1; continue

    prob_b = 1-prob_a

    # 找賠率
    odds_a = odds_b = None
    if has_odds:
        found = odds_df[
            (odds_df["fighter_1"]==fa_raw)&(odds_df["fighter_2"]==fb_raw)
        ]
        flipped=False
        if len(found)==0:
            found = odds_df[
                (odds_df["fighter_1"]==fb_raw)&(odds_df["fighter_2"]==fa_raw)
            ]
            flipped=True
        if len(found)==0:
            found = odds_df[
                (odds_df["fighter_1"]==fa)&(odds_df["fighter_2"]==fb)
            ]
        if len(found)==0:
            found = odds_df[
                (odds_df["fighter_1"]==fb)&(odds_df["fighter_2"]==fa)
            ]
            flipped=True

        if len(found)>0:
            o1=float(found["odds_1"].mean())
            o2=float(found["odds_2"].mean())
            if flipped: o1,o2=o2,o1
            odds_a=o1; odds_b=o2

    # EV 計算
    edge_a=edge_b=None; mkt_a=mkt_b=None
    if odds_a and odds_b and abs(odds_a)>=100 and abs(odds_b)>=100:
        mkt_raw_a=us_to_prob(odds_a); mkt_raw_b=us_to_prob(odds_b)
        mkt_a,mkt_b=remove_vig(mkt_raw_a,mkt_raw_b)
        edge_a=prob_a-mkt_a; edge_b=prob_b-mkt_b

    n_a = sequences[fa]["n_fights"]
    n_b = sequences[fb]["n_fights"]

    results.append({
        "date":         str(row["DATE"])[:10],
        "fighter_a":    fa_raw,
        "fighter_b":    fb_raw,
        "fa_matched":   fa,
        "fb_matched":   fb,
        "division":     div,
        "prob_a":       prob_a,
        "prob_b":       prob_b,
        "actual_a_wins": actual_a_wins,
        "odds_a":       odds_a,
        "odds_b":       odds_b,
        "mkt_a":        mkt_a,
        "mkt_b":        mkt_b,
        "edge_a":       edge_a,
        "edge_b":       edge_b,
        "n_a":          n_a,
        "n_b":          n_b,
    })

res = pd.DataFrame(results)
print(f"成功預測: {len(res)} 場")
print(f"跳過（找不到序列）: {skipped_no_seq} 場")

# ─────────────────────────────────────────────
# 7. 目標一：模型表現
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("目標一：模型表現")
print(f"{'='*65}")

correct = (res["prob_a"]>0.5)==res["actual_a_wins"]
print(f"整體勝負準確率: {correct.sum()}/{len(res)} = {correct.mean()*100:.1f}%")

# 過濾場次不足的
for min_f in [3,5,7,10]:
    mask = (res["n_a"]>=min_f)&(res["n_b"]>=min_f)
    sub = res[mask]
    if len(sub)==0: continue
    c = ((sub["prob_a"]>0.5)==sub["actual_a_wins"]).sum()
    print(f"雙方都 >={min_f} 場: {c}/{len(sub)} = {c/len(sub)*100:.1f}%")

# 模型信心分組
print(f"\n模型信心分組準確率:")
bins = [(0.6,0.7,"60-70%"),(0.7,0.8,"70-80%"),(0.8,0.9,"80-90%"),(0.9,1.0,"90%+")]
for lo,hi,label in bins:
    mask = (res["prob_a"].clip(1-hi,1-lo+0.001)!=res["prob_a"])&False  # placeholder
    # 取 max(prob_a, prob_b) 在這個區間的比賽
    max_p = res[["prob_a","prob_b"]].max(axis=1)
    mask = (max_p>=lo)&(max_p<hi)
    sub = res[mask]
    if len(sub)==0: continue
    # 模型預測是 prob_a 還是 prob_b 更高
    pred_a = sub["prob_a"]>=sub["prob_b"]
    act_a  = sub["actual_a_wins"]
    c = ((pred_a&act_a)|(~pred_a&~act_a)).sum()
    print(f"  信心 {label}: {c}/{len(sub)} = {c/len(sub)*100:.1f}%")

# ─────────────────────────────────────────────
# 8. 目標二：賺錢表現
# ─────────────────────────────────────────────
if not has_odds or res["edge_a"].isna().all():
    print(f"\n⚠️  無賠率數據，跳過 EV 分析")
else:
    odds_res = res.dropna(subset=["edge_a","edge_b"]).copy()
    print(f"\n{'='*65}")
    print(f"目標二：賺錢表現（有賠率的 {len(odds_res)} 場）")
    print(f"{'='*65}")

    # 每場選擇最大 edge 方向
    def get_best(r):
        if abs(r["edge_a"])>=abs(r["edge_b"]):
            return r["edge_a"],r["odds_a"],r["actual_a_wins"],r["mkt_a"]
        else:
            return r["edge_b"],r["odds_b"],not r["actual_a_wins"],r["mkt_b"]

    for threshold,label in [(0.05,"Edge>5%"),(0.10,"Edge>10%"),(0.15,"Edge>15%")]:
        bets=[]
        for _,r in odds_res.iterrows():
            edge,odds,wins,mkt = get_best(r)
            if edge<=threshold: continue
            dec = us_to_dec(odds)
            kelly = min(max(0,edge/(dec-1)),0.20)
            bets.append({"edge":edge,"odds":odds,"dec":dec,
                        "kelly":kelly,"wins":wins,"mkt":mkt,
                        "bout":f"{r['fighter_a']} vs {r['fighter_b']}"})

        if not bets:
            print(f"\n{label}: 無符合條件的比賽")
            continue

        bdf=pd.DataFrame(bets)
        win_n=bdf["wins"].sum(); total=len(bdf)
        flat_pnl=sum((r["dec"]-1) if r["wins"] else -1 for _,r in bdf.iterrows())

        # Kelly 模擬（$10,000 起始）
        bankroll=10000.
        for _,r in bdf.iterrows():
            bet=bankroll*r["kelly"]
            if r["wins"]: bankroll+=bet*(r["dec"]-1)
            else: bankroll-=bet

        print(f"\n{label} ({total} 場)")
        print(f"  獲勝: {win_n}/{total} = {win_n/total*100:.1f}%")
        print(f"  平押損益: {flat_pnl:+.2f} 單位  ROI: {flat_pnl/total*100:+.1f}%")
        print(f"  Kelly $10,000 → ${bankroll:,.0f}  ({(bankroll/10000-1)*100:+.1f}%)")

        print(f"\n  {'比賽':<40} {'Edge':>6} {'賠率':>7} {'結果':>4}")
        print(f"  {'-'*60}")
        for _,r in bdf.sort_values("edge",ascending=False).iterrows():
            us = f"{int(r['odds']):+d}" if r["odds"]>0 else str(int(r["odds"]))
            print(f"  {r['bout'][:38]:<38} {r['edge']*100:>+5.1f}% {us:>6} {'✅' if r['wins'] else '❌'}")

# ─────────────────────────────────────────────
# 9. 場次過濾後的 EV
# ─────────────────────────────────────────────
if has_odds and not res["edge_a"].isna().all():
    odds_res_f = odds_res[(odds_res["n_a"]>=5)&(odds_res["n_b"]>=5)]
    if len(odds_res_f)>0:
        print(f"\n{'='*65}")
        print(f"雙方都 >=5 場的 EV 分析（{len(odds_res_f)} 場）")
        print(f"{'='*65}")
        bets=[]
        for _,r in odds_res_f.iterrows():
            edge,odds,wins,mkt=get_best(r)
            if edge<=0.05: continue
            dec=us_to_dec(odds)
            kelly=min(max(0,edge/(dec-1)),0.20)
            bets.append({"edge":edge,"odds":odds,"dec":dec,
                        "kelly":kelly,"wins":wins})
        if bets:
            bdf=pd.DataFrame(bets)
            wn=bdf["wins"].sum(); tn=len(bdf)
            fp=sum((r["dec"]-1) if r["wins"] else -1 for _,r in bdf.iterrows())
            bk=10000.
            for _,r in bdf.iterrows():
                bt=bk*r["kelly"]
                if r["wins"]: bk+=bt*(r["dec"]-1)
                else: bk-=bt
            print(f"Edge>5%, 雙方>=5場: {wn}/{tn}={wn/tn*100:.1f}%, "
                  f"ROI={fp/tn*100:+.1f}%, Kelly $10k→${bk:,.0f}")

# ─────────────────────────────────────────────
# 10. 儲存完整結果
# ─────────────────────────────────────────────
res.to_csv("data/processed/backtest_2026_full.csv",index=False)
print(f"\n完整結果儲存至 data/processed/backtest_2026_full.csv")
print(f"\n✅ 2026 年回測完成")
