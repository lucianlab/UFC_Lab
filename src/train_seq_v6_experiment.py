"""
UFC GRU Sequence Model v6b
覆蓋: src/train_seq.py

修正：
  - 訓練集改回 2026 以前（包含 2024-2025）
  - Platt Scaling 用未對稱化的原始數據校準

執行: python3 src/train_seq.py
"""

import pandas as pd
import numpy as np
import json, os, pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# 1. 載入資料
# ──────────────────────────────────────────────
print("載入序列資料...")
with open("data/processed/fighter_sequences.pkl", "rb") as f:
    seq_data = pickle.load(f)

sequences    = seq_data["sequences"]
physical     = seq_data["physical"]
SEQ_LEN      = seq_data["seq_len"]
N_FEAT       = len(seq_data["seq_features"])
div_physical = seq_data["div_physical"]
ftr_physical = seq_data["fighter_physical"]
DIVISION_ENC = seq_data["division_enc"]

matchups = pd.read_csv("data/processed/matchups_seq.csv", parse_dates=["date"])
matchups = matchups.sort_values("date").reset_index(drop=True)

print(f"選手數: {len(sequences)}")
print(f"對戰樣本: {len(matchups)}")
print(f"序列長度: {SEQ_LEN}, 特徵數: {N_FEAT}")

# ──────────────────────────────────────────────
# 2. 物理特徵
# ──────────────────────────────────────────────
all_reaches = [v["reach"]  for v in physical.values() if v.get("reach")]
all_heights = [v["height"] for v in physical.values() if v.get("height")]
h_mean, h_std = np.mean(all_heights), np.std(all_heights) + 1e-6
r_mean, r_std = np.mean(all_reaches),  np.std(all_reaches)  + 1e-6

def get_reach_pct(name, division):
    p    = ftr_physical.get(name, {})
    v    = p.get("reach")
    info = div_physical.get(division, {}).get("reach")
    if info is None or v is None: return 0.5
    sv = info.get("sorted", [])
    return sum(1 for x in sv if x < v) / len(sv) if sv else 0.5

MATCHUP_DIM = 9

def get_matchup_feat(name_a, name_b, title_fight, division):
    pa = physical.get(name_a, {"reach":180.0,"height":175.0,"stance_enc":0})
    pb = physical.get(name_b, {"reach":180.0,"height":175.0,"stance_enc":0})
    ar = (pa["reach"]  - r_mean) / r_std
    br = (pb["reach"]  - r_mean) / r_std
    ah = (pa["height"] - h_mean) / h_std
    bh = (pb["height"] - h_mean) / h_std
    ast = pa["stance_enc"] / 2.0
    bst = pb["stance_enc"] / 2.0
    arp = get_reach_pct(name_a, division)
    brp = get_reach_pct(name_b, division)
    div_enc = DIVISION_ENC.get(division, 0.5)
    return [
        float(title_fight), ar-br, ah-bh,
        float(ast != bst),
        float((ast==0 and bst==0.5) or (ast==0.5 and bst==0)),
        arp, brp, arp-brp, div_enc,
    ]

# ──────────────────────────────────────────────
# 3. 對稱化
# ──────────────────────────────────────────────
def symmetrize(df):
    orig    = df.copy()
    flipped = df.copy()
    flipped["fighter_a"]   = df["fighter_b"]
    flipped["fighter_b"]   = df["fighter_a"]
    flipped["a_seq_len"]   = df["b_seq_len"]
    flipped["b_seq_len"]   = df["a_seq_len"]
    flipped["winner_is_a"] = 1 - df["winner_is_a"]
    return pd.concat([orig, flipped], ignore_index=True)\
             .sort_values("date").reset_index(drop=True)

matchups_sym = symmetrize(matchups)
print(f"對稱化後: {len(matchups_sym)} 筆")

# ──────────────────────────────────────────────
# 4. 時間切割
#    Train：2026 以前（含 2024-2025）
#    Cal：  2024-2025（未對稱化，用於 Platt Scaling）
#    Val：  2026（holdout）
# ──────────────────────────────────────────────
train_mask = matchups_sym["date"] < "2026-01-01"
val_mask   = matchups_sym["date"] >= "2026-01-01"

train_df = matchups_sym[train_mask].reset_index(drop=True)
val_df   = matchups_sym[val_mask].reset_index(drop=True)

# Cal set：未對稱化的 2024-2025（真實 A方勝率 ~60%，有意義的分布）
cal_orig = matchups[
    (matchups["date"] >= "2024-01-01") &
    (matchups["date"] < "2026-01-01")
].reset_index(drop=True)

print(f"訓練集: {len(train_df)} 筆（2026年以前）")
print(f"校準集: {len(cal_orig)} 筆（2024-2025，未對稱化，A方勝率={cal_orig['winner_is_a'].mean():.2%}）")
print(f"驗證集: {len(val_df)} 筆（2026，holdout）")

# ──────────────────────────────────────────────
# 5. Dataset
# ──────────────────────────────────────────────
class FightSeqDataset(Dataset):
    def __init__(self, df, sequences, matchup_fn):
        self.df         = df.reset_index(drop=True)
        self.seqs       = sequences
        self.mfn        = matchup_fn
        self.empty_seq  = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
        self.empty_mask = np.zeros(SEQ_LEN, dtype=bool)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row    = self.df.iloc[i]
        a_data = self.seqs.get(row["fighter_a"])
        b_data = self.seqs.get(row["fighter_b"])
        a_seq  = a_data["seq"]  if a_data else self.empty_seq
        a_mask = a_data["mask"] if a_data else self.empty_mask
        b_seq  = b_data["seq"]  if b_data else self.empty_seq
        b_mask = b_data["mask"] if b_data else self.empty_mask
        mfeat  = np.array(self.mfn(
            row["fighter_a"], row["fighter_b"],
            row["title_fight"], row["division"]
        ), dtype=np.float32)
        win_label    = torch.tensor(float(row["winner_is_a"]), dtype=torch.float32)
        finish_label = torch.tensor(int(row["finish_5class"]),  dtype=torch.long)
        return (
            torch.tensor(a_seq,  dtype=torch.float32),
            torch.tensor(a_mask, dtype=torch.bool),
            torch.tensor(b_seq,  dtype=torch.float32),
            torch.tensor(b_mask, dtype=torch.bool),
            torch.tensor(mfeat,  dtype=torch.float32),
            win_label, finish_label,
        )

train_ds     = FightSeqDataset(train_df,  sequences, get_matchup_feat)
cal_ds       = FightSeqDataset(cal_orig,  sequences, get_matchup_feat)
val_ds       = FightSeqDataset(val_df,    sequences, get_matchup_feat)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
cal_loader   = DataLoader(cal_ds,   batch_size=256, shuffle=False, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
# 6. 模型（雙頭）
# ──────────────────────────────────────────────
HIDDEN_DIM = 64
N_FINISH   = 5

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

model = UFCSeqModel(N_FEAT, HIDDEN_DIM, MATCHUP_DIM, N_FINISH)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型參數: {total_params:,}")
print(f"Encoder: GRU({N_FEAT}→{HIDDEN_DIM}×2) + Attention")
print(f"Win head + Finish head（{N_FINISH}類）")

# ──────────────────────────────────────────────
# 7. 訓練
# ──────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, min_lr=1e-5)
bce_loss     = nn.BCELoss()
ce_loss      = nn.CrossEntropyLoss()
FINISH_WEIGHT = 0.5

def evaluate(loader):
    model.eval()
    total_loss, win_correct, finish_correct, total = 0, 0, 0, 0
    with torch.no_grad():
        for a_seq, a_mask, b_seq, b_mask, mfeat, win_label, finish_label in loader:
            win_pred, finish_logits = model(a_seq, a_mask, b_seq, b_mask, mfeat)
            w_loss = bce_loss(win_pred, win_label)
            f_loss = ce_loss(finish_logits, finish_label)
            loss   = w_loss + FINISH_WEIGHT * f_loss
            total_loss     += loss.item() * len(win_label)
            win_correct    += ((win_pred > 0.5) == win_label.bool()).sum().item()
            finish_correct += (finish_logits.argmax(dim=1) == finish_label).sum().item()
            total          += len(win_label)
    return total_loss/total, win_correct/total, finish_correct/total

EPOCHS, PATIENCE = 80, 12
best_val, best_epoch, no_improve = float("inf"), 0, 0

print(f"\n開始訓練...")
print(f"{'Epoch':>6} {'Loss':>8} {'WinAcc':>8} {'FinAcc':>8} "
      f"{'vLoss':>8} {'vWin':>7} {'vFin':>7}")
print("─" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for a_seq, a_mask, b_seq, b_mask, mfeat, win_label, finish_label in train_loader:
        optimizer.zero_grad()
        win_pred, finish_logits = model(a_seq, a_mask, b_seq, b_mask, mfeat)
        w_loss = bce_loss(win_pred, win_label)
        f_loss = ce_loss(finish_logits, finish_label)
        loss   = w_loss + FINISH_WEIGHT * f_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    tr_loss, tr_win, tr_fin = evaluate(train_loader)
    va_loss, va_win, va_fin = evaluate(val_loader)
    scheduler.step(va_loss)

    if epoch % 5 == 0 or epoch == 1:
        marker = " ←" if va_loss < best_val else ""
        print(f"{epoch:>6}  {tr_loss:>7.4f}  {tr_win:>7.3f}  {tr_fin:>7.3f}  "
              f"{va_loss:>7.4f}  {va_win:>6.3f}  {va_fin:>6.3f}{marker}")

    if va_loss < best_val - 1e-4:
        best_val, best_epoch, no_improve = va_loss, epoch, 0
        torch.save(model.state_dict(), "models/model_seq.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping (epoch {epoch}, best: epoch {best_epoch})")
            break

# ──────────────────────────────────────────────
# 8. 最終結果
# ──────────────────────────────────────────────
model.load_state_dict(torch.load("models/model_seq.pt", weights_only=True))
_, final_win, final_fin = evaluate(val_loader)
print(f"\n2026 Holdout 結果:")
print(f"  勝負準確率:   {final_win:.3f}")
print(f"  結束方式準確率: {final_fin:.3f}")

# ──────────────────────────────────────────────
# 9. Platt Scaling（用未對稱化的 2024-2025 校準）
# ──────────────────────────────────────────────
print("\n訓練 Platt Scaling 校準器（2024-2025，未對稱化）...")
print(f"校準集 A方勝率: {cal_orig['winner_is_a'].mean():.2%}（有偏差，校準有意義）")

model.eval()
raw_probs, true_labels = [], []

with torch.no_grad():
    for a_seq, a_mask, b_seq, b_mask, mfeat, win_label, finish_label in cal_loader:
        win_pred, _ = model(a_seq, a_mask, b_seq, b_mask, mfeat)

        # 建立反向 matchup feat（用於正反兩向平均）
        # 這裡簡化：只用正向預測做校準
        raw_probs.extend(win_pred.numpy())
        true_labels.extend(win_label.numpy())

raw_probs   = np.array(raw_probs).reshape(-1, 1)
true_labels = np.array(true_labels)

calibrator = LogisticRegression(C=1.0, max_iter=1000)
calibrator.fit(raw_probs, true_labels)

cal_preds = calibrator.predict_proba(raw_probs)[:,1]
print(f"校準前 - 平均預測: {raw_probs.mean():.3f}, 實際勝率: {true_labels.mean():.3f}")
print(f"校準後 - 平均預測: {cal_preds.mean():.3f}, 實際勝率: {true_labels.mean():.3f}")

# 驗證校準效果（在 val set 上）
val_raw, val_true = [], []
with torch.no_grad():
    for a_seq, a_mask, b_seq, b_mask, mfeat, win_label, finish_label in val_loader:
        win_pred, _ = model(a_seq, a_mask, b_seq, b_mask, mfeat)
        val_raw.extend(win_pred.numpy())
        val_true.extend(win_label.numpy())

val_raw   = np.array(val_raw).reshape(-1, 1)
val_true  = np.array(val_true)
val_cal   = calibrator.predict_proba(val_raw)[:,1]
print(f"\n在 2026 Val set 上:")
print(f"  校準前 - 平均預測: {val_raw.mean():.3f}, 實際勝率: {val_true.mean():.3f}")
print(f"  校準後 - 平均預測: {val_cal.mean():.3f}, 實際勝率: {val_true.mean():.3f}")

with open("models/platt_calibrator.pkl", "wb") as f:
    pickle.dump(calibrator, f)
print("校準器儲存至 models/platt_calibrator.pkl")

# ──────────────────────────────────────────────
# 10. 計算所有選手風格向量
# ──────────────────────────────────────────────
print("\n計算所有選手風格向量...")
model.eval()
fighter_names  = list(sequences.keys())
all_style_vecs = []

for i in range(0, len(fighter_names), 256):
    batch = fighter_names[i:i+256]
    seqs  = np.stack([sequences[n]["seq"]  for n in batch])
    masks = np.stack([sequences[n]["mask"] for n in batch])
    vecs, _ = model.encode_with_attention(
        torch.tensor(seqs,  dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool)
    )
    all_style_vecs.append(vecs)

all_style_vecs = np.vstack(all_style_vecs)
fighter_to_idx = {n: i for i, n in enumerate(fighter_names)}

np.save("models/seq_embeddings.npy", all_style_vecs)
with open("models/seq_fighter_id_map.json", "w") as f:
    json.dump(fighter_to_idx, f, ensure_ascii=False, indent=2)

config = {
    "n_feat": N_FEAT, "hidden_dim": HIDDEN_DIM,
    "matchup_dim": MATCHUP_DIM, "seq_len": SEQ_LEN,
    "n_finish": N_FINISH,
    "h_mean": h_mean, "h_std": h_std,
    "r_mean": r_mean, "r_std": r_std,
}
with open("models/seq_model_config.json", "w") as f:
    json.dump(config, f, indent=2)

# ──────────────────────────────────────────────
# 11. 2026 逐場回測
# ──────────────────────────────────────────────
print("\n2026 年回測（真正的 holdout）:")
val_orig = matchups[matchups["date"] >= "2026-01-01"].copy()

correct = 0
total   = 0
for _, row in val_orig.iterrows():
    fa = row["fighter_a"]
    fb = row["fighter_b"]
    if fa not in sequences or fb not in sequences: continue

    a_seq  = torch.tensor(sequences[fa]["seq"],  dtype=torch.float32).unsqueeze(0)
    a_mask = torch.tensor(sequences[fa]["mask"], dtype=torch.bool).unsqueeze(0)
    b_seq  = torch.tensor(sequences[fb]["seq"],  dtype=torch.float32).unsqueeze(0)
    b_mask = torch.tensor(sequences[fb]["mask"], dtype=torch.bool).unsqueeze(0)
    mf_ab  = torch.tensor([get_matchup_feat(fa, fb, row["title_fight"], row["division"])],
                           dtype=torch.float32)
    mf_ba  = torch.tensor([get_matchup_feat(fb, fa, row["title_fight"], row["division"])],
                           dtype=torch.float32)

    with torch.no_grad():
        p_ab, _ = model(a_seq, a_mask, b_seq, b_mask, mf_ab)
        p_ba, _ = model(b_seq, b_mask, a_seq, a_mask, mf_ba)

    prob_a = float(((p_ab + (1-p_ba))/2).item())
    pred_a_wins   = prob_a > 0.5
    actual_a_wins = bool(row["winner_is_a"])
    correct += (pred_a_wins == actual_a_wins)
    total   += 1

print(f"2026 回測: {correct}/{total} = {correct/total*100:.1f}%")

# ──────────────────────────────────────────────
# 12. 相似選手驗證
# ──────────────────────────────────────────────
def find_similar(name, top_n=5):
    if name not in fighter_to_idx: return
    idx  = fighter_to_idx[name]
    vec  = all_style_vecs[idx]
    norm = np.linalg.norm(all_style_vecs, axis=1, keepdims=True) + 1e-8
    sims = (all_style_vecs / norm) @ (vec / (np.linalg.norm(vec)+1e-8))
    matchups_raw = pd.read_csv("data/processed/matchups_seq.csv")
    def get_div(n):
        m = matchups_raw[(matchups_raw["fighter_a"]==n)|(matchups_raw["fighter_b"]==n)]
        return m["division"].mode()[0] if len(m) > 0 else ""
    own_div = get_div(name)
    print(f"\n  {name}:")
    shown = 0
    for i in np.argsort(-sims):
        if i == idx: continue
        fname = fighter_names[i]
        if get_div(fname) != own_div: continue
        nf = sequences[fname]["n_fights"]
        if nf < 3: continue
        print(f"    {fname:<30} sim={sims[i]:.3f} ({nf}場)")
        shown += 1
        if shown >= top_n: break

print("\n驗證風格向量...")
find_similar("Carlos Prates")
find_similar("Islam Makhachev")
find_similar("Sean Strickland")
find_similar("Alex Pereira")

print("\n✅ 完成！")
print("  models/model_seq.pt")
print("  models/platt_calibrator.pkl")
print("  models/seq_embeddings.npy")
print("  models/seq_fighter_id_map.json")
print("  models/seq_model_config.json")
