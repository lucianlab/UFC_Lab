"""
UFC GRU Sequence Model v8
覆蓋: src/train_seq.py

v8 變更：
  - N_FEAT 從 pkl 自動讀取（28→33，不需手動改）
  - HIDDEN_DIM 64→48（資料量約 7000 筆，減少過擬合）
  - weight_decay 1e-4→2e-4（更強正則化）
  - PATIENCE 12→15（給模型更多收斂時間）
  - 其餘架構不變（GRU×2層 + Attention + Platt Scaling）

執行前：python3 src/data_prep_seq.py
執行：  python3 src/train_seq.py
"""

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
print("載入 matchup_samples.pkl ...")
with open("data/processed/matchup_samples.pkl", "rb") as f:
    all_samples = pickle.load(f)

with open("data/processed/fighter_sequences.pkl", "rb") as f:
    seq_data = pickle.load(f)

SEQ_LEN      = seq_data["seq_len"]
N_FEAT       = len(seq_data["seq_features"])   # 自動對應 v8 的 33
physical     = seq_data["physical"]
ftr_physical = seq_data["fighter_physical"]
div_physical = seq_data["div_physical"]
DIVISION_ENC = seq_data["division_enc"]

print(f"總 sample 數: {len(all_samples)}")
print(f"序列長度: {SEQ_LEN}, 特徵數: {N_FEAT}")

# ──────────────────────────────────────────────
# 2. 時間切割
# ──────────────────────────────────────────────
train_samples = [s for s in all_samples if s["date"] <  "2024-01-01"]
cal_samples   = [s for s in all_samples if s["date"] >= "2024-01-01"]

print(f"訓練集: {len(train_samples)} 筆（2024年以前）")
print(f"校準/驗證集: {len(cal_samples)} 筆（2024-2025）")

if len(train_samples) == 0 or len(cal_samples) == 0:
    raise ValueError("訓練集或驗證集為空，請確認 matchup_samples.pkl 有正確的日期範圍")

# ──────────────────────────────────────────────
# 3. 物理特徵（matchup features，9維，不變）
# ──────────────────────────────────────────────
all_reaches = [v["reach"]  for v in physical.values() if v.get("reach")]
all_heights = [v["height"] for v in physical.values() if v.get("height")]
h_mean, h_std = np.mean(all_heights), np.std(all_heights) + 1e-6
r_mean, r_std = np.mean(all_reaches),  np.std(all_reaches)  + 1e-6

MATCHUP_DIM = 9

def get_reach_pct(name, division):
    p    = ftr_physical.get(name, {})
    v    = p.get("reach")
    info = div_physical.get(division, {}).get("reach")
    if info is None or v is None: return 0.5
    sv = info.get("sorted", [])
    return sum(1 for x in sv if x < v) / len(sv) if sv else 0.5

def get_matchup_feat(name_a, name_b, title_fight, division):
    pa  = physical.get(name_a, {"reach":180.0,"height":175.0,"stance_enc":0})
    pb  = physical.get(name_b, {"reach":180.0,"height":175.0,"stance_enc":0})
    ar  = (pa["reach"]  - r_mean) / r_std
    br  = (pb["reach"]  - r_mean) / r_std
    ah  = (pa["height"] - h_mean) / h_std
    bh  = (pb["height"] - h_mean) / h_std
    ast = pa["stance_enc"] / 2.0
    bst = pb["stance_enc"] / 2.0
    arp = get_reach_pct(name_a, division)
    brp = get_reach_pct(name_b, division)
    return [
        float(title_fight), ar-br, ah-bh,
        float(ast != bst),
        float((ast==0 and bst==0.5) or (ast==0.5 and bst==0)),
        arp, brp, arp-brp,
        DIVISION_ENC.get(division, 0.5),
    ]

# ──────────────────────────────────────────────
# 4. Dataset（從 snapshot 直接讀，不查全域 dict）
# ──────────────────────────────────────────────
class FightSnapshotDataset(Dataset):
    def __init__(self, samples, symmetrize=False):
        self.items = []
        for s in samples:
            self.items.append(s)
            if symmetrize:
                flipped = dict(s)
                flipped["fighter_a"]   = s["fighter_b"]
                flipped["fighter_b"]   = s["fighter_a"]
                flipped["seq_a"]       = s["seq_b"]
                flipped["mask_a"]      = s["mask_b"]
                flipped["seq_b"]       = s["seq_a"]
                flipped["mask_b"]      = s["mask_a"]
                flipped["n_hist_a"]    = s["n_hist_b"]
                flipped["n_hist_b"]    = s["n_hist_a"]
                flipped["winner_is_a"] = 1 - s["winner_is_a"]
                self.items.append(flipped)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        s     = self.items[i]
        mfeat = np.array(get_matchup_feat(
            s["fighter_a"], s["fighter_b"],
            s["title_fight"], s["division"]
        ), dtype=np.float32)
        return (
            torch.tensor(s["seq_a"],  dtype=torch.float32),
            torch.tensor(s["mask_a"], dtype=torch.bool),
            torch.tensor(s["seq_b"],  dtype=torch.float32),
            torch.tensor(s["mask_b"], dtype=torch.bool),
            torch.tensor(mfeat,       dtype=torch.float32),
            torch.tensor(float(s["winner_is_a"]), dtype=torch.float32),
        )

train_ds     = FightSnapshotDataset(train_samples, symmetrize=True)
val_ds       = FightSnapshotDataset(cal_samples,   symmetrize=True)
cal_ds       = FightSnapshotDataset(cal_samples,   symmetrize=False)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
cal_loader   = DataLoader(cal_ds,   batch_size=256, shuffle=False, num_workers=0)

print(f"訓練集（對稱化後）: {len(train_ds)} 筆")
print(f"驗證集（對稱化後）: {len(val_ds)} 筆")

# ──────────────────────────────────────────────
# 5. 模型（HIDDEN_DIM 64→48）
# ──────────────────────────────────────────────
HIDDEN_DIM = 48   # v8: 減小以匹配資料量，減少過擬合

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, gru_out, mask):
        scores  = self.attn(gru_out).squeeze(-1).masked_fill(~mask, -1e9)
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
        self.encoder   = FighterEncoder(n_feat, hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim*2 + matchup_dim, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
    def forward(self, a_seq, a_mask, b_seq, b_mask, matchup):
        sa, _ = self.encoder(a_seq, a_mask)
        sb, _ = self.encoder(b_seq, b_mask)
        return self.predictor(torch.cat([sa, sb, matchup], dim=1)).squeeze(1)
    def encode_with_attention(self, seq, mask):
        with torch.no_grad():
            style, weights = self.encoder(seq, mask)
            return style.numpy(), weights.numpy()

model = UFCSeqModel(N_FEAT, HIDDEN_DIM, MATCHUP_DIM)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型參數: {total_params:,}")
print(f"Encoder: GRU({N_FEAT}→{HIDDEN_DIM}×2) + Attention")
print(f"Predictor: {HIDDEN_DIM*2+MATCHUP_DIM} → 128 → 64 → 1")

# ──────────────────────────────────────────────
# 6. 訓練（weight_decay 提高，patience 延長）
# ──────────────────────────────────────────────
optimizer = torch.optim.Adam(
    model.parameters(), lr=5e-4, weight_decay=2e-4)   # v8: 2e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, min_lr=1e-5)
criterion = nn.BCELoss()

def evaluate(loader):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for a_seq, a_mask, b_seq, b_mask, mfeat, label in loader:
            pred  = model(a_seq, a_mask, b_seq, b_mask, mfeat)
            loss  = criterion(pred, label)
            total_loss += loss.item() * len(label)
            correct    += ((pred > 0.5) == label.bool()).sum().item()
            total      += len(label)
    return total_loss / total, correct / total

EPOCHS  = 100
PATIENCE = 15    # v8: 延長
best_val, best_epoch, no_improve = float("inf"), 0, 0

print(f"\n開始訓練（HIDDEN={HIDDEN_DIM}, weight_decay=2e-4, patience={PATIENCE}）...")
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} "
      f"{'Val Loss':>10} {'Val Acc':>9}")
print("─" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for a_seq, a_mask, b_seq, b_mask, mfeat, label in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(a_seq, a_mask, b_seq, b_mask, mfeat), label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    tr_loss, tr_acc = evaluate(train_loader)
    va_loss, va_acc = evaluate(val_loader)
    scheduler.step(va_loss)

    if epoch % 5 == 0 or epoch == 1:
        marker = " ←best" if va_loss < best_val else ""
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_acc:>9.3f}  "
              f"{va_loss:>9.4f}  {va_acc:>8.3f}{marker}")

    if va_loss < best_val - 1e-4:
        best_val, best_epoch, no_improve = va_loss, epoch, 0
        torch.save(model.state_dict(), "models/model_seq.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping (epoch {epoch}, best: epoch {best_epoch})")
            break

# ──────────────────────────────────────────────
# 7. 最終結果
# ──────────────────────────────────────────────
model.load_state_dict(torch.load("models/model_seq.pt", weights_only=True))
_, final_val_acc = evaluate(val_loader)
print(f"\n✅ Val accuracy (2024-2025, true no-leakage): {final_val_acc:.3f}")

# ──────────────────────────────────────────────
# 8. Platt Scaling（未對稱化的 cal set）
# ──────────────────────────────────────────────
print("\n訓練 Platt Scaling 校準器...")
model.eval()
raw_probs, true_labels = [], []
with torch.no_grad():
    for a_seq, a_mask, b_seq, b_mask, mfeat, label in cal_loader:
        pred = model(a_seq, a_mask, b_seq, b_mask, mfeat)
        raw_probs.extend(pred.numpy())
        true_labels.extend(label.numpy())

raw_arr  = np.array(raw_probs).reshape(-1, 1)
true_arr = np.array(true_labels)
cal_lr   = LogisticRegression(C=1.0, max_iter=1000)
cal_lr.fit(raw_arr, true_arr)
cal_preds = cal_lr.predict_proba(raw_arr)[:,1]
print(f"校準前均值: {raw_arr.mean():.3f} → 校準後: {cal_preds.mean():.3f} "
      f"（實際: {true_arr.mean():.3f}）")

with open("models/platt_calibrator.pkl", "wb") as f:
    pickle.dump(cal_lr, f)

# ──────────────────────────────────────────────
# 9. 風格向量（全量序列）
# ──────────────────────────────────────────────
print("\n計算風格向量...")
sequences_full = seq_data["sequences"]
fighter_names  = list(sequences_full.keys())
all_vecs = []
model.eval()
for i in range(0, len(fighter_names), 256):
    batch = fighter_names[i:i+256]
    seqs  = np.stack([sequences_full[n]["seq"]  for n in batch])
    masks = np.stack([sequences_full[n]["mask"] for n in batch])
    vecs, _ = model.encode_with_attention(
        torch.tensor(seqs,  dtype=torch.float32),
        torch.tensor(masks, dtype=torch.bool))
    all_vecs.append(vecs)

all_vecs = np.vstack(all_vecs)
np.save("models/seq_embeddings.npy", all_vecs)
with open("models/seq_fighter_id_map.json", "w") as f:
    json.dump({n: i for i, n in enumerate(fighter_names)},
              f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────
# 10. 儲存 config
# ──────────────────────────────────────────────
config = {
    "n_feat":      N_FEAT,
    "hidden_dim":  HIDDEN_DIM,
    "matchup_dim": MATCHUP_DIM,
    "seq_len":     SEQ_LEN,
    "h_mean": h_mean, "h_std": h_std,
    "r_mean": r_mean, "r_std": r_std,
    "version": "v8",
}
with open("models/seq_model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n✅ 完成！")
print(f"  models/model_seq.pt  (hidden={HIDDEN_DIM}, n_feat={N_FEAT})")
print(f"  models/platt_calibrator.pkl")
print(f"  models/seq_embeddings.npy")
print(f"  models/seq_model_config.json")
print(f"\n2026 年回測請執行: python3 src/backtest_2026.py")
