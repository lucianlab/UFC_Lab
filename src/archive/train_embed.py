"""
UFC Fighter Style Encoder — v3
覆蓋: src/train_embed.py

執行: python3 src/train_embed.py
輸出:
  models/model_embed.pt
  models/fighter_embeddings.npy
  models/fighter_id_map.json
  models/embed_scaler.pkl
  models/embed_feature_cols.json
"""

import pandas as pd
import numpy as np
import json, os, pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# 1. 載入資料
# ──────────────────────────────────────────────
df = pd.read_csv("data/processed/fights_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"載入 {len(df)} 場比賽")

# encoder 輸入：每個選手自己的風格數據
FIGHTER_STATS = [
    "splm", "sapm", "str_acc", "str_def",
    "td_avg", "td_avg_acc", "td_def", "sub_avg",
    "win_rate", "reach", "experience",
    "pure_ko_rate", "tko_rate", "finish_rate",
    "ko_vulnerability", "momentum", "splm_trend",
]

# predictor 輔助輸入：兩人差值
DELTA_FEATURES = [
    "title_fight",
    "delta_wins", "delta_losses", "delta_win_rate", "delta_experience",
    "delta_splm", "delta_sapm", "delta_str_acc", "delta_str_def",
    "delta_td_avg", "delta_td_avg_acc", "delta_td_def", "delta_sub_avg",
    "delta_height", "delta_reach",
    "southpaw_matchup", "same_stance",
    "r_stance_enc", "b_stance_enc",
    "delta_pure_ko_rate", "delta_tko_rate", "delta_finish_rate",
    "delta_ko_vulnerability", "delta_momentum", "delta_splm_trend",
    "delta_last3_win_rate", "delta_last3_splm",
    "delta_effective_td", "delta_effective_sub",
    "delta_ko_power", "delta_ctrl_dominance",
]

# 確認欄位存在，過濾掉資料裡沒有的
available_cols = set(df.columns)
FIGHTER_STATS  = [s for s in FIGHTER_STATS  if f"r_{s}" in available_cols]
DELTA_FEATURES = [s for s in DELTA_FEATURES if s in available_cols]

N_STATS   = len(FIGHTER_STATS)
N_DELTA   = len(DELTA_FEATURES)
STYLE_DIM = 16

print(f"Fighter stats: {N_STATS} 個")
print(f"Delta features: {N_DELTA} 個")

# ──────────────────────────────────────────────
# 2. 對稱化
# ──────────────────────────────────────────────
def symmetrize(df):
    orig    = df.copy()
    flipped = df.copy()

    # delta 欄位全部取負
    for col in [c for c in df.columns if c.startswith("delta_")]:
        flipped[col] = -df[col]

    # stance 對調
    flipped["r_stance_enc"] = df["b_stance_enc"]
    flipped["b_stance_enc"] = df["r_stance_enc"]

    # 勝負對調
    flipped["winner_is_red"] = 1 - df["winner_is_red"]

    # fighter stats 對調（只對調資料裡有的欄位）
    for s in FIGHTER_STATS:
        r_col, b_col = f"r_{s}", f"b_{s}"
        if r_col in df.columns and b_col in df.columns:
            orig_r = df[r_col].copy()
            orig_b = df[b_col].copy()
            flipped[r_col] = orig_b
            flipped[b_col] = orig_r

    return pd.concat([orig, flipped], ignore_index=True)\
             .sort_values("date").reset_index(drop=True)

df_sym = symmetrize(df)
print(f"對稱化後: {len(df_sym)} 筆 (紅角勝率: {df_sym['winner_is_red'].mean():.2%})")

# ──────────────────────────────────────────────
# 3. 標準化
# ──────────────────────────────────────────────
r_stat_cols = [f"r_{s}" for s in FIGHTER_STATS]
b_stat_cols = [f"b_{s}" for s in FIGHTER_STATS]

all_stats = np.vstack([
    df_sym[r_stat_cols].values,
    df_sym[b_stat_cols].values
]).astype(np.float32)

stats_scaler = StandardScaler()
stats_scaler.fit(all_stats)

delta_scaler = StandardScaler()
delta_scaler.fit(df_sym[DELTA_FEATURES].values.astype(np.float32))

with open("models/embed_scaler.pkl", "wb") as f:
    pickle.dump({"stats": stats_scaler, "delta": delta_scaler}, f)

with open("models/embed_feature_cols.json", "w") as f:
    json.dump({"fighter_stats": FIGHTER_STATS,
               "delta_features": DELTA_FEATURES}, f, indent=2)

r_stats_scaled = stats_scaler.transform(df_sym[r_stat_cols].values.astype(np.float32))
b_stats_scaled = stats_scaler.transform(df_sym[b_stat_cols].values.astype(np.float32))
delta_scaled   = delta_scaler.transform(df_sym[DELTA_FEATURES].values.astype(np.float32))
y = df_sym["winner_is_red"].values.astype(np.float32)

# ──────────────────────────────────────────────
# 4. Train / Val split（時間序列）
# ──────────────────────────────────────────────
split = int(len(df_sym) * 0.85)

# ──────────────────────────────────────────────
# 5. Dataset
# ──────────────────────────────────────────────
class FightDataset(Dataset):
    def __init__(self, r, b, d, y):
        self.r = torch.tensor(r, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.d = torch.tensor(d, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.r[i], self.b[i], self.d[i], self.y[i]

train_ds = FightDataset(r_stats_scaled[:split], b_stats_scaled[:split],
                        delta_scaled[:split],   y[:split])
val_ds   = FightDataset(r_stats_scaled[split:], b_stats_scaled[split:],
                        delta_scaled[split:],   y[split:])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

# ──────────────────────────────────────────────
# 6. 模型
# ──────────────────────────────────────────────
class StyleEncoder(nn.Module):
    def __init__(self, n_stats, style_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_stats, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, style_dim), nn.Tanh(),
        )
    def forward(self, x): return self.net(x)

class UFCStyleModel(nn.Module):
    def __init__(self, n_stats, style_dim, n_delta):
        super().__init__()
        self.encoder = StyleEncoder(n_stats, style_dim)
        self.predictor = nn.Sequential(
            nn.Linear(style_dim*2+n_delta, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, r, b, d):
        return self.predictor(
            torch.cat([self.encoder(r), self.encoder(b), d], dim=1)
        ).squeeze(1)
    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()

model = UFCStyleModel(N_STATS, STYLE_DIM, N_DELTA)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型參數: {total_params:,}")
print(f"Encoder: {N_STATS} → 32 → {STYLE_DIM}")
print(f"Predictor: {STYLE_DIM*2+N_DELTA} → 64 → 32 → 1")

# ──────────────────────────────────────────────
# 7. 訓練
# ──────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=6, factor=0.5, min_lr=1e-5
)
criterion = nn.BCELoss()

def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for r, b, d, label in loader:
            pred  = model(r, b, d)
            loss  = criterion(pred, label)
            total_loss += loss.item() * len(label)
            correct    += ((pred > 0.5) == label.bool()).sum().item()
            total      += len(label)
    return total_loss / total, correct / total

EPOCHS, PATIENCE = 80, 12
best_val, best_epoch, no_improve = float("inf"), 0, 0

print(f"\n開始訓練 (最多 {EPOCHS} epochs, patience={PATIENCE})...")
print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
print("─" * 55)

for epoch in range(1, EPOCHS+1):
    model.train()
    for r, b, d, label in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(r, b, d), label)
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
        torch.save(model.state_dict(), "models/model_embed.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping (epoch {epoch}, best: epoch {best_epoch})")
            break

# ──────────────────────────────────────────────
# 8. 最終結果
# ──────────────────────────────────────────────
model.load_state_dict(torch.load("models/model_embed.pt", weights_only=True))
_, final_val_acc = evaluate(val_loader)
print(f"\n最終 val accuracy : {final_val_acc:.3f}")
print(f"前版 val accuracy : 0.772")
diff = final_val_acc - 0.772
print(f"差距              : {diff:+.3f}")

# ──────────────────────────────────────────────
# 9. 計算所有選手的風格向量
# ──────────────────────────────────────────────
print("\n計算所有選手的風格向量...")

fighter_stats_rows = {}
for _, row in df.sort_values("date").iterrows():
    for corner in ["r", "b"]:
        name = row[f"{corner}_name"]
        fighter_stats_rows[name] = [
            row.get(f"{corner}_{s}", 0.0) for s in FIGHTER_STATS
        ]

fighter_names = list(fighter_stats_rows.keys())
stats_matrix  = np.array(list(fighter_stats_rows.values()), dtype=np.float32)
stats_scaled  = stats_scaler.transform(stats_matrix)
stats_tensor  = torch.tensor(stats_scaled, dtype=torch.float32)

all_style_vecs = model.encode(stats_tensor)
np.save("models/fighter_embeddings.npy", all_style_vecs)

fighter_to_idx = {name: i for i, name in enumerate(fighter_names)}
with open("models/fighter_id_map.json", "w") as f:
    json.dump(fighter_to_idx, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────
# 10. 相似選手驗證
# ──────────────────────────────────────────────
def find_similar(name, top_n=6):
    if name not in fighter_to_idx:
        print(f"  找不到: {name}"); return
    idx  = fighter_to_idx[name]
    vec  = all_style_vecs[idx]
    norm = np.linalg.norm(all_style_vecs, axis=1, keepdims=True) + 1e-8
    sims = (all_style_vecs / norm) @ (vec / (np.linalg.norm(vec) + 1e-8))
    top  = np.argsort(-sims)[1:top_n+1]
    print(f"\n  {name}:")
    for i in top:
        n_fights = len(df[(df["r_name"]==fighter_names[i])|(df["b_name"]==fighter_names[i])])
        print(f"    {fighter_names[i]:<30} sim={sims[i]:.3f}  ({n_fights}場)")

print("\n驗證風格向量...")
find_similar("Sean Strickland")
find_similar("Khamzat Chimaev")
find_similar("Islam Makhachev")
find_similar("Carlos Prates")

print("\n✅ 完成！")
print("  models/model_embed.pt")
print("  models/fighter_embeddings.npy")
print("  models/fighter_id_map.json")
print("  models/embed_scaler.pkl")
print("  models/embed_feature_cols.json")