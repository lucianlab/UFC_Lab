"""
UFC Matchup Prediction — Step 2: Model Training
執行方式: python3 src/train.py
輸出:
  models/model_win.json
  models/model_method.json
  models/model_round.json
  models/feature_cols.json
"""

import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────
# 1. 載入資料
# ──────────────────────────────────────────────
df = pd.read_csv("data/processed/fights_clean.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"載入 {len(df)} 場比賽")

FEATURE_COLS = [
    "title_fight",
    "delta_wins", "delta_losses", "delta_win_rate", "delta_experience",
    "delta_splm", "delta_sapm",
    "delta_str_acc", "delta_str_def",
    "delta_td_avg", "delta_td_avg_acc", "delta_td_def",
    "delta_sub_avg",
    "delta_height", "delta_reach",
    "southpaw_matchup", "same_stance",
    "r_stance_enc", "b_stance_enc",
]

# ──────────────────────────────────────────────
# 2. 對稱化 — 解決紅角偏差問題
#    把每場比賽複製一份，紅藍對調，讓訓練資料 50/50
# ──────────────────────────────────────────────
def symmetrize(df):
    original = df.copy()
    flipped  = df.copy()

    for col in [c for c in df.columns if c.startswith("delta_")]:
        flipped[col] = -df[col]

    flipped["r_stance_enc"]  = df["b_stance_enc"]
    flipped["b_stance_enc"]  = df["r_stance_enc"]
    flipped["r_win_rate"]    = df["b_win_rate"]
    flipped["b_win_rate"]    = df["r_win_rate"]
    flipped["r_experience"]  = df["b_experience"]
    flipped["b_experience"]  = df["r_experience"]
    flipped["winner_is_red"] = 1 - df["winner_is_red"]

    return pd.concat([original, flipped], ignore_index=True).sort_values("date").reset_index(drop=True)

df_sym = symmetrize(df)
print(f"對稱化後: {len(df_sym)} 筆 (紅角勝率: {df_sym['winner_is_red'].mean():.2%})")

# ──────────────────────────────────────────────
# 3. 準備特徵矩陣
# ──────────────────────────────────────────────
X        = df_sym[FEATURE_COLS].values
y_win    = df_sym["winner_is_red"].values
y_method = df_sym["finish_method"].fillna(2).astype(int).values

# 輪次：只用有明確輪次的資料，label 從 0 開始
round_mask    = df_sym["finish_round"].notna()
X_round       = df_sym.loc[round_mask, FEATURE_COLS].values
y_round       = (df_sym.loc[round_mask, "finish_round"].astype(int).clip(1, 5) - 1).values

tscv = TimeSeriesSplit(n_splits=5)

# ──────────────────────────────────────────────
# 4. 訓練函數
# ──────────────────────────────────────────────
def train_and_evaluate(X, y, task_name, objective, eval_metric, num_class=None):
    print(f"\n{'='*50}")
    print(f"訓練: {task_name}")

    params = dict(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        objective        = objective,
        eval_metric      = eval_metric,
        random_state     = 42,
        n_jobs           = -1,
    )
    if num_class:
        params["num_class"] = num_class

    model = xgb.XGBClassifier(**params)
    accs, losses = [], []

    for fold, (tr, te) in enumerate(tscv.split(X)):
        model.fit(X[tr], y[tr],
                  eval_set=[(X[te], y[te])],
                  verbose=False)
        pred  = model.predict(X[te])
        proba = model.predict_proba(X[te])
        acc   = accuracy_score(y[te], pred)
        ll    = log_loss(y[te], proba)
        accs.append(acc)
        losses.append(ll)
        print(f"  Fold {fold+1}: accuracy={acc:.3f}, logloss={ll:.3f}")

    print(f"  平均 accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  平均 logloss : {np.mean(losses):.3f}")

    # 最終模型用全部資料訓練
    model.fit(X, y, verbose=False)
    return model

# ──────────────────────────────────────────────
# 5. 訓練三個模型
# ──────────────────────────────────────────────
model_win    = train_and_evaluate(X, y_win,    "勝負預測",   "binary:logistic",  "logloss")
model_method = train_and_evaluate(X, y_method, "結束方式預測", "multi:softprob",  "mlogloss", num_class=3)
model_round  = train_and_evaluate(X_round, y_round, "結束輪次預測", "multi:softprob", "mlogloss", num_class=5)

# ──────────────────────────────────────────────
# 6. Feature Importance
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("Feature Importance (勝負模型)")
fi = sorted(zip(FEATURE_COLS, model_win.feature_importances_), key=lambda x: -x[1])
for name, score in fi:
    bar = "█" * int(score * 300)
    print(f"  {name:<25} {bar} {score:.4f}")

# ──────────────────────────────────────────────
# 7. 儲存
# ──────────────────────────────────────────────
model_win.save_model("models/model_win.json")
model_method.save_model("models/model_method.json")
model_round.save_model("models/model_round.json")

with open("models/feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print("\n✅ 完成！模型已儲存至 models/")
