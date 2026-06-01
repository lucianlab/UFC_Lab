from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import json
import math
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════
#  Universe 選手資料
# ══════════════════════════════════════════════════════
DATA_PATH = Path(__file__).parent / "data" / "fighters.json"
with open(DATA_PATH, encoding="utf-8") as f:
    FIGHTERS = json.load(f)
_by_name = {f["name"]: f for f in FIGHTERS}

# ══════════════════════════════════════════════════════
#  Fighter Builder 資料
# ══════════════════════════════════════════════════════
BUILDER_PATH = Path(__file__).parent / "data" / "clean" / "fighter_builder_features.csv"
BUILDER_DF   = pd.read_csv(BUILDER_PATH)

# ── PCT_TO_RAW: 玩家輸入欄位 → CSV pct 欄位 ──────────
# v3: 13 features, 移除 height, 新增 body/td_accuracy
# KNN 直接在 pct 空間算距離 (已經是 1-10 百分位,不需 quantile 反查)
PCT_COLS = [
    'pct_reach',
    'pct_striking_power', 'pct_striking_volume',
    'pct_striking_accuracy', 'pct_striking_defense',
    'pct_leg_kicks', 'pct_clinch', 'pct_body',
    'pct_td_frequency', 'pct_td_accuracy', 'pct_td_defense',
    'pct_submission', 'pct_control', 'pct_ground_pound',
]

# 啟動時驗證所有欄位都在 BUILDER_DF 且不全是 NaN
def _valid_pct_cols():
    valid = []
    for col in PCT_COLS:
        if col not in BUILDER_DF.columns:
            print(f"WARNING: {col} not in BUILDER_DF, skipping")
            continue
        if BUILDER_DF[col].dropna().empty:
            print(f"WARNING: {col} is all NaN, skipping")
            continue
        valid.append(col)
    return valid

KNN_COLS = _valid_pct_cols()
print(f"KNN_COLS ({len(KNN_COLS)}): {KNN_COLS}")

# ── z-score 標準化 (在 pct 空間做,確保各維度權重相等) ─
_means = {}
_stds  = {}
for col in KNN_COLS:
    vals = BUILDER_DF[col].dropna()
    _means[col] = float(vals.mean())
    _stds[col]  = float(vals.std()) if float(vals.std()) > 0 else 1.0

# ── 預建 KNN matrix ────────────────────────────────────
def _build_matrix():
    rows = []
    for _, row in BUILDER_DF.iterrows():
        vec = []
        for col in KNN_COLS:
            val = row[col] if pd.notna(row[col]) else _means[col]
            vec.append((float(val) - _means[col]) / _stds[col])
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

_MATRIX = _build_matrix()
print(f"KNN matrix: {_MATRIX.shape}, nan: {np.isnan(_MATRIX).any()}")

# ══════════════════════════════════════════════════════
#  排行榜
# ══════════════════════════════════════════════════════
LEADERBOARD_PATH = Path(__file__).parent / "data" / "leaderboard.json"

def _load_leaderboard():
    if LEADERBOARD_PATH.exists():
        with open(LEADERBOARD_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_leaderboard(data):
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ══════════════════════════════════════════════════════
#  工具函數
# ══════════════════════════════════════════════════════
def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except:
        return default

def safe_int(val, default=0):
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else int(v)
    except:
        return default

def safe_str(val, default=''):
    try:
        if val is None: return default
        if isinstance(val, float) and math.isnan(val): return default
        return str(val)
    except:
        return default

def clean_json(obj):
    if isinstance(obj, dict):   return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

# ══════════════════════════════════════════════════════
#  既有端點
# ══════════════════════════════════════════════════════
@app.get("/api/fighters")
def get_fighters():
    return FIGHTERS

@app.get("/api/fighters/{name}")
def get_fighter(name: str):
    fighter = _by_name.get(name)
    if fighter is None:
        raise HTTPException(status_code=404, detail="Fighter not found")
    return fighter

# ══════════════════════════════════════════════════════
#  Fighter Builder
# ══════════════════════════════════════════════════════
class BuilderInput(BaseModel):
    # 玩家輸入的 13 個 pct 分數 (1-10)
    # 直接對應 CSV 的 pct_* 欄位,不需要 quantile 反查
    pct_reach:             Optional[float] = 5.0
    pct_striking_power:    Optional[float] = 5.0
    pct_striking_volume:   Optional[float] = 5.0
    pct_striking_accuracy: Optional[float] = 5.0
    pct_striking_defense:  Optional[float] = 5.0
    pct_leg_kicks:         Optional[float] = 5.0
    pct_clinch:            Optional[float] = 5.0
    pct_body:              Optional[float] = 5.0
    pct_td_frequency:      Optional[float] = 5.0
    pct_td_accuracy:       Optional[float] = 5.0
    pct_td_defense:        Optional[float] = 5.0
    pct_submission:        Optional[float] = 5.0
    pct_control:           Optional[float] = 5.0
    pct_ground_pound:      Optional[float] = 5.0

def _input_to_zvec(inp: BuilderInput) -> np.ndarray:
    """玩家 pct 輸入 → z-score 標準化向量"""
    vec = []
    for col in KNN_COLS:
        pct_val = safe_float(getattr(inp, col, 5.0), 5.0)
        pct_val = max(1.0, min(10.0, pct_val))
        z = (pct_val - _means[col]) / _stds[col]
        vec.append(z)
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

@app.post("/api/nearest")
def find_nearest(inp: BuilderInput, k: int = 5):
    query = _input_to_zvec(inp)

    if query.shape[0] != _MATRIX.shape[1]:
        raise HTTPException(status_code=500,
            detail=f"Vector dim mismatch: query={query.shape[0]}, matrix={_MATRIX.shape[1]}")

    # 純歐氏距離,不做 tier 加權
    diffs = _MATRIX - query
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    # 取最近 50 個,從裡面找 tier 最高的 k 個展示
    top50_idx  = np.argsort(dists)[:50]
    candidates = []
    for idx in top50_idx:
        row = BUILDER_DF.iloc[idx]
        candidates.append({
            "idx":        int(idx),
            "tier_label": safe_str(row.get('tier_label'), 'E') or 'E',
            "tier_score": safe_int(row.get('tier_score'), 0),
            "distance":   float(dists[idx]),
        })

    # 在 top50 裡按 tier_score 降序,取前 k 個
    # 這樣保證結果是「離你最近的 50 個人裡最知名的」
    candidates.sort(key=lambda x: (-x['tier_score'], x['distance']))
    top_k = candidates[:k]

    results = []
    for c in top_k:
        idx = c['idx']
        row = BUILDER_DF.iloc[idx]
        results.append({
            "name":       safe_str(row.get('name'), 'Unknown'),
            "wc":         safe_str(row.get('wc'), ''),
            "tier_label": c['tier_label'],
            "tier_score": c['tier_score'],
            "best_rank":  None if pd.isna(row.get('best_rank')) else safe_float(row['best_rank']),
            "win_rate":   None if pd.isna(row.get('win_rate'))  else round(safe_float(row['win_rate']), 3),
            "distance":   round(c['distance'], 3),
            "pct": {
                col: round(safe_float(row.get(col), 5.0), 1)
                for col in KNN_COLS
            }
        })

    return clean_json({"matches": results})

# ══════════════════════════════════════════════════════
#  排行榜
# ══════════════════════════════════════════════════════
class LeaderboardEntry(BaseModel):
    username:    str
    tier_score:  int
    nearest:     str
    points_used: int
    features:    dict

@app.get("/api/leaderboard")
def get_leaderboard():
    data = _load_leaderboard()
    data.sort(key=lambda x: (-x['tier_score'], x.get('points_used', 999)))
    return data[:50]

@app.post("/api/leaderboard")
def post_leaderboard(entry: LeaderboardEntry):
    data     = _load_leaderboard()
    existing = next((i for i, e in enumerate(data)
                     if e['username'] == entry.username), None)
    new_entry = entry.dict()
    if existing is not None:
        old = data[existing]
        if (entry.tier_score > old['tier_score'] or
            (entry.tier_score == old['tier_score'] and
             entry.points_used < old.get('points_used', 999))):
            data[existing] = new_entry
    else:
        data.append(new_entry)
    _save_leaderboard(data)
    return {"ok": True, "entries": len(data)}
