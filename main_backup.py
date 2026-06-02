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

# ── PCT_TO_RAW: 玩家 feature 名稱 → CSV 欄位名稱 ──────
# 注意: gas_tank 在 fighter_vectors 全是 NaN,不能用
# pct_cardio 保留在 UI,但不參與 KNN
PCT_TO_RAW = {
    'pct_height':            'height_cm',
    'pct_reach':             'reach_cm',
    'pct_striking_power':    'ko_rate',
    'pct_striking_volume':   'sig_per_r',
    'pct_striking_accuracy': 'sig_acc',
    'pct_striking_defense':  'str_def',
    'pct_leg_kicks':         'leg_pct',
    'pct_clinch':            'clinch_pct',
    'pct_takedown_offense':  'td_per_r',
    'pct_takedown_defense':  'td_def',
    'pct_ground_control':    'ctrl_per_r',
    'pct_submission':        'sub_rate',
    'pct_ground_pound':      'gnp_per_r',
    # 'pct_cardio': 'gas_tank' ← 全是 NaN,不加入
}

# ── KNN_FEATURES: 只包含實際有資料的欄位 ──────────────
# 必須是 PCT_TO_RAW values 的子集,且欄位在 BUILDER_DF 裡不能全是 NaN
def _valid_knn_features():
    valid = []
    for raw_col in PCT_TO_RAW.values():
        if raw_col not in BUILDER_DF.columns:
            print(f"WARNING: {raw_col} not in BUILDER_DF, skipping")
            continue
        non_null = BUILDER_DF[raw_col].dropna()
        if len(non_null) == 0:
            print(f"WARNING: {raw_col} is all NaN, skipping")
            continue
        valid.append(raw_col)
    return valid

KNN_FEATURES = _valid_knn_features()
print(f"KNN_FEATURES ({len(KNN_FEATURES)}): {KNN_FEATURES}")

# ── z-score 標準化參數 (只算 KNN_FEATURES) ────────────
_feat_means = {}
_feat_stds  = {}
for col in KNN_FEATURES:
    vals = BUILDER_DF[col].dropna()
    _feat_means[col] = float(vals.mean())
    _feat_stds[col]  = float(vals.std()) if float(vals.std()) > 0 else 1.0

# ── 預建 KNN matrix (N x len(KNN_FEATURES)) ──────────
def _build_matrix():
    rows = []
    for _, row in BUILDER_DF.iterrows():
        vec = []
        for col in KNN_FEATURES:
            val = row[col] if pd.notna(row[col]) else _feat_means[col]
            vec.append((float(val) - _feat_means[col]) / _feat_stds[col])
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

_BUILDER_MATRIX = _build_matrix()
print(f"KNN matrix shape: {_BUILDER_MATRIX.shape}, any nan: {np.isnan(_BUILDER_MATRIX).any()}")

# ── Tier 加權 ─────────────────────────────────────────
TIER_WEIGHTS = {
    'S': 3.0, 'A+': 2.5, 'A': 2.0,
    'B+': 1.7, 'B': 1.4,
    'C+': 1.2, 'C': 1.1,
    'D+': 1.0, 'D': 1.0, 'E': 0.8,
}

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
    """遞迴把所有 nan/inf/numpy 型別轉成 JSON 安全型別"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
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
    pct_height:            Optional[float] = 5.0
    pct_reach:             Optional[float] = 5.0
    pct_striking_power:    Optional[float] = 5.0
    pct_striking_volume:   Optional[float] = 5.0
    pct_striking_accuracy: Optional[float] = 5.0
    pct_striking_defense:  Optional[float] = 5.0
    pct_leg_kicks:         Optional[float] = 5.0
    pct_clinch:            Optional[float] = 5.0
    pct_takedown_offense:  Optional[float] = 5.0
    pct_takedown_defense:  Optional[float] = 5.0
    pct_ground_control:    Optional[float] = 5.0
    pct_submission:        Optional[float] = 5.0
    pct_ground_pound:      Optional[float] = 5.0
    pct_cardio:            Optional[float] = 5.0  # UI 用,不參與 KNN

def _pct_to_raw(pct_col: str, pct_val: float) -> Optional[float]:
    """
    玩家 1-10 分 → 原始數值
    若欄位不在 KNN_FEATURES 回傳 None (呼叫方跳過)
    """
    raw_col = PCT_TO_RAW.get(pct_col)
    if raw_col is None or raw_col not in _feat_means:
        return None
    q      = max(0.01, min(0.99, (pct_val - 1) / 9.0))
    result = BUILDER_DF[raw_col].quantile(q)
    if pd.isna(result):
        return _feat_means[raw_col]
    return float(result)

def _input_to_zvec(inp: BuilderInput) -> np.ndarray:
    """玩家輸入 → 標準化向量,長度 = len(KNN_FEATURES)"""
    vec = []
    for pct_col in PCT_TO_RAW:          # 只遍歷 PCT_TO_RAW (不含 pct_cardio)
        raw_col = PCT_TO_RAW[pct_col]
        if raw_col not in _feat_means:  # 安全檢查
            continue
        pct_val = safe_float(getattr(inp, pct_col, 5.0), 5.0)
        raw_val = _pct_to_raw(pct_col, pct_val)
        if raw_val is None:
            raw_val = _feat_means[raw_col]
        z = (raw_val - _feat_means[raw_col]) / _feat_stds[raw_col]
        vec.append(z)
    arr = np.array(vec, dtype=np.float32)
    # 最終保險:把殘留 nan 換成 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

@app.post("/api/nearest")
def find_nearest(inp: BuilderInput, k: int = 5):
    query = _input_to_zvec(inp)

    # 維度檢查
    if query.shape[0] != _BUILDER_MATRIX.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=f"Vector dim mismatch: query={query.shape[0]}, matrix={_BUILDER_MATRIX.shape[1]}"
        )

    diffs = _BUILDER_MATRIX - query
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    # Tier 加權 KNN
    weighted = []
    for i, (dist, (_, row)) in enumerate(zip(dists, BUILDER_DF.iterrows())):
        tier   = safe_str(row.get('tier_label'), 'E') or 'E'
        weight = TIER_WEIGHTS.get(tier, 1.0)
        weighted.append((float(dist) / weight, i))

    weighted.sort(key=lambda x: x[0])

    # pct 欄位清單 (包含 pct_cardio 供 UI 顯示)
    all_pct_cols = list(PCT_TO_RAW.keys()) + ['pct_cardio']

    results = []
    for w_dist, idx in weighted[:k]:
        row = BUILDER_DF.iloc[idx]
        results.append({
            "name":       safe_str(row.get('name'), 'Unknown'),
            "wc":         safe_str(row.get('wc'), ''),
            "tier_label": safe_str(row.get('tier_label'), 'E') or 'E',
            "tier_score": safe_int(row.get('tier_score'), 0),
            "best_rank":  None if pd.isna(row.get('best_rank')) else safe_float(row['best_rank']),
            "win_rate":   None if pd.isna(row.get('win_rate'))  else round(safe_float(row['win_rate']), 3),
            "distance":   round(safe_float(w_dist), 3),
            "pct": {
                col: round(safe_float(row.get(col), 5.0), 1)
                for col in all_pct_cols
                if col in BUILDER_DF.columns
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
