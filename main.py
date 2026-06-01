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

# ── 載入 Universe 選手資料 ─────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "fighters.json"
with open(DATA_PATH, encoding="utf-8") as f:
    FIGHTERS = json.load(f)
_by_name = {f["name"]: f for f in FIGHTERS}

# ── 載入 Fighter Builder 資料 ─────────────────────────
BUILDER_PATH = Path(__file__).parent / "data" / "clean" / "fighter_builder_features.csv"
BUILDER_DF   = pd.read_csv(BUILDER_PATH)

KNN_FEATURES = [
    'height_cm', 'reach_cm',
    'ko_rate', 'sig_per_r', 'sig_acc', 'str_def',
    'leg_pct', 'clinch_pct',
    'td_per_r', 'td_def', 'ctrl_per_r',
    'sub_rate', 'gnp_per_r', 'gas_tank',
]

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
    'pct_cardio':            'gas_tank',
}

TIER_WEIGHTS = {
    'S': 3.0, 'A+': 2.5, 'A': 2.0,
    'B+': 1.7, 'B': 1.4,
    'C+': 1.2, 'C': 1.1,
    'D+': 1.0, 'D': 1.0, 'E': 0.8,
}

# z-score 標準化參數
_feat_means = {}
_feat_stds  = {}
for col in KNN_FEATURES:
    vals = BUILDER_DF[col].dropna()
    _feat_means[col] = float(vals.mean())
    _feat_stds[col]  = float(vals.std()) if float(vals.std()) > 0 else 1.0

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

# ── 排行榜 ─────────────────────────────────────────────
LEADERBOARD_PATH = Path(__file__).parent / "data" / "leaderboard.json"

def _load_leaderboard():
    if LEADERBOARD_PATH.exists():
        with open(LEADERBOARD_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_leaderboard(data):
    with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── 工具函數 ───────────────────────────────────────────
def safe_float(val, default=0.0):
    """任何值都安全轉成 float,nan/inf 回傳 default"""
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except:
        return default

def safe_int(val, default=0):
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return int(v)
    except:
        return default

def safe_str(val, default=''):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return str(val)

def clean_json(obj):
    """遞迴把所有 nan/inf/numpy 型別轉成 JSON 安全的 Python 型別"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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
    pct_cardio:            Optional[float] = 5.0

def _pct_to_raw(pct_col: str, pct_val: float) -> float:
    raw_col = PCT_TO_RAW[pct_col]
    q = max(0.01, min(0.99, (pct_val - 1) / 9.0))
    return float(BUILDER_DF[raw_col].quantile(q))

def _input_to_zvec(inp: BuilderInput) -> np.ndarray:
    vec = []
    for pct_col, raw_col in PCT_TO_RAW.items():
        pct_val = getattr(inp, pct_col) or 5.0
        raw_val = _pct_to_raw(pct_col, pct_val)
        z = (raw_val - _feat_means[raw_col]) / _feat_stds[raw_col]
        vec.append(z)
    return np.array(vec, dtype=np.float32)


@app.post("/api/nearest")
def find_nearest(inp: BuilderInput, k: int = 5):
    query = _input_to_zvec(inp)

    diffs = _BUILDER_MATRIX - query
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    weighted = []
    for i, (dist, (_, row)) in enumerate(zip(dists, BUILDER_DF.iterrows())):
        tier   = safe_str(row.get('tier_label'), 'E') or 'E'
        weight = TIER_WEIGHTS.get(tier, 1.0)
        weighted.append((float(dist) / weight, i))

    weighted.sort(key=lambda x: x[0])

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
                pct_col: round(safe_float(row.get(pct_col), 5.0), 1)
                for pct_col in PCT_TO_RAW.keys()
            }
        })

    # clean_json 保底:確保所有 numpy 型別和 nan 都被清掉
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
    data = _load_leaderboard()
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
