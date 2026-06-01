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

# ── 載入 Universe 選手資料 ──────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "fighters.json"
with open(DATA_PATH, encoding="utf-8") as f:
    FIGHTERS = json.load(f)
_by_name = {f["name"]: f for f in FIGHTERS}

# ── 載入 Fighter Builder 資料 ──────────────────────────
BUILDER_PATH = Path(__file__).parent / "data" / "clean" / "fighter_builder_features.csv"
BUILDER_DF   = pd.read_csv(BUILDER_PATH)

# KNN 用的 raw feature 欄位 (14個,順序固定)
KNN_FEATURES = [
    'height_cm', 'reach_cm',
    'ko_rate', 'sig_per_r', 'sig_acc', 'str_def',
    'leg_pct', 'clinch_pct',
    'td_per_r', 'td_def', 'ctrl_per_r',
    'sub_rate', 'gnp_per_r', 'gas_tank',
]

# 預先計算每個 feature 的 mean/std 用於 z-score 標準化
# 標準化確保不同單位的 feature 有相同權重
_feat_means = {}
_feat_stds  = {}
for col in KNN_FEATURES:
    vals = BUILDER_DF[col].dropna()
    _feat_means[col] = float(vals.mean())
    _feat_stds[col]  = float(vals.std()) if float(vals.std()) > 0 else 1.0

# 預先把 builder df 標準化成 numpy array (加速 KNN)
def _build_matrix():
    rows = []
    for _, row in BUILDER_DF.iterrows():
        vec = []
        for col in KNN_FEATURES:
            val = row[col] if pd.notna(row[col]) else _feat_means[col]
            vec.append((val - _feat_means[col]) / _feat_stds[col])
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

_BUILDER_MATRIX = _build_matrix()

# tier 加權:知名選手有更大的「引力」
TIER_WEIGHTS = {
    'S': 3.0, 'A+': 2.5, 'A': 2.0,
    'B+': 1.7, 'B': 1.4,
    'C+': 1.2, 'C': 1.1,
    'D+': 1.0, 'D': 1.0, 'E': 0.8,
}

# ── 排行榜儲存 ─────────────────────────────────────────
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
#  Fighter Builder 端點
# ══════════════════════════════════════════════════════

class BuilderInput(BaseModel):
    # 玩家輸入的 14 個 feature,值為 1-10 的百分位分數
    # 未提供的 feature 用 5 (中位數) 填補
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


# pct(1-10) → raw value 的對應欄位
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

def _pct_to_raw_value(pct_col: str, pct_val: float) -> float:
    """
    玩家給的 1-10 百分位分數 → 原始數值
    用 quantile 從 builder df 反查對應的真實數值
    """
    raw_col = PCT_TO_RAW[pct_col]
    q = (pct_val - 1) / 9.0   # 1→0.0, 10→1.0
    q = max(0.01, min(0.99, q))
    return float(BUILDER_DF[raw_col].quantile(q))

def _input_to_zvec(inp: BuilderInput) -> np.ndarray:
    """玩家輸入 → 標準化向量 (14維)"""
    vec = []
    for pct_col, raw_col in PCT_TO_RAW.items():
        pct_val = getattr(inp, pct_col) or 5.0
        raw_val = _pct_to_raw_value(pct_col, pct_val)
        z = (raw_val - _feat_means[raw_col]) / _feat_stds[raw_col]
        vec.append(z)
    return np.array(vec, dtype=np.float32)

def safe_float(val, default=5.0):
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except:
        return default

@app.post("/api/nearest")
def find_nearest(inp: BuilderInput, k: int = 5):
    """
    輸入: 14個百分位分數 (1-10)
    輸出: 最近的 k 個真實選手 + 加權距離
    加權KNN: tier越高的選手引力越大
    """
    query = _input_to_zvec(inp)

    # 計算所有選手的歐氏距離
    diffs = _BUILDER_MATRIX - query          # (N, 14)
    dists = np.sqrt((diffs ** 2).sum(axis=1))  # (N,)

    # 加權:距離 / tier_weight → tier 高的選手等效更近
    weighted = []
    for i, (dist, (_, row)) in enumerate(zip(dists, BUILDER_DF.iterrows())):
        tier   = row.get('tier_label', 'E') or 'E'
        weight = TIER_WEIGHTS.get(tier, 1.0)
        weighted.append((float(dist) / weight, i))

    weighted.sort(key=lambda x: x[0])
    top_k = weighted[:k]

    results = []
    for w_dist, idx in top_k:
        row = BUILDER_DF.iloc[idx]
        results.append({
            "name":        row['name'],
            "wc":          row.get('wc', ''),
            "tier_label":  row.get('tier_label', 'E'),
            "tier_score":  int(row.get('tier_score', 0)),
            "best_rank":   None if pd.isna(row.get('best_rank')) else float(row['best_rank']),
            "win_rate":    None if pd.isna(row.get('win_rate'))  else round(float(row['win_rate']), 3),
            "distance":    round(w_dist, 3),
            # pct features 方便前端顯示雷達圖
            "pct": {
                pct_col: round(safe_float(row.get(pct_col), 5.0), 1)
                for pct_col in PCT_TO_RAW.keys()
            }
        })

    return {"matches": results}


# ══════════════════════════════════════════════════════
#  排行榜端點
# ══════════════════════════════════════════════════════

class LeaderboardEntry(BaseModel):
    username:   str
    tier_score: int        # 找到的最近選手的 tier_score
    nearest:    str        # 最近選手名稱
    points_used: int       # Compete mode: 用了幾點
    features:   dict       # 玩家的 14 個 feature 值


@app.get("/api/leaderboard")
def get_leaderboard():
    data = _load_leaderboard()
    # 按 tier_score 降序,同分按 points_used 升序
    data.sort(key=lambda x: (-x['tier_score'], x.get('points_used', 999)))
    return data[:50]   # 只回傳前 50


@app.post("/api/leaderboard")
def post_leaderboard(entry: LeaderboardEntry):
    data = _load_leaderboard()

    # 同一個 username 只保留最高分
    existing = next((i for i, e in enumerate(data)
                     if e['username'] == entry.username), None)

    new_entry = entry.dict()

    if existing is not None:
        if (entry.tier_score > data[existing]['tier_score'] or
            (entry.tier_score == data[existing]['tier_score'] and
             entry.points_used < data[existing].get('points_used', 999))):
            data[existing] = new_entry
    else:
        data.append(new_entry)

    _save_leaderboard(data)
    return {"ok": True, "entries": len(data)}
