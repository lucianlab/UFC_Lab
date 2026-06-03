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

PCT_COLS = [
    'pct_reach',
    'pct_striking_power', 'pct_striking_volume',
    'pct_striking_accuracy', 'pct_striking_defense',
    'pct_leg_kicks', 'pct_clinch', 'pct_body',
    'pct_td_frequency', 'pct_td_accuracy', 'pct_td_defense',
    'pct_submission', 'pct_control', 'pct_ground_pound',
]

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

_means = {}
_stds  = {}
for col in KNN_COLS:
    vals = BUILDER_DF[col].dropna()
    _means[col] = float(vals.mean())
    _stds[col]  = float(vals.std()) if float(vals.std()) > 0 else 1.0

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
#  VS Mode — ML 勝率預測
# ══════════════════════════════════════════════════════
import xgboost as xgb

VS_MODEL_PATH  = Path(__file__).parent / "data" / "model_win_v2.json"
VS_COLS_PATH   = Path(__file__).parent / "data" / "feature_cols_v2.json"
BUILDER_FEATS_PATH = Path(__file__).parent / "data" / "clean" / "fighter_builder_features.csv"
FA_PATH        = Path(__file__).parent / "data" / "clean" / "fighters_all.csv"

_vs_model  = None
_vs_cols   = None
_vs_bf     = None  # builder features df
_vs_fa     = None  # fighters_all df

def _load_vs_assets():
    global _vs_model, _vs_cols, _vs_bf, _vs_fa
    if _vs_model is not None:
        return
    _vs_model = xgb.XGBClassifier()
    _vs_model.load_model(str(VS_MODEL_PATH))
    with open(VS_COLS_PATH) as f:
        _vs_cols = json.load(f)
    _vs_bf = pd.read_csv(BUILDER_FEATS_PATH)
    _vs_fa = pd.read_csv(FA_PATH)
    print(f"VS model loaded: {len(_vs_cols)} features")

try:
    _load_vs_assets()
except Exception as e:
    print(f"WARNING: VS model not loaded: {e}")

def _get_fighter_stats(name):
    """從 builder_features 取選手統計，對應 model_win_v2 的特徵"""
    if _vs_bf is None:
        return None
    rows = _vs_bf[_vs_bf['name'] == name]
    if len(rows) == 0:
        return None
    return rows.iloc[0]

def _build_vs_vector(red_name, blue_name):
    """建立 delta 特徵向量"""
    r = _get_fighter_stats(red_name)
    b = _get_fighter_stats(blue_name)
    if r is None or b is None:
        return None, None, None

    fa_r = _vs_fa[_vs_fa['name'] == red_name]
    fa_b = _vs_fa[_vs_fa['name'] == blue_name]

    def fa_val(df, col):
        if len(df) == 0: return np.nan
        v = df.iloc[0].get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    # 建立對應 feature_cols_v2 的特徵 dict
    # 這些欄位從 fighter_builder_features 取，用 pct_ 欄位做差值
    # 對應關係：feature_cols_v2 的 delta_xxx 對應 builder 的原始統計
    stats_map = {
        'win_rate':            ('win_rate', 'win_rate'),
        'ko_rate':             ('pct_striking_power', 'pct_striking_power'),
        'sub_rate':            ('pct_submission', 'pct_submission'),
        'finish_rate':         ('pct_striking_power', 'pct_striking_power'),  # proxy
        'sig_per_r':           ('pct_striking_volume', 'pct_striking_volume'),
        'sig_acc':             ('pct_striking_accuracy', 'pct_striking_accuracy'),
        'str_def':             ('pct_striking_defense', 'pct_striking_defense'),
        'td_per_r':            ('pct_td_frequency', 'pct_td_frequency'),
        'td_acc':              ('pct_td_accuracy', 'pct_td_accuracy'),
        'td_def':              ('pct_td_defense', 'pct_td_defense'),
        'ctrl_per_r':          ('pct_control', 'pct_control'),
        'sub_per_r':           ('pct_submission', 'pct_submission'),
        'ground_pct':          ('pct_ground_pound', 'pct_ground_pound'),
        'gas_tank':            ('pct_striking_volume', 'pct_striking_volume'),  # proxy
        'ctrl_received_per_r': ('pct_td_defense', 'pct_td_defense'),  # proxy
        'sos':                 ('win_rate', 'win_rate'),  # proxy
        'sos_td_def':          ('pct_td_defense', 'pct_td_defense'),
        'recent_win_rate':     ('win_rate', 'win_rate'),
        'recent_finish_rate':  ('pct_striking_power', 'pct_striking_power'),
        'recent_sig_per_r':    ('pct_striking_volume', 'pct_striking_volume'),
        'recent_td_per_r':     ('pct_td_frequency', 'pct_td_frequency'),
        'win_streak':          ('win_rate', 'win_rate'),
        'lose_streak':         ('win_rate', 'win_rate'),
        'days_since_last':     ('win_rate', 'win_rate'),
        'modal_shift':         ('pct_ground_pound', 'pct_ground_pound'),
        'age':                 ('win_rate', 'win_rate'),
    }

    vec = {}
    for col in _vs_cols:
        if col.startswith('delta_'):
            key = col[6:]  # remove 'delta_'
            if key in stats_map:
                r_col, b_col = stats_map[key]
                rv = safe_float(r.get(r_col), 5.0)
                bv = safe_float(b.get(b_col), 5.0)
                vec[col] = rv - bv
            elif key == 'in_prime':
                vec[col] = 0.0
            elif key == 'past_prime':
                vec[col] = 0.0
            elif key == 'reach':
                vec[col] = fa_val(fa_r, 'reach_cm') - fa_val(fa_b, 'reach_cm')
                if np.isnan(vec[col]): vec[col] = 0.0
            elif key == 'height':
                vec[col] = fa_val(fa_r, 'height_cm') - fa_val(fa_b, 'height_cm')
                if np.isnan(vec[col]): vec[col] = 0.0
            else:
                vec[col] = 0.0
        elif col == 'southpaw_matchup':
            r_stance = str(fa_val(fa_r, 'stance') or '')
            b_stance = str(fa_val(fa_b, 'stance') or '')
            vec[col] = float((r_stance == 'Southpaw') != (b_stance == 'Southpaw'))
        elif col == 'same_stance':
            r_stance = str(fa_val(fa_r, 'stance') or '')
            b_stance = str(fa_val(fa_b, 'stance') or '')
            vec[col] = float(r_stance == b_stance)
        elif col.startswith('f1_') or col.startswith('f2_'):
            # 風格克制乘積項
            if col == 'f1_td_threat_vs_f2_def':
                vec[col] = safe_float(r.get('pct_td_frequency'), 5.0) * safe_float(b.get('pct_td_defense'), 5.0)
            elif col == 'f2_td_threat_vs_f1_def':
                vec[col] = safe_float(b.get('pct_td_frequency'), 5.0) * safe_float(r.get('pct_td_defense'), 5.0)
            elif col == 'f1_str_threat_vs_f2_def':
                vec[col] = safe_float(r.get('pct_striking_volume'), 5.0) * safe_float(b.get('pct_striking_defense'), 5.0)
            elif col == 'f2_str_threat_vs_f1_def':
                vec[col] = safe_float(b.get('pct_striking_volume'), 5.0) * safe_float(r.get('pct_striking_defense'), 5.0)
            elif col == 'f1_sub_vs_f2_ctrl':
                vec[col] = safe_float(r.get('pct_submission'), 5.0) * safe_float(b.get('pct_control'), 5.0)
            elif col == 'f1_ko_vs_f2_absorb':
                vec[col] = safe_float(r.get('pct_striking_power'), 5.0) * safe_float(b.get('pct_striking_defense'), 5.0)
            else:
                vec[col] = 0.0
        else:
            vec[col] = 0.0

    X = pd.DataFrame([vec])[_vs_cols]
    X = X.fillna(0)
    return X, r, b

class VsInput(BaseModel):
    red:  str
    blue: str

@app.post("/api/vs_predict")
def vs_predict(inp: VsInput):
    if _vs_model is None:
        raise HTTPException(status_code=503, detail="VS model not loaded")

    X, r_row, b_row = _build_vs_vector(inp.red, inp.blue)
    if X is None:
        raise HTTPException(status_code=404,
            detail=f"Fighter not found in builder features")

    prob = float(_vs_model.predict_proba(X)[0][1])

    # SHAP top 5 features
    try:
        import shap
        explainer = shap.TreeExplainer(_vs_model)
        shap_vals = explainer.shap_values(X)
        sv = shap_vals[0]
        feat_shap = sorted(zip(_vs_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features = [{"feature": f, "shap": round(float(v), 4)} for f, v in feat_shap]
    except Exception:
        top_features = []

    # 量級不同警告
    r_wc = safe_str(r_row.get('wc'), '') if r_row is not None else ''
    b_wc = safe_str(b_row.get('wc'), '') if b_row is not None else ''
    cross_wc = (r_wc != b_wc and r_wc != '' and b_wc != '')

    return clean_json({
        "red_win_prob":  round(prob, 3),
        "blue_win_prob": round(1 - prob, 3),
        "top_features":  top_features,
        "cross_wc":      cross_wc,
    })

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
    if isinstance(obj, dict):       return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):       return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):   return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

def _row_to_result(row, dist):
    return {
        "name":       safe_str(row.get('name'), 'Unknown'),
        "wc":         safe_str(row.get('wc'), ''),
        "tier_label": safe_str(row.get('tier_label'), 'E') or 'E',
        "tier_score": safe_int(row.get('tier_score'), 0),
        "best_rank":  None if pd.isna(row.get('best_rank')) else safe_float(row['best_rank']),
        "win_rate":   None if pd.isna(row.get('win_rate'))  else round(safe_float(row['win_rate']), 3),
        "distance":   round(float(dist), 3),
        "pct": {
            col: round(safe_float(row.get(col), 5.0), 1)
            for col in KNN_COLS
        }
    }

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
    vec = []
    for col in KNN_COLS:
        pct_val = safe_float(getattr(inp, col, 5.0), 5.0)
        pct_val = max(1.0, min(10.0, pct_val))
        z = (pct_val - _means[col]) / _stds[col]
        vec.append(z)
    arr = np.array(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

@app.post("/api/nearest")
def find_nearest(inp: BuilderInput):
    query = _input_to_zvec(inp)

    if query.shape[0] != _MATRIX.shape[1]:
        raise HTTPException(status_code=500,
            detail=f"Vector dim mismatch: query={query.shape[0]}, matrix={_MATRIX.shape[1]}")

    diffs = _MATRIX - query
    dists = np.sqrt((diffs ** 2).sum(axis=1))

    # ── 你的戰士：純距離最近的 1 個，不管 tier ────────
    closest_idx  = int(np.argmin(dists))
    your_fighter = _row_to_result(BUILDER_DF.iloc[closest_idx], dists[closest_idx])

    # ── 最近知名選手：top30 裡 tier 最高的 2 個 ───────
    # 排除已經是「你的戰士」的那個
    top30_idx = np.argsort(dists)[:30]
    notable_candidates = []
    for idx in top30_idx:
        if idx == closest_idx:
            continue
        row = BUILDER_DF.iloc[idx]
        tier  = safe_str(row.get('tier_label'), 'E') or 'E'
        score = safe_int(row.get('tier_score'), 0)
        # 只考慮 B tier 以上
        if tier in ('D+', 'D', 'E'):
            continue
        notable_candidates.append({
            'idx':   int(idx),
            'score': score,
            'dist':  float(dists[idx]),
            'tier':  tier,
        })

    # 按 tier_score 降序，同分按距離升序
    notable_candidates.sort(key=lambda x: (-x['score'], x['dist']))
    notable = []
    for c in notable_candidates[:2]:
        notable.append(_row_to_result(BUILDER_DF.iloc[c['idx']], c['dist']))

    return clean_json({
        "your_fighter": your_fighter,
        "notable":      notable,
    })

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
