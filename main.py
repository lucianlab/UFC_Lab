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
#  VS Mode — ML 預測（勝負 + 方式 + 回合）
# ══════════════════════════════════════════════════════
import xgboost as xgb

VS_WIN_PATH    = Path(__file__).parent / "data" / "model_win_v2.json"
VS_METHOD_PATH = Path(__file__).parent / "data" / "model_method_v2.json"
VS_ROUND_PATH  = Path(__file__).parent / "data" / "model_round_v2.json"
VS_COLS_PATH   = Path(__file__).parent / "data" / "feature_cols_v2.json"
BUILDER_FEATS_PATH = Path(__file__).parent / "data" / "clean" / "fighter_builder_features.csv"
FA_PATH        = Path(__file__).parent / "data" / "clean" / "fighters_all.csv"

_vs_win    = None
_vs_method = None
_vs_round  = None
_vs_cols   = None
_vs_bf     = None
_vs_fa     = None

def _load_vs_assets():
    global _vs_win, _vs_method, _vs_round, _vs_cols, _vs_bf, _vs_fa
    if _vs_win is not None:
        return
    _vs_win = xgb.XGBClassifier()
    _vs_win.load_model(str(VS_WIN_PATH))
    _vs_method = xgb.XGBClassifier()
    _vs_method.load_model(str(VS_METHOD_PATH))
    _vs_round = xgb.XGBClassifier()
    _vs_round.load_model(str(VS_ROUND_PATH))
    with open(VS_COLS_PATH) as f:
        _vs_cols = json.load(f)
    _vs_bf = pd.read_csv(BUILDER_FEATS_PATH)
    _vs_fa = pd.read_csv(FA_PATH)
    print(f"VS models loaded: win/method/round, {len(_vs_cols)} features")

try:
    _load_vs_assets()
except Exception as e:
    print(f"WARNING: VS models not loaded: {e}")

def _get_fighter_stats(name):
    if _vs_bf is None: return None
    rows = _vs_bf[_vs_bf['name'] == name]
    return rows.iloc[0] if len(rows) > 0 else None

def _build_vs_vector(red_name, blue_name):
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

    stats_map = {
        'win_rate':            ('win_rate', 'win_rate'),
        'ko_rate':             ('pct_striking_power', 'pct_striking_power'),
        'sub_rate':            ('pct_submission', 'pct_submission'),
        'finish_rate':         ('pct_striking_power', 'pct_striking_power'),
        'sig_per_r':           ('pct_striking_volume', 'pct_striking_volume'),
        'sig_acc':             ('pct_striking_accuracy', 'pct_striking_accuracy'),
        'str_def':             ('pct_striking_defense', 'pct_striking_defense'),
        'td_per_r':            ('pct_td_frequency', 'pct_td_frequency'),
        'td_acc':              ('pct_td_accuracy', 'pct_td_accuracy'),
        'td_def':              ('pct_td_defense', 'pct_td_defense'),
        'ctrl_per_r':          ('pct_control', 'pct_control'),
        'sub_per_r':           ('pct_submission', 'pct_submission'),
        'ground_pct':          ('pct_ground_pound', 'pct_ground_pound'),
        'gas_tank':            ('pct_striking_volume', 'pct_striking_volume'),
        'ctrl_received_per_r': ('pct_td_defense', 'pct_td_defense'),
        'sos':                 ('win_rate', 'win_rate'),
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
            key = col[6:]
            if key in stats_map:
                rv = safe_float(r.get(stats_map[key][0]), 5.0)
                bv = safe_float(b.get(stats_map[key][1]), 5.0)
                vec[col] = rv - bv
            elif key in ('in_prime', 'past_prime'):
                vec[col] = 0.0
            elif key == 'reach':
                v = fa_val(fa_r, 'reach_cm') - fa_val(fa_b, 'reach_cm')
                vec[col] = 0.0 if np.isnan(v) else v
            elif key == 'height':
                v = fa_val(fa_r, 'height_cm') - fa_val(fa_b, 'height_cm')
                vec[col] = 0.0 if np.isnan(v) else v
            else:
                vec[col] = 0.0
        elif col == 'southpaw_matchup':
            rs = str(fa_val(fa_r, 'stance') or '')
            bs = str(fa_val(fa_b, 'stance') or '')
            vec[col] = float((rs == 'Southpaw') != (bs == 'Southpaw'))
        elif col == 'same_stance':
            rs = str(fa_val(fa_r, 'stance') or '')
            bs = str(fa_val(fa_b, 'stance') or '')
            vec[col] = float(rs == bs)
        elif col == 'delta_str_penetration':
            r_str  = safe_float(r.get('sig_per_r'), 5.0)
            b_sdef = safe_float(b.get('str_def'), 0.5)
            b_str  = safe_float(b.get('sig_per_r'), 5.0)
            r_sdef = safe_float(r.get('str_def'), 0.5)
            f1_sp  = r_str * (1 - b_sdef)
            f2_sp  = b_str * (1 - r_sdef)
            vec[col] = f1_sp - f2_sp
        elif col == 'delta_td_penetration':
            r_td   = safe_float(r.get('td_per_r'), 1.0)
            b_tdef = safe_float(b.get('td_def'), 0.5)
            b_td   = safe_float(b.get('td_per_r'), 1.0)
            r_tdef = safe_float(r.get('td_def'), 0.5)
            f1_tp  = r_td * (1 - b_tdef)
            f2_tp  = b_td * (1 - r_tdef)
            vec[col] = f1_tp - f2_tp
        elif col == 'delta_str_vs_td':
            r_str  = safe_float(r.get('sig_per_r'), 5.0)
            b_tdef = safe_float(b.get('td_def'), 0.5)
            b_str  = safe_float(b.get('sig_per_r'), 5.0)
            r_tdef = safe_float(r.get('td_def'), 0.5)
            f1     = r_str * (1 - b_tdef)
            f2     = b_str * (1 - r_tdef)
            vec[col] = f1 - f2
        elif col == 'f1_td_threat_vs_f2_def':
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

    X = pd.DataFrame([vec])[_vs_cols].fillna(0)
    return X, r, b

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
    if isinstance(obj, dict):        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):    return bool(obj)
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
#  VS Mode endpoint
# ══════════════════════════════════════════════════════
class VsInput(BaseModel):
    red:  str
    blue: str

@app.post("/api/vs_predict")
def vs_predict(inp: VsInput):
    if _vs_win is None:
        raise HTTPException(status_code=503, detail="VS models not loaded")

    X_fwd, r_row, b_row = _build_vs_vector(inp.red,  inp.blue)
    X_rev, _,     _     = _build_vs_vector(inp.blue, inp.red)

    if X_fwd is None or X_rev is None:
        raise HTTPException(status_code=404,
            detail=f"Fighter not found: {inp.red} or {inp.blue}")

    # 對稱化：兩方向平均
    win_fwd  = float(_vs_win.predict_proba(X_fwd)[0][1])
    win_rev  = float(_vs_win.predict_proba(X_rev)[0][0])
    win_prob = (win_fwd + win_rev) / 2

    method_probs_fwd = _vs_method.predict_proba(X_fwd)[0]
    method_probs_rev = _vs_method.predict_proba(X_rev)[0]
    method_probs     = (method_probs_fwd + method_probs_rev[::-1][:len(method_probs_fwd)]) / 2
    method_idx       = int(np.argmax(method_probs))
    method_map       = {0: "Decision", 1: "KO/TKO", 2: "Submission"}

    round_probs_fwd  = _vs_round.predict_proba(X_fwd)[0]
    round_probs_rev  = _vs_round.predict_proba(X_rev)[0]
    round_probs      = (round_probs_fwd + round_probs_rev) / 2
    pred_round       = int(np.argmax(round_probs)) + 1

    win_sig    = abs(win_prob - 0.5) * 2
    method_sig = float(np.max(method_probs))
    round_sig  = float(np.max(round_probs))
    confidence = round(win_sig * 0.5 + method_sig * 0.3 + round_sig * 0.2, 3)
    if   confidence >= 0.65: confidence_label = "HIGH"
    elif confidence >= 0.50: confidence_label = "MODERATE"
    else:                    confidence_label = "LOW"

    # SHAP
    try:
        import shap
        explainer  = shap.TreeExplainer(_vs_win)
        shap_vals  = explainer.shap_values(X_fwd)
        sv         = shap_vals[0]
        feat_shap  = sorted(zip(_vs_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features = [{"feature": f, "shap": round(float(v), 4)} for f, v in feat_shap]
    except Exception:
        top_features = []

    r_wc   = safe_str(r_row.get('wc'), '') if r_row is not None else ''
    b_wc   = safe_str(b_row.get('wc'), '') if b_row is not None else ''
    cross_wc = (r_wc != b_wc and r_wc != '' and b_wc != '')

    return clean_json({
        "red_win_prob":     round(win_prob, 3),
        "blue_win_prob":    round(1 - win_prob, 3),
        "method":           method_map[method_idx],
        "method_probs":     {"decision": round(method_probs[0], 3),
                             "ko":       round(method_probs[1], 3),
                             "sub":      round(method_probs[2], 3)},
        "pred_round":       pred_round,
        "confidence":       confidence,
        "confidence_label": confidence_label,
        "top_features":     top_features,
        "cross_wc":         cross_wc,
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

    closest_idx  = int(np.argmin(dists))
    your_fighter = _row_to_result(BUILDER_DF.iloc[closest_idx], dists[closest_idx])

    top30_idx = np.argsort(dists)[:30]
    notable_candidates = []
    for idx in top30_idx:
        if idx == closest_idx:
            continue
        row   = BUILDER_DF.iloc[idx]
        tier  = safe_str(row.get('tier_label'), 'E') or 'E'
        score = safe_int(row.get('tier_score'), 0)
        if tier in ('D+', 'D', 'E'):
            continue
        notable_candidates.append({
            'idx':   int(idx),
            'score': score,
            'dist':  float(dists[idx]),
            'tier':  tier,
        })

    notable_candidates.sort(key=lambda x: (-x['score'], x['dist']))
    notable = []
    for c in notable_candidates[:2]:
        notable.append(_row_to_result(BUILDER_DF.iloc[c['idx']], c['dist']))

    return clean_json({
        "your_fighter": your_fighter,
        "notable":      notable,
    })

# ══════════════════════════════════════════════════════
#  排行榜端點
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

# ══════════════════════════════════════════════════════
#  DNA Mode — 三個靜態資料端點
# ══════════════════════════════════════════════════════
DNA_YEARLY_PATH   = Path(__file__).parent / "data" / "dna_yearly.csv"
DNA_FIGHTERS_PATH = Path(__file__).parent / "data" / "dna_fighters.csv"
DNA_CHAMPS_PATH   = Path(__file__).parent / "data" / "dna_champions.csv"

def _load_dna():
    result = {}

    # yearly
    try:
        df = pd.read_csv(DNA_YEARLY_PATH)
        records = []
        for _, row in df.iterrows():
            records.append({
                "year":          int(row["year"]),
                "n_fighters":    int(row["n_fighters"]),
                "pct_striking":  safe_float(row.get("pct_striking"),  0),
                "pct_grappling": safe_float(row.get("pct_grappling"), 0),
                "pct_balanced":  safe_float(row.get("pct_balanced"),  0),
                "str_winrate":   None if pd.isna(row.get("str_winrate"))  else safe_float(row["str_winrate"],  0),
                "grp_winrate":   None if pd.isna(row.get("grp_winrate"))  else safe_float(row["grp_winrate"],  0),
                "winrate_diff":  None if pd.isna(row.get("winrate_diff")) else safe_float(row["winrate_diff"], 0),
                "is_partial":    bool(row.get("is_partial", False)),
            })
        result["yearly"] = records
        print(f"DNA yearly loaded: {len(records)} years")
    except Exception as e:
        print(f"WARNING: DNA yearly not loaded: {e}")
        result["yearly"] = []

    # fighters
    try:
        df = pd.read_csv(DNA_FIGHTERS_PATH)
        records = []
        for _, row in df.iterrows():
            records.append({
                "fighter":          safe_str(row.get("fighter"), ""),
                "wc":               safe_str(row.get("wc"), ""),
                "tier_label":       safe_str(row.get("tier_label"), "E"),
                "ever_champion":    bool(row.get("ever_champion", False)),
                "win_rate":         None if pd.isna(row.get("win_rate")) else safe_float(row["win_rate"], 0),
                "meta_score_pct":   safe_float(row.get("meta_score_pct"), 50),
                "striking_signal":  safe_float(row.get("striking_signal"), 0),
                "grappling_signal": safe_float(row.get("grappling_signal"), 0),
            })
        result["fighters"] = records
        print(f"DNA fighters loaded: {len(records)} fighters")
    except Exception as e:
        print(f"WARNING: DNA fighters not loaded: {e}")
        result["fighters"] = []

    # champions
    try:
        df = pd.read_csv(DNA_CHAMPS_PATH)
        records = []
        for _, row in df.iterrows():
            if pd.isna(row.get("title_year")):
                continue
            records.append({
                "champion":         safe_str(row.get("champion"), ""),
                "wc":               safe_str(row.get("wc"), ""),
                "title_year":       int(row["title_year"]),
                "meta_score_pct":   safe_float(row.get("meta_score_pct"), 50),
                "title_fights_won": safe_int(row.get("title_fights_won"), 1),
                "radius":           safe_float(row.get("radius"), 6),
                "n_reigns":         safe_int(row.get("n_reigns"), 1),
            })
        result["champions"] = records
        print(f"DNA champions loaded: {len(records)} champions")
    except Exception as e:
        print(f"WARNING: DNA champions not loaded: {e}")
        result["champions"] = []

    return result

_DNA = _load_dna()

@app.get("/api/dna/yearly")
def get_dna_yearly():
    return _DNA.get("yearly", [])

@app.get("/api/dna/fighters")
def get_dna_fighters():
    return _DNA.get("fighters", [])

@app.get("/api/dna/champions")
def get_dna_champions():
    return _DNA.get("champions", [])


# ══════════════════════════════════════════════════════
#  HELIX — Graph 資料端點
# ══════════════════════════════════════════════════════
HELIX_PATH = Path(__file__).parent / "data" / "helix_graph.json"
_HELIX = None

def _load_helix():
    global _HELIX
    if not HELIX_PATH.exists():
        print("WARNING: helix_graph.json 不存在，跳過")
        return
    with open(HELIX_PATH, encoding="utf-8") as f:
        _HELIX = json.load(f)
    print(f"HELIX loaded: {_HELIX['meta']['nodes']} nodes, {_HELIX['meta']['edges']} edges")

try:
    _load_helix()
except Exception as e:
    print(f"WARNING: HELIX not loaded: {e}")

@app.get("/api/helix/meta")
def get_helix_meta():
    """Graph 的 meta 資訊（nodes/edges 數量等）"""
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")
    return _HELIX["meta"]

@app.get("/api/helix/graph")
def get_helix_graph(ranked_only: bool = True):
    """
    取得 graph 資料
    ranked_only=true  → 只回傳曾排名的 node（預設，前端常駐）
    ranked_only=false → 全部 node
    edge 只回傳兩端 node 都在回傳集合內的
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    nodes = _HELIX["nodes"]
    edges = _HELIX["edges"]

    if ranked_only:
        nodes = [n for n in nodes if n.get("is_ranked")]

    node_ids = {n["id"] for n in nodes}
    edges = [
        e for e in edges
        if e["winner"] in node_ids and e["loser"] in node_ids
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }

@app.get("/api/helix/fighter/{name}")
def get_helix_fighter(name: str, depth: int = 1):
    """
    取得某選手的局部 graph（鄰居展開）
    depth=1 → 直接對手（預設）
    depth=2 → 對手的對手
    回傳包含 unranked 的完整鄰居
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    # 建立快速查詢 dict
    node_map = {n["id"]: n for n in _HELIX["nodes"]}

    if name not in node_map:
        raise HTTPException(status_code=404, detail=f"Fighter '{name}' not found in graph")

    # BFS 展開到指定 depth
    visited = {name}
    frontier = {name}

    all_edges = _HELIX["edges"]
    # 建立adjacency（雙向方便BFS）
    adj = {}
    for e in all_edges:
        adj.setdefault(e["winner"], []).append(e["loser"])
        adj.setdefault(e["loser"], []).append(e["winner"])

    for _ in range(min(depth, 2)):  # 最多 depth=2 避免爆量
        next_frontier = set()
        for node in frontier:
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        frontier = next_frontier

    # 收集相關 nodes 和 edges
    result_nodes = [node_map[n] for n in visited if n in node_map]
    result_edges = [
        e for e in all_edges
        if e["winner"] in visited and e["loser"] in visited
    ]

    # beat chain：從 name 出發，只取出邊（他打贏的人）
    beat_chain = []
    for e in all_edges:
        if e["winner"] == name:
            beat_chain.append({
                "opponent": e["loser"],
                "bouts":    e["bouts"],
                "count":    e["count"],
                "weight":   e["weight"],
            })
    beat_chain.sort(key=lambda x: -x["weight"])

    # loss chain：輸給誰
    loss_chain = []
    for e in all_edges:
        if e["loser"] == name:
            loss_chain.append({
                "opponent": e["winner"],
                "bouts":    e["bouts"],
                "count":    e["count"],
            })

    return {
        "fighter":    node_map[name],
        "nodes":      result_nodes,
        "edges":      result_edges,
        "beat_chain": beat_chain,
        "loss_chain": loss_chain,
    }

@app.get("/api/helix/top")
def get_helix_top(n: int = 50, wc: str = None):
    """
    PageRank 前 n 名的選手 + 彼此之間的 edge
    wc: 按量級過濾（可選）
    """
    if _HELIX is None:
        raise HTTPException(status_code=503, detail="HELIX data not available")

    nodes = sorted(_HELIX["nodes"], key=lambda x: -x["pr"])

    if wc and wc.lower() != "all":
        # 用 beat_chain 的 wc 資訊過濾 edge，node 本身沒有固定量級
        # 這裡先回傳全量級的 top n，wc filter 留給前端
        pass

    top_nodes = nodes[:n]
    top_ids   = {nd["id"] for nd in top_nodes}

    top_edges = [
        e for e in _HELIX["edges"]
        if e["winner"] in top_ids and e["loser"] in top_ids
    ]

    return {
        "nodes":        top_nodes,
        "edges":        top_edges,
        "community_leaders": _HELIX.get("community_leaders", []),
    }


# ══════════════════════════════════════════════════════
#  Gemini 2.0 Flash — VS Narrative
# ══════════════════════════════════════════════════════
import os

# SHAP feature → 可讀描述
SHAP_LABELS = {
    "delta_win_rate":          "win rate gap",
    "delta_ko_rate":           "KO rate gap",
    "delta_sub_rate":          "submission rate gap",
    "delta_finish_rate":       "finishing rate gap",
    "delta_sig_per_r":         "striking volume gap",
    "delta_sig_acc":           "striking accuracy gap",
    "delta_str_def":           "striking defense gap",
    "delta_td_per_r":          "takedown volume gap",
    "delta_td_avg_acc":        "takedown accuracy gap",
    "delta_td_def":            "takedown defense gap",
    "delta_ctrl_per_r":        "control time gap",
    "delta_sub_per_r":         "submission attempt gap",
    "delta_ground_pct":        "ground game gap",
    "delta_gas_tank":          "cardio gap",
    "delta_sos":               "strength of schedule gap",
    "delta_recent_win_rate":   "recent form gap",
    "delta_recent_sig_per_r":  "recent striking gap",
    "delta_age":               "age gap",
    "delta_reach":             "reach gap",
    "f1_td_threat_vs_f2_def":  "red's takedown threat vs blue's defense",
    "f2_td_threat_vs_f1_def":  "blue's takedown threat vs red's defense",
    "f1_str_threat_vs_f2_def": "red's striking threat vs blue's defense",
    "f2_str_threat_vs_f1_def": "blue's striking threat vs red's defense",
    "f1_sub_vs_f2_ctrl":       "red's submission vs blue's control",
    "f1_ko_vs_f2_absorb":      "red's KO power vs blue's durability",
}

def _shap_to_english(features: list, red_name: str, blue_name: str) -> str:
    parts = []
    for f in features[:3]:
        feat  = f.get("feature", "")
        shap  = f.get("shap", 0)
        label = SHAP_LABELS.get(feat, feat.replace("delta_","").replace("_"," "))
        favor = red_name.split()[-1] if shap > 0 else blue_name.split()[-1]
        parts.append(f"{favor} leads in {label}")
    return "; ".join(parts)

def _get_archetype(name: str) -> str:
    f = _by_name.get(name, {})
    return f.get("archetype", "Balanced")

def _fmt_stats(name: str) -> str:
    f = _by_name.get(name, {})
    sig  = f.get("sig_per_r", 0) or 0
    td   = f.get("td_per_r",  0) or 0
    ko   = round((f.get("ko_rate",  0) or 0) * 100)
    sub  = round((f.get("sub_rate", 0) or 0) * 100)
    sdef = round((f.get("str_def",  0) or 0) * 100)
    tdef = round((f.get("td_def",   0) or 0) * 100)
    return (f"{sig:.1f} sig.strikes/rd, {td:.1f} TDs/rd, "
            f"{ko}% KO rate, {sub}% sub rate, "
            f"{sdef}% str.def, {tdef}% TD def")

class NarrativeInput(BaseModel):
    red:              str
    blue:             str
    red_win_prob:     float
    method:           str
    pred_round:       int
    confidence_label: str
    top_features:     list = []

@app.post("/api/vs_narrative")
async def vs_narrative(inp: NarrativeInput):
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return {"narrative": ""}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        winner     = inp.red  if inp.red_win_prob >= 0.5 else inp.blue
        loser      = inp.blue if inp.red_win_prob >= 0.5 else inp.red
        win_pct    = round(inp.red_win_prob * 100) if inp.red_win_prob >= 0.5 else round((1-inp.red_win_prob)*100)
        method_str = inp.method
        round_str  = f"R{inp.pred_round}" if inp.method != "Decision" else "decision"
        shap_str   = _shap_to_english(inp.top_features, inp.red, inp.blue)
        red_arch   = _get_archetype(inp.red)
        blue_arch  = _get_archetype(inp.blue)
        red_stats  = _fmt_stats(inp.red)
        blue_stats = _fmt_stats(inp.blue)

        red_desc  = ARCHETYPE_DESC.get(red_arch,  red_arch)
        blue_desc = ARCHETYPE_DESC.get(blue_arch, blue_arch)
        method_human = {"KO/TKO": "knockout", "Submission": "submission", "Decision": "decision"}.get(method_str, method_str)

        prompt = (
            "You are a passionate UFC analyst writing a pre-fight breakdown. "
            "Write 3-4 sentences with genuine enthusiasm for the sport. "
            "Do NOT mention specific past fights or career history. Analyze only from the data below.\n\n"
            f"FIGHTERS:\n"
            f"- {inp.red}: {red_desc}. Stats: {red_stats}\n"
            f"- {inp.blue}: {blue_desc}. Stats: {blue_stats}\n\n"
            f"MODEL OUTPUT: {winner} wins - {win_pct}% probability, predicted {method_human} ({round_str})\n"
            f"KEY FACTORS THE MODEL IS WEIGHTING: {shap_str}\n\n"
            "Write a flowing 3-4 sentence fight breakdown. "
            "First sentence: describe the stylistic matchup in vivid, human terms - what kind of fight does this set up? "
            "Second sentence: each fighter's path to victory based on their style and the key factors. "
            "Third sentence: what the model sees in the numbers that drives its prediction. "
            "Optional fourth sentence: the most likely decisive moment or dimension. "
            "Use fighter last names only. Write with the energy of someone who loves this sport "
            "- flowing prose, not bullet points, no hedging."
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=220,
                temperature=0.7,
            )
        )
        text = response.text.strip().replace("\n", " ")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        result = ". ".join(sentences[:4]) + ("." if sentences else "")
        return {"narrative": result}

    except Exception as e:
        print(f"Gemini error: {e}")
        return {"narrative": ""}

