"""
build_model_v2.py
==================
乾淨重寫的 UFC 勝負預測模型
- 零 data leakage（每場比賽只用賽前歷史）
- TimeSeriesSplit cross-validation
- Feature names 存進模型
- 輸出 model_win_v2.json + feature_cols_v2.json

跑法：
  cd ~/UFC
  python3 build_model_v2.py

輸出：
  data/model_win_v2.json
  data/feature_cols_v2.json
  (console 顯示 AUC + feature importance)
"""

import pandas as pd
import numpy as np
import json, os, warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════
BASE       = os.path.expanduser('~/UFC')
FIGHTS_CSV = os.path.join(BASE, 'data', 'clean', 'fights_all_rounds.csv')
FA_CSV     = os.path.join(BASE, 'data', 'clean', 'fighters_all.csv')
OUT_MODEL  = os.path.join(BASE, 'data', 'model_win_v2.json')
OUT_COLS   = os.path.join(BASE, 'data', 'feature_cols_v2.json')

MIN_FIGHTS_HISTORY = 3   # 至少要有幾場歷史才能算特徵
RECENT_N           = 3   # 近期狀態用最近幾場

# ══════════════════════════════════════════════════════
# 1. 讀資料
# ══════════════════════════════════════════════════════
print("=== 1. 讀資料 ===")
df  = pd.read_csv(FIGHTS_CSV, low_memory=False)
fa  = pd.read_csv(FA_CSV)

df['date'] = pd.to_datetime(df['date'])

# 數值欄位清理
num_cols = ['kd','sig_str_landed','sig_str_attempted','td_landed','td_attempted',
            'ctrl_sec','sub_att','head_landed','body_landed','leg_landed',
            'dist_landed','clinch_landed','ground_landed','ground_attempted',
            'sig_str_pct','td_pct']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['round_i'] = df['round'].str.extract(r'(\d+)').astype(float)

print(f"  比賽回合rows: {len(df)}")
print(f"  獨立場次: {df['bout'].nunique()}")
print(f"  日期範圍: {df['date'].min().date()} ~ {df['date'].max().date()}")

# ══════════════════════════════════════════════════════
# 2. 建立場次層級資料（每場每人一行）
# ══════════════════════════════════════════════════════
print("\n=== 2. 建立場次層級 ===")

# 對手同場數據（用於防守計算）
opp_cols = ['sig_str_landed','sig_str_attempted','td_landed','td_attempted','ctrl_sec']
opp = (df.groupby(['bout','fighter'])[opp_cols]
         .sum()
         .reset_index()
         .rename(columns={'fighter':'opponent', **{c:'opp_'+c for c in opp_cols}}))

# 選手本身場次統計
bout_stats = df.groupby(['bout','fighter']).agg(
    date          = ('date', 'first'),
    opponent      = ('opponent', 'first'),
    won           = ('won', 'first'),
    method        = ('method', 'first'),
    finish_round  = ('finish_round', 'first'),
    age_at_fight  = ('age_at_fight', 'first'),
    n_rounds      = ('round_i', 'max'),
    # 打擊
    sig_landed    = ('sig_str_landed', 'sum'),
    sig_attempted = ('sig_str_attempted', 'sum'),
    kd            = ('kd', 'sum'),
    # 摔技
    td_landed     = ('td_landed', 'sum'),
    td_attempted  = ('td_attempted', 'sum'),
    ctrl_sec      = ('ctrl_sec', 'sum'),
    sub_att       = ('sub_att', 'sum'),
    # 位置分布
    head_landed   = ('head_landed', 'sum'),
    body_landed   = ('body_landed', 'sum'),
    leg_landed    = ('leg_landed', 'sum'),
    dist_landed   = ('dist_landed', 'sum'),
    clinch_landed = ('clinch_landed', 'sum'),
    ground_landed = ('ground_landed', 'sum'),
    # 後段體能
    early_sig     = ('sig_str_landed', lambda x: x[df.loc[x.index,'round_i'] <= 2].sum()),
    late_sig      = ('sig_str_landed', lambda x: x[df.loc[x.index,'round_i'] >= 3].sum()),
).reset_index()

# Merge 對手數據
bout_stats = bout_stats.merge(opp, on=['bout','opponent'], how='left')

# 衍生欄位
bout_stats['sig_acc']     = bout_stats['sig_landed'] / bout_stats['sig_attempted'].replace(0, np.nan)
bout_stats['td_acc']      = bout_stats['td_landed']  / bout_stats['td_attempted'].replace(0, np.nan)
bout_stats['str_def']     = 1 - (bout_stats['opp_sig_str_landed'] / bout_stats['opp_sig_str_attempted'].replace(0, np.nan))
bout_stats['td_def']      = 1 - (bout_stats['opp_td_landed'] / bout_stats['opp_td_attempted'].replace(0, np.nan))
bout_stats['ctrl_received'] = bout_stats['opp_ctrl_sec']

# 結束方式
m = bout_stats['method'].astype(str)
bout_stats['is_ko']  = bout_stats['won'].astype(float) == 1
bout_stats['is_ko']  = bout_stats['is_ko'] & m.str.contains('KO|TKO', na=False)
bout_stats['is_sub'] = (bout_stats['won'].astype(float) == 1) & m.str.fullmatch('Submission', na=False)
bout_stats['is_finish'] = bout_stats['is_ko'] | bout_stats['is_sub']

# gas_tank: 後段/前段打擊比（體能代理）
bout_stats['gas_tank'] = bout_stats['late_sig'] / bout_stats['early_sig'].replace(0, np.nan)

# 地板比例
total_pos = bout_stats['dist_landed'] + bout_stats['clinch_landed'] + bout_stats['ground_landed']
bout_stats['ground_pct'] = bout_stats['ground_landed'] / total_pos.replace(0, np.nan)

bout_stats['won']  = pd.to_numeric(bout_stats['won'], errors='coerce')
bout_stats['date'] = pd.to_datetime(bout_stats['date'])
bout_stats = bout_stats.sort_values(['fighter','date']).reset_index(drop=True)

print(f"  場次層級rows: {len(bout_stats)}")

# ══════════════════════════════════════════════════════
# 3. 賽前歷史統計函數（核心：zero leakage）
# ══════════════════════════════════════════════════════
print("\n=== 3. 計算賽前特徵 ===")

STAT_COLS = ['sig_landed','sig_attempted','sig_acc','kd',
             'td_landed','td_attempted','td_acc',
             'ctrl_sec','ctrl_received','sub_att',
             'str_def','td_def','gas_tank','ground_pct',
             'is_ko','is_sub','is_finish','won']

def get_prefight_stats(fighter_name, fight_date, all_bouts):
    """只取該場比賽日期之前的歷史，回傳生涯統計 + 近期統計"""
    hist = all_bouts[
        (all_bouts['fighter'] == fighter_name) &
        (all_bouts['date'] < fight_date)
    ].sort_values('date')

    n = len(hist)
    if n < MIN_FIGHTS_HISTORY:
        return None

    # 生涯統計
    career = {}
    career['n_fights']     = n
    career['win_rate']     = hist['won'].mean()
    career['ko_rate']      = hist['is_ko'].mean()
    career['sub_rate']     = hist['is_sub'].mean()
    career['finish_rate']  = hist['is_finish'].mean()

    # 每回合平均（用總量/場數估計）
    career['sig_per_r']    = hist['sig_landed'].mean()
    career['sig_acc']      = hist['sig_acc'].median()
    career['str_def']      = hist['str_def'].median()
    career['td_per_r']     = hist['td_landed'].mean()
    career['td_acc']       = hist['td_acc'].median()
    career['td_def']       = hist['td_def'].median()
    career['ctrl_per_r']   = hist['ctrl_sec'].mean()
    career['sub_per_r']    = hist['sub_att'].mean()
    career['ground_pct']   = hist['ground_pct'].median()
    career['gas_tank']     = hist['gas_tank'].median()
    career['ctrl_received_per_r'] = hist['ctrl_received'].mean()

    # 近期 N 場狀態
    recent = hist.tail(RECENT_N)
    career['recent_win_rate']    = recent['won'].mean()
    career['recent_finish_rate'] = recent['is_finish'].mean()
    career['recent_sig_per_r']   = recent['sig_landed'].mean()
    career['recent_td_per_r']    = recent['td_landed'].mean()

    # 連勝/連敗
    streak = 0
    for w in hist['won'].iloc[::-1]:
        if pd.isna(w): break
        if w == 1: streak += 1
        else: break
    career['win_streak'] = streak

    lose_streak = 0
    for w in hist['won'].iloc[::-1]:
        if pd.isna(w): break
        if w == 0: lose_streak += 1
        else: break
    career['lose_streak'] = lose_streak

    # 對手強度（SOS）：過去對手的平均勝率
    opp_names = hist['opponent'].tolist()
    opp_wr = []
    for opp_name in opp_names:
        opp_hist = all_bouts[
            (all_bouts['fighter'] == opp_name) &
            (all_bouts['date'] < fight_date)
        ]
        if len(opp_hist) >= 2:
            opp_wr.append(opp_hist['won'].mean())
    career['sos'] = np.mean(opp_wr) if opp_wr else np.nan

    # 摔技維度的 SOS：過去對手的 td_def 平均
    opp_td_defs = []
    for opp_name in opp_names:
        opp_hist = all_bouts[
            (all_bouts['fighter'] == opp_name) &
            (all_bouts['date'] < fight_date)
        ]
        if len(opp_hist) >= 2:
            opp_td_defs.append(opp_hist['td_def'].median())
    career['sos_td_def'] = np.mean(opp_td_defs) if opp_td_defs else np.nan

    # 模態轉移（Chimaev 指標）：後段 ground_pct / 前段 ground_pct
    # 數字 < 1 代表體力下降後放棄摔技
    if n >= 3:
        early_gp = hist.head(max(1, n//2))['ground_pct'].median()
        late_gp  = hist.tail(max(1, n//2))['ground_pct'].median()
        career['modal_shift'] = late_gp / (early_gp + 1e-6)
    else:
        career['modal_shift'] = 1.0

    # 休息天數
    last_fight_date = hist['date'].max()
    career['days_since_last'] = (fight_date - last_fight_date).days

    # 年齡巔峰期（26-32）
    age = hist['age_at_fight'].iloc[-1] if pd.notna(hist['age_at_fight'].iloc[-1]) else np.nan
    career['age']          = age
    career['in_prime']     = 1 if (pd.notna(age) and 26 <= age <= 32) else 0
    career['past_prime']   = 1 if (pd.notna(age) and age > 35) else 0

    return career

# ══════════════════════════════════════════════════════
# 4. 建立訓練資料集
# ══════════════════════════════════════════════════════
print("\n=== 4. 建立訓練資料集（最慢的步驟）===")

# 每場比賽取兩個選手
bouts_unique = (bout_stats[['bout','date','fighter','opponent','won','finish_round','method']]
                .drop_duplicates()
                .sort_values('date')
                .reset_index(drop=True))

# 兩個方向都保留（A vs B 和 B vs A），讓模型學到對稱關係
# 這樣不管誰是 red/blue，預測結果都一致
bouts_dedup = bouts_unique.copy()
bouts_dedup = bouts_dedup[bouts_dedup['date'] >= '2013-01-01']
print(f"  訓練樣本數（雙向）: {len(bouts_dedup)}")

records = []
fighter_names = []
opponent_names = []
skipped = 0

for idx, row in bouts_dedup.iterrows():
    f1_name = row['fighter']
    f2_name = row['opponent']
    date    = row['date']
    bout    = row['bout']

    # 取兩個選手的賽前統計
    s1 = get_prefight_stats(f1_name, date, bout_stats)
    s2 = get_prefight_stats(f2_name, date, bout_stats)

    if s1 is None or s2 is None:
        skipped += 1
        continue

    # 標籤：fighter 贏了嗎
    won_row = bouts_unique[
        (bouts_unique['bout'] == bout) &
        (bouts_unique['fighter'] == f1_name)
    ]
    if len(won_row) == 0:
        skipped += 1
        continue
    label_raw = won_row.iloc[0]['won']
    if pd.isna(label_raw):
        skipped += 1
        continue
    label = int(label_raw)

    # 結束方式和回合：永遠從贏方那行取（確保跟 won label 一致）
    winner_row = bouts_unique[
        (bouts_unique['bout'] == bout) &
        (bouts_unique['won'] == 1)
    ]
    if len(winner_row) == 0:
        finish_round = np.nan
        method_label = 0
    else:
        finish_round_raw = winner_row.iloc[0].get('finish_round', np.nan)
        finish_round = int(finish_round_raw) if pd.notna(finish_round_raw) else np.nan
        method_raw = str(winner_row.iloc[0].get('method', ''))
        if 'KO' in method_raw or 'TKO' in method_raw:
            method_label = 1
        elif 'Submission' in method_raw:
            method_label = 2
        else:
            method_label = 0

    # 差值特徵
    rec = {'label': label, 'label_round': finish_round,
           'label_method': method_label, 'date': date, 'bout': bout}

    # 基本差值
    for key in ['win_rate','ko_rate','sub_rate','finish_rate',
                'sig_per_r','sig_acc','str_def',
                'td_per_r','td_acc','td_def',
                'ctrl_per_r','sub_per_r','ground_pct','gas_tank',
                'ctrl_received_per_r','sos','sos_td_def',
                'recent_win_rate','recent_finish_rate',
                'recent_sig_per_r','recent_td_per_r',
                'win_streak','lose_streak','days_since_last',
                'modal_shift','age']:
        v1 = s1.get(key, np.nan)
        v2 = s2.get(key, np.nan)
        rec[f'delta_{key}'] = (v1 if pd.notna(v1) else 0) - (v2 if pd.notna(v2) else 0)

    # Boolean 差值
    rec['delta_in_prime']   = s1['in_prime']   - s2['in_prime']
    rec['delta_past_prime'] = s1['past_prime']  - s2['past_prime']

    # 風格克制乘積項（關鍵：不是差值而是交互作用）
    rec['f1_td_threat_vs_f2_def']  = s1['td_per_r']  * s2['td_def']   # f1 摔技威脅 vs f2 防摔
    rec['f2_td_threat_vs_f1_def']  = s2['td_per_r']  * s1['td_def']   # 反向
    rec['f1_str_threat_vs_f2_def'] = s1['sig_per_r'] * s2['str_def']  # f1 打擊量 vs f2 防打擊
    rec['f2_str_threat_vs_f1_def'] = s2['sig_per_r'] * s1['str_def']  # 反向
    rec['f1_sub_vs_f2_ctrl']       = s1['sub_per_r'] * s2['ctrl_per_r']  # f1 鎖技 vs f2 控制
    rec['f1_ko_vs_f2_absorb']      = s1['ko_rate']   * s2['ctrl_received_per_r']  # KO力 vs 被打中率

    records.append(rec)
    fighter_names.append(f1_name)
    opponent_names.append(f2_name)

    if idx % 500 == 0:
        print(f"  處理中... {idx}/{len(bouts_dedup)}")

print(f"  完成：{len(records)} 筆訓練樣本，跳過 {skipped} 筆（歷史不足）")

# ══════════════════════════════════════════════════════
# 5. 加入硬件特徵（臂展、身高）
# ══════════════════════════════════════════════════════
print("\n=== 5. 加入硬件特徵 ===")

train_df = pd.DataFrame(records)

# Merge fighters_all 拿臂展身高
fa_map = fa.set_index('name')[['reach_cm','height_cm','stance']].to_dict('index')

def get_fa(name, col):
    return fa_map.get(name, {}).get(col, np.nan)

print(f"  訓練集 shape: {train_df.shape}")
print(f"  勝負比例: {train_df['label'].mean():.3f} (應接近 0.5)")

# ══════════════════════════════════════════════════════
# 6. 特徵選擇 + 填補 NaN
# ══════════════════════════════════════════════════════
print("\n=== 6. 特徵準備 ===")

FEATURE_COLS = [c for c in train_df.columns
                if c not in ['label','label_round','label_method','date','bout']]

X = train_df[FEATURE_COLS].copy()
y_win    = train_df['label'].astype(int)
y_round  = train_df['label_round']
y_method = train_df['label_method'].astype(int)

# NaN 填補：用中位數
for col in X.columns:
    med = X[col].median()
    X[col] = X[col].fillna(med if pd.notna(med) else 0)

print(f"  特徵數: {len(FEATURE_COLS)}")
print(f"  NaN 剩餘: {X.isna().sum().sum()}")

# ══════════════════════════════════════════════════════
# 7. 訓練函數
# ══════════════════════════════════════════════════════
def train_model(X_s, y_s, task='binary', n_classes=None):
    """通用訓練函數，支援 binary / multiclass / regression"""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, accuracy_score

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    if task == 'binary':
        eval_metric = 'logloss'
        objective   = 'binary:logistic'
        num_class   = None
    elif task == 'multiclass':
        eval_metric = 'mlogloss'
        objective   = 'multi:softprob'
        num_class   = n_classes
    else:
        eval_metric = 'rmse'
        objective   = 'reg:squarederror'
        num_class   = None

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_s)):
        X_tr, X_val = X_s.iloc[tr_idx], X_s.iloc[val_idx]
        y_tr, y_val = y_s.iloc[tr_idx], y_s.iloc[val_idx]

        params = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                      use_label_encoder=False, eval_metric=eval_metric,
                      random_state=42, objective=objective)
        if num_class:
            params['num_class'] = num_class

        m = XGBClassifier(**params) if task != 'regression' else __import__('xgboost').XGBRegressor(**params)
        m.fit(X_tr, y_tr)

        if task == 'binary':
            prob = m.predict_proba(X_val)[:,1]
            score = roc_auc_score(y_val, prob)
            print(f"    Fold {fold+1}: AUC={score:.4f}")
        elif task == 'multiclass':
            pred = m.predict(X_val)
            score = accuracy_score(y_val, pred)
            print(f"    Fold {fold+1}: ACC={score:.4f}")
        else:
            from sklearn.metrics import mean_absolute_error
            pred = m.predict(X_val)
            score = mean_absolute_error(y_val, pred)
            print(f"    Fold {fold+1}: MAE={score:.4f}")
        scores.append(score)

    print(f"    平均: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # 全資料重訓
    params2 = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                   gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                   use_label_encoder=False, eval_metric=eval_metric,
                   random_state=42, objective=objective)
    if num_class:
        params2['num_class'] = num_class

    final = XGBClassifier(**params2) if task != 'regression' else __import__('xgboost').XGBRegressor(**params2)
    final.fit(X_s, y_s)
    final.get_booster().feature_names = FEATURE_COLS
    return final, np.mean(scores)

# 按時間排序
time_order = train_df['date'].argsort().values
X_sorted      = X.iloc[time_order].reset_index(drop=True)
y_win_sorted  = y_win.iloc[time_order].reset_index(drop=True)
y_meth_sorted = y_method.iloc[time_order].reset_index(drop=True)
y_round_all   = y_round.iloc[time_order].reset_index(drop=True)

# 回合預測：只用有效回合數（1-5）的樣本，label 轉 0-indexed (0-4)
round_mask     = y_round_all.notna() & y_round_all.between(1, 5)
X_round        = X_sorted[round_mask].reset_index(drop=True)
y_round_sorted = (y_round_all[round_mask].astype(int) - 1).reset_index(drop=True)  # 0-4

print(f"  Round 訓練樣本: {len(y_round_sorted)}, 分布: {y_round_sorted.value_counts().sort_index().to_dict()}")

print("\n=== 7a. 訓練勝負模型 ===")
model_win, auc_win = train_model(X_sorted, y_win_sorted, task='binary')

print("\n=== 7b. 訓練結束方式模型 (0=Dec, 1=KO, 2=Sub) ===")
model_method, acc_method = train_model(X_sorted, y_meth_sorted, task='multiclass', n_classes=3)

print("\n=== 7c. 訓練結束回合模型 (1-5) ===")
model_round, acc_round = train_model(X_round, y_round_sorted, task='multiclass', n_classes=5)

# ══════════════════════════════════════════════════════
# 8. Feature Importance（勝負模型）
# ══════════════════════════════════════════════════════
print("\n=== 8. Feature Importance - 勝負模型 (Top 20) ===")
fi = dict(zip(FEATURE_COLS, model_win.feature_importances_))
fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
for name, score in fi_sorted[:20]:
    bar = '█' * int(score * 500)
    print(f"  {name:40s} {score:.4f}  {bar}")

# ══════════════════════════════════════════════════════
# 9. 存檔
# ══════════════════════════════════════════════════════
print(f"\n=== 9. 存檔 ===")
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

OUT_MODEL_METHOD = os.path.join(BASE, 'data', 'model_method_v2.json')
OUT_MODEL_ROUND  = os.path.join(BASE, 'data', 'model_round_v2.json')

model_win.save_model(OUT_MODEL)
model_method.save_model(OUT_MODEL_METHOD)
model_round.save_model(OUT_MODEL_ROUND)

with open(OUT_COLS, 'w') as f:
    json.dump(FEATURE_COLS, f, indent=2)

print(f"  → {OUT_MODEL}  (AUC {auc_win:.4f})")
print(f"  → {OUT_MODEL_METHOD}  (ACC {acc_method:.4f})")
print(f"  → {OUT_MODEL_ROUND}  (ACC {acc_round:.4f})")
print(f"  → {OUT_COLS}")
print(f"\n完成。特徵數: {len(FEATURE_COLS)}, 訓練樣本: {len(X_sorted)}")
