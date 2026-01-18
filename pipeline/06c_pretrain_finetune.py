"""
Phase 6c: Synthetic pretrain + Real finetune experiments.
Three approaches compared:
  1. GBM warm-start (pretrain on synth, continue training on real)
  2. Calibration layer (synth model + isotonic calibration on real)
  3. Sample reweighting (joint training, real samples upweighted)
Baselines: synth-only, real-only, all-data (no distinction).
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED, CHART_PALETTE

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_imputed.csv'))
TARGET = 'mh_risk_class'
COMPOSITE = 'mh_composite'
META = 'source_dataset'
feat_cols = [c for c in df.columns if c not in [TARGET, COMPOSITE, META]]

REAL_DS = ['ds1', 'ds4', 'ds7']
SYNTH_DS = ['ds2', 'ds3', 'ds5', 'ds6', 'ds8']

le = LabelEncoder()
y_all = le.fit_transform(df[TARGET].values)
X_all = df[feat_cols].values
source = df[META].values

real_mask = np.isin(source, REAL_DS)
synth_mask = np.isin(source, SYNTH_DS)

X_real, y_real = X_all[real_mask], y_all[real_mask]
X_synth, y_synth = X_all[synth_mask], y_all[synth_mask]

print(f'Real: {X_real.shape[0]} rows, Synth: {X_synth.shape[0]} rows')
print(f'Real class dist: {np.bincount(y_real)}')
print(f'Synth class dist: {np.bincount(y_synth)}')

# We evaluate on real data using 5-fold CV on real subset.
# For pretrain methods, synth is always fully used for pretraining,
# then each CV fold's real-train split is used for finetuning,
# and real-test split for evaluation.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
results = []

# ============================================================
# Baseline 1: Real-only
# ============================================================
print('\n--- Baseline: Real-only ---')
scores = cross_val_score(
    RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                           random_state=RANDOM_SEED, n_jobs=-1),
    X_real, y_real, cv=cv, scoring='f1_macro')
print(f'  F1m = {scores.mean():.4f} ± {scores.std():.4f}')
results.append(('Real-only (RF)', scores.mean(), scores.std()))

# ============================================================
# Baseline 2: Synth-only, tested on real
# ============================================================
print('\n--- Baseline: Synth-only → eval on real ---')
f1s = []
for train_idx, test_idx in cv.split(X_real, y_real):
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_synth, y_synth)
    y_pred = clf.predict(X_real[test_idx])
    f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
f1s = np.array(f1s)
print(f'  F1m = {f1s.mean():.4f} ± {f1s.std():.4f}')
results.append(('Synth-only → Real (RF)', f1s.mean(), f1s.std()))

# ============================================================
# Baseline 3: All data, no distinction
# ============================================================
print('\n--- Baseline: All data combined ---')
f1s = []
for train_idx, test_idx in cv.split(X_real, y_real):
    X_tr = np.vstack([X_synth, X_real[train_idx]])
    y_tr = np.concatenate([y_synth, y_real[train_idx]])
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_real[test_idx])
    f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
f1s = np.array(f1s)
print(f'  F1m = {f1s.mean():.4f} ± {f1s.std():.4f}')
results.append(('All combined (RF)', f1s.mean(), f1s.std()))

# ============================================================
# Method 1: XGBoost warm-start
# ============================================================
print('\n--- Method 1: XGBoost warm-start (synth pretrain → real finetune) ---')
f1s = []
for train_idx, test_idx in cv.split(X_real, y_real):
    # Phase 1: pretrain on synth
    pretrain = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
    pretrain.fit(X_synth, y_synth)

    # Phase 2: finetune on real (continue from pretrained model)
    finetune = xgb.XGBClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
    finetune.fit(X_real[train_idx], y_real[train_idx],
                 xgb_model=pretrain.get_booster())

    y_pred = finetune.predict(X_real[test_idx])
    f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
f1s = np.array(f1s)
print(f'  F1m = {f1s.mean():.4f} ± {f1s.std():.4f}')
results.append(('XGB warm-start (synth→real)', f1s.mean(), f1s.std()))

# ============================================================
# Method 2: Calibration (synth model + isotonic on real)
# ============================================================
print('\n--- Method 2: Synth model + Isotonic calibration on real ---')
f1s = []
for train_idx, test_idx in cv.split(X_real, y_real):
    base = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                   random_state=RANDOM_SEED, n_jobs=-1)
    base.fit(X_synth, y_synth)

    # Synth-trained model's probabilities → isotonic calibration on real
    from sklearn.isotonic import IsotonicRegression
    probs_train = base.predict_proba(X_real[train_idx])
    probs_test = base.predict_proba(X_real[test_idx])

    # Per-class isotonic calibration
    cal_probs = np.zeros_like(probs_test)
    for c in range(probs_train.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(probs_train[:, c], (y_real[train_idx] == c).astype(float))
        cal_probs[:, c] = ir.predict(probs_test[:, c])

    y_pred = cal_probs.argmax(axis=1)
    f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
f1s = np.array(f1s)
print(f'  F1m = {f1s.mean():.4f} ± {f1s.std():.4f}')
results.append(('Calibrated (synth+real isotonic)', f1s.mean(), f1s.std()))

# ============================================================
# Method 3: Sample reweighting
# ============================================================
print('\n--- Method 3: Sample reweighting (real 10x) ---')
f1s = []
for train_idx, test_idx in cv.split(X_real, y_real):
    X_tr = np.vstack([X_synth, X_real[train_idx]])
    y_tr = np.concatenate([y_synth, y_real[train_idx]])
    weights = np.concatenate([
        np.ones(len(X_synth)),
        np.ones(len(train_idx)) * 10.0
    ])
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr, sample_weight=weights)
    y_pred = clf.predict(X_real[test_idx])
    f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
f1s = np.array(f1s)
print(f'  F1m = {f1s.mean():.4f} ± {f1s.std():.4f}')
results.append(('Reweighted (real 10x)', f1s.mean(), f1s.std()))

# ============================================================
# Summary table + chart
# ============================================================
res_df = pd.DataFrame(results, columns=['Method', 'F1_macro_mean', 'F1_macro_std'])
res_df = res_df.sort_values('F1_macro_mean', ascending=False)
print('\n' + '='*60)
print('PRETRAIN+FINETUNE COMPARISON (eval on real data)')
print('='*60)
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(OUTPUT_DIR, 'pretrain_finetune_results.csv'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = []
for m in res_df['Method']:
    if 'Baseline' in m or 'only' in m.lower() or 'combined' in m.lower():
        colors.append('#9E9E9E')
    else:
        colors.append(CHART_PALETTE[0])
# all bars
y_pos = np.arange(len(res_df))
ax.barh(y_pos, res_df['F1_macro_mean'],
        xerr=res_df['F1_macro_std'], height=0.6,
        color=['#9E9E9E' if ('only' in m.lower() or 'combined' in m.lower())
               else CHART_PALETTE[0] for m in res_df['Method']])
ax.set_yticks(y_pos)
ax.set_yticklabels(res_df['Method'])
ax.set_xlabel('F1 macro (on real data)')
ax.set_title('Synthetic Pretrain → Real Finetune: Method Comparison')
for i, (_, row) in enumerate(res_df.iterrows()):
    ax.text(row['F1_macro_mean'] + 0.01, i, f'{row["F1_macro_mean"]:.3f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_pretrain_finetune.png'))
plt.close()
print('\nChart saved.')
