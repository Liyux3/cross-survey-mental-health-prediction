"""
Phase 6d: Deep-dive into synthetic vs real distribution shift and adaptation.
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
df['y_enc'] = le.fit_transform(df[TARGET])
X_all = df[feat_cols].apply(pd.to_numeric, errors='coerce').values
y_all = df['y_enc'].values
source = df[META].values

real_mask = np.isin(source, REAL_DS)
synth_mask = np.isin(source, SYNTH_DS)
X_real, y_real = X_all[real_mask].astype(float), y_all[real_mask]
X_synth, y_synth = X_all[synth_mask].astype(float), y_all[synth_mask]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ============================================================
# 1. Feature Distribution Shift: KS test + visualization
# ============================================================
print('='*60)
print('1. FEATURE DISTRIBUTION SHIFT (Real vs Synthetic)')
print('='*60)

# For numeric-ish features, compute KS statistic
ks_results = []
for i, col in enumerate(feat_cols):
    real_vals = pd.to_numeric(pd.Series(X_real[:, i]), errors='coerce').values
    synth_vals = pd.to_numeric(pd.Series(X_synth[:, i]), errors='coerce').values
    real_vals = real_vals[~np.isnan(real_vals)]
    synth_vals = synth_vals[~np.isnan(synth_vals)]
    if len(real_vals) > 10 and len(synth_vals) > 10:
        ks_stat, p_val = stats.ks_2samp(real_vals, synth_vals)
        mean_diff = real_vals.mean() - synth_vals.mean()
        std_real, std_synth = real_vals.std(), synth_vals.std()
        ks_results.append({
            'Feature': col, 'KS_stat': ks_stat, 'p_value': p_val,
            'Mean_real': real_vals.mean(), 'Mean_synth': synth_vals.mean(),
            'Mean_diff': mean_diff,
            'Std_real': std_real, 'Std_synth': std_synth,
        })

ks_df = pd.DataFrame(ks_results).sort_values('KS_stat', ascending=False)
print('\nTop features by distribution shift (KS statistic):')
print(ks_df[['Feature', 'KS_stat', 'Mean_real', 'Mean_synth', 'Mean_diff']].head(15).to_string(index=False))

# Visualization: top 6 shifted features
top_shifted = ks_df.head(6)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, feat in enumerate(top_shifted):
    ax = axes[idx // 3, idx % 3]
    fi = feat_cols.index(feat)
    r_vals = X_real[:, fi][~np.isnan(X_real[:, fi])]
    s_vals = X_synth[:, fi][~np.isnan(X_synth[:, fi])]
    ax.hist(r_vals, bins=30, alpha=0.6, color='#2196F3', label='Real', density=True)
    ax.hist(s_vals, bins=30, alpha=0.6, color='#FF9800', label='Synthetic', density=True)
    ks_val = ks_df[ks_df['Feature'] == feat]['KS_stat'].values[0]
    ax.set_title(f'{feat}\nKS={ks_val:.3f}')
    ax.legend(fontsize=8)
fig.suptitle('Feature Distributions: Real vs Synthetic (top 6 shift)', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_distribution_shift.png'))
plt.close()
print('\nChart saved: eval_distribution_shift.png')

# ============================================================
# 2. Target distribution shift
# ============================================================
print('\n' + '='*60)
print('2. TARGET DISTRIBUTION SHIFT')
print('='*60)

real_composite = df[real_mask][COMPOSITE]
synth_composite = df[synth_mask][COMPOSITE]
ks_target, p_target = stats.ks_2samp(real_composite, synth_composite)
print(f'  Real composite:  mean={real_composite.mean():.3f}, std={real_composite.std():.3f}')
print(f'  Synth composite: mean={synth_composite.mean():.3f}, std={synth_composite.std():.3f}')
print(f'  KS stat: {ks_target:.3f}, p={p_target:.2e}')

print(f'\n  Class distribution:')
real_cls = df[real_mask][TARGET].value_counts(normalize=True).sort_index()
synth_cls = df[synth_mask][TARGET].value_counts(normalize=True).sort_index()
for cls in sorted(df[TARGET].unique()):
    r = real_cls.get(cls, 0)
    s = synth_cls.get(cls, 0)
    print(f'    {cls}: Real={r:.3f}, Synth={s:.3f}, gap={r-s:+.3f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(real_composite, bins=30, alpha=0.7, color='#2196F3', label='Real', density=True)
axes[0].hist(synth_composite, bins=30, alpha=0.7, color='#FF9800', label='Synthetic', density=True)
axes[0].set_title(f'MH Composite Distribution (KS={ks_target:.3f})')
axes[0].legend()
axes[0].axvline(0.33, color='gray', ls='--', lw=0.8)
axes[0].axvline(0.66, color='gray', ls='--', lw=0.8)

cls_comp = pd.DataFrame({'Real': real_cls, 'Synthetic': synth_cls})
cls_comp.plot(kind='bar', ax=axes[1], color=['#2196F3', '#FF9800'])
axes[1].set_title('Class Distribution: Real vs Synthetic')
axes[1].set_ylabel('Proportion')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_target_shift.png'))
plt.close()
print('Chart saved: eval_target_shift.png')

# ============================================================
# 3. Per-class performance breakdown across methods
# ============================================================
print('\n' + '='*60)
print('3. PER-CLASS BREAKDOWN')
print('='*60)

methods = {}

# Collect per-fold per-class F1 for key methods
def run_method(name, fit_fn):
    per_class_f1 = {c: [] for c in le.classes_}
    for train_idx, test_idx in cv.split(X_real, y_real):
        y_pred = fit_fn(train_idx, test_idx)
        for ci, c in enumerate(le.classes_):
            mask_c = y_real[test_idx] == ci
            if mask_c.sum() > 0:
                tp = ((y_pred == ci) & mask_c).sum()
                fp = ((y_pred == ci) & ~mask_c).sum()
                fn = ((y_pred != ci) & mask_c).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                per_class_f1[c].append(f1)
    return {c: np.mean(v) for c, v in per_class_f1.items()}

def real_only(train_idx, test_idx):
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_real[train_idx], y_real[train_idx])
    return clf.predict(X_real[test_idx])

def synth_only(train_idx, test_idx):
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_synth, y_synth)
    return clf.predict(X_real[test_idx])

def reweighted(train_idx, test_idx):
    X_tr = np.vstack([X_synth, X_real[train_idx]])
    y_tr = np.concatenate([y_synth, y_real[train_idx]])
    w = np.concatenate([np.ones(len(X_synth)), np.ones(len(train_idx)) * 10.0])
    clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr, sample_weight=w)
    return clf.predict(X_real[test_idx])

def warmstart(train_idx, test_idx):
    pre = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                              random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
    pre.fit(X_synth, y_synth)
    ft = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.05,
                             random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
    ft.fit(X_real[train_idx], y_real[train_idx], xgb_model=pre.get_booster())
    return ft.predict(X_real[test_idx])

for name, fn in [('Real-only', real_only), ('Synth-only', synth_only),
                  ('Reweighted', reweighted), ('Warm-start', warmstart)]:
    r = run_method(name, fn)
    methods[name] = r
    print(f'\n  {name}:')
    for c, v in r.items():
        print(f'    {c}: F1={v:.3f}')

# Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(le.classes_))
width = 0.2
for i, (name, vals) in enumerate(methods.items()):
    f1s = [vals[c] for c in le.classes_]
    ax.bar(x + i * width, f1s, width, label=name)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(le.classes_)
ax.set_ylabel('F1 Score')
ax.set_title('Per-Class F1: Adaptation Methods Compared')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_perclass_adaptation.png'))
plt.close()
print('\nChart saved: eval_perclass_adaptation.png')

# ============================================================
# 4. Reweighting sensitivity curve
# ============================================================
print('\n' + '='*60)
print('4. REWEIGHTING SENSITIVITY')
print('='*60)

weight_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
weight_results = []

for w_val in weight_values:
    f1s = []
    for train_idx, test_idx in cv.split(X_real, y_real):
        X_tr = np.vstack([X_synth, X_real[train_idx]])
        y_tr = np.concatenate([y_synth, y_real[train_idx]])
        w = np.concatenate([np.ones(len(X_synth)), np.ones(len(train_idx)) * w_val])
        clf = RandomForestClassifier(200, max_depth=15, class_weight='balanced',
                                      random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_tr, y_tr, sample_weight=w)
        y_pred = clf.predict(X_real[test_idx])
        f1s.append(f1_score(y_real[test_idx], y_pred, average='macro', zero_division=0))
    mean_f1 = np.mean(f1s)
    weight_results.append((w_val, mean_f1, np.std(f1s)))
    print(f'  weight={w_val:3d}x: F1m={mean_f1:.4f} ± {np.std(f1s):.4f}')

# Add baselines for reference
real_only_f1 = methods['Real-only']
real_only_macro = np.mean(list(real_only_f1.values()))

fig, ax = plt.subplots(figsize=(10, 5))
ws = [r[0] for r in weight_results]
f1s = [r[1] for r in weight_results]
stds = [r[2] for r in weight_results]
ax.plot(ws, f1s, 'o-', color=CHART_PALETTE[0], label='Reweighted (synth+real)')
ax.fill_between(ws, [f-s for f,s in zip(f1s,stds)], [f+s for f,s in zip(f1s,stds)],
                alpha=0.2, color=CHART_PALETTE[0])
ax.axhline(real_only_macro, color='#F44336', ls='--', lw=1.5, label='Real-only baseline')
ax.set_xlabel('Real sample weight multiplier')
ax.set_ylabel('F1 macro (on real data)')
ax.set_title('Reweighting Sensitivity: How much to trust real vs synthetic?')
ax.legend()
ax.set_xscale('log')
ax.set_xticks(weight_values)
ax.set_xticklabels([str(v) for v in weight_values])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_reweight_sensitivity.png'))
plt.close()
print('\nChart saved: eval_reweight_sensitivity.png')

# ============================================================
# 5. Warm-start: what was retained?
# ============================================================
print('\n' + '='*60)
print('5. WARM-START ANALYSIS: Feature importance before/after finetune')
print('='*60)

pre = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                          random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
pre.fit(X_synth, y_synth)
imp_pre = pre.feature_importances_

ft = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.05,
                         random_state=RANDOM_SEED, eval_metric='mlogloss', n_jobs=-1)
ft.fit(X_real, y_real, xgb_model=pre.get_booster())
imp_post = ft.feature_importances_

# Rank correlation between pre and post
from scipy.stats import spearmanr
rho, p = spearmanr(imp_pre, imp_post)
print(f'  Spearman rank correlation (pre vs post importance): rho={rho:.3f}, p={p:.3e}')

# Top features comparison
top_pre = np.argsort(imp_pre)[::-1][:10]
top_post = np.argsort(imp_post)[::-1][:10]
overlap = set(top_pre) & set(top_post)
print(f'  Top-10 overlap: {len(overlap)}/10')
print(f'  Pre  top-5: {[feat_cols[i] for i in top_pre[:5]]}')
print(f'  Post top-5: {[feat_cols[i] for i in top_post[:5]]}')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_n = 12
idx_pre = np.argsort(imp_pre)[-top_n:]
axes[0].barh([feat_cols[i] for i in idx_pre], imp_pre[idx_pre], color='#FF9800')
axes[0].set_title('Pretrained on Synthetic')

idx_post = np.argsort(imp_post)[-top_n:]
axes[1].barh([feat_cols[i] for i in idx_post], imp_post[idx_post], color='#2196F3')
axes[1].set_title('After Finetune on Real')

fig.suptitle(f'Feature Importance Shift (Spearman rho={rho:.3f})', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_warmstart_importance.png'))
plt.close()
print('Chart saved: eval_warmstart_importance.png')

# ============================================================
# Summary
# ============================================================
print('\n' + '='*60)
print('DEEP-DIVE SUMMARY')
print('='*60)
print(f'1. Distribution shift:')
print(f'   Most shifted features: {", ".join(ks_df.head(3)["Feature"].tolist())}')
print(f'   Target KS={ks_target:.3f}, real skews toward High Risk ({real_cls.get("High",0):.1%}) vs synth ({synth_cls.get("High",0):.1%})')
print(f'2. Per-class: synthetic helps most with... (see per-class chart)')
print(f'3. Optimal reweight: ~{weight_results[np.argmax([r[1] for r in weight_results])][0]}x')
print(f'4. Warm-start retention: rho={rho:.3f} ({"moderate" if rho > 0.5 else "weak"} preservation)')
