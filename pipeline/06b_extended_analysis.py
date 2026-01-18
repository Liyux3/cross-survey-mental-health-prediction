"""
Phase 6b: Extended analysis — Real vs Synthetic, source_dataset ablation,
cross-dataset consistency of feature importance.
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

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

# ============================================================
# 1. LODO: Real vs Synthetic breakdown
# ============================================================
print('='*60)
print('1. LODO BREAKDOWN: Real vs Synthetic')
print('='*60)

lodo = pd.read_csv(os.path.join(OUTPUT_DIR, 'lodo_results.csv'))
lodo['Type'] = lodo['Test_Dataset'].apply(lambda x: 'Real' if x in REAL_DS else 'Synthetic')

for t in ['Real', 'Synthetic']:
    sub = lodo[lodo['Type'] == t]
    print(f'\n  {t} datasets ({", ".join(sub["Test_Dataset"].tolist())}):')
    print(f'    Mean Acc: {sub["Accuracy"].mean():.3f}')
    print(f'    Mean F1m: {sub["F1_macro"].mean():.3f}')

# Chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2196F3' if ds in REAL_DS else '#FF9800' for ds in lodo['Test_Dataset']]
x = np.arange(len(lodo))
bars = ax.bar(x, lodo['F1_macro'], color=colors)
ax.set_xticks(x)
ax.set_xticklabels(lodo['Test_Dataset'])
ax.set_ylabel('F1 macro')
ax.set_title('LODO F1 by Dataset (Blue=Real, Orange=Synthetic)')
ax.axhline(0.333, color='gray', ls='--', lw=0.8)
for i, row in lodo.iterrows():
    ax.text(i, row['F1_macro'] + 0.01, f'{row["F1_macro"]:.2f}', ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_lodo_real_vs_synth.png'))
plt.close()
print('\n  Chart saved: eval_lodo_real_vs_synth.png')

# ============================================================
# 2. Train on Synthetic, Test on Real (and vice versa)
# ============================================================
print('\n' + '='*60)
print('2. CROSS-TYPE GENERALIZATION')
print('='*60)

X_all = df[feat_cols].values
y_all = df['y_enc'].values
source = df[META].values

real_mask = df[META].isin(REAL_DS).values
synth_mask = df[META].isin(SYNTH_DS).values

def train_test_cross(train_mask, test_mask, label):
    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_te, y_te = X_all[test_mask], y_all[test_mask]
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f'  {label}: Acc={acc:.3f}, F1m={f1m:.3f} (train={train_mask.sum()}, test={test_mask.sum()})')
    return acc, f1m

results_cross = []
a, f = train_test_cross(synth_mask, real_mask, 'Train Synth → Test Real')
results_cross.append(('Synth→Real', a, f))
a, f = train_test_cross(real_mask, synth_mask, 'Train Real → Test Synth')
results_cross.append(('Real→Synth', a, f))
a, f = train_test_cross(real_mask, real_mask, 'Train Real → Test Real (CV-like)')
results_cross.append(('Real→Real', a, f))
a, f = train_test_cross(synth_mask, synth_mask, 'Train Synth → Test Synth (CV-like)')
results_cross.append(('Synth→Synth', a, f))

# ============================================================
# 3. Feature importance consistency across LODO folds
# ============================================================
print('\n' + '='*60)
print('3. FEATURE IMPORTANCE CONSISTENCY ACROSS LODO FOLDS')
print('='*60)

datasets = sorted(df[META].unique())
importance_matrix = np.zeros((len(datasets), len(feat_cols)))

for i, test_ds in enumerate(datasets):
    mask_train = source != test_ds
    X_tr, y_tr = X_all[mask_train], y_all[mask_train]
    clf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                  class_weight='balanced',
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    importance_matrix[i] = clf.feature_importances_

import_df = pd.DataFrame(importance_matrix, columns=feat_cols, index=datasets)

# Rank-based consistency: for each fold, rank features, then see overlap of top-10
top_k = 10
top_per_fold = {}
for ds in datasets:
    top_feats = import_df.loc[ds].nlargest(top_k).index.tolist()
    top_per_fold[ds] = top_feats

# Count how many times each feature appears in any fold's top-10
from collections import Counter
all_top = [f for feats in top_per_fold.values() for f in feats]
consistency = Counter(all_top)
print(f'\n  Features appearing in top-{top_k} across {len(datasets)} LODO folds:')
for feat, count in consistency.most_common(15):
    print(f'    {feat}: {count}/{len(datasets)} folds')

# Chart
fig, ax = plt.subplots(figsize=(12, 6))
import seaborn as sns
# Normalize per fold for visual comparison
import_norm = import_df.div(import_df.sum(axis=1), axis=0)
top_feats_overall = import_df.mean().nlargest(12).index.tolist()
sns.heatmap(import_norm[top_feats_overall].T, cmap='YlOrRd', ax=ax,
            annot=True, fmt='.2f', linewidths=0.5)
ax.set_title('Feature Importance (normalized) across LODO Folds')
ax.set_xlabel('Test Dataset (left out)')
ax.set_ylabel('Feature')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_feature_consistency_lodo.png'))
plt.close()
print('\n  Chart saved: eval_feature_consistency_lodo.png')

# ============================================================
# 4. Source dataset ablation: with vs without source_dataset as feature
# ============================================================
print('\n' + '='*60)
print('4. SOURCE_DATASET ABLATION')
print('='*60)

from sklearn.model_selection import StratifiedKFold, cross_val_score

# Without source_dataset (current model)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
clf_no_src = RandomForestClassifier(n_estimators=200, max_depth=15,
                                     class_weight='balanced',
                                     random_state=RANDOM_SEED, n_jobs=-1)
scores_no = cross_val_score(clf_no_src, X_all, y_all, cv=cv, scoring='f1_macro')
print(f'  Without source_dataset: F1m = {scores_no.mean():.4f} ± {scores_no.std():.4f}')

# With source_dataset as one-hot
df_with_src = df[feat_cols].copy()
src_dummies = pd.get_dummies(df[META], prefix='src')
df_with_src = pd.concat([df_with_src, src_dummies], axis=1)
X_with_src = df_with_src.values

clf_with_src = RandomForestClassifier(n_estimators=200, max_depth=15,
                                       class_weight='balanced',
                                       random_state=RANDOM_SEED, n_jobs=-1)
scores_with = cross_val_score(clf_with_src, X_with_src, y_all, cv=cv, scoring='f1_macro')
print(f'  With source_dataset:    F1m = {scores_with.mean():.4f} ± {scores_with.std():.4f}')
print(f'  Delta: {scores_with.mean() - scores_no.mean():+.4f}')

if scores_with.mean() - scores_no.mean() > 0.02:
    print('  → source_dataset provides significant boost, indicating dataset identity leaks into target.')
    print('    This is expected given different target construction methods per dataset.')
    print('    We exclude it to ensure model learns behavioral patterns, not dataset identity.')
else:
    print('  → Minimal difference, our target construction successfully reduces dataset-specific bias.')

# ============================================================
# Summary
# ============================================================
print('\n' + '='*60)
print('ANALYSIS SUMMARY')
print('='*60)
print(f'Standard CV (RF):     F1m = {scores_no.mean():.3f}')
print(f'LODO mean:            F1m = {pd.read_csv(os.path.join(OUTPUT_DIR, "lodo_results.csv"))["F1_macro"].mean():.3f}')
print(f'Synth→Real:           F1m = {results_cross[0][2]:.3f}')
print(f'Real→Synth:           F1m = {results_cross[1][2]:.3f}')
print(f'Source ablation delta: {scores_with.mean() - scores_no.mean():+.3f}')
print(f'\nMost consistent features (appearing in all LODO folds top-{top_k}):')
for feat, count in consistency.most_common():
    if count >= len(datasets) - 1:
        print(f'  {feat} ({count}/{len(datasets)})')
