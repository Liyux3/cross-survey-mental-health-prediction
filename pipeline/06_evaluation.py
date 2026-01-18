"""
Phase 6: Evaluation & Interpretation — SHAP, PDP, metrics charts.
Input: pipeline/output/master_imputed.csv, saved models
Output: charts in pipeline/output/
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED, CHART_PALETTE

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'
})

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_imputed.csv'))
TARGET = 'mh_risk_class'
COMPOSITE = 'mh_composite'
META = 'source_dataset'
feat_cols = [c for c in df.columns if c not in [TARGET, COMPOSITE, META]]

X = df[feat_cols].values
le_target = LabelEncoder()
y = le_target.fit_transform(df[TARGET].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

model = joblib.load(os.path.join(OUTPUT_DIR, 'best_tree_classifier.pkl'))
y_pred = model.predict(X_test)

# ============================================================
# 1. Confusion Matrix
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix (Hold-out)')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_confusion_matrix.png'))
plt.close()
print('1/5 confusion matrix')

# ============================================================
# 2. SHAP Summary Plot
# ============================================================
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# shap_values_raw shape: (n_samples, n_features, n_classes) — transpose to list of per-class arrays
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    n_classes = shap_values_raw.shape[2]
    shap_values = [shap_values_raw[:, :, i] for i in range(n_classes)]
else:
    shap_values = shap_values_raw

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feat_cols,
                  class_names=le_target.classes_.tolist(),
                  show=False, max_display=20)
plt.title('SHAP Summary (all classes)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eval_shap_summary.png'))
plt.close()
print('2/5 SHAP summary')

# Per-class SHAP
for i, cls_name in enumerate(le_target.classes_):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values[i], X_test, feature_names=feat_cols,
                      show=False, max_display=15)
    plt.title(f'SHAP — {cls_name} Risk')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'eval_shap_{cls_name.lower()}.png'))
    plt.close()
print('   per-class SHAP done')

# ============================================================
# 3. SHAP Dependence Plot (top features)
# ============================================================
mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:5]

fig, axes = plt.subplots(1, min(5, len(top_idx)), figsize=(20, 4))
if len(top_idx) == 1:
    axes = [axes]
for ax_i, fi in enumerate(top_idx[:5]):
    cls_idx = min(2, len(shap_values) - 1)
    shap.dependence_plot(fi, shap_values[cls_idx], X_test,
                         feature_names=feat_cols, ax=axes[ax_i], show=False)
    axes[ax_i].set_title(feat_cols[fi])
fig.suptitle('SHAP Dependence — High Risk Class', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_shap_dependence.png'))
plt.close()
print('3/5 SHAP dependence')

# ============================================================
# 4. Feature Importance: SHAP vs Permutation vs Built-in
# ============================================================
from sklearn.inspection import permutation_importance

perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                               random_state=RANDOM_SEED, scoring='f1_macro',
                               n_jobs=-1)

builtin_imp = model.feature_importances_

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# SHAP
top_n = 15
shap_imp = mean_abs_shap
idx_shap = np.argsort(shap_imp)[-top_n:]
axes[0].barh([feat_cols[i] for i in idx_shap], shap_imp[idx_shap], color=CHART_PALETTE[0])
axes[0].set_title('SHAP (mean |SHAP|)')

# Permutation
idx_perm = np.argsort(perm.importances_mean)[-top_n:]
axes[1].barh([feat_cols[i] for i in idx_perm], perm.importances_mean[idx_perm],
             xerr=perm.importances_std[idx_perm], color=CHART_PALETTE[1])
axes[1].set_title('Permutation Importance')

# Built-in
idx_builtin = np.argsort(builtin_imp)[-top_n:]
axes[2].barh([feat_cols[i] for i in idx_builtin], builtin_imp[idx_builtin],
             color=CHART_PALETTE[2])
axes[2].set_title('Built-in (Gini) Importance')

fig.suptitle('Feature Importance Comparison', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_feature_importance_comparison.png'))
plt.close()
print('4/5 feature importance comparison')

# ============================================================
# 5. LODO Results Bar Chart
# ============================================================
lodo = pd.read_csv(os.path.join(OUTPUT_DIR, 'lodo_results.csv'))
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(lodo))
w = 0.35
ax.bar(x - w/2, lodo['Accuracy'], w, label='Accuracy', color=CHART_PALETTE[0])
ax.bar(x + w/2, lodo['F1_macro'], w, label='F1 macro', color=CHART_PALETTE[1])
ax.set_xticks(x)
ax.set_xticklabels(lodo['Test_Dataset'], rotation=0)
ax.set_ylabel('Score')
ax.set_title('LODO: Per-Dataset Generalization')
ax.legend()
ax.axhline(0.333, color='gray', ls='--', lw=0.8, label='Random baseline')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eval_lodo_bar.png'))
plt.close()
print('5/5 LODO bar chart')

# ============================================================
# Print top features summary
# ============================================================
print('\n=== Top 10 Features by Mean |SHAP| ===')
for i in np.argsort(mean_abs_shap)[::-1][:10]:
    print(f'  {feat_cols[i]}: {mean_abs_shap[i]:.4f}')

print('\nAll evaluation charts saved to output/')
