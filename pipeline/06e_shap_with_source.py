"""
Quick SHAP run with source_dataset included (ordinal encoded, single column).
Generates eval_shap_summary_with_source.png for presentation use.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'
})

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_imputed.csv'))
TARGET = 'mh_risk_class'
COMPOSITE = 'mh_composite'

le_src = LabelEncoder()
df['source_dataset_enc'] = le_src.fit_transform(df['source_dataset'])

feat_cols = [c for c in df.columns if c not in [TARGET, COMPOSITE, 'source_dataset']]
print(f"Features ({len(feat_cols)}): includes source_dataset_enc")

X = df[feat_cols].values
le_target = LabelEncoder()
y = le_target.fit_transform(df[TARGET].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

from sklearn.metrics import f1_score, accuracy_score
y_pred = rf.predict(X_test)
print(f"Acc: {accuracy_score(y_test, y_pred):.3f}, F1m: {f1_score(y_test, y_pred, average='macro'):.3f}")

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

sv_array = np.array(shap_values)  # could be (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
print(f"shap_values shape: {sv_array.shape}")
if sv_array.ndim == 3 and sv_array.shape[1] == len(feat_cols):
    # (n_samples, n_features, n_classes)
    mean_abs_shap = np.abs(sv_array).mean(axis=(0, 2))
elif sv_array.ndim == 3 and sv_array.shape[2] == len(feat_cols):
    # (n_classes, n_samples, n_features)
    mean_abs_shap = np.abs(sv_array).mean(axis=(0, 1))
else:
    # fallback: list of arrays
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
print(f"mean_abs_shap shape: {mean_abs_shap.shape}")
rank = np.argsort(mean_abs_shap)[::-1]
print("\nTop 15 by mean |SHAP|:")
for i in range(min(15, len(rank))):
    idx = rank[i]
    marker = " ← source_dataset" if feat_cols[idx] == 'source_dataset_enc' else ""
    print(f"  {i+1:2d}. {feat_cols[idx]}: {mean_abs_shap[idx]:.4f}{marker}")

src_idx = feat_cols.index('source_dataset_enc')
src_rank = int(np.sum(mean_abs_shap > mean_abs_shap[src_idx])) + 1
print(f"\nsource_dataset_enc rank: {src_rank}/{len(feat_cols)}")

# Convert to list-of-arrays format for summary_plot
sv_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Bar chart (multi-class overview)
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_list, X_test, feature_names=feat_cols,
                  class_names=le_target.classes_, show=False, max_display=20)
plt.title('SHAP Summary (with source_dataset)')
plt.savefig(os.path.join(OUTPUT_DIR, 'eval_shap_summary_with_source.png'))
plt.close()
print(f"\nSaved: eval_shap_summary_with_source.png")

# Beeswarm per class (shows direction)
for i, cls in enumerate(le_target.classes_):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[:, :, i], X_test, feature_names=feat_cols,
                      show=False, max_display=20)
    plt.title(f'SHAP Beeswarm — {cls} Risk (with source_dataset)')
    plt.savefig(os.path.join(OUTPUT_DIR, f'eval_shap_beeswarm_{cls.lower()}_with_source.png'))
    plt.close()
    print(f"Saved: eval_shap_beeswarm_{cls.lower()}_with_source.png")

