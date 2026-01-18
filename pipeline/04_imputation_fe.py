"""
Phase 4b: Imputation comparison + feature engineering.
Input: pipeline/output/master.csv
Output: pipeline/output/master_imputed.csv (best imputation method applied)
        pipeline/output/imputation_comparison.csv (results table)

Strategy for structural missing (features only in 1-2 datasets):
- Features present in <=2 datasets (>80% missing) are dropped from the shared model.
  They contribute zero information for samples from other datasets, and imputing 80%+
  would inject noise. These features still appear in per-dataset analysis if needed.
- Features present in 3+ datasets are imputed.
- Universal features (age, gender, screen_time, platform) have <20% missing, imputed normally.
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master.csv'))
print(f'Loaded: {df.shape}')

TARGET = 'mh_risk_class'
COMPOSITE = 'mh_composite'
META = 'source_dataset'

# ============================================================
# Step 1: Identify which features to keep
# ============================================================
feature_cols = [c for c in df.columns if c not in [TARGET, COMPOSITE, META]]
miss_rate = df[feature_cols].isnull().mean()

# Keep features with <80% overall missing (present in 3+ datasets effectively)
keep_features = miss_rate[miss_rate < 0.80].index.tolist()
drop_features = miss_rate[miss_rate >= 0.80].index.tolist()

print(f'\nKeeping {len(keep_features)} features (< 80% missing)')
print(f'Dropping {len(drop_features)} features (>= 80% missing, dataset-specific)')
print(f'Dropped: {drop_features}')

df_work = df[keep_features + [TARGET, COMPOSITE, META]].copy()

# ============================================================
# Step 2: Encode categoricals for imputation
# ============================================================
cat_cols = df_work[keep_features].select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in keep_features if c not in cat_cols]

print(f'\nCategorical: {cat_cols}')
print(f'Numeric: {len(num_cols)} columns')

# Label encode categoricals (will decode after imputation)
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    mask = df_work[col].notna()
    encoded = le.fit_transform(df_work.loc[mask, col].astype(str))
    new_series = pd.Series(np.nan, index=df_work.index, dtype=float)
    new_series.loc[mask] = encoded.astype(float)
    df_work[col] = new_series
    le_dict[col] = le

all_feat = num_cols + cat_cols

# ============================================================
# Step 3: Compare imputation methods
# ============================================================
X_raw = df_work[all_feat].values
y = df_work[TARGET].values

def evaluate_imputation(X_imp, y, name):
    """Quick RF classification to compare imputation quality."""
    valid = ~pd.isna(y)
    X_v, y_v = X_imp[valid], y[valid]
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                 class_weight='balanced', n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X_v, y_v, cv=cv, scoring='f1_macro')
    print(f'  {name}: F1_macro = {scores.mean():.4f} ± {scores.std():.4f}')
    return scores.mean(), scores.std()

results = []

# Method 1: Median/Mode
print('\n--- Imputation Comparison (5-fold RF, balanced) ---')
imp_median = SimpleImputer(strategy='median')
X_med = imp_median.fit_transform(X_raw)
m, s = evaluate_imputation(X_med, y, 'Median/Mode')
results.append(('Median/Mode', m, s))

# Method 2: KNN (k=5)
imp_knn = KNNImputer(n_neighbors=5)
X_knn = imp_knn.fit_transform(X_raw)
m, s = evaluate_imputation(X_knn, y, 'KNN (k=5)')
results.append(('KNN (k=5)', m, s))

# Method 3: MICE (IterativeImputer)
imp_mice = IterativeImputer(max_iter=10, random_state=RANDOM_SEED, sample_posterior=False)
X_mice = imp_mice.fit_transform(X_raw)
m, s = evaluate_imputation(X_mice, y, 'MICE')
results.append(('MICE', m, s))

# Save comparison
res_df = pd.DataFrame(results, columns=['Method', 'F1_macro_mean', 'F1_macro_std'])
res_df.to_csv(os.path.join(OUTPUT_DIR, 'imputation_comparison.csv'), index=False)
print(f'\nImputation comparison saved.')

# ============================================================
# Step 4: Apply best imputation method
# ============================================================
best_method = res_df.loc[res_df['F1_macro_mean'].idxmax(), 'Method']
print(f'\nBest method: {best_method}')

if 'KNN' in best_method:
    X_final = X_knn
elif 'MICE' in best_method:
    X_final = X_mice
else:
    X_final = X_med

# Rebuild DataFrame
df_imputed = pd.DataFrame(X_final, columns=all_feat, index=df_work.index)

# Decode categoricals back
for col in cat_cols:
    le = le_dict[col]
    df_imputed[col] = le.inverse_transform(df_imputed[col].round().astype(int).clip(
        0, len(le.classes_) - 1))

df_imputed[TARGET] = df_work[TARGET].values
df_imputed[COMPOSITE] = df_work[COMPOSITE].values
df_imputed[META] = df_work[META].values

# ============================================================
# Step 5: Feature engineering
# ============================================================

# Interaction: screen_time x late_night_usage (where available, otherwise imputed)
if 'late_night_usage' in df_imputed.columns:
    df_imputed['screen_x_latenight'] = (
        df_imputed['daily_screen_time_hours'] * df_imputed['late_night_usage'])

# One-hot encode categoricals
cat_cols_final = df_imputed.select_dtypes(include=['object']).columns.tolist()
cat_cols_final = [c for c in cat_cols_final if c not in [TARGET, META]]

df_encoded = pd.get_dummies(df_imputed, columns=cat_cols_final, drop_first=False)

# Drop any remaining NaN in target
df_encoded = df_encoded.dropna(subset=[TARGET])

print(f'\nFinal dataset: {df_encoded.shape}')
print(f'Class distribution:\n{df_encoded[TARGET].value_counts()}')

out_path = os.path.join(OUTPUT_DIR, 'master_imputed.csv')
df_encoded.to_csv(out_path, index=False)
print(f'Saved: {out_path}')

# Feature list for reference
feat_final = [c for c in df_encoded.columns if c not in [TARGET, COMPOSITE, META]]
print(f'\nFinal feature count: {len(feat_final)}')
