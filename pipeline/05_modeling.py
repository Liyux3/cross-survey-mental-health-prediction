"""
Phase 5: Classification (primary) + Regression (secondary) + LODO.
Input: pipeline/output/master_imputed.csv
Output: pipeline/output/model_results.csv, pipeline/output/lodo_results.csv, saved models
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, cross_validate)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor,
                               StackingClassifier)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master_imputed.csv'))
print(f'Loaded: {df.shape}')

TARGET = 'mh_risk_class'
COMPOSITE = 'mh_composite'
META = 'source_dataset'

feat_cols = [c for c in df.columns if c not in [TARGET, COMPOSITE, META]]
print(f'Features: {len(feat_cols)}')

X = df[feat_cols].values
y_cls = df[TARGET].values
y_reg = df[COMPOSITE].values
source = df[META].values

le_target = LabelEncoder()
y_cls_enc = le_target.fit_transform(y_cls)

# ============================================================
# 5.1 Classification: Stratified 5-fold CV
# ============================================================
print('\n' + '='*60)
print('CLASSIFICATION (3-class, stratified 5-fold CV)')
print('='*60)

models_cls = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                    random_state=RANDOM_SEED))
    ]),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, class_weight='balanced',
                                            random_state=RANDOM_SEED),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                            class_weight='balanced',
                                            random_state=RANDOM_SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1,
                                                     random_state=RANDOM_SEED),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                   random_state=RANDOM_SEED, use_label_encoder=False,
                                   eval_metric='mlogloss', n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cls_results = []

for name, model in models_cls.items():
    scoring = {'f1_macro': 'f1_macro', 'f1_weighted': 'f1_weighted',
               'accuracy': 'accuracy'}
    scores = cross_validate(model, X, y_cls_enc, cv=cv, scoring=scoring, n_jobs=-1)
    row = {
        'Model': name,
        'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        'F1_macro': f"{scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}",
        'F1_weighted': f"{scores['test_f1_weighted'].mean():.4f} ± {scores['test_f1_weighted'].std():.4f}",
        'F1_macro_mean': scores['test_f1_macro'].mean(),
    }
    cls_results.append(row)
    print(f"  {name}: Acc={row['Accuracy']}, F1m={row['F1_macro']}")

cls_df = pd.DataFrame(cls_results).sort_values('F1_macro_mean', ascending=False)

# ============================================================
# 5.1b Stacking: top 2-3 models
# ============================================================
print('\n--- Stacking Ensemble ---')
top2 = cls_df['Model'].head(2).tolist()
estimators = [(n, models_cls[n]) for n in top2]
stacker = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced',
                                       random_state=RANDOM_SEED),
    cv=3, n_jobs=-1
)
scores = cross_validate(stacker, X, y_cls_enc, cv=cv,
                         scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
                         n_jobs=-1)
stack_row = {
    'Model': f'Stacking({"+".join(top2)})',
    'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
    'F1_macro': f"{scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}",
    'F1_weighted': 'N/A',
    'F1_macro_mean': scores['test_f1_macro'].mean(),
}
cls_results.append(stack_row)
print(f"  {stack_row['Model']}: Acc={stack_row['Accuracy']}, F1m={stack_row['F1_macro']}")

cls_df = pd.DataFrame(cls_results).sort_values('F1_macro_mean', ascending=False)
cls_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_results.csv'), index=False)
print(f'\nClassification results saved.')

# ============================================================
# 5.2 Train best model on 80/20 holdout for final metrics
# ============================================================
print('\n--- Final Hold-out Evaluation (best model) ---')
best_name = cls_df.iloc[0]['Model']
if 'Stacking' in best_name:
    best_model = stacker
else:
    best_model = models_cls[best_name]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls_enc, test_size=0.2, stratify=y_cls_enc, random_state=RANDOM_SEED)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f'\nBest model: {best_name}')
print(f'Hold-out Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Hold-out F1 macro: {f1_score(y_test, y_pred, average="macro"):.4f}')
print(f'\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Save best model
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_classifier.pkl'))
joblib.dump(le_target, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

# Also train best non-stacking for SHAP later (tree-based preferred)
tree_models = [r for r in cls_results if r['Model'] in
               ['XGBoost', 'GradientBoosting', 'RandomForest']]
tree_models.sort(key=lambda x: x['F1_macro_mean'], reverse=True)
best_tree_name = tree_models[0]['Model']
best_tree = models_cls[best_tree_name]
best_tree.fit(X_train, y_train)
joblib.dump(best_tree, os.path.join(OUTPUT_DIR, 'best_tree_classifier.pkl'))
print(f'\nBest tree model for SHAP: {best_tree_name}')

# ============================================================
# 5.3 LODO (Leave-One-Dataset-Out)
# ============================================================
print('\n' + '='*60)
print('LODO (Leave-One-Dataset-Out) Validation')
print('='*60)

# Use best tree model type for consistency
lodo_results = []
datasets = sorted(df[META].unique())

for test_ds in datasets:
    mask_test = source == test_ds
    mask_train = ~mask_test

    X_tr, y_tr = X[mask_train], y_cls_enc[mask_train]
    X_te, y_te = X[mask_test], y_cls_enc[mask_test]

    if len(np.unique(y_te)) < 2:
        print(f'  {test_ds}: skipped (single class in test)')
        continue

    model = models_cls[best_tree_name].__class__(**models_cls[best_tree_name].get_params())
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average='macro', zero_division=0)
    lodo_results.append({
        'Test_Dataset': test_ds,
        'N_test': mask_test.sum(),
        'N_train': mask_train.sum(),
        'Accuracy': acc,
        'F1_macro': f1m,
    })
    print(f'  {test_ds} (n={mask_test.sum()}): Acc={acc:.3f}, F1m={f1m:.3f}')

lodo_df = pd.DataFrame(lodo_results)
lodo_df.to_csv(os.path.join(OUTPUT_DIR, 'lodo_results.csv'), index=False)
print(f'\nLODO mean Acc: {lodo_df["Accuracy"].mean():.3f}')
print(f'LODO mean F1m: {lodo_df["F1_macro"].mean():.3f}')

# ============================================================
# 5.4 Regression (Secondary Track)
# ============================================================
print('\n' + '='*60)
print('REGRESSION (secondary)')
print('='*60)

models_reg = {
    'Ridge': Pipeline([('scaler', StandardScaler()),
                        ('reg', Ridge(alpha=1.0, random_state=RANDOM_SEED))]),
    'Lasso': Pipeline([('scaler', StandardScaler()),
                        ('reg', Lasso(alpha=0.001, random_state=RANDOM_SEED, max_iter=5000))]),
    'RF_Regressor': RandomForestRegressor(n_estimators=200, max_depth=15,
                                           random_state=RANDOM_SEED, n_jobs=-1),
    'XGB_Regressor': xgb.XGBRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=0.1, random_state=RANDOM_SEED,
                                        n_jobs=-1),
}

reg_results = []
for name, model in models_reg.items():
    cv_r2 = cross_val_score(model, X, y_reg, cv=5, scoring='r2')
    cv_neg_mse = cross_val_score(model, X, y_reg, cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(-cv_neg_mse.mean())
    row = {
        'Model': name,
        'R2': f"{cv_r2.mean():.4f} ± {cv_r2.std():.4f}",
        'RMSE': f"{rmse:.4f}",
        'R2_mean': cv_r2.mean(),
    }
    reg_results.append(row)
    print(f"  {name}: R²={row['R2']}, RMSE={row['RMSE']}")

reg_df = pd.DataFrame(reg_results).sort_values('R2_mean', ascending=False)
reg_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_results.csv'), index=False)

print('\n=== Done ===')
print(f'Outputs in {OUTPUT_DIR}')
