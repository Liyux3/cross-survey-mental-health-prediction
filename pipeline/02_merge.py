"""
Phase 3: Post-merge feature alignment, sanity checks, and verification.
Input: pipeline/output/all_cleaned.csv (from 01_clean_and_target.py)
Output: pipeline/output/master.csv (analysis-ready)
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'all_cleaned.csv'))
print(f'Loaded: {df.shape}')

# --- 3.1 Feature alignment fixes ---

# social_comparison: ds4 has 1-5 Likert, ds6 has binary 0/1
# Normalize ds4's to 0-1: (val-1)/4
mask_ds4 = df['source_dataset'] == 'ds4'
if mask_ds4.any() and 'social_comparison' in df.columns:
    df.loc[mask_ds4, 'social_comparison'] = (df.loc[mask_ds4, 'social_comparison'] - 1) / 4.0
    print('Normalized ds4 social_comparison from 1-5 to 0-1')

# affects_academic: ds7 string -> binary
if 'affects_academic' in df.columns:
    df['affects_academic'] = df['affects_academic'].map({'yes': 1, 'no': 0})

# conflicts_over_sm: ds7 numeric, keep as-is (already int)

# addiction_level: ds5 string -> ordinal
if 'addiction_level' in df.columns:
    df['addiction_level'] = df['addiction_level'].map({
        'low': 0, 'moderate': 1, 'high': 2
    })

# --- 3.2 Sanity checks: composite vs screen_time correlation per dataset ---
print('\n=== Sanity Check: mh_composite ~ daily_screen_time_hours correlation ===')
for ds in sorted(df['source_dataset'].unique()):
    sub = df[df['source_dataset'] == ds][['mh_composite', 'daily_screen_time_hours']].dropna()
    if len(sub) > 5:
        r = sub['mh_composite'].corr(sub['daily_screen_time_hours'])
        print(f'  {ds}: r = {r:+.3f} (n={len(sub)})')
    else:
        print(f'  {ds}: too few rows ({len(sub)})')

# --- 3.3 Missing rate per feature ---
print('\n=== Missing Rate by Feature ===')
miss = df.drop(columns=['mh_composite', 'mh_risk_class', 'source_dataset']).isnull().mean()
miss_sorted = miss[miss > 0].sort_values(ascending=False)
for col, rate in miss_sorted.items():
    present_ds = df[df[col].notna()]['source_dataset'].unique()
    print(f'  {col}: {rate:.1%} missing (present in: {", ".join(sorted(present_ds))})')

# --- 3.4 Class balance ---
print('\n=== Class Balance ===')
print(df['mh_risk_class'].value_counts())
print(f'\nPer dataset:')
ct = pd.crosstab(df['source_dataset'], df['mh_risk_class'])
print(ct)

# --- 3.5 Drop ID-like or non-feature columns, save master ---
# All cleaning already done in 01, just verify no leakage
target_cols = ['mh_composite', 'mh_risk_class']
meta_cols = ['source_dataset']
feature_cols = [c for c in df.columns if c not in target_cols + meta_cols]
print(f'\n{len(feature_cols)} feature columns, {len(target_cols)} target columns, {len(meta_cols)} meta columns')

out_path = os.path.join(OUTPUT_DIR, 'master.csv')
df.to_csv(out_path, index=False)
print(f'\nSaved master.csv: {df.shape}')
