"""
Phase 4a: EDA — distributions, correlations, key visualizations.
Input: pipeline/output/master.csv
Output: charts in pipeline/output/
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, CHART_PALETTE

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'
})

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'master.csv'))

# ============================================================
# 1. Target composite distribution per dataset
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
for i, ds in enumerate(sorted(df['source_dataset'].unique())):
    ax = axes[i // 4, i % 4]
    sub = df[df['source_dataset'] == ds]['mh_composite']
    ax.hist(sub, bins=30, color=CHART_PALETTE[i], alpha=0.8, edgecolor='white')
    ax.set_title(f'{ds} (n={len(sub)})')
    ax.axvline(0.33, color='gray', ls='--', lw=0.8)
    ax.axvline(0.66, color='gray', ls='--', lw=0.8)
    ax.set_xlim(0, 1)
fig.suptitle('MH Composite Distribution per Dataset', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_composite_dist.png'))
plt.close()
print('1/6 composite distributions')

# ============================================================
# 2. Class balance stacked bar
# ============================================================
ct = pd.crosstab(df['source_dataset'], df['mh_risk_class'], normalize='index')
ct = ct[['Low', 'Moderate', 'High']]
fig, ax = plt.subplots(figsize=(10, 5))
ct.plot(kind='bar', stacked=True, color=['#4CAF50', '#FFC107', '#F44336'], ax=ax)
ax.set_ylabel('Proportion')
ax.set_title('Class Distribution per Dataset')
ax.legend(title='Risk Class')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_class_balance.png'))
plt.close()
print('2/6 class balance')

# ============================================================
# 3. Universal features: screen_time and age distributions
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (col, label) in enumerate([
    ('daily_screen_time_hours', 'Daily Screen Time (hours)'),
    ('age', 'Age')
]):
    ax = axes[i]
    for j, ds in enumerate(sorted(df['source_dataset'].unique())):
        sub = df[df['source_dataset'] == ds][col].dropna()
        if len(sub) > 10:
            ax.hist(sub, bins=25, alpha=0.4, label=ds, color=CHART_PALETTE[j])
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.set_title(f'{label} Distribution by Dataset')

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_universal_features.png'))
plt.close()
print('3/6 universal features')

# ============================================================
# 4. Correlation heatmap: numeric features vs composite
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['mh_composite', 'mh_risk_class']]
corr_with_target = df[numeric_cols + ['mh_composite']].corr()['mh_composite'].drop('mh_composite')
corr_with_target = corr_with_target.dropna().sort_values()

fig, ax = plt.subplots(figsize=(10, max(6, len(corr_with_target) * 0.3)))
colors = ['#F44336' if v > 0 else '#2196F3' for v in corr_with_target.values]
ax.barh(corr_with_target.index, corr_with_target.values, color=colors, height=0.7)
ax.set_xlabel('Correlation with MH Composite')
ax.set_title('Feature Correlations with Target (all data pooled)')
ax.axvline(0, color='black', lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_corr_with_target.png'))
plt.close()
print('4/6 correlation with target')

# ============================================================
# 5. Screen time vs composite — KDE density heatmap
# ============================================================
tmp_st = df[['daily_screen_time_hours', 'mh_composite']].dropna()
r_st = tmp_st['daily_screen_time_hours'].corr(tmp_st['mh_composite'])

fig, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Key Feature – Mental Health Composite Interactions',
             fontsize=14, fontweight='bold')

ax = axes5[0]
sns.kdeplot(data=tmp_st, x='daily_screen_time_hours', y='mh_composite',
            fill=True, cmap='YlOrRd', thresh=0.02, levels=15, ax=ax)
z = np.polyfit(tmp_st['daily_screen_time_hours'], tmp_st['mh_composite'], 1)
xs = np.linspace(tmp_st['daily_screen_time_hours'].quantile(0.01),
                 tmp_st['daily_screen_time_hours'].quantile(0.99), 100)
ax.plot(xs, np.poly1d(z)(xs), color='#111827', linewidth=2.5, linestyle='--')
ax.set_title(f'Screen Time vs MH Composite\n(r = {r_st:.3f})',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Daily Screen Time (hours)', fontsize=10)
ax.set_ylabel('MH Composite (higher = worse)', fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

tmp_ln = df[['late_night_usage', 'mh_composite']].dropna()
r_ln = tmp_ln['late_night_usage'].corr(tmp_ln['mh_composite'])
ax = axes5[1]
sns.kdeplot(data=tmp_ln, x='late_night_usage', y='mh_composite',
            fill=True, cmap='YlGn', thresh=0.02, levels=15, ax=ax)
z2 = np.polyfit(tmp_ln['late_night_usage'], tmp_ln['mh_composite'], 1)
xs2 = np.linspace(tmp_ln['late_night_usage'].quantile(0.01),
                  tmp_ln['late_night_usage'].quantile(0.99), 100)
ax.plot(xs2, np.poly1d(z2)(xs2), color='#111827', linewidth=2.5, linestyle='--')
ax.set_title(f'Late Night Usage vs MH Composite\n(r = {r_ln:.3f})',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Late Night Usage', fontsize=10)
ax.set_ylabel('MH Composite (higher = worse)', fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'eda_screentime_vs_composite.png'))
plt.close()
print('5/6 KDE density: screen_time + late_night vs composite')

# ============================================================
# 6. Missing rate heatmap (dataset × feature)
# ============================================================
feature_cols = [c for c in df.columns if c not in ['mh_composite', 'mh_risk_class', 'source_dataset']]
miss_by_ds = df.groupby('source_dataset')[feature_cols].apply(lambda x: x.isnull().mean())
miss_by_ds = miss_by_ds[miss_by_ds.max(axis=0).sort_values(ascending=False).index]
cols_with_miss = miss_by_ds.columns[miss_by_ds.max() > 0]

if len(cols_with_miss) > 0:
    fig, ax = plt.subplots(figsize=(max(12, len(cols_with_miss) * 0.4), 5))
    sns.heatmap(miss_by_ds[cols_with_miss], cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Missing Rate'}, linewidths=0.5)
    ax.set_title('Missing Rate: Dataset × Feature')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'eda_missing_heatmap.png'))
    plt.close()
print('6/6 missing heatmap')

# ============================================================
# Summary stats
# ============================================================
print('\n=== Key Stats ===')
print(f'Total rows: {len(df)}')
print(f'Features: {len(feature_cols)}')
print(f'Numeric features: {len(numeric_cols)}')
cat_cols = df[feature_cols].select_dtypes(include=['object']).columns
print(f'Categorical features: {len(cat_cols)} ({", ".join(cat_cols)})')
print(f'\nScreen time range: {df["daily_screen_time_hours"].min():.1f} - {df["daily_screen_time_hours"].max():.1f} hours')
print(f'Age range: {df["age"].min():.0f} - {df["age"].max():.0f}')
print(f'\nFeatures with >90% missing (dataset-specific, will rely on imputation or drop):')
high_miss = miss_by_ds.columns[miss_by_ds.mean() > 0.9]
for c in high_miss:
    present = miss_by_ds.index[miss_by_ds[c] < 1.0].tolist()
    print(f'  {c} — only in {", ".join(present)}')
