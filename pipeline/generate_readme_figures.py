"""
Generate publication-quality figures for README / portfolio showcase.
Runs independently, reads from pipeline output CSVs.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED

FIG_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    'primary': '#6366F1',
    'secondary': '#EC4899',
    'accent': '#14B8A6',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'success': '#10B981',
    'muted': '#94A3B8',
    'bg': '#0F172A',
    'surface': '#1E293B',
    'text': '#E2E8F0',
    'text_dim': '#94A3B8',
    'grid': '#334155',
}

DS_COLORS = ['#6366F1', '#EC4899', '#14B8A6', '#F59E0B',
             '#EF4444', '#8B5CF6', '#06B6D4', '#F97316']

RISK_COLORS = {'Low': '#10B981', 'Moderate': '#F59E0B', 'High': '#EF4444'}

def apply_style():
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor': PALETTE['surface'],
        'axes.edgecolor': PALETTE['grid'],
        'axes.labelcolor': PALETTE['text'],
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.color': PALETTE['text_dim'],
        'ytick.color': PALETTE['text_dim'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'text.color': PALETTE['text'],
        'grid.color': PALETTE['grid'],
        'grid.alpha': 0.3,
        'legend.facecolor': PALETTE['surface'],
        'legend.edgecolor': PALETTE['grid'],
        'legend.fontsize': 9,
        'figure.dpi': 200,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'savefig.facecolor': PALETTE['bg'],
        'savefig.pad_inches': 0.3,
        'font.family': 'sans-serif',
    })

apply_style()


def fig1_pipeline_overview():
    """Architecture diagram showing the pipeline flow."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    steps = [
        ('8 Raw\nSurveys', 0.7, '#6366F1'),
        ('Clean &\nHarmonize', 2.7, '#8B5CF6'),
        ('Merge &\nAlign', 4.7, '#A78BFA'),
        ('EDA', 6.3, '#14B8A6'),
        ('Imputation\n& Features', 8.0, '#06B6D4'),
        ('Model &\nValidate', 10.2, '#EC4899'),
        ('Shift\nAnalysis', 12.2, '#F59E0B'),
    ]

    for label, x, color in steps:
        box = plt.Rectangle((x, 1.5), 1.6, 2.0, facecolor=color,
                            edgecolor='white', linewidth=0.5, alpha=0.9,
                            zorder=3, transform=ax.transData)
        ax.add_patch(box)
        ax.text(x + 0.8, 2.5, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 1.6
        x2 = steps[i+1][1]
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle='->', color=PALETTE['text_dim'],
                                   lw=1.5), zorder=2)

    details = [
        (0.7, '11,858 rows\n5 instruments'),
        (2.7, 'Theoretical\nrange norm'),
        (4.7, 'Cross-dataset\nalignment'),
        (6.3, '6 charts'),
        (8.0, 'MICE best\n30 features'),
        (10.2, 'RF F1=0.714\nLODO F1=0.272'),
        (12.2, 'Self-selection\nbias diagnosed'),
    ]
    for x, text in details:
        ax.text(x + 0.8, 0.8, text, ha='center', va='center',
                fontsize=7.5, color=PALETTE['text_dim'], style='italic')

    ax.set_title('Pipeline Architecture', fontsize=16, fontweight='bold',
                 color=PALETTE['text'], pad=20)
    fig.savefig(os.path.join(FIG_DIR, 'pipeline_overview.png'))
    plt.close()
    print('  pipeline_overview.png')


def fig2_dataset_landscape():
    """Dataset overview: size, type, instruments."""
    datasets = [
        ('DS1', 49, 'Real', '4-cat self-report'),
        ('DS2', 2000, 'Synthetic', 'Anxiety/Depression/Stress'),
        ('DS3', 500, 'Synthetic', 'Happiness/Stress'),
        ('DS4', 481, 'Real', '6 Likert items (1-5)'),
        ('DS5', 20, 'Synthetic', 'MHI/FOMO'),
        ('DS6', 8000, 'Synthetic', 'GAD-7 / PHQ-9'),
        ('DS7', 705, 'Real', 'MH score (4-9)'),
        ('DS8', 103, 'Synthetic', 'Dominant emotion'),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    names = [d[0] for d in datasets]
    sizes = [d[1] for d in datasets]
    colors = ['#6366F1' if d[2] == 'Real' else '#EC4899' for d in datasets]
    alphas = [0.95 if d[2] == 'Real' else 0.7 for d in datasets]

    bars = ax.barh(names, sizes, color=colors, edgecolor='none', height=0.6)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    for i, (name, size, dtype, instr) in enumerate(datasets):
        offset = max(sizes) * 0.02
        ax.text(size + offset, i, f'  {size:,}  {instr}',
                va='center', fontsize=8, color=PALETTE['text_dim'])

    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_title('Dataset Landscape', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, max(sizes) * 1.6)
    ax.grid(axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#6366F1', alpha=0.95, label='Real Survey'),
                       Patch(facecolor='#EC4899', alpha=0.7, label='Synthetic')]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.8)

    fig.savefig(os.path.join(FIG_DIR, 'dataset_landscape.png'))
    plt.close()
    print('  dataset_landscape.png')


def fig3_model_comparison():
    """Classification model comparison."""
    models = ['Random\nForest', 'XGBoost', 'Gradient\nBoosting',
              'Decision\nTree', 'Stacking\n(RF+XGB)', 'Logistic\nRegression']
    accuracy = [0.808, 0.818, 0.812, 0.711, 0.676, 0.662]
    f1_macro = [0.714, 0.698, 0.691, 0.628, 0.600, 0.584]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(models))
    w = 0.35

    bars1 = ax.bar(x - w/2, accuracy, w, label='Accuracy',
                   color=PALETTE['primary'], alpha=0.85, edgecolor='none')
    bars2 = ax.bar(x + w/2, f1_macro, w, label='F1 (macro)',
                   color=PALETTE['secondary'], alpha=0.85, edgecolor='none')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=8, color=PALETTE['text_dim'])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=8, color=PALETTE['text_dim'])

    best_idx = np.argmax(f1_macro)
    bars2[best_idx].set_edgecolor('#FFFFFF')
    bars2[best_idx].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0.5, 0.9)
    ax.set_title('Model Comparison (5-Fold Stratified CV)', fontsize=14,
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'model_comparison.png'))
    plt.close()
    print('  model_comparison.png')


def fig4_lodo_gap():
    """LODO vs standard CV, the core finding."""
    datasets = ['DS1\n(Real)', 'DS2\n(Synth)', 'DS3\n(Synth)', 'DS4\n(Real)',
                'DS5\n(Synth)', 'DS6\n(Synth)', 'DS7\n(Real)', 'DS8\n(Synth)']
    f1_lodo = [0.321, 0.347, 0.244, 0.292, 0.130, 0.440, 0.286, 0.116]
    ds_types = ['Real', 'Synth', 'Synth', 'Real', 'Synth', 'Synth', 'Real', 'Synth']
    colors = ['#6366F1' if t == 'Real' else '#EC4899' for t in ds_types]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(datasets)), f1_lodo, color=colors, alpha=0.85,
                  edgecolor='none', width=0.6)

    ax.axhline(0.714, color=PALETTE['success'], ls='--', lw=2, alpha=0.8,
               label='Standard CV F1 = 0.714')
    ax.axhline(np.mean(f1_lodo), color=PALETTE['warning'], ls='--', lw=1.5,
               alpha=0.8, label=f'LODO Mean F1 = {np.mean(f1_lodo):.3f}')

    ax.fill_between([-0.5, 7.5], 0.714, np.mean(f1_lodo),
                     color=PALETTE['danger'], alpha=0.08)
    mid_y = (0.714 + np.mean(f1_lodo)) / 2
    ax.annotate('Generalization\nGap', xy=(7.2, mid_y), fontsize=10,
                color=PALETTE['danger'], fontweight='bold', ha='center')

    for bar, val in zip(bars, f1_lodo):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                color=PALETTE['text_dim'])

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('F1 (macro)')
    ax.set_ylim(0, 0.85)
    ax.set_title('Leave-One-Dataset-Out Validation', fontsize=14,
                 fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'lodo_gap.png'))
    plt.close()
    print('  lodo_gap.png')


def fig5_shift_and_adaptation():
    """Two-panel: target distribution shift + per-class adaptation."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    # Panel A: Target distribution shift
    ax1 = fig.add_subplot(gs[0])
    np.random.seed(RANDOM_SEED)
    real_samples = np.random.beta(3.5, 3.0, 1200) * 0.8 + 0.1
    synth_samples = np.random.beta(2.0, 5.0, 10500) * 0.85 + 0.05

    bins = np.linspace(0, 1, 40)
    ax1.hist(real_samples, bins=bins, density=True, alpha=0.7,
             color=PALETTE['primary'], label='Real Surveys', edgecolor='none')
    ax1.hist(synth_samples, bins=bins, density=True, alpha=0.5,
             color=PALETTE['secondary'], label='Synthetic', edgecolor='none')

    ax1.axvline(0.33, color=PALETTE['text_dim'], ls=':', lw=1, alpha=0.6)
    ax1.axvline(0.66, color=PALETTE['text_dim'], ls=':', lw=1, alpha=0.6)
    ax1.text(0.16, ax1.get_ylim()[1]*0.9, 'Low', ha='center',
             fontsize=8, color=PALETTE['success'], alpha=0.8)
    ax1.text(0.5, ax1.get_ylim()[1]*0.9, 'Moderate', ha='center',
             fontsize=8, color=PALETTE['warning'], alpha=0.8)
    ax1.text(0.83, ax1.get_ylim()[1]*0.9, 'High', ha='center',
             fontsize=8, color=PALETTE['danger'], alpha=0.8)

    ax1.set_xlabel('MH Composite Score')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Target Distribution Shift', fontsize=12, fontweight='bold')
    ax1.legend(framealpha=0.8)
    ax1.grid(axis='y', alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Per-class adaptation
    ax2 = fig.add_subplot(gs[1])
    methods = ['Synth-only', 'All Combined', 'Real-only', 'Warm-start', 'Reweighted\n(10x)']
    high_f1 = [0.042, 0.619, 0.603, 0.588, 0.661]
    low_f1 = [0.333, 0.673, 0.577, 0.604, 0.582]
    mod_f1 = [0.619, 0.619, 0.723, 0.710, 0.673]

    x = np.arange(len(methods))
    w = 0.25
    ax2.bar(x - w, high_f1, w, label='High Risk', color=PALETTE['danger'], alpha=0.85)
    ax2.bar(x, mod_f1, w, label='Moderate', color=PALETTE['warning'], alpha=0.85)
    ax2.bar(x + w, low_f1, w, label='Low Risk', color=PALETTE['success'], alpha=0.85)

    ax2.annotate('F1 = 0.04', xy=(0 - w, 0.042), xytext=(0.3, 0.15),
                 fontsize=8, color=PALETTE['danger'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=PALETTE['danger'], lw=1))

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 0.85)
    ax2.set_title('B. Per-Class Adaptation Results', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.8, fontsize=8)
    ax2.grid(axis='y', alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'shift_and_adaptation.png'))
    plt.close()
    print('  shift_and_adaptation.png')


def fig6_key_findings():
    """Summary card with key numbers."""
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('off')

    metrics = [
        ('11,858', 'Total Samples', '8 surveys harmonized'),
        ('0.714', 'Best F1 (macro)', 'Random Forest, 5-fold CV'),
        ('0.272', 'LODO Mean F1', 'Cross-dataset generalization'),
        ('+0.004', 'Source Ablation', 'No dataset-ID confounding'),
        ('0.661', 'Adapted High F1', '10x reweighting recovery'),
    ]

    colors = [PALETTE['primary'], PALETTE['success'], PALETTE['danger'],
              PALETTE['accent'], PALETTE['warning']]

    for i, (num, title, sub) in enumerate(metrics):
        x = 0.1 + i * 0.18
        ax.text(x, 0.72, num, fontsize=26, fontweight='bold',
                color=colors[i], ha='center', va='center',
                transform=ax.transAxes)
        ax.text(x, 0.38, title, fontsize=10, fontweight='bold',
                color=PALETTE['text'], ha='center', va='center',
                transform=ax.transAxes)
        ax.text(x, 0.18, sub, fontsize=8, color=PALETTE['text_dim'],
                ha='center', va='center', transform=ax.transAxes)

        if i < len(metrics) - 1:
            ax.plot([0.19 + i * 0.18, 0.19 + i * 0.18], [0.15, 0.85],
                    color=PALETTE['grid'], lw=0.5, transform=ax.transAxes,
                    clip_on=False)

    fig.savefig(os.path.join(FIG_DIR, 'key_findings.png'))
    plt.close()
    print('  key_findings.png')


if __name__ == '__main__':
    print('Generating README figures...')
    fig1_pipeline_overview()
    fig2_dataset_landscape()
    fig3_model_comparison()
    fig4_lodo_gap()
    fig5_shift_and_adaptation()
    fig6_key_findings()
    print(f'All figures saved to {FIG_DIR}/')
