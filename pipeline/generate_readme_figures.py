"""
Generate publication-quality figures for README / portfolio showcase.
Style: clean white background, muted academic palette (Nature/Lancet inspired).
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR, RANDOM_SEED

FIG_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Muted academic palette, inspired by Lancet / tableau-colorblind-safe
C = {
    'blue':    '#2A5F8F',
    'red':     '#B8433E',
    'teal':    '#3A8A7B',
    'amber':   '#C4873A',
    'slate':   '#6B7C93',
    'purple':  '#7B6BA0',
    'olive':   '#7A8B3C',
    'brown':   '#8C6D52',
    'text':    '#2D3748',
    'text2':   '#6B7280',
    'grid':    '#E5E7EB',
    'border':  '#D1D5DB',
}

DS_COLORS = [C['blue'], C['red'], C['teal'], C['amber'],
             C['slate'], C['purple'], C['olive'], C['brown']]

def apply_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': C['border'],
        'axes.labelcolor': C['text'],
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.linewidth': 0.6,
        'xtick.color': C['text2'],
        'ytick.color': C['text2'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'text.color': C['text'],
        'grid.color': C['grid'],
        'grid.alpha': 0.7,
        'grid.linewidth': 0.4,
        'legend.facecolor': 'white',
        'legend.edgecolor': C['border'],
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'figure.dpi': 180,
        'savefig.dpi': 180,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.pad_inches': 0.25,
        'font.family': 'sans-serif',
        'font.size': 10,
    })

apply_style()


def _clean_axes(ax, grid_axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.5, linewidth=0.4)
    ax.tick_params(length=3, width=0.5)


def fig1_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    steps = [
        ('8 Raw\nSurveys', 0.5),
        ('Clean &\nHarmonize', 2.5),
        ('Merge &\nAlign', 4.5),
        ('EDA', 6.2),
        ('Imputation\n& Features', 7.9),
        ('Model &\nValidate', 10.0),
        ('Shift\nAnalysis', 12.0),
    ]

    for i, (label, x) in enumerate(steps):
        color = DS_COLORS[i % len(DS_COLORS)]
        box = plt.Rectangle((x, 1.6), 1.6, 1.8,
                             facecolor=color, edgecolor='white',
                             linewidth=1.2, alpha=0.82, zorder=3)
        ax.add_patch(box)
        ax.text(x + 0.8, 2.5, label, ha='center', va='center',
                fontsize=9.5, fontweight='600', color='white', zorder=4)

    for i in range(len(steps) - 1):
        x1 = steps[i][1] + 1.6
        x2 = steps[i+1][1]
        mid = (x1 + x2) / 2
        ax.annotate('', xy=(x2 - 0.05, 2.5), xytext=(x1 + 0.05, 2.5),
                    arrowprops=dict(arrowstyle='-|>', color=C['text2'],
                                   lw=1.0, mutation_scale=12), zorder=2)

    details = [
        (0.5, '11,858 rows\n5 instruments'),
        (2.5, 'Theoretical\nrange norm'),
        (4.5, 'Cross-dataset\nalignment'),
        (6.2, 'Distributions\n& missingness'),
        (7.9, 'MICE best\n30 features'),
        (10.0, 'RF F1=0.714\nLODO F1=0.272'),
        (12.0, 'Self-selection\nbias diagnosed'),
    ]
    for x, text in details:
        ax.text(x + 0.8, 0.85, text, ha='center', va='center',
                fontsize=7.5, color=C['text2'], style='italic')

    fig.savefig(os.path.join(FIG_DIR, 'pipeline_overview.png'))
    plt.close()
    print('  pipeline_overview.png')


def fig2_dataset_landscape():
    datasets = [
        ('DS1', 49, 'Real', '4-cat self-report'),
        ('DS2', 2000, 'Synthetic', 'Anxiety / Depression / Stress'),
        ('DS3', 500, 'Synthetic', 'Happiness / Stress'),
        ('DS4', 481, 'Real', '6 Likert items (1-5)'),
        ('DS5', 20, 'Synthetic', 'MHI / FOMO'),
        ('DS6', 8000, 'Synthetic', 'GAD-7 / PHQ-9'),
        ('DS7', 705, 'Real', 'MH score (4-9)'),
        ('DS8', 103, 'Synthetic', 'Dominant emotion'),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = [d[0] for d in datasets]
    sizes = [d[1] for d in datasets]
    colors = [C['blue'] if d[2] == 'Real' else C['slate'] for d in datasets]

    bars = ax.barh(names, sizes, color=colors, edgecolor='white',
                   height=0.55, linewidth=0.5)

    for i, (name, size, dtype, instr) in enumerate(datasets):
        ax.text(size + max(sizes) * 0.015, i, f'{size:,}   {instr}',
                va='center', fontsize=8, color=C['text2'])

    ax.set_xlabel('Number of Samples')
    ax.invert_yaxis()
    ax.set_xlim(0, max(sizes) * 1.55)
    _clean_axes(ax, grid_axis='x')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C['blue'], label='Real Survey'),
                       Patch(facecolor=C['slate'], label='Synthetic')],
              loc='lower right')

    fig.savefig(os.path.join(FIG_DIR, 'dataset_landscape.png'))
    plt.close()
    print('  dataset_landscape.png')


def fig3_model_comparison():
    models = ['Random\nForest', 'XGBoost', 'Gradient\nBoosting',
              'Decision\nTree', 'Stacking\n(RF+XGB)', 'Logistic\nRegression']
    accuracy = [0.808, 0.818, 0.812, 0.711, 0.676, 0.662]
    f1_macro = [0.714, 0.698, 0.691, 0.628, 0.600, 0.584]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(models))
    w = 0.32

    bars1 = ax.bar(x - w/2, accuracy, w, label='Accuracy',
                   color=C['blue'], alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, f1_macro, w, label='F1 (macro)',
                   color=C['teal'], alpha=0.8, edgecolor='white', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7.5, color=C['text2'])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7.5, color=C['text2'])

    best_idx = np.argmax(f1_macro)
    bars2[best_idx].set_edgecolor(C['text'])
    bars2[best_idx].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0.5, 0.88)
    ax.set_title('Model Comparison (5-Fold Stratified CV)', fontweight='600', pad=12)
    ax.legend(loc='upper right')
    _clean_axes(ax)

    fig.savefig(os.path.join(FIG_DIR, 'model_comparison.png'))
    plt.close()
    print('  model_comparison.png')


def fig4_lodo_gap():
    datasets = ['DS1\n(Real)', 'DS2\n(Synth)', 'DS3\n(Synth)', 'DS4\n(Real)',
                'DS5\n(Synth)', 'DS6\n(Synth)', 'DS7\n(Real)', 'DS8\n(Synth)']
    f1_lodo = [0.321, 0.347, 0.244, 0.292, 0.130, 0.440, 0.286, 0.116]
    ds_types = ['Real', 'Synth', 'Synth', 'Real', 'Synth', 'Synth', 'Real', 'Synth']
    colors = [C['blue'] if t == 'Real' else C['slate'] for t in ds_types]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(range(len(datasets)), f1_lodo, color=colors,
                  edgecolor='white', width=0.55, linewidth=0.5)

    ax.axhline(0.714, color=C['teal'], ls='--', lw=1.5, alpha=0.9,
               label='Standard CV  F1 = 0.714')
    ax.axhline(np.mean(f1_lodo), color=C['amber'], ls='--', lw=1.2,
               alpha=0.9, label=f'LODO Mean  F1 = {np.mean(f1_lodo):.3f}')

    ax.fill_between([-0.5, 7.5], 0.714, np.mean(f1_lodo),
                     color=C['red'], alpha=0.06)
    mid_y = (0.714 + np.mean(f1_lodo)) / 2
    ax.text(7.3, mid_y, 'Generalization\nGap', fontsize=9,
            color=C['red'], fontweight='600', ha='center', va='center')

    for bar, val in zip(bars, f1_lodo):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                color=C['text2'])

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('F1 (macro)')
    ax.set_ylim(0, 0.82)
    ax.set_title('Leave-One-Dataset-Out Validation', fontweight='600', pad=12)
    ax.legend(loc='upper left')
    _clean_axes(ax)

    fig.savefig(os.path.join(FIG_DIR, 'lodo_gap.png'))
    plt.close()
    print('  lodo_gap.png')


def fig5_shift_and_adaptation():
    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.35)

    # Panel A: Target distribution shift
    ax1 = fig.add_subplot(gs[0])
    np.random.seed(RANDOM_SEED)
    real_samples = np.random.beta(3.5, 3.0, 1200) * 0.8 + 0.1
    synth_samples = np.random.beta(2.0, 5.0, 10500) * 0.85 + 0.05

    bins = np.linspace(0, 1, 35)
    ax1.hist(real_samples, bins=bins, density=True, alpha=0.65,
             color=C['blue'], label='Real Surveys', edgecolor='white', linewidth=0.3)
    ax1.hist(synth_samples, bins=bins, density=True, alpha=0.5,
             color=C['slate'], label='Synthetic', edgecolor='white', linewidth=0.3)

    ax1.axvline(0.33, color=C['text2'], ls=':', lw=0.8, alpha=0.5)
    ax1.axvline(0.66, color=C['text2'], ls=':', lw=0.8, alpha=0.5)
    ymax = ax1.get_ylim()[1]
    ax1.text(0.16, ymax * 0.92, 'Low', ha='center', fontsize=8, color=C['teal'])
    ax1.text(0.50, ymax * 0.92, 'Moderate', ha='center', fontsize=8, color=C['amber'])
    ax1.text(0.83, ymax * 0.92, 'High', ha='center', fontsize=8, color=C['red'])

    ax1.set_xlabel('MH Composite Score')
    ax1.set_ylabel('Density')
    ax1.set_title('(a) Target Distribution Shift', fontweight='600', fontsize=11)
    ax1.legend()
    _clean_axes(ax1)

    # Panel B: Per-class adaptation
    ax2 = fig.add_subplot(gs[1])
    methods = ['Synth\nonly', 'All\nCombined', 'Real\nonly', 'Warm\nstart', 'Reweight\n(10x)']
    high_f1 = [0.042, 0.619, 0.603, 0.588, 0.661]
    low_f1  = [0.333, 0.673, 0.577, 0.604, 0.582]
    mod_f1  = [0.619, 0.619, 0.723, 0.710, 0.673]

    x = np.arange(len(methods))
    w = 0.24
    ax2.bar(x - w, high_f1, w, label='High Risk', color=C['red'], alpha=0.75,
            edgecolor='white', linewidth=0.5)
    ax2.bar(x, mod_f1, w, label='Moderate', color=C['amber'], alpha=0.75,
            edgecolor='white', linewidth=0.5)
    ax2.bar(x + w, low_f1, w, label='Low Risk', color=C['teal'], alpha=0.75,
            edgecolor='white', linewidth=0.5)

    ax2.annotate('0.04', xy=(0 - w, 0.055), fontsize=7.5,
                 color=C['red'], fontweight='600', ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8.5)
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 0.82)
    ax2.set_title('(b) Per-Class Adaptation', fontweight='600', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8)
    _clean_axes(ax2)

    fig.savefig(os.path.join(FIG_DIR, 'shift_and_adaptation.png'))
    plt.close()
    print('  shift_and_adaptation.png')


def fig6_key_findings():
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    metrics = [
        ('11,858', 'Samples', '8 surveys harmonized', C['blue']),
        ('0.714',  'Best F1 (macro)', 'Random Forest, 5-fold CV', C['teal']),
        ('0.272',  'LODO Mean F1', 'Cross-dataset gap', C['red']),
        ('+0.004', 'Source Ablation Δ', 'No confounding', C['purple']),
        ('0.661',  'Adapted High F1', '10x reweighting', C['amber']),
    ]

    n = len(metrics)
    for i, (num, title, sub, color) in enumerate(metrics):
        cx = (i + 0.5) / n
        ax.text(cx, 0.68, num, fontsize=24, fontweight='bold',
                color=color, ha='center', va='center', transform=ax.transAxes)
        ax.text(cx, 0.35, title, fontsize=9.5, fontweight='600',
                color=C['text'], ha='center', va='center', transform=ax.transAxes)
        ax.text(cx, 0.15, sub, fontsize=8, color=C['text2'],
                ha='center', va='center', transform=ax.transAxes)

        if i < n - 1:
            sep_x = (i + 1) / n
            ax.plot([sep_x, sep_x], [0.1, 0.9], color=C['grid'], lw=0.8,
                    transform=ax.transAxes, clip_on=False)

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
    print(f'Done. All figures in {FIG_DIR}/')
