<p align="center">
  <h1 align="center">Cross-Survey Mental Health Risk Prediction</h1>
  <p align="center">
    Harmonizing 8 heterogeneous mental health surveys into a unified risk prediction pipeline,<br>
    with cross-dataset validation exposing the real-synthetic generalization gap.
  </p>
  <p align="center">
    <a href="#key-findings">Findings</a> &middot;
    <a href="#pipeline-architecture">Pipeline</a> &middot;
    <a href="#results">Results</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="report.pdf">Report</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="sklearn">
    <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost">
    <img src="https://img.shields.io/badge/SHAP-4B0082?style=flat-square" alt="SHAP">
    <img src="https://img.shields.io/github/license/Liyux3/cross-survey-mental-health-prediction?style=flat-square&color=10B981" alt="License">
  </p>
</p>

<br>

<p align="center">
  <img src="figures/key_findings.png" alt="Key Findings" width="100%">
</p>

## The Problem

Mental health surveys measure the same construct with wildly different instruments: GAD-7 (0-21), PHQ-9 (0-27), Likert scales (1-5), binary yes/no. Naively rescaling all scores to a common range introduces confounders: models learn to predict *which dataset a sample came from* rather than actual mental health risk.

This project tackles the harmonization problem with **theoretically grounded normalization**, then rigorously validates whether the resulting model generalizes across datasets or merely memorizes dataset identity.

## Key Findings

- **Source ablation delta of +0.004** confirms our target construction eliminates dataset-identity confounding
- **LODO validation** (train on 7, test on the 8th) exposes a generalization gap: standard CV F1 = 0.714 vs LODO mean F1 = 0.272
- The gap traces to **self-selection bias**: real survey respondents over-report MH concerns (32% High Risk) vs synthetic data (4% High Risk)
- **Reweighting real samples 10x** recovers High Risk F1 from 0.04 to 0.66
- **7 features remain stable** across all 8 LODO folds: screen time, late-night usage, social comparison, sleep duration, age, gender

## Dataset Landscape

<p align="center">
  <img src="figures/dataset_landscape.png" alt="Dataset Landscape" width="85%">
</p>

8 Kaggle survey datasets (3 real, 5 synthetic), totaling **11,858 samples**. Each uses different MH instruments, from standardized clinical scales (GAD-7, PHQ-9) to ad-hoc Likert items. After harmonization: **30 features**, **3-class target** (Low / Moderate / High Risk).

<details>
<summary><b>Dataset Details</b></summary>
<br>

| # | Source | Rows | MH Instruments | Type |
|---|--------|-----:|---------------|:----:|
| 1 | Google Forms survey | 49 | 4-category self-report | Real |
| 2 | CDC/WHO distribution synthetic | 2,000 | Anxiety/Depression/Stress (0-20), Mood (1-10) | Synth |
| 3 | Designed-correlation synthetic | 500 | Happiness (1-10), Stress (1-10) | Synth |
| 4 | University stats course survey | 481 | 6 Likert items (1-5) | Real |
| 5 | Synthetic generator | 20 | MHI (25-90), FOMO (2-10) | Synth |
| 6 | Scientifically grounded synthetic | 8,000 | GAD-7 (0-21), PHQ-9 (0-27) | Synth |
| 7 | Student survey | 705 | MH score (4-9) | Real |
| 8 | Synthetic (E. BULUT) | 103 | Dominant emotion (6-cat) | Synth |

**Target construction:** Per-dataset MH variables are normalized to [0, 1] using theoretical instrument ranges (not observed min-max), then averaged into a composite score. Subjective psychological states (anxiety, depression, stress) become the target; observable behaviors (screen time, sleep duration) become features.
</details>

## Pipeline Architecture

<p align="center">
  <img src="figures/pipeline_overview.png" alt="Pipeline Architecture" width="100%">
</p>

| Step | Script | What it does |
|:----:|--------|-------------|
| 1 | `01_clean_and_target.py` | Per-dataset cleaning, column standardization, composite target construction |
| 2 | `02_merge.py` | Cross-dataset feature alignment, sanity checks |
| 3 | `03_eda.py` | Distribution analysis, correlation mapping, missingness profiling |
| 4 | `04_imputation_fe.py` | Imputation comparison (Median vs KNN vs **MICE**), interaction features |
| 5 | `05_modeling.py` | 6-model classification, LODO validation, regression baseline |
| 6a | `06_evaluation.py` | SHAP analysis, confusion matrices, feature importance |
| 6b | `06b_extended_analysis.py` | Real vs synthetic split, source ablation, feature consistency |
| 6c | `06c_pretrain_finetune.py` | Synthetic-to-real transfer: reweighting, warm-start, calibration |
| 6d | `06d_deepdive_shift.py` | KS-test distribution shift diagnosis, per-class adaptation |
| 6e | `06e_shap_with_source.py` | SHAP with dataset identity as feature (confounder verification) |

## Results

### Model Comparison

<p align="center">
  <img src="figures/model_comparison.png" alt="Model Comparison" width="85%">
</p>

Random Forest achieves the best F1 macro (0.714) by balancing minority class recall. XGBoost leads on accuracy (0.818) but under-detects High Risk. Stacking underperforms because the base models (RF + XGB) are too similar for the LR meta-learner to exploit.

### The Generalization Gap

<p align="center">
  <img src="figures/lodo_gap.png" alt="LODO Gap" width="85%">
</p>

Standard 5-fold CV reports F1 = 0.714. LODO reveals the model cannot generalize across datasets (mean F1 = 0.272). Within-type transfer works well (Real-to-Real F1 = 0.925), but cross-type fails, pointing to a systematic shift rather than random noise.

### Distribution Shift and Adaptation

<p align="center">
  <img src="figures/shift_and_adaptation.png" alt="Shift and Adaptation" width="100%">
</p>

**Root cause:** Self-selection bias in voluntary MH surveys. Respondents with mental health concerns are overrepresented in real data, creating a target distribution mismatch (KS = 0.456). Synthetic generators approximate general population distributions instead.

**Fix:** Reweighting real samples 10x during training recovers High Risk detection (F1: 0.04 to 0.66) with minimal trade-off on other classes. Warm-start transfer preserves feature ranking (Spearman rho = 0.736) but does not improve calibration.

<details>
<summary><b>More Results: Feature Importance and Stability</b></summary>
<br>

**SHAP Top 5:** late-night usage (0.100), screen_time x late_night interaction (0.089), social comparison (0.059), daily screen time (0.053), sleep duration (0.022).

When dataset identity is added as a feature, it ranks 5th (SHAP = 0.030), confirming it carries some signal but is not dominant. The source ablation delta (+0.004 F1) corroborates this: our target construction successfully prevents the model from relying on dataset identity.

**Feature stability across LODO folds:** 7 features appear in the top-10 importance across all 8 folds, suggesting the model captures genuine behavioral-MH associations rather than dataset-specific artifacts.
</details>

## Quick Start

```bash
git clone https://github.com/Liyux3/cross-survey-mental-health-prediction.git
cd cross-survey-mental-health-prediction
pip install -r requirements.txt
```

Run the pipeline sequentially:

```bash
cd pipeline
python 01_clean_and_target.py    # Clean + harmonize → output/all_cleaned.csv
python 02_merge.py               # Align features    → output/master.csv
python 03_eda.py                 # EDA charts        → output/eda_*.png
python 04_imputation_fe.py       # MICE imputation   → output/master_imputed.csv
python 05_modeling.py            # Train + LODO      → output/*.csv, *.pkl
python 06_evaluation.py          # SHAP + eval       → output/eval_*.png
python 06b_extended_analysis.py  # Real vs synthetic  → output/eval_*.png
python 06c_pretrain_finetune.py  # Domain adaptation  → output/eval_*.png
python 06d_deepdive_shift.py     # Shift diagnosis    → output/eval_*.png
python 06e_shap_with_source.py   # Confounder check   → output/eval_*.png
```

<details>
<summary><b>Requirements</b></summary>
<br>

Python 3.9+ with: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`

Full list in [`requirements.txt`](requirements.txt).
</details>

## Report

The full 6-page technical report with methodology details, justifications, and additional analyses is available at [`report.pdf`](report.pdf).

## License

[MIT](LICENSE)
