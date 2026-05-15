"""
Add a new survey dataset to the pipeline using a YAML config file.

Usage:
    python add_dataset.py my_survey.yaml              # clean + merge only
    python add_dataset.py my_survey.yaml --retrain     # also retrain model

The config describes column mappings and MH target variables with their
theoretical ranges. The script normalizes targets to [0,1] (1 = worst),
averages them into a composite, and appends to the existing merged data.
"""
import os, sys, argparse
import pandas as pd
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, OUTPUT_DIR, normalize_platform, normalize_gender

STANDARD_FEATURES = [
    'age', 'gender', 'daily_screen_time_hours', 'sleep_duration_hours',
    'primary_platform', 'social_comparison', 'late_night_usage',
    'dominant_content_type', 'activity_type',
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(config):
    filepath = os.path.join(DATA_DIR, config['file'])
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f'Unsupported file format: {filepath}')


def clean_dataset(df, config):
    col_map = config.get('columns', {})
    ds_id = config['id']
    out = pd.DataFrame()

    # Age
    if 'age' in col_map and col_map['age'] in df.columns:
        age_col = col_map['age']
        if 'age_mapping' in config:
            out['age'] = df[age_col].map(config['age_mapping'])
        else:
            out['age'] = pd.to_numeric(df[age_col], errors='coerce')

    # Gender
    if 'gender' in col_map and col_map['gender'] in df.columns:
        out['gender'] = df[col_map['gender']].apply(normalize_gender)

    # Screen time
    if 'daily_screen_time_hours' in col_map and col_map['daily_screen_time_hours'] in df.columns:
        vals = pd.to_numeric(df[col_map['daily_screen_time_hours']], errors='coerce')
        if config.get('screen_time_unit') == 'minutes':
            vals = vals / 60.0
        out['daily_screen_time_hours'] = vals

    # Sleep
    if 'sleep_duration_hours' in col_map and col_map['sleep_duration_hours'] in df.columns:
        out['sleep_duration_hours'] = pd.to_numeric(
            df[col_map['sleep_duration_hours']], errors='coerce')

    # Platform
    if 'primary_platform' in col_map and col_map['primary_platform'] in df.columns:
        out['primary_platform'] = df[col_map['primary_platform']].apply(normalize_platform)

    # Social comparison
    if 'social_comparison' in col_map and col_map['social_comparison'] in df.columns:
        out['social_comparison'] = pd.to_numeric(
            df[col_map['social_comparison']], errors='coerce')

    # Late night usage
    if 'late_night_usage' in col_map and col_map['late_night_usage'] in df.columns:
        out['late_night_usage'] = pd.to_numeric(
            df[col_map['late_night_usage']], errors='coerce')

    # Content type
    if 'dominant_content_type' in col_map and col_map['dominant_content_type'] in df.columns:
        out['dominant_content_type'] = df[col_map['dominant_content_type']].str.strip().str.lower()

    # Activity type
    if 'activity_type' in col_map and col_map['activity_type'] in df.columns:
        out['activity_type'] = df[col_map['activity_type']].str.strip().str.lower()

    # Any extra columns not in standard features
    for key, src_col in col_map.items():
        if key not in STANDARD_FEATURES and src_col in df.columns and key not in out.columns:
            out[key] = df[src_col]

    # MH composite from target variables
    targets = config.get('mh_targets', [])
    if not targets:
        raise ValueError('No mh_targets defined in config. Need at least one MH variable.')

    normalized = []
    for t in targets:
        col = t['column']
        if col not in df.columns:
            print(f'  Warning: target column {col} not found, skipping')
            continue
        lo, hi = t['range']
        vals = pd.to_numeric(df[col], errors='coerce')

        if t['direction'] == 'higher_is_worse':
            norm = (vals - lo) / (hi - lo)
        elif t['direction'] == 'higher_is_better':
            norm = (hi - vals) / (hi - lo)
        else:
            raise ValueError(f'Unknown direction: {t["direction"]}')

        normalized.append(norm.clip(0, 1))

    if not normalized:
        raise ValueError('No valid MH target columns found in data.')

    out['mh_composite'] = np.mean(normalized, axis=0)
    out['mh_risk_class'] = pd.cut(
        out['mh_composite'],
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=['Low', 'Moderate', 'High']
    )
    out['source_dataset'] = ds_id

    return out.dropna(subset=['mh_composite']).reset_index(drop=True)


def merge_into_existing(new_df, ds_id):
    existing_path = os.path.join(OUTPUT_DIR, 'all_cleaned.csv')
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        existing = existing[existing['source_dataset'] != ds_id]
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(existing_path, index=False)
    return merged


def retrain_model(merged_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import joblib

    # Run same pipeline as 02-04 inline
    TARGET = 'mh_risk_class'
    META = ['mh_composite', 'source_dataset']
    drop_cols = [c for c in merged_df.columns
                 if merged_df[c].isna().mean() > 0.8
                 and c not in [TARGET] + META]
    df = merged_df.drop(columns=drop_cols)

    cat_cols = df.select_dtypes(include='object').columns.difference([TARGET] + META)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    feat_cols = [c for c in df.columns if c not in [TARGET] + META]
    X = df[feat_cols].fillna(df[feat_cols].median())

    if 'daily_screen_time_hours' in X.columns and 'late_night_usage' in X.columns:
        X['screen_x_latenight'] = X['daily_screen_time_hours'] * X['late_night_usage']
        feat_cols.append('screen_x_latenight')

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])

    model = RandomForestClassifier(
        n_estimators=30, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1)
    model.fit(X.values, y)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X.values, y, cv=cv, scoring='f1_macro')

    joblib.dump(model, os.path.join(OUTPUT_DIR, 'model_compact.pkl'), compress=3)
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))

    return scores.mean(), scores.std(), len(feat_cols)


def main():
    parser = argparse.ArgumentParser(description='Add a new survey dataset')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model after adding the dataset')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f'Loading {config["file"]}...')

    df = load_data(config)
    print(f'  Raw data: {len(df)} rows, {len(df.columns)} columns')

    cleaned = clean_dataset(df, config)
    print(f'  Cleaned: {len(cleaned)} rows')
    print(f'  Composite range: [{cleaned["mh_composite"].min():.3f}, '
          f'{cleaned["mh_composite"].max():.3f}]')
    print(f'  Class distribution: {cleaned["mh_risk_class"].value_counts().to_dict()}')

    cleaned.to_csv(os.path.join(OUTPUT_DIR, f'cleaned_{config["id"]}.csv'), index=False)

    merged = merge_into_existing(cleaned, config['id'])
    print(f'\nMerged total: {len(merged)} rows, '
          f'{merged["source_dataset"].nunique()} datasets')

    if args.retrain:
        print('\nRetraining model...')
        f1_mean, f1_std, n_feat = retrain_model(merged)
        print(f'  F1 macro: {f1_mean:.3f} +/- {f1_std:.3f} ({n_feat} features)')
        print(f'  Model saved to {OUTPUT_DIR}/model_compact.pkl')
    else:
        print('\nSkipping retrain. Run with --retrain to update the model.')


if __name__ == '__main__':
    main()
