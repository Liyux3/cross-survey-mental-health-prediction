import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PLATFORM_MAP = {
    'instagram': 'instagram',
    'tiktok': 'tiktok',
    'twitter': 'twitter_x', 'x': 'twitter_x', 'twitter/x': 'twitter_x',
    'facebook': 'facebook',
    'youtube': 'youtube',
    'snapchat': 'snapchat',
    'linkedin': 'linkedin',
    'reddit': 'reddit',
    'whatsapp': 'whatsapp',
    'pinterest': 'pinterest',
    'telegram': 'telegram',
    'discord': 'discord',
}

GENDER_MAP = {
    'male': 'male', 'm': 'male',
    'female': 'female', 'f': 'female',
    'non-binary': 'other', 'nonbinary': 'other', 'transgender female': 'other',
    'transgender male': 'other', 'other': 'other', 'prefer not to say': 'other',
}

CHART_PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0',
                 '#00BCD4', '#E91E63', '#795548']

def normalize_platform(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    for key, mapped in PLATFORM_MAP.items():
        if key in s:
            return mapped
    return 'other'

def normalize_gender(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    return GENDER_MAP.get(s, 'other')
