"""
Phase 1+2: Per-dataset cleaning + MH composite target construction.
Each dataset cleaned independently, target composite computed, then saved as cleaned CSV.
"""
import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (DATA_DIR, OUTPUT_DIR, normalize_platform, normalize_gender,
                   RANDOM_SEED)


def clean_ds1():
    """#1 dataset.csv: 49 rows, real survey (Google Forms)"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'))
    df.columns = df.columns.str.strip()

    age_map = {'18-25': 21.5, '26-35': 30.5, '36-45': 40.5, '46-60': 53.0}
    screen_map = {
        'Less than 1 hour': 0.5, '1-2 hours': 1.5, '2-3 hours': 2.5,
        '3-5 hours': 4.0, 'More than 5 hours': 6.0,
    }
    mh_map = {
        'Yes, negatively': 0.83,
        'Not sure': 0.50,
        'Yes, positively': 0.17,
        'No, it has no effect': 0.0,
    }

    age_col = [c for c in df.columns if 'age' in c.lower()][0]
    occ_col = [c for c in df.columns if 'occupation' in c.lower()][0]
    loc_col = [c for c in df.columns if 'live' in c.lower() or 'location' in c.lower()][0]
    screen_col = [c for c in df.columns if 'hours' in c.lower() and 'social' in c.lower()][0]
    platform_col = [c for c in df.columns if 'platform' in c.lower()][0]
    mh_col = [c for c in df.columns if 'mental' in c.lower()][0]

    out = pd.DataFrame()
    out['age'] = df[age_col].map(age_map)
    out['occupation'] = df[occ_col].str.strip()
    out['location_type'] = df[loc_col].str.strip().str.lower().str.replace(' area', '')
    out['daily_screen_time_hours'] = df[screen_col].str.strip().map(screen_map)
    out['primary_platform'] = df[platform_col].apply(normalize_platform)
    out['mh_composite'] = df[mh_col].str.strip().map(mh_map)
    out['source_dataset'] = 'ds1'
    return out


def clean_ds2():
    """#2 digital_diet_mental_health.csv: 2000 rows, synthetic"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'digital_diet_mental_health.csv'))

    out = pd.DataFrame()
    out['age'] = df['age'].astype(float)
    out['gender'] = df['gender'].apply(normalize_gender)
    out['daily_screen_time_hours'] = df['daily_screen_time_hours'].astype(float)
    out['social_media_hours'] = df['social_media_hours'].astype(float)
    out['phone_usage_hours'] = df['phone_usage_hours'].astype(float)
    out['laptop_usage_hours'] = df['laptop_usage_hours'].astype(float)
    out['tablet_usage_hours'] = df['tablet_usage_hours'].astype(float)
    out['tv_usage_hours'] = df['tv_usage_hours'].astype(float)
    out['work_related_hours'] = df['work_related_hours'].astype(float)
    out['entertainment_hours'] = df['entertainment_hours'].astype(float)
    out['gaming_hours'] = df['gaming_hours'].astype(float)
    out['sleep_duration_hours'] = df['sleep_duration_hours'].astype(float)
    out['physical_activity_hours_per_week'] = df['physical_activity_hours_per_week'].astype(float)
    out['location_type'] = df['location_type'].str.strip().str.lower()
    out['uses_wellness_apps'] = df['uses_wellness_apps'].astype(int)
    out['eats_healthy'] = df['eats_healthy'].astype(int)
    out['caffeine_intake_mg'] = df['caffeine_intake_mg_per_day'].astype(float)
    out['mindfulness_minutes'] = df['mindfulness_minutes_per_day'].astype(float)

    # Target: anxiety/20, depression/20, (stress-1)/9, (10-mood)/9, (10-sleep_quality)/9
    # EXCLUDE mental_health_score (uniform random, r<0.05 with everything)
    t_anxiety = df['weekly_anxiety_score'] / 20.0
    t_depression = df['weekly_depression_score'] / 20.0
    t_stress = (df['stress_level'] - 1) / 9.0
    t_mood = (10 - df['mood_rating']) / 9.0
    t_sleep = (10 - df['sleep_quality']) / 9.0
    out['mh_composite'] = (t_anxiety + t_depression + t_stress + t_mood + t_sleep) / 5.0

    out['source_dataset'] = 'ds2'
    return out


def clean_ds3():
    """#3 Mental_Health_Balance: 500 rows, synthetic"""
    df = pd.read_excel(os.path.join(DATA_DIR,
        'Mental_Health_and_Social_Media_Balance_Dataset.xlsx'))

    out = pd.DataFrame()
    out['age'] = df['Age'].astype(float)
    out['gender'] = df['Gender'].apply(normalize_gender)
    out['daily_screen_time_hours'] = df['Daily_Screen_Time(hrs)'].astype(float)
    out['days_without_sm'] = df['Days_Without_Social_Media'].astype(int)
    out['exercise_frequency'] = df['Exercise_Frequency(week)'].astype(int)
    out['primary_platform'] = df['Social_Media_Platform'].apply(normalize_platform)

    # Target: (10-happiness)/9, (stress-1)/9, (10-sleep_quality)/9
    t_happy = (10 - df['Happiness_Index(1-10)']) / 9.0
    t_stress = (df['Stress_Level(1-10)'] - 1) / 9.0
    t_sleep = (10 - df['Sleep_Quality(1-10)']) / 9.0
    out['mh_composite'] = (t_happy + t_stress + t_sleep) / 3.0

    out['source_dataset'] = 'ds3'
    return out


def clean_ds4():
    """#4 smmh.xlsx: 481 rows, real survey (university)"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'smmh.xlsx'))
    cols = df.columns.tolist()

    out = pd.DataFrame()
    out['age'] = pd.to_numeric(df[cols[1]], errors='coerce')
    out['gender'] = df[cols[2]].apply(normalize_gender)
    out['relationship_status'] = df[cols[3]].str.strip().str.lower()
    out['occupation'] = df[cols[4]].str.strip()

    org = df[cols[5]].fillna('Unknown')
    out['organization'] = org.str.strip()

    # Screen time: categorical -> midpoint
    st_map = {
        'Less than an Hour': 0.5,
        'Between 1 and 2 hours': 1.5,
        'Between 2 and 3 hours': 2.5,
        'Between 3 and 4 hours': 3.5,
        'Between 4 and 5 hours': 4.5,
        'More than 5 hours': 6.0,
    }
    out['daily_screen_time_hours'] = df[cols[8]].str.strip().map(st_map)

    # Platform: multi-select, take first after splitting
    def first_platform(val):
        if pd.isna(val): return np.nan
        parts = str(val).split(',')
        return normalize_platform(parts[0].strip())
    out['primary_platform'] = df[cols[7]].apply(first_platform)

    # Behavioral features (1-5 Likert)
    out['purposeless_usage'] = df[cols[9]].astype(float)
    out['distracted_while_busy'] = df[cols[10]].astype(float)
    out['restless_without_sm'] = df[cols[11]].astype(float)
    out['distractibility'] = df[cols[12]].astype(float)
    out['social_comparison'] = df[cols[15]].astype(float)
    out['validation_seeking'] = df[cols[17]].astype(float)

    # Target: Q13(worries), Q14(concentration), Q16(comparison_feeling),
    #         Q18(depression), Q19(interest_fluct), Q20(sleep_issues) — all 1-5
    t_worry = (df[cols[13]] - 1) / 4.0
    t_conc = (df[cols[14]] - 1) / 4.0
    t_comp = (df[cols[16]] - 1) / 4.0
    t_dep = (df[cols[18]] - 1) / 4.0
    t_int = (df[cols[19]] - 1) / 4.0
    t_sleep = (df[cols[20]] - 1) / 4.0
    out['mh_composite'] = (t_worry + t_conc + t_comp + t_dep + t_int + t_sleep) / 6.0

    out['source_dataset'] = 'ds4'
    return out


def clean_ds5():
    """#5 social_media_addiction.xlsx: 20 rows, synthetic"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'social_media_addiction.xlsx'))

    out = pd.DataFrame()
    out['age'] = df['Age'].astype(float)
    out['gender'] = df['Gender'].apply(normalize_gender)
    out['primary_platform'] = df['Platform'].apply(normalize_platform)
    out['daily_screen_time_hours'] = df['Daily_Usage_Time_min'] / 60.0
    out['posts_per_day'] = df['Posts_Per_Day'].astype(float)
    out['likes_received_daily'] = df['Likes_Received_Daily'].astype(float)
    out['comments_received_daily'] = df['Comments_Received_Daily'].astype(float)
    out['messages_sent_daily'] = df['Messages_Sent_Daily'].astype(float)
    out['scroll_rate_ppm'] = df['Scroll_Rate_ppm'].astype(float)
    out['addiction_level'] = df['Addiction_Level'].str.strip().str.lower()
    out['emotional_state'] = df['Emotional_State_Post_Usage'].str.strip().str.lower()

    # Target: (90-MHI)/65, (FOMO-1)/9, (productivity_loss-1)/9
    # MHI observed 25-90, treat as 25-90 theoretical for now
    t_mhi = (90 - df['Mental_Health_Index']) / 65.0
    t_fomo = (df['FOMO_Score'] - 1) / 9.0
    t_prod = (df['Productivity_Loss_Score'] - 1) / 9.0
    out['mh_composite'] = (t_mhi + t_fomo + t_prod) / 3.0

    out['source_dataset'] = 'ds5'
    return out


def clean_ds6():
    """#6 social_media_mental_health.xlsx: 8000 rows, synthetic (GAD-7/PHQ-9)"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'social_media_mental_health.xlsx'))

    out = pd.DataFrame()
    out['age'] = df['Age'].astype(float)
    out['gender'] = df['Gender'].apply(normalize_gender)
    out['primary_platform'] = df['Primary_Platform'].apply(normalize_platform)
    out['daily_screen_time_hours'] = df['Daily_Screen_Time_Hours'].astype(float)
    out['dominant_content_type'] = df['Dominant_Content_Type'].str.strip().str.lower()
    out['activity_type'] = df['Activity_Type'].str.strip().str.lower()
    out['late_night_usage'] = df['Late_Night_Usage'].astype(int)
    out['social_comparison'] = df['Social_Comparison_Trigger'].astype(float)
    out['sleep_duration_hours'] = df['Sleep_Duration_Hours'].astype(float)

    # Target: GAD_7/21, PHQ_9/27
    out['mh_composite'] = (df['GAD_7_Score'] / 21.0 + df['PHQ_9_Score'] / 27.0) / 2.0

    out['source_dataset'] = 'ds6'
    return out


def clean_ds7():
    """#7 Students Social Media Addiction.xlsx: 705 rows, real survey"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'Students Social Media Addiction.xlsx'))

    out = pd.DataFrame()
    out['age'] = df['Age'].astype(float)
    out['gender'] = df['Gender'].apply(normalize_gender)
    out['academic_level'] = df['Academic_Level'].str.strip().str.lower()
    out['country'] = df['Country'].str.strip()
    out['daily_screen_time_hours'] = df['Avg_Daily_Usage_Hours'].astype(float)
    out['primary_platform'] = df['Most_Used_Platform'].apply(normalize_platform)
    out['affects_academic'] = df['Affects_Academic_Performance'].str.strip().str.lower()
    out['sleep_duration_hours'] = df['Sleep_Hours_Per_Night'].astype(float)
    out['relationship_status'] = df['Relationship_Status'].str.strip().str.lower()
    out['conflicts_over_sm'] = df['Conflicts_Over_Social_Media']
    out['addicted_score'] = df['Addicted_Score'].astype(float)

    # Target: (9 - MH_Score) / 5, range observed 4-9, higher=better → reverse
    out['mh_composite'] = (9 - df['Mental_Health_Score']) / 5.0

    out['source_dataset'] = 'ds7'
    return out


def clean_ds8():
    """#8 test.xlsx: 206 rows (103 valid), synthetic"""
    df = pd.read_excel(os.path.join(DATA_DIR, 'test.xlsx'))
    df = df.dropna(how='all').reset_index(drop=True)

    out = pd.DataFrame()
    out['age'] = pd.to_numeric(df['Age'], errors='coerce')
    out['gender'] = df['Gender'].apply(normalize_gender)
    out['primary_platform'] = df['Platform'].apply(normalize_platform)
    out['daily_screen_time_hours'] = pd.to_numeric(
        df['Daily_Usage_Time (minutes)'], errors='coerce') / 60.0
    out['posts_per_day'] = pd.to_numeric(df['Posts_Per_Day'], errors='coerce')
    out['likes_received_daily'] = pd.to_numeric(df['Likes_Received_Per_Day'], errors='coerce')
    out['comments_received_daily'] = pd.to_numeric(df['Comments_Received_Per_Day'], errors='coerce')
    out['messages_sent_daily'] = pd.to_numeric(df['Messages_Sent_Per_Day'], errors='coerce')

    emotion_map = {
        'anxiety': 0.83, 'sadness': 0.83, 'anger': 0.83,
        'boredom': 0.50, 'neutral': 0.33, 'happiness': 0.17,
    }
    out['mh_composite'] = df['Dominant_Emotion'].str.strip().str.lower().map(emotion_map)

    out['source_dataset'] = 'ds8'
    return out


def discretize(composite):
    """3-class: Low [0, 0.33), Moderate [0.33, 0.66), High [0.66, 1.0]"""
    return pd.cut(composite,
                  bins=[-0.001, 0.33, 0.66, 1.001],
                  labels=['Low', 'Moderate', 'High'])


def main():
    cleaners = [clean_ds1, clean_ds2, clean_ds3, clean_ds4,
                clean_ds5, clean_ds6, clean_ds7, clean_ds8]
    all_dfs = []

    for i, fn in enumerate(cleaners, 1):
        print(f'Processing dataset #{i}...')
        df = fn()
        df['mh_risk_class'] = discretize(df['mh_composite'])

        out_path = os.path.join(OUTPUT_DIR, f'cleaned_ds{i}.csv')
        df.to_csv(out_path, index=False)
        print(f'  → {len(df)} rows, composite range: '
              f'[{df["mh_composite"].min():.3f}, {df["mh_composite"].max():.3f}], '
              f'class dist: {df["mh_risk_class"].value_counts().to_dict()}')
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(os.path.join(OUTPUT_DIR, 'all_cleaned.csv'), index=False)
    print(f'\nMerged: {merged.shape[0]} rows, {merged.shape[1]} columns')
    print(f'Overall class distribution:\n{merged["mh_risk_class"].value_counts()}')
    print(f'\nComposite stats per dataset:')
    print(merged.groupby('source_dataset')['mh_composite'].describe().round(3))


if __name__ == '__main__':
    main()
