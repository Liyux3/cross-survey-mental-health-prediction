"""
Streamlit demo for cross-survey mental health risk prediction.
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join('output', 'best_classifier.pkl')
LE_PATH = os.path.join('output', 'label_encoder.pkl')

FEATURE_COLS = [
    'age', 'daily_screen_time_hours', 'sleep_duration_hours',
    'social_comparison', 'late_night_usage', 'screen_x_latenight',
    'primary_platform_discord', 'primary_platform_facebook',
    'primary_platform_instagram', 'primary_platform_linkedin',
    'primary_platform_other', 'primary_platform_pinterest',
    'primary_platform_reddit', 'primary_platform_snapchat',
    'primary_platform_telegram', 'primary_platform_tiktok',
    'primary_platform_twitter_x', 'primary_platform_whatsapp',
    'primary_platform_youtube',
    'gender_female', 'gender_male', 'gender_other',
    'dominant_content_type_educational/tech',
    'dominant_content_type_entertainment/comedy',
    'dominant_content_type_gaming',
    'dominant_content_type_lifestyle/fashion',
    'dominant_content_type_news/politics',
    'dominant_content_type_self-help/motivation',
    'activity_type_active', 'activity_type_passive',
]

PLATFORMS = [
    'Instagram', 'YouTube', 'TikTok', 'Facebook', 'Twitter/X',
    'Snapchat', 'Reddit', 'WhatsApp', 'LinkedIn', 'Discord',
    'Telegram', 'Pinterest', 'Other',
]

PLATFORM_COL_MAP = {
    'Instagram': 'primary_platform_instagram',
    'YouTube': 'primary_platform_youtube',
    'TikTok': 'primary_platform_tiktok',
    'Facebook': 'primary_platform_facebook',
    'Twitter/X': 'primary_platform_twitter_x',
    'Snapchat': 'primary_platform_snapchat',
    'Reddit': 'primary_platform_reddit',
    'WhatsApp': 'primary_platform_whatsapp',
    'LinkedIn': 'primary_platform_linkedin',
    'Discord': 'primary_platform_discord',
    'Telegram': 'primary_platform_telegram',
    'Pinterest': 'primary_platform_pinterest',
    'Other': 'primary_platform_other',
}

CONTENT_TYPES = [
    'Entertainment/Comedy', 'Educational/Tech', 'News/Politics',
    'Lifestyle/Fashion', 'Gaming', 'Self-help/Motivation',
]

CONTENT_COL_MAP = {
    'Entertainment/Comedy': 'dominant_content_type_entertainment/comedy',
    'Educational/Tech': 'dominant_content_type_educational/tech',
    'News/Politics': 'dominant_content_type_news/politics',
    'Lifestyle/Fashion': 'dominant_content_type_lifestyle/fashion',
    'Gaming': 'dominant_content_type_gaming',
    'Self-help/Motivation': 'dominant_content_type_self-help/motivation',
}

RISK_COLORS = {'Low': '#3A8A7B', 'Moderate': '#C4873A', 'High': '#B8433E'}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le


def build_feature_vector(age, screen_time, sleep, social_comp, late_night,
                         gender, platform, content_type, activity):
    row = {col: 0.0 for col in FEATURE_COLS}

    row['age'] = age
    row['daily_screen_time_hours'] = screen_time
    row['sleep_duration_hours'] = sleep
    row['social_comparison'] = social_comp
    row['late_night_usage'] = late_night
    row['screen_x_latenight'] = screen_time * late_night

    if platform in PLATFORM_COL_MAP:
        row[PLATFORM_COL_MAP[platform]] = 1.0

    gender_col = f'gender_{gender.lower()}'
    if gender_col in row:
        row[gender_col] = 1.0

    if content_type in CONTENT_COL_MAP:
        row[CONTENT_COL_MAP[content_type]] = 1.0

    activity_col = f'activity_type_{activity.lower()}'
    if activity_col in row:
        row[activity_col] = 1.0

    return pd.DataFrame([row], columns=FEATURE_COLS)


def main():
    st.set_page_config(page_title='MH Risk Predictor', layout='wide')

    st.title('Mental Health Risk Prediction')
    st.caption('Predict mental health risk level from social media usage patterns. '
               'Based on a Random Forest model trained on 11,858 harmonized survey samples.')

    model, le = load_model()

    col1, col2 = st.columns([1, 1], gap='large')

    with col1:
        st.subheader('Your Profile')

        age = st.slider('Age', 13, 65, 22)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])

        st.subheader('Social Media Usage')

        screen_time = st.slider('Daily screen time (hours)', 0.0, 13.0, 4.0, 0.5)
        late_night = st.slider('Late-night usage intensity', 0.0, 1.5, 0.4, 0.1,
                               help='How much you use social media late at night (0 = none, 1.5 = heavy)')
        social_comp = st.slider('Social comparison tendency', 0.0, 1.0, 0.3, 0.1,
                                help='How often you compare yourself to others on social media')
        platform = st.selectbox('Primary platform', PLATFORMS)
        content_type = st.selectbox('Dominant content type', CONTENT_TYPES)
        activity = st.radio('Usage style', ['Active', 'Passive'],
                            help='Active = posting, commenting. Passive = scrolling, watching.')

        st.subheader('Health')
        sleep = st.slider('Sleep duration (hours)', 3.0, 12.0, 7.0, 0.5)

    X = build_feature_vector(age, screen_time, sleep, social_comp, late_night,
                             gender, platform, content_type, activity)

    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_label = le.classes_[pred_idx]
    pred_conf = proba[pred_idx]

    with col2:
        st.subheader('Prediction')

        risk_color = RISK_COLORS.get(pred_label, '#6B7C93')
        st.markdown(
            f'<div style="text-align:center; padding: 1.5rem; '
            f'border: 2px solid {risk_color}; border-radius: 12px; '
            f'margin-bottom: 1.5rem;">'
            f'<p style="font-size: 1rem; color: #6B7280; margin:0;">Predicted Risk Level</p>'
            f'<p style="font-size: 2.5rem; font-weight: 700; color: {risk_color}; '
            f'margin: 0.3rem 0;">{pred_label} Risk</p>'
            f'<p style="font-size: 1rem; color: #6B7280; margin:0;">'
            f'Confidence: {pred_conf:.1%}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.subheader('Class Probabilities')
        prob_df = pd.DataFrame({
            'Risk Level': le.classes_,
            'Probability': proba,
        })
        for _, r in prob_df.iterrows():
            label = r['Risk Level']
            p = r['Probability']
            color = RISK_COLORS.get(label, '#6B7C93')
            st.markdown(
                f'<div style="margin-bottom: 0.5rem;">'
                f'<span style="color: {color}; font-weight: 600;">{label}</span>'
                f'<span style="float: right; color: #6B7280;">{p:.1%}</span>'
                f'<div style="background: #E5E7EB; border-radius: 4px; height: 8px; margin-top: 4px;">'
                f'<div style="background: {color}; width: {p*100:.1f}%; height: 100%; '
                f'border-radius: 4px;"></div></div></div>',
                unsafe_allow_html=True,
            )

        st.subheader('Top Contributing Factors')
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:6]
        top_features = []
        for idx in top_idx:
            fname = FEATURE_COLS[idx]
            val = X.iloc[0, idx]
            imp = importances[idx]
            display_name = fname.replace('_', ' ').replace('primary platform ', '').replace('dominant content type ', '').title()
            top_features.append({'Feature': display_name, 'Your Value': f'{val:.2f}', 'Importance': f'{imp:.3f}'})

        st.dataframe(pd.DataFrame(top_features), hide_index=True, use_container_width=True)

        with st.expander('About this model'):
            st.markdown(
                '**Model:** Random Forest (100 trees, class-weight balanced)\n\n'
                '**Training data:** 11,858 samples from 8 harmonized mental health surveys\n\n'
                '**Performance:** F1 macro = 0.714 (5-fold stratified CV)\n\n'
                '**Limitation:** LODO validation shows limited cross-dataset generalization '
                '(mean F1 = 0.272), primarily due to self-selection bias in voluntary surveys. '
                'Predictions should be treated as indicative, not diagnostic.\n\n'
                '[View full report](https://github.com/Liyux3/cross-survey-mental-health-prediction/blob/main/report.pdf) '
                '| [Source code](https://github.com/Liyux3/cross-survey-mental-health-prediction)'
            )


if __name__ == '__main__':
    main()
