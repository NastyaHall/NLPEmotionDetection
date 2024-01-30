import streamlit as st 
import altair as alt
from textblob import TextBlob
import pandas as pd 
import numpy as np 
import joblib

pipe_lr = joblib.load(open('models/emotion_classifier_pipe_lr_30_january_2024.pkl', 'rb'))
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😱", "happy":"🤗", "joy":"😄", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"🤦‍♀️", "surprise":"😮"}

def get_text_polarity(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 4)

def get_text_subjectivity(text):
    blob = TextBlob(text)
    return round(blob.sentiment.subjectivity, 4)

def predict_emotions(text):
    subjectivity = get_text_subjectivity(text)
    polarity = get_text_polarity(text)
    results = pipe_lr.predict([text])

    if results[0] == 'joy' or results[0] == 'sadness':
        if subjectivity >= 0 and subjectivity < 0.1:
            return 'neutral'
        
        if polarity < 0:
            return 'sadness'
    
    return results[0]

def get_prediction_proba(text):
    results = pipe_lr.predict_proba([text])
    return results

def main():
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon="🦋",
        layout="wide",
    )
    st.title('Emotion Detector')
    with st.form(key='emotion_detection_form'):
        raw_text = st.text_area('Type Here')
        submit_text = st.form_submit_button(label='Detect The Emotion')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = 'emotions', 'probability'

        if prediction == 'neutral' or prediction == 'sadness':

            if get_text_subjectivity(raw_text) >= 0 and get_text_subjectivity(raw_text) < 0.1:
                max_index = proba_df_clean['probability'].idxmax()
                proba_df_clean.loc[max_index, 'probability'], proba_df_clean.loc[proba_df_clean['emotions'] == 'neutral', 'probability'] = proba_df_clean.loc[proba_df_clean['emotions'] == 'neutral', 'probability'].values[0], proba_df_clean.loc[max_index, 'probability']

            if get_text_polarity(raw_text) < 0:
                max_index = proba_df_clean['probability'].idxmax()
                proba_df_clean.loc[max_index, 'probability'], proba_df_clean.loc[proba_df_clean['emotions'] == 'sadness', 'probability'] = proba_df_clean.loc[proba_df_clean['emotions'] == 'sadness', 'probability'].values[0], proba_df_clean.loc[max_index, 'probability']

        with col1:
            st.subheader('Prediction')
            emotion_emoji = emotions_emoji_dict[prediction]
            st.code(f'{prediction}: {emotion_emoji}')
            st.code(f'Confidence: {proba_df_clean.loc[proba_df_clean["probability"].idxmax(), "probability"]}')
            st.code(f'Subjectivity: {get_text_subjectivity(raw_text)}')
            st.code(f'Polarity: {get_text_polarity(raw_text)}')

        with col2:
            st.subheader('Probability')
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()