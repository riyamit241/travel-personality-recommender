import streamlit as st
import pandas as pd
from travel_recommender import TravelRecommender

# Load data as done in travel_recommender.py
df = pd.read_csv("clean_travel_data.csv")
recommender = TravelRecommender(df)

st.title("Personality-based Travel Recommender")

valid_mbti_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP',
                    'INFJ', 'INFP', 'ENFJ', 'ENFP',
                    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
                    'ISTP', 'ISFP', 'ESTP', 'ESFP']

personality_type = st.text_input("Enter your 16 Personalities type (e.g., ENFP):")

if st.button("Get Recommendations"):
    pt = personality_type.upper()
    if pt not in valid_mbti_types:
        st.error("Please enter a valid MBTI type (e.g., ENFP)")
    else:
        try:
            recs = recommender.get_recommendations(pt, top_n=5)
            # display recommendations ...
        except Exception as e:
            st.error(f"Error: {e}")