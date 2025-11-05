import streamlit as st
import pandas as pd
from recommender import TravelRecommender

# Load data as done in travel_recommender.py
df = pd.read_csv("your_data.csv")
recommender = TravelRecommender(df)

st.title("Personality-based Travel Recommender")

personality_type = st.text_input("Enter your 16 Personalities type (e.g., ENFP):")

if st.button("Get Recommendations"):
    if len(personality_type) == 4:
        try:
            recs = recommender.get_recommendations(personality_type.upper(), top_n=5)
            st.success(f"Top 5 destinations for {personality_type.upper()}:")
            for i, rec in enumerate(recs, 1):
                st.write(f"{i}. {rec['city']}, {rec['country']} - Rating: {rec['rating']}, Cost: ${rec['cost_per_day']}/day")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter a valid 4-letter MBTI type")
