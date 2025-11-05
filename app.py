import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from travel_recommender import get_recommendations

import streamlit as st
import pandas as pd
import numpy as np
from travel_recommender import get_recommendations  # make sure this function exists

# ------------------------------------------------------------
# Basic page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Travel Personality Recommender", page_icon="🌎", layout="centered")

st.title("🌍 Travel Personality Recommender System")
st.markdown("""
Find travel destinations that match your personality type and preferences!
Simply enter your MBTI type and let the recommender do the rest.
""")

# ------------------------------------------------------------
# User input section
# ------------------------------------------------------------
personality = st.selectbox(
    "Select your MBTI personality type:",
    [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ],
)

travel_type = st.multiselect(
    "Preferred travel types:",
    ["Beach", "City", "Adventure", "Nature", "Cultural", "Wellness", "Nightlife", "Relaxation"]
)

budget = st.radio("Select your budget level:", ["Budget", "Mid-range", "Luxury"])
duration = st.selectbox("Trip duration:", ["Weekend", "Short trip", "One week", "Long trip"])

if st.button("✨ Get Recommendations"):
    try:
        # Call your recommender function here
        recommendations = get_recommendations(
            mbti_type=personality,
            preferences=travel_type,
            budget=budget,
            duration=duration
        )

        if isinstance(recommendations, pd.DataFrame):
            st.success("Here are your top travel destinations:")
            st.dataframe(recommendations[["destination_name", "city", "country"]].head(10))
        else:
            st.info("No suitable matches found — try adjusting your preferences!")

    except Exception as e:
        st.error(f"⚠️ Oops! Something went wrong: {e}")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Created by Riya Mittal · Travel Personality Recommender · 2025")
