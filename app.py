import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open("rainfall_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Rainfall Prediction", layout="centered")
st.title("🌧️ Rainfall Prediction App")
st.markdown("Enter weather features to predict whether it will rain tomorrow.")

# Sidebar input
st.sidebar.header("Enter Weather Data")

def user_input():
    sunshine = st.sidebar.slider("Sunshine (hours)", 0.0, 15.0, 7.5)
    humidity_9am = st.sidebar.slider("Humidity at 9am (%)", 0.0, 100.0, 70.0)
    cloud_3pm = st.sidebar.slider("Cloud at 3pm (0–8 oktas)", 0.0, 8.0, 4.0)
    
    data = {
        'Sunshine': sunshine,
        'Humidity9am': humidity_9am,
        'Cloud3pm': cloud_3pm
    }
    return pd.DataFrame([data])

input_df = user_input()

# Show input
st.subheader("Input Features")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

# Show result
st.subheader("Prediction")
st.write("🌧️ Rain Tomorrow" if prediction == 1 else "☀️ No Rain Tomorrow")

st.subheader("Confidence")
st.write(f"Probability of Rain: {proba:.2%}")
