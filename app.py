# app.py

import streamlit as st
import pandas as pd
import joblib
import requests
import os

# === Utility: Download model from Google Drive ===
def download_from_gdrive(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

# === Download rainfall_rf_model.pkl ===
download_from_gdrive("1GXyWnqIrxjMejPPmfAv4BzOY7Dq5eZNS", "rainfall_rf_model.pkl")

# === Load model and local preprocessor files ===
model = joblib.load("rainfall_rf_model.pkl")
imputer = joblib.load("imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🌧️ Rainfall Prediction App")

# === Input Fields ===
user_input = {}

numeric_features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
    'Pressure3pm', 'Temp9am', 'Temp3pm'
]

for feature in numeric_features:
    user_input[feature] = st.number_input(f"{feature}:", value=0.0)

categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for cat in categorical_features:
    options = label_encoders[cat].classes_.tolist()
    user_val = st.selectbox(cat, options)
    user_input[cat] = label_encoders[cat].transform([user_val])[0]

# === Predict ===
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        input_df = imputer.transform(input_df)
        prediction = model.predict(input_df)[0]
        result = "Yes 🌧️" if prediction == 1 else "No ☀️"
        st.success(f"Prediction: Will it rain tomorrow? {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
