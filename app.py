# app.py

import streamlit as st
import pandas as pd
import joblib
import requests
import os

# === Helper: Download file from Google Drive ===
def download_from_gdrive(file_id, destination):
    if not os.path.exists(destination):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)

# === Download the model from Google Drive ===
download_from_gdrive("1QThjYFwE9axeNt59e5QBlQ3qvxaUhCHv", "rainfall_rf_model.pkl")

# === Load model and preprocessing objects ===
model = joblib.load("rainfall_rf_model.pkl")
imputer = joblib.load("imputer.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

# === Streamlit UI ===
st.title("🌧️ Rainfall Prediction App")

# === User Inputs ===
user_input = {}

# Define input fields — same as training data
numerical_features = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
    'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
    'Pressure3pm', 'Temp9am', 'Temp3pm'
]

for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}:", value=0.0)

categorical_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for feature in categorical_features:
    options = label_encoders[feature].classes_.tolist()
    value = st.selectbox(feature, options)
    encoded = label_encoders[feature].transform([value])[0]
    user_input[feature] = encoded

# === Prediction Button ===
if st.button("Predict"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([user_input])

        # Reorder columns to match training
        input_df = input_df[feature_order]

        # Apply imputer
        input_df = imputer.transform(input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Yes 🌧️" if prediction == 1 else "No ☀️"
        st.success(f"Prediction: Will it rain tomorrow? {result}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
