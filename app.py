# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessors
model = joblib.load("rainfall_rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
imputer = joblib.load("imputer.pkl")

st.title("Rainfall Prediction App 🌧️")

# Input fields
user_input = {}
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
            'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
            'Pressure3pm', 'Temp9am', 'Temp3pm']
for feat in features:
    user_input[feat] = st.number_input(f"{feat}", value=0.0)

# Categorical inputs
cat_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for cat in cat_features:
    options = label_encoders[cat].classes_.tolist()
    user_val = st.selectbox(cat, options)
    user_input[cat] = label_encoders[cat].transform([user_val])[0]

# Prediction
if st.button("Predict Rain Tomorrow"):
    input_df = pd.DataFrame([user_input])
    input_df = imputer.transform(input_df)
    pred = model.predict(input_df)[0]
    result = "Yes 🌧️" if pred == 1 else "No ☀️"
    st.success(f"Prediction: It will rain tomorrow? {result}")
