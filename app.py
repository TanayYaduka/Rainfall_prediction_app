# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load("rainfall_rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
imputer = joblib.load("imputer.pkl")

# Define all features in same order as used during training
num_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                'Pressure3pm', 'Temp9am', 'Temp3pm']
cat_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
all_features = num_features + cat_features

st.title("Rainfall Prediction App 🌧️")

# Collect user input
user_input = {}

# Numerical Inputs
st.subheader("Numerical Weather Data")
for feat in num_features:
    user_input[feat] = st.number_input(f"{feat}", value=0.0)

# Categorical Inputs
st.subheader("Categorical Weather Data")
for cat in cat_features:
    options = label_encoders[cat].classes_.tolist()
    user_val = st.selectbox(f"{cat}", options)
    user_input[cat] = label_encoders[cat].transform([user_val])[0]

# Prediction
if st.button("Predict Rain Tomorrow"):
    try:
        # Create DataFrame in the same order as during training
        input_df = pd.DataFrame([[user_input[feat] for feat in all_features]],
                                columns=all_features)
        
        # Impute missing values
        input_df_imputed = imputer.transform(input_df)
        
        # Predict
        pred = model.predict(input_df_imputed)[0]
        result = "Yes 🌧️" if pred == 1 else "No ☀️"
        st.success(f"Prediction: Will it rain tomorrow? {result}")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
