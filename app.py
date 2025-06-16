{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f336300f-2f22-4a1f-8197-c7185f7aefe6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-06-16 17:21:09.035 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-06-16 17:21:09.052 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load model and preprocessors\n",
    "model = joblib.load(\"rainfall_rf_model.pkl\")\n",
    "label_encoders = joblib.load(\"label_encoders.pkl\")\n",
    "imputer = joblib.load(\"imputer.pkl\")\n",
    "\n",
    "st.title(\"Rainfall Prediction App 🌧️\")\n",
    "\n",
    "# Input fields\n",
    "user_input = {}\n",
    "features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am',\n",
    "            'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',\n",
    "            'Pressure3pm', 'Temp9am', 'Temp3pm']\n",
    "for feat in features:\n",
    "    user_input[feat] = st.number_input(f\"{feat}\", value=0.0)\n",
    "\n",
    "# Categorical inputs\n",
    "cat_features = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']\n",
    "for cat in cat_features:\n",
    "    options = label_encoders[cat].classes_.tolist()\n",
    "    user_val = st.selectbox(cat, options)\n",
    "    user_input[cat] = label_encoders[cat].transform([user_val])[0]\n",
    "\n",
    "# Prediction\n",
    "if st.button(\"Predict Rain Tomorrow\"):\n",
    "    input_df = pd.DataFrame([user_input])\n",
    "    input_df = imputer.transform(input_df)\n",
    "    pred = model.predict(input_df)[0]\n",
    "    result = \"Yes 🌧️\" if pred == 1 else \"No ☀️\"\n",
    "    st.success(f\"Prediction: It will rain tomorrow? {result}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "106c21a0-0f8e-4b92-852b-a5ff5ef4eeeb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
