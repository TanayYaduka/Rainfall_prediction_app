
# 🌧️ Rainfall Prediction App

This Streamlit app predicts **whether it will rain tomorrow** using weather conditions such as temperature, wind, humidity, and pressure. The model is trained on the official [Australian Weather dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

🚀 **Deployed App**: [Rainfall Prediction App on Streamlit](https://rainfall-prediction-app-tanay-yaduka.streamlit.app/)

---

## 🔍 Features

- Real-time input for weather parameters
- Auto-encodes categorical values
- Imputes missing data using the same strategy as the training pipeline
- Displays a simple Yes/No prediction with emoji for clarity
- Downloads the trained model directly from Google Drive (no need to upload it to GitHub)

---

## 🧠 Model Details

- **Model Used**: Random Forest Classifier
- **Training Features**:
  - Numerical: MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Temp9am, Temp3pm
  - Categorical: WindGustDir, WindDir9am, WindDir3pm, RainToday
- **Preprocessing**:
  - Label encoding for categorical features
  - SimpleImputer for missing values
  - Feature order preserved via `feature_order.pkl`

---

## 📦 Files Included

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `imputer.pkl` | Scikit-learn imputer for numerical features |
| `label_encoders.pkl` | Encoders for categorical variables |
| `feature_order.pkl` | List of feature column order used during training |
| `requirements.txt` | Required Python libraries |

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rainfall-prediction-app.git
   cd rainfall-prediction-app
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Google Drive Model Link

The model is downloaded at runtime from Google Drive:

- **Model File**: [`rainfall_rf_model.pkl`](https://drive.google.com/file/d/1QThjYFwE9axeNt59e5QBlQ3qvxaUhCHv/view?usp=sharing)

Ensure this file is publicly accessible with "Anyone with the link can view" permission.

---

## ✅ Hosted On

- [Streamlit Cloud](https://streamlit.io/cloud)
- Live App: [Click here to try it out](https://rainfall-prediction-app-tanay-yaduka.streamlit.app/)

---

## 👤 Author

**Tanay Yaduka**  


---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it.
