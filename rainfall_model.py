# rainfall_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("weatherAUS.csv")

# Drop columns with high missing values and data leakage
df = df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date', 'Location', 'RISK_MM'], axis=1)

# Drop rows with missing target
df.dropna(subset=["RainTomorrow"], inplace=True)

# Encode categorical features
label_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separate features and target
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# Impute missing values
imp = SimpleImputer(strategy='mean')
X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = RandomForestClassifier(max_depth=16, n_estimators=100, random_state=12345)
model.fit(X_train, y_train)

# Save model and preprocessors
joblib.dump(model, "rainfall_rf_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imp, "imputer.pkl")

print("✅ Model training complete and saved successfully!")
