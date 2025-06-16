# rainfall_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("weatherAUS.csv")

# Drop columns with too many missing values or not needed
columns_to_drop = ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date', 'Location', 'RISK_MM']
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# Drop rows where target is missing
df.dropna(subset=["RainTomorrow"], inplace=True)

# Encode categorical columns
label_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split features and target
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=12345)
model.fit(X_train, y_train)

# Save feature order to file
feature_order = X.columns.tolist()

# Save model and preprocessing objects
joblib.dump(feature_order, "feature_order.pkl")
joblib.dump(model, "rainfall_rf_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ Model training complete and files saved!")
