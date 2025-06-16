{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76d66b02-882d-486b-a563-929b4726af38",
   "metadata": {},
   "outputs": [],
   "source": [
    "# rainfall_model.py\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "import joblib\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(\"weatherAUS.csv\")\n",
    "\n",
    "# Drop columns with too many missing values\n",
    "df = df.drop(['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Date', 'Location'], axis=1)\n",
    "\n",
    "# Drop rows with missing target\n",
    "df.dropna(subset=[\"RainTomorrow\"], inplace=True)\n",
    "\n",
    "# Encode categorical features\n",
    "label_cols = df.select_dtypes(include=['object']).columns\n",
    "label_encoders = {}\n",
    "for col in label_cols:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col].astype(str))\n",
    "    label_encoders[col] = le\n",
    "\n",
    "# Separate features and target\n",
    "X = df.drop(\"RainTomorrow\", axis=1)\n",
    "y = df[\"RainTomorrow\"]\n",
    "\n",
    "# Impute missing values\n",
    "imp = SimpleImputer(strategy='mean')\n",
    "X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
    "\n",
    "# Train model\n",
    "model = RandomForestClassifier(max_depth=16, n_estimators=100, random_state=12345)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save model and preprocessors\n",
    "joblib.dump(model, \"rainfall_rf_model.pkl\")\n",
    "joblib.dump(label_encoders, \"label_encoders.pkl\")\n",
    "joblib.dump(imp, \"imputer.pkl\")\n",
    "print(\"Model training complete and saved!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ef69b66-f22d-4ac1-b341-359d0d3f9079",
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
