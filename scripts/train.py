import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import json

# Load and prepare data
clinical_df = pd.read_csv("../data/processed/train_ready/clinical_df_v1.csv")
clinical_df.sort_values('timestamp', inplace=True)  

# Strategic features (removed patient_id)
features = [
    'hour',
    'is_night',
    'pm25_24h_avg',
    'no2_exceedance',
    'adherence_trend',
    'gema_risk_score',
    'district_risk',
    'age',
    'has_COPD'
]
target = 'puffs'

X = clinical_df[features]
y = clinical_df[target]

# Temporal split (60% train, 20% validation, 20% test)
n = len(X)
train_end = int(n * 0.6)
val_end = train_end + int(n * 0.2)

X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

# Save datasets to folder
split_dir = "../data/processed/splits"
os.makedirs(split_dir, exist_ok=True)

pd.concat([X_train, y_train], axis=1).to_csv(f"{split_dir}/train_data.csv", index=False)
pd.concat([X_val, y_val], axis=1).to_csv(f"{split_dir}/val_data.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv(f"{split_dir}/test_data.csv", index=False)

# Load model_config
with open("../config/model_config.json", "r") as f:
    params = json.load(f)

# Initialize the model
model = XGBRegressor(**params)

# Train with validation monitoring
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Final evaluation on test set
y_pred = model.predict(X_test)
print(f"\nTest Performance:")
print(f"- MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"- R²: {r2_score(y_test, y_pred):.2f}")

# Clinical utility check
accuracy_within_2 = np.mean(np.abs(y_test - y_pred) <= 2)
print(f"- Predictions within ±2 puffs: {accuracy_within_2:.1%}")

# Save model with timestamp
model_dir = "../model/experiments"
os.makedirs(model_dir, exist_ok=True)
model_path = f"{model_dir}/xgb_{datetime.now().strftime('%Y%m%d-%H%M')}_mae_{mean_absolute_error(y_test, y_pred):.1f}.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")