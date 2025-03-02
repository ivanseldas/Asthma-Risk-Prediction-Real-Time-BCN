import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load pre-split data
def load_split_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['puffs'])  
    y = df['puffs']                 
    return X, y

# Load all splits
X_train, y_train = load_split_data("../data/processed/splits/train_data.csv")
X_val, y_val = load_split_data("../data/processed/splits/val_data.csv") 
X_test, y_test = load_split_data("../data/processed/splits/test_data.csv")

# Load model
model_path = f"../model/production/xgb_hyp20250302-1821_mae_0.5.pkl"
model = joblib.load(model_path)

# Fit model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 1. Core Performance Metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
        'Error Std': np.std(y_true - y_pred),
        'Max Error': np.max(np.abs(y_true - y_pred))
    }

# Generate predictions
test_preds = model.predict(X_test)

# Calculate metrics
metrics = calculate_metrics(y_test, test_preds)
print("Model Performance Metrics:")
print(pd.DataFrame([metrics]))

# 2. Clinical Impact Analysis (Updated for DataFrame Input)
def clinical_error_analysis(df):
    # Critical thresholds (GEMA guidelines)
    df['true_risk'] = np.where(df['true'] > 2, 'High Risk', 'Normal')
    df['pred_risk'] = np.where(df['pred'] > 2, 'High Risk', 'Normal')
    
    confusion = pd.crosstab(df['true_risk'], df['pred_risk'])
    
    sensitivity = confusion.loc['High Risk', 'High Risk']/confusion.loc['High Risk'].sum()
    specificity = confusion.loc['Normal', 'Normal']/confusion.loc['Normal'].sum()
    
    return {
        'confusion_matrix': confusion,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'clinical_accuracy': (confusion.values.diagonal().sum()/confusion.values.sum())
    }

# Prepare error analysis dataframe
error_df = X_test.copy()
error_df['true'] = y_test.values
error_df['pred'] = test_preds

# Generate clinical metrics
clinical_metrics = clinical_error_analysis(error_df)
print("\nClinical Impact Metrics:")
print(pd.DataFrame([clinical_metrics]))