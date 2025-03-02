import pandas as pd 
import numpy as np
from datetime import datetime
import joblib
import json
from scipy.stats import loguniform, randint


from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# 1. Load model, train & test data, and hyperparams
model_path = f"../model/experiments/xgb_20250302-1630_mae_0.5.pkl"
model = joblib.load(model_path)

# Load train, test and validation data
train = pd.read_csv("../data/processed/splits/train_data.csv")
test = pd.read_csv("../data/processed/splits/test_data.csv")
# val = pd.read_csv("../data/processed/splits/val_data.csv")

target = 'puffs'

X_train = train.drop(columns=target)
y_train = train[target]

X_test = test.drop(columns=target)
y_test = test[target]

# Load hyperparams
def load_hyperparams(json_file):
    with open(json_file, "r") as f:
        params = json.load(f)
    
    param_dist = {}
    for key, value in params.items():
        if isinstance(value, dict):
            if value["type"] == "randint":
                param_dist[key] = randint(value["lower"], value["upper"])
            elif value["type"] == "loguniform":
                param_dist[key] = loguniform(value["low"], value["high"])
        else:
            param_dist[key] = value  # Directly use lists or other values
    
    return param_dist

param_dist = load_hyperparams("../config/hyperparams.json")

# 2. Bayesian Optimization with Time-Series CV
tuner = RandomizedSearchCV(
    estimator=XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',  
        enable_categorical=False  
    ),
    param_distributions=param_dist,
    n_iter=1,
    scoring='neg_mean_absolute_error',
    cv=TimeSeriesSplit(n_splits=3),
    verbose=3,
    random_state=42,
    error_score='raise'  
)

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

tuner.fit(X_train, y_train)

# 3. Optimization Results
best_params = tuner.best_params_
print(f"Best Parameters: {best_params}")
print(f"Best MAE: {-tuner.best_score_:.3f}")

# 4. Final Model Training
optimized_model = XGBRegressor(
    objective='reg:squarederror',
    **best_params,
    early_stopping_rounds=20,
    eval_metric=['mae', 'rmse']
)

optimized_model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

# 5. Comparative Clinical Evaluation
baseline_mae = mean_absolute_error(y_test, model.predict(X_test))
optimized_mae = mean_absolute_error(y_test, optimized_model.predict(X_test))

# 6. Save best performing model
model_opt_dir = "../model/production"
if optimized_mae < baseline_mae:
    model_opt_path = f"{model_opt_dir}/xgb_hyp{datetime.now().strftime('%Y%m%d-%H%M')}_mae_{optimized_mae:.1f}.pkl"
else:
    model_opt_path = f"{model_opt_dir}/xgb_base{datetime.now().strftime('%Y%m%d-%H%M')}_mae_{baseline_mae:.1f}.pkl"
    
joblib.dump(optimized_mae, model_opt_path)

print(f"""
Comparative Results:
- Baseline MAE: {baseline_mae:.3f}
- Optimized MAE: {optimized_mae:.3f}
- Improvement: {(1 - optimized_mae/baseline_mae)*100:.1f}%
- Model Saved: {model_opt_path}
""")