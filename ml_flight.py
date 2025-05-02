import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import clickhouse_connect

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
import mlflow

from tqdm.auto import tqdm
import time
# from optuna.visualization import plot_optimization_history, plot_param_importances

# -------------------------------
# Config
# -------------------------------
DB_NAME = 'flights_data'
TABLE_NAME = f'{DB_NAME}.flight_2022'
DROP_COLUMNS_FILE = 'flight_figs/columns_to_drop.txt'
TARGET_COLUMN = 'Cancelled'
OUTPUT_METRICS = 'ml_model_metrics.json'
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "Flight ML Models"

# -------------------------------
# Helpers
# -------------------------------

def connect_to_clickhouse():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST'),
        port=os.getenv('CLICKHOUSE_PORT'),
        username=os.getenv('CLICKHOUSE_USER'),
        password=os.getenv('CLICKHOUSE_PASSWORD')
    )

CONFIG_FILE = 'column_config.json'

# -------------------------------
# Helpers
# -------------------------------

def read_column_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config["columns_to_load"], config.get("columns_to_drop", [])

def get_all_columns(client, table):
    query = f"SELECT * FROM {table} LIMIT 1"
    return list(client.query_df(query).columns)

def load_selected_columns(client, table, use_columns):
    query = f"SELECT {', '.join(use_columns)} FROM {table} WHERE {TARGET_COLUMN} IS NOT NULL"
    return client.query_df(query)

# -------------------------------
# Preprocessing
# -------------------------------
def convert_booleans_and_categories(df):
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            # Binary column - likely boolean
            df[col] = df[col].map({True: 1, False: 0}) if df[col].dtype == 'bool' else pd.Categorical(df[col]).codes
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.Categorical(df[col]).codes
    return df


def apply_yeo_johnson(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    transformer = PowerTransformer(method='yeo-johnson')
    df[numeric_cols] = transformer.fit_transform(df[numeric_cols])
    return df

# -------------------------------
# Model Training
# -------------------------------

def evaluate_models(X, y):
    results = {}
    models = {
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'lightgbm': LGBMClassifier(),
        'catboost': CatBoostClassifier(verbose=0)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'cross_val_score': float(np.mean(cross_val_score(model, X, y, cv=5)))
        }

    return results


def objective(trial, model_name, X_train, y_train):
    if model_name == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
    elif model_name == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
        }
        model = LGBMClassifier(**params)
    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 200),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'verbose': 0
        }
        model = CatBoostClassifier(**params)
    else:
        raise ValueError("Unknown model")

    return cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc').mean()

def tune_model(model_name, X_train, y_train, n_trials=20):
    study_name = f"{model_name}_study"
    storage = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train), n_trials=n_trials)
    
    return study.best_params



# -------------------------------
# Main
# -------------------------------

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("🔌 Connecting to ClickHouse...")
    client = connect_to_clickhouse()

    print("📖 Reading column config...")
    cols_to_load, cols_to_drop = read_column_config(CONFIG_FILE)
    print(f"✅ Columns to load: {cols_to_load}")
    print(f"🗑️ Columns to drop after load: {cols_to_drop}")

    print("📥 Loading data from DB...")
    df = load_selected_columns(client, TABLE_NAME, cols_to_load)

    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    X = convert_booleans_and_categories(X)
    print("🧼 Applying Yeo-Johnson transform...")
    X = apply_yeo_johnson(X)

    results = {}

    print("🏁 Starting model training and evaluation...")
    for model_name in tqdm(['xgboost', 'lightgbm', 'catboost'], desc="Training models"):
        print(f"\n⚙️ Tuning {model_name} with Optuna...")
        start_time = time.time()

        best_params = tune_model(model_name, X, y)
        print(f"🎯 Best params for {model_name}: {best_params}")

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)

            model, metrics = train_and_evaluate(model_name, X, y, best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, f"{model_name}_model")

            elapsed = time.time() - start_time
            metrics["training_time_sec"] = round(elapsed, 2)
            mlflow.log_metric("training_time_sec", elapsed)

            results[model_name] = {
                "best_params": best_params,
                "metrics": metrics
            }

    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"📊 Saved all metrics to {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
