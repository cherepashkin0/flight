import os
import uuid
import json
import time
import numpy as np
import pandas as pd
import clickhouse_connect
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import optuna
import wandb
from optuna.visualization import plot_optimization_history, plot_param_importances

import random

def generate_readable_name():
    adjectives = [
        "brave", "curious", "gentle", "bold", "quiet", "lucky", "fierce",
        "bright", "silly", "wise", "wild", "calm", "cheerful", "kind"
    ]
    nouns = [
        "lion", "panda", "eagle", "otter", "dolphin", "tiger", "koala",
        "owl", "fox", "penguin", "bear", "falcon", "shark", "whale"
    ]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:4]}"

# -------------------------------
# Config
# -------------------------------
DB_NAME = 'flights_data'
TABLE_NAME = f'{DB_NAME}.flight_2022_preprocessed'
TARGET_COLUMN = 'Cancelled'
OUTPUT_METRICS = 'ml_model_metrics.json'
WANDB_PROJECT = "Flight_ML_Models"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

# -------------------------------
# Output Directory
# -------------------------------
ARTIFACT_DIR = 'artifacts'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------
# ClickHouse Helpers
# -------------------------------
def connect_to_clickhouse():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST'),
        port=os.getenv('CLICKHOUSE_PORT'),
        username=os.getenv('CLICKHOUSE_USER'),
        password=os.getenv('CLICKHOUSE_PASSWORD')
    )

def load_data(client, table):
    query = f"SELECT * FROM {table} WHERE {TARGET_COLUMN} IS NOT NULL"
    return client.query_df(query)

# -------------------------------
# Objective Function
# -------------------------------
def objective(trial, model_name, X_train, y_train):
    if model_name == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 5, 10),
            'max_depth': trial.suggest_int('max_depth', 2, 3),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
    elif model_name == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 5, 10),
            'max_depth': trial.suggest_int('max_depth', 2, 3),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
            'random_state': 42
        }
        model = LGBMClassifier(**params)
    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 5, 10),
            'depth': trial.suggest_int('depth', 2, 3),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2),
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
    else:
        raise ValueError("Unknown model")

    auc = cross_val_score(model, X_train, y_train, cv=2, scoring='roc_auc').mean()
    wandb.log({"optuna_auc": auc})
    print(f"[{model_name}] Trial {trial.number} AUC: {auc:.4f}")
    return auc

# -------------------------------
# Tune and Evaluate
# -------------------------------
def tune_model(model_name, X_train, y_train, n_trials=5):
    study_name = f"{model_name}_study"
    study_path = os.path.join(ARTIFACT_DIR, f"{study_name}.db")
    storage = f"sqlite:///{study_path}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train), n_trials=n_trials)

    print(f"🏆 Best trial for {model_name}: Value = {study.best_value:.4f}, Params = {study.best_params}")

    fig1 = plot_optimization_history(study)
    fig1_path = os.path.join(ARTIFACT_DIR, f"{model_name}_opt_history.png")
    fig1.write_image(fig1_path)

    fig2 = plot_param_importances(study)
    fig2_path = os.path.join(ARTIFACT_DIR, f"{model_name}_param_importance.png")
    fig2.write_image(fig2_path)

    return study.best_params, [fig1_path, fig2_path]

def train_and_evaluate(model_name, X, y, best_params):
    if model_name == 'xgboost':
        model = XGBClassifier(**best_params, random_state=42)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(**best_params, random_state=42)
    elif model_name == 'catboost':
        model = CatBoostClassifier(**best_params, random_seed=42)
    else:
        raise ValueError("Unknown model")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cross_val_score': float(np.mean(cross_val_score(model, X, y, cv=5)))
    }

    for k, v in metrics.items():
        wandb.log({k: v})

    return model, metrics

# -------------------------------
# Main Function
# -------------------------------
def main():
    print("🔌 Connecting to ClickHouse...")
    client = connect_to_clickhouse()

    print("📥 Loading preprocessed data...")
    df = load_data(client, TABLE_NAME)

    y = df[TARGET_COLUMN].astype(int)
    pre_flight_cols = [
        'CRSDepTime', 'CRSArrTime', 'Distance',
        'Year','Quarter','Month','DayOfWeek',
        'DOT_ID_Marketing_Airline','Flight_Number_Marketing_Airline',
        'Flight_Number_Operating_Airline',
        'OriginAirportID','OriginCityMarketID','OriginStateFips','OriginWac',
        'DestAirportID','DestCityMarketID','DestStateFips',
        'FlightDate_year','FlightDate_month','FlightDate_day',
        'FlightDate_weekday','FlightDate_dayofyear'
    ]
    X = df[pre_flight_cols]

    results = {}

    print("🏁 Starting model training and evaluation...")
    for model_name in tqdm(['xgboost', 'lightgbm', 'catboost'], desc="Training models"):
        print(f"\n⚙️ Tuning {model_name} with Optuna...")
        start_time = time.time()

        run_name = f"{model_name}_{generate_readable_name()}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            entity=WANDB_ENTITY,
            config={},
            reinit=True
        )

        tune_start = time.time()
        best_params, image_paths = tune_model(model_name, X, y)
        tune_duration = round(time.time() - tune_start, 2)
        wandb.log({"optuna_tuning_time_sec": tune_duration})

        print(f"🎯 Best params for {model_name}: {best_params}")
        wandb.log(best_params)

        model, metrics = train_and_evaluate(model_name, X, y, best_params)

        elapsed = round(time.time() - start_time, 2)
        wandb.log({"training_time_sec": elapsed})

        for path in image_paths:
            wandb.log({os.path.basename(path): wandb.Image(path)})

        wandb.finish()

        results[model_name] = {
            "best_params": best_params,
            "metrics": metrics
        }

    output_path = os.path.join(ARTIFACT_DIR, OUTPUT_METRICS)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"📊 Saved all metrics to {output_path}")

if __name__ == "__main__":
    main()
