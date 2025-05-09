import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, confusion_matrix
)

import xgboost as xgb
import mlflow
import mlflow.sklearn

# === CONFIGURATION ===
IMPUTE_MISSING = False
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

POST_EVENT = []

def load_data_and_columns():
    print("Loading columns to drop...")
    with open('columns_to_drop_id.txt', 'r') as f:
        cols1 = [line.strip() for line in f]
    with open('columns_to_drop_high_correlation.txt', 'r') as f:
        cols2 = [line.strip() for line in f]
    with open('columns_to_drop_leakage.txt', 'r') as f:
        cols3 = [line.strip() for line in f]

    cols_to_drop = set(cols1 + cols2 + cols3 + POST_EVENT)

    project_id = os.environ.get('GCP_PROJECT_ID')
    if not project_id:
        raise EnvironmentError("GCP_PROJECT_ID environment variable not set.")

    query = f"""
        SELECT * EXCEPT ({', '.join(f'`{col}`' for col in cols_to_drop)})
        FROM `{project_id}.flights.flights_all`
        LIMIT 1000000
    """
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def prepare_data(df):
    y = df['Cancelled']
    X = df.drop(columns=['Cancelled', 'FlightDate'])
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

def drop_categoricals(X):
    # Drop object-type columns and convert the rest to float64 to avoid pandas.NA issues
    X_clean = X.select_dtypes(exclude='object').copy()
    for col in X_clean.columns:
        if pd.api.types.is_integer_dtype(X_clean[col]) and X_clean[col].isna().any():
            X_clean[col] = X_clean[col].astype('float64')
    return X_clean

def build_pipeline(feature_names):
    numeric_cols = feature_names.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('drop_cat', FunctionTransformer(drop_categoricals, validate=False), feature_names.columns.tolist())
    ])

    steps = [('preprocess', preprocessor)]
    if IMPUTE_MISSING:
        steps.append(('impute', SimpleImputer(strategy='mean')))

    steps.append(('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        tree_method='hist',
        n_jobs=-1,
        random_state=SEED
    )))
    return Pipeline(steps)

def evaluate(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_scores[best_idx],
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def plot_feature_importance(model, out_file='xgb_feature_importance.png'):
    importance = model.get_booster().get_score(importance_type='weight')
    if not importance:
        return
    imp_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values('Importance')
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['Feature'], imp_df['Importance'])
    plt.title("XGBoost Feature Importance (weight)")
    plt.tight_layout()
    plt.savefig(out_file)
    return imp_df

def main():
    df = load_data_and_columns()
    if 'Cancelled' not in df.columns:
        raise ValueError("Target column 'Cancelled' not found.")

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    class_ratio = np.bincount(y_train)
    print(f"Class distribution: {class_ratio[0]}: {class_ratio[1]}")

    # ✅ Ensure experiment exists
    experiment_name = "Default"
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        pipeline = build_pipeline(X_train)

        # Adjust scale_pos_weight dynamically
        pipeline.named_steps['xgb'].set_params(
            scale_pos_weight=class_ratio[0] / class_ratio[1]
        )

        print("Training model...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        duration = time.time() - start
        print(f"Training finished in {duration:.2f} seconds")

        # Predict probabilities
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_proba)

        # Log params
        mlflow.log_params({
            'model': 'XGBoost',
            'dataset_size': len(df),
            'train_ratio': 0.8,
            'scale_pos_weight': round(class_ratio[0] / class_ratio[1], 2),
            'impute': IMPUTE_MISSING
        })

        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, float):
                mlflow.log_metric(k, v)

        # Log confusion matrix as text
        cm_str = f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}\n" \
                 f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}"
        with open("confusion_matrix.txt", "w") as f:
            f.write(cm_str)
        mlflow.log_artifact("confusion_matrix.txt")

        # Plot and log feature importances
        imp_df = plot_feature_importance(pipeline.named_steps['xgb'])
        if imp_df is not None:
            imp_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("xgb_feature_importance.png")
            mlflow.log_artifact("feature_importance.csv")

        # Log trained model
        mlflow.sklearn.log_model(pipeline, "model")

        print("\nEvaluation Metrics:")
        for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            print(f"{k}: {metrics[k]:.4f}")

        print("Run completed and logged to MLflow.")

if __name__ == "__main__":
    main()
