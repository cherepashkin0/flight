import os
import time
import random
import numpy as np
import pandas as pd
from google.cloud import bigquery
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.impute import SimpleImputer

# === CONFIGURATION ===
IMPUTE_MISSING = False  # Set to True to enable missing value imputation

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# === POST-EVENT FEATURES TO DROP ===
POST_EVENT = [
    # 'TaxiOut', 'TaxiIn', 'DepDelay', 'DepDelayMinutes',
    # 'ArrDelay', 'ActualElapsedTime', 'AirTime', 
    # 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups',
    # 'Diverted', 'DivAirportLandings', 'DivReachedDest', 
    # 'CancellationCode', 'CRSArrTime'
]


def load_data_and_columns():
    print("Loading columns to drop...")
    with open('columns_to_drop_id.txt', 'r') as f:
        cols_to_drop_1 = [line.strip() for line in f]
    with open('columns_to_drop_high_correlation.txt', 'r') as f:
        cols_to_drop_2 = [line.strip() for line in f]
    with open('columns_to_drop_leakage.txt', 'r') as f:
        cols_to_drop_3 = [line.strip() for line in f]

    cols_to_drop = set(cols_to_drop_1 + cols_to_drop_2 + cols_to_drop_3 + POST_EVENT)

    project_id = os.environ.get('GCP_PROJECT_ID')
    if not project_id:
        raise EnvironmentError("GCP_PROJECT_ID environment variable is not set.")

    drop_list = ', '.join(f'`{col}`' for col in cols_to_drop)
    table_ref = f"`{project_id}.flights.flights_all`"
    query = f"""
        SELECT * EXCEPT ({drop_list})
        FROM {table_ref}
        LIMIT 1000000
    """

    print(f"Executing BigQuery query to load data (dropping {len(cols_to_drop)} columns)...")
    print(f"Dropping columns: {cols_to_drop}")
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns from BigQuery.")
    print(f"Columns names going to process further: {df.columns}")

    return df


def prepare_data(df):
    y = df['Cancelled']
    X = df.drop(columns=['Cancelled', 'FlightDate'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


def preprocess_features(X_train, X_test):
    print("Preprocessing features...")
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    datetime_columns = [col for col in X_train.columns if X_train[col].dtype == 'datetime64[ns]']
    categorical_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']

    for col in datetime_columns:
        for df in (X_train_processed, X_test_processed):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek

        X_train_processed.drop(columns=[col], inplace=True)
        X_test_processed.drop(columns=[col], inplace=True)

    if categorical_columns:
        print(f"Dropping {len(categorical_columns)} categorical columns")
        X_train_processed.drop(columns=categorical_columns, inplace=True)
        X_test_processed.drop(columns=categorical_columns, inplace=True)

    X_train_processed = X_train_processed.fillna(np.nan)
    X_test_processed = X_test_processed.fillna(np.nan)

    if IMPUTE_MISSING:
        missing_cols = X_train_processed.columns[X_train_processed.isna().any()]
        print(f"Imputing {len(missing_cols)} columns with missing values")

        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train_processed)
        X_test_imputed = imputer.transform(X_test_processed)

        X_train_processed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

    return X_train_processed, X_test_processed


def train_xgboost_model(X_train, y_train):
    print("Training XGBoost model...")
    class_counts = np.bincount(y_train)
    imbalance_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1
    print(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': imbalance_ratio,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': SEED
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    X_train_main, X_valid, y_train_main, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    dtrain_main = xgb.DMatrix(X_train_main, label=y_train_main)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    print("Training model...")
    watchlist = [(dtrain_main, 'train'), (dvalid, 'valid')]
    model = xgb.train(params, dtrain_main, num_boost_round=100,
                      evals=watchlist, early_stopping_rounds=50, verbose_eval=25)

    print(f"Best iteration: {model.best_iteration}")
    final_model = xgb.train(params, dtrain, num_boost_round=model.best_iteration)
    print("Training completed.")

    return final_model


def calculate_naive_baseline(y_test):
    print("\nCalculating naive baseline metrics...")
    majority_class = np.argmax(np.bincount(y_test))
    baseline_pred = np.full_like(y_test, majority_class)

    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)

    baseline_proba = np.zeros(len(y_test))
    baseline_proba.fill(majority_class)
    try:
        baseline_roc_auc = roc_auc_score(y_test, baseline_proba)
    except:
        baseline_roc_auc = 0.5

    print("Naive Baseline Metrics:")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Precision: {baseline_precision:.4f}")
    print(f"Recall: {baseline_recall:.4f}")
    print(f"ROC-AUC: {baseline_roc_auc:.4f}")

    return {
        'accuracy': baseline_accuracy,
        'precision': baseline_precision,
        'recall': baseline_recall,
        'roc_auc': baseline_roc_auc
    }


def evaluate_model(model, X_test, y_test, baseline_metrics):
    print("\nEvaluating model on test set...")
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"Best F1 threshold: {best_threshold:.4f}")

    y_pred = (y_pred_proba >= best_threshold).astype(int)

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': 2 * precision_score(y_test, y_pred) * recall_score(y_test, y_pred) /
              (precision_score(y_test, y_pred) + recall_score(y_test, y_pred) + 1e-10),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print("Test Set Metrics:")
    for k, v in test_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

    print("\nComparison with Baseline:")
    print("Metric     | Model     | Baseline  | Improvement")
    print("-----------|-----------|-----------|------------")
    for k in ['accuracy', 'precision', 'recall', 'roc_auc']:
        diff = test_metrics[k] - baseline_metrics[k]
        print(f"{k.capitalize():<10} | {test_metrics[k]:.4f}    | {baseline_metrics[k]:.4f}    | {diff * 100:.2f}%")

    return test_metrics


def visualize_feature_importance(model, feature_names):
    feature_importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values(by='Importance', ascending=False)

    top_features = importance_df.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Feature Importance (weight)')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png')
    print("Feature importance visualization saved to 'xgb_feature_importance.png'")

    importance_df.to_csv("xgb_feature_importance.csv", index=False)
    print("Full feature importances saved to 'xgb_feature_importance.csv'")
    return importance_df


def main():
    df = load_data_and_columns()
    if 'Cancelled' not in df.columns:
        raise ValueError("Target variable 'Cancelled' not found.")

    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
    model = train_xgboost_model(X_train_processed, y_train)
    baseline_metrics = calculate_naive_baseline(y_test)
    test_metrics = evaluate_model(model, X_test_processed, y_test, baseline_metrics)
    importance_df = visualize_feature_importance(model, X_train_processed.columns)

    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")

    try:
        model.save_model('flight_cancellation_xgb_model.json')
        print("Model saved to 'flight_cancellation_xgb_model.json'")

        with open('xgb_model_summary.txt', 'w') as f:
            f.write("=== FLIGHT CANCELLATION PREDICTION MODEL SUMMARY ===\n\n")
            f.write("Model: XGBoost\n")
            f.write(f"Dataset size: {len(X_train) + len(X_test)} flights\n")
            f.write(f"Class imbalance: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.2f}:1\n\n")
            f.write("=== PERFORMANCE METRICS ===\n")
            for k in test_metrics:
                f.write(f"{k.capitalize()}: {test_metrics[k]:.4f}\n")
            f.write("\n=== TOP PREDICTIVE FEATURES ===\n")
            for i, row in importance_df.head(10).iterrows():
                f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}\n")
        print("Performance summary saved to 'xgb_model_summary.txt'")
    except Exception as e:
        print(f"Error saving model or summary: {e}")

    print("\nModel training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
