import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import fbeta_score
from google.cloud import bigquery
import optuna
import os
import mlflow
import yaml
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
from datetime import datetime, timezone
import logging
import json  # NEW: for writing roles mapping

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Set up logging
log_file = Path(__file__).with_suffix(".log")  # same location, scriptname.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # keep console output
        logging.FileHandler(log_file, mode="w", encoding="utf-8")  # log to file
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SAMPLE_SIZE = config['project']['sample_size']
project_id = os.environ.get('GCP_PROJECT_ID')
table_name = config['project']['table']
dataset_name = config['project']['dataset']
table_id = f'{project_id}.{dataset_name}.{table_name}'

def get_numeric_columns():
    """Get numeric columns from the describe CSV, excluding skip_additional columns"""
    describe_df = pd.read_csv(config['data']['describe_csv_path'])
    numeric_cols = describe_df.loc[describe_df['Role'] == 'num', 'Column_Name'].tolist()
    target_col = describe_df.loc[describe_df['Role'] == 'tgt', 'Column_Name'].tolist()[0]
    
    # NEW: Get skip columns from config and filter them out
    skip_cols = config.get('columns', {}).get('skip_additional', [])
    logger.info(f"Columns to skip from config: {skip_cols}")
    
    # Filter out skip columns from numeric columns
    original_count = len(numeric_cols)
    numeric_cols = [col for col in numeric_cols if col not in skip_cols]
    skipped_count = original_count - len(numeric_cols)
    
    logger.info(f"Filtered out {skipped_count} columns. Using {len(numeric_cols)} numeric features.")
    if skipped_count > 0:
        skipped_numeric = [col for col in skip_cols if col in describe_df.loc[describe_df['Role'] == 'num', 'Column_Name'].tolist()]
        logger.info(f"Actually skipped numeric columns: {skipped_numeric}")
    
    return numeric_cols, target_col

# NEW: build mapping {column: role} filtered to used columns; keep CSV order
def build_roles_mapping(used_columns):
    """
    Read describe CSV and return an Ordered mapping {column: role}
    only for columns present in `used_columns`, preserving CSV order.
    """
    describe_df = pd.read_csv(config['data']['describe_csv_path'])
    df = describe_df[describe_df['Column_Name'].isin(used_columns)]
    # preserve order as in CSV
    mapping = {row['Column_Name']: row['Role'] for _, row in df.iterrows()}
    return mapping

# NEW: persist roles mapping locally and return file path
def save_roles_mapping_locally(roles_mapping, filename="columns_roles.json"):
    """
    Save mapping to a pretty-printed JSON file next to the script.
    """
    out_path = Path(__file__).with_name(filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roles_mapping, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved columns→roles mapping locally: {out_path}")
    return str(out_path)

def load_data_from_bigquery(numeric_cols, target_col, sample_size=None):
    """Load only numeric data from BigQuery"""
    client = bigquery.Client(project=project_id)
    all_columns = numeric_cols + [target_col]
    
    if sample_size:
        pos = sample_size // 10
        neg = sample_size - pos
        cols = ", ".join(all_columns)
        
        query = f"""
        WITH balanced_sample AS (
            SELECT {cols}
            FROM `{table_id}`
            WHERE {target_col} IS TRUE
            ORDER BY RAND()
            LIMIT {pos}
        ),
        normal_sample AS (
            SELECT {cols}
            FROM `{table_id}`
            WHERE {target_col} IS FALSE
            ORDER BY RAND()
            LIMIT {neg}
        )
        SELECT {cols}
        FROM (
            SELECT * FROM balanced_sample
            UNION ALL
            SELECT * FROM normal_sample
        )
        ORDER BY RAND()
        """
    else:
        query = f"SELECT {', '.join(all_columns)} FROM `{table_id}`"
    
    logger.info("Loading data from BigQuery...")
    df = client.query(query).to_dataframe()
    logger.info(f"Loaded data shape: {df.shape}")
    return df

def create_preprocessor(numeric_cols):
    """Create simple preprocessor for numeric data"""
    return Pipeline([
        ('scaler', StandardScaler())
    ])

def objective_lightgbm(trial, X_train, y_train, numeric_cols):
    """Optuna objective for LightGBM with 3 main hyperparameters"""
    # Calculate class weights for imbalanced data
    counts = y_train.value_counts()
    n_pos = int(counts.get(True, 0))
    n_neg = int(counts.get(False, 0))
    scale_pos_weight = n_neg / max(n_pos, 1)
    
    # Only 3 main hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'scale_pos_weight': scale_pos_weight,
        'verbosity': -1,
        'seed': 42
    }
    
    # Create model and pipeline
    model = lgb.LGBMClassifier(**params)
    preprocessor = create_preprocessor(numeric_cols)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)
        score = fbeta_score(y_val, y_pred, beta=2.0)
        scores.append(score)
    
    return np.mean(scores)

def train_final_model(best_params, X_train, X_test, y_train, y_test, numeric_cols, roles_mapping):  # NEW: roles_mapping
    """Train final model with best parameters and log to MLflow"""
    
    # Create final model
    model = lgb.LGBMClassifier(**best_params, objective='binary', seed=42)
    preprocessor = create_preprocessor(numeric_cols)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train final model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    f2_score = fbeta_score(y_test, y_pred, beta=2.0)
    
    # Extract feature importances
    lgbm_model = pipeline.named_steps['classifier']
    feature_importances = lgbm_model.feature_importances_
    
    # Create feature importances DataFrame
    importance_df = pd.DataFrame({
        'feature_name': numeric_cols,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Log to MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(best_params)
        
        # NEW: Log skip columns as parameter for traceability
        skip_cols = config.get('columns', {}).get('skip_additional', [])
        mlflow.log_param("skip_additional_columns", skip_cols)
        mlflow.log_param("num_features_used", len(numeric_cols))
        
        # Log metrics
        mlflow.log_metric("f2_score", f2_score)
        
        # Save feature importances as CSV temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            importance_df.to_csv(f.name, index=False)
            temp_csv_path = f.name
        
        # Log feature importances CSV as artifact
        mlflow.log_artifact(temp_csv_path, artifact_path="feature_importances.csv")
        os.unlink(temp_csv_path)

        # NEW: save roles mapping as JSON (local + MLflow artifact)
        local_roles_path = save_roles_mapping_locally(roles_mapping, filename="columns_roles.json")
        mlflow.log_artifact(local_roles_path, artifact_path="columns_roles.json")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="lightgbm_model"
        )
        
        # Get artifact URIs
        model_uri = mlflow.get_artifact_uri("lightgbm_model")
        feature_importance_uri = mlflow.get_artifact_uri("feature_importances.csv")
        roles_mapping_uri = mlflow.get_artifact_uri("columns_roles.json")  # NEW
        
        logger.info(f"Model logged to MLflow with F2 score: {f2_score:.4f}")
        logger.info(f"Model URI: {model_uri}")
        logger.info(f"Feature importances URI: {feature_importance_uri}")
        logger.info(f"Roles mapping URI: {roles_mapping_uri}")  # NEW
        logger.info(f"Top 5 features: {importance_df.head()['feature_name'].tolist()}")
        
        return pipeline, f2_score, model_uri, feature_importance_uri

def log_metrics_to_bigquery(f2_score, model_uri):
    """Log metrics to BigQuery table"""
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{config['bq_logging']['dataset']}.{config['bq_logging']['target_table']}"
    
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_name": config['project']['name'],
        "model_name": "lightgbm",
        "metric_name": "f2_score",
        "metric_value": f2_score,
        "pipeline_uri": model_uri
    }
    
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        logger.error(f"Error inserting row to BigQuery: {errors}")
    else:
        logger.info("Metrics logged to BigQuery successfully")

def main():
    """Main execution function"""
    try:
        logger.info("=== Starting ML Pipeline ===")
        
        # Get numeric columns (now with skip filtering)
        numeric_cols, target_col = get_numeric_columns()
        logger.info(f"Using {len(numeric_cols)} numeric features after filtering skip_additional columns")
        
        # Build roles mapping ONLY for used columns (numeric + target)
        used_columns = list(numeric_cols) + [target_col]  # what model really sees
        roles_mapping = build_roles_mapping(used_columns)  # NEW
        logger.info(f"Columns→roles mapping (used): {roles_mapping}")  # NEW
        
        # Load data
        df = load_data_from_bigquery(numeric_cols, target_col, SAMPLE_SIZE)
        df = df.dropna(subset=[target_col])
        
        X = df[numeric_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Optuna optimization
        logger.info("Starting Optuna optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective_lightgbm(trial, X_train, y_train, numeric_cols),
            n_trials=config['optuna']['n_trials']
        )
        
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best F2 score: {study.best_value:.4f}")
        
        # Train final model
        pipeline, f2_score, model_uri, feature_importance_uri = train_final_model(
            study.best_params, X_train, X_test, y_train, y_test, numeric_cols, roles_mapping  # NEW
        )
        
        # Log metrics to BigQuery
        log_metrics_to_bigquery(f2_score, model_uri)
        
        logger.info("=== Pipeline completed successfully ===")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
