import os
import json
import logging
import warnings
import uuid
import numpy as np
import time
from datetime import datetime

import yaml
import dask
import clickhouse_connect
import dask.dataframe as dd
from dask.delayed import delayed
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from xgboost.dask import DaskDMatrix, train as xgb_dask_train, predict as xgb_dask_predict
from lightgbm.dask import DaskLGBMClassifier
from catboost import CatBoostClassifier

import pandas as pd
import wandb

# -------------------------------
# Suppress Specific Warnings
# -------------------------------
warnings.filterwarnings(
    "ignore",
    message=".*No further splits with positive gain.*",
    category=UserWarning
)

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------------------
# Define helper functions outside of main()
# -------------------------------
@delayed
def fetch_partition(host, port, username, password, table_name, target_column, offset, length):
    """Fetch a partition of data from ClickHouse.
    This function creates a new connection for each partition to avoid pickling issues.
    """
    # Create a new connection for each partition
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password
    )
    return client.query_df(
        f"SELECT * FROM {table_name} WHERE {target_column} IS NOT NULL LIMIT {length} OFFSET {offset}"
    )

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate classification metrics."""
    y_pred = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics

# -------------------------------
# Main Function
# -------------------------------
def main():
    start_time = time.time()
    
    # Start Dask client under guard
    client = Client()
    logging.info("Dask dashboard: %s", client.dashboard_link)
    
    # Log worker info
    worker_info = client.ncores()
    logging.info("Dask workers: %d, Total cores: %d", 
                len(worker_info), sum(worker_info.values()))

    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Check if this is a dry run
    DRY_RUN = cfg.get('dry_run', False)
    SAMPLE_FRACTION = cfg.get('sample_fraction', 0.1) if DRY_RUN else 1.0
    
    if DRY_RUN:
        logging.info("Running in DRY RUN mode with %.1f%% of data", SAMPLE_FRACTION * 100)
    else:
        logging.info("Running with FULL dataset")

    DB_NAME = cfg['db']['name']
    TABLE_NAME = f"{DB_NAME}.{cfg['db']['table']}"
    TARGET_COLUMN = cfg['target_column']
    OUTPUT_METRICS = cfg['output']['metrics_file']
    ARTIFACT_DIR = cfg['output']['artifact_dir']
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    WANDB_PROJECT = cfg['wandb']['project']
    WANDB_ENTITY = cfg['wandb']['entity'] or os.getenv('WANDB_ENTITY')
    FIXED_PARAMS = cfg['fixed_params']  # Use fixed parameters instead of grid search
    PART_SIZE = cfg.get('batch_size', 100_000)
    RUN_SUFFIX = uuid.uuid4().hex[:4]

    # ClickHouse connection parameters
    CH_HOST = os.getenv('CLICKHOUSE_HOST')
    CH_PORT = os.getenv('CLICKHOUSE_PORT')
    CH_USER = os.getenv('CLICKHOUSE_USER')
    CH_PASS = os.getenv('CLICKHOUSE_PASSWORD')

    # Create a client for metadata operations only
    click_client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASS
    )

    # Estimate total rows
    count_df = click_client.query_df(
        f"SELECT count() AS cnt FROM {TABLE_NAME} WHERE {TARGET_COLUMN} IS NOT NULL"
    )
    TOTAL_ROWS = int(count_df['cnt'].iloc[0])
    logging.info("Total rows: %d", TOTAL_ROWS)
    
    # In dry run mode, use a sample of the data
    if DRY_RUN:
        SAMPLE_ROWS = int(TOTAL_ROWS * SAMPLE_FRACTION)
        logging.info("Using %d rows for dry run (%.1f%% sample)", 
                    SAMPLE_ROWS, SAMPLE_FRACTION * 100)
        
        # For dry run, get a random sample rather than sequential rows
        sample_query = f"""
        SELECT * FROM {TABLE_NAME} 
        WHERE {TARGET_COLUMN} IS NOT NULL 
        ORDER BY rand() 
        LIMIT {SAMPLE_ROWS}
        """
        
        # Get sample data directly instead of using partitioned approach
        sample_df = click_client.query_df(sample_query)
        logging.info("Sample data shape: %s", sample_df.shape)
        
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(sample_df, npartitions=min(20, len(sample_df) // 10000 + 1))
        logging.info("Dask DataFrame partitions: %d", ddf.npartitions)
    else:
        # Build Dask DataFrame via delayed partitions - passing connection params not the connection
        partitions = [
            fetch_partition(
                CH_HOST, CH_PORT, CH_USER, CH_PASS, 
                TABLE_NAME, TARGET_COLUMN, 
                off, PART_SIZE
            ) 
            for off in range(0, TOTAL_ROWS, PART_SIZE)
        ]
        
        # Get the schema by running a small query directly
        sample_df = click_client.query_df(
            f"SELECT * FROM {TABLE_NAME} WHERE {TARGET_COLUMN} IS NOT NULL LIMIT 1"
        )
        meta = sample_df.iloc[0:0]  # Empty DataFrame with correct schema
        
        ddf = dd.from_delayed(partitions, meta=meta)
        logging.info("Dask DataFrame partitions: %d", ddf.npartitions)

    # Features and labels
    features = [c for c in ddf.columns if c != TARGET_COLUMN]
    X = ddf[features]
    y = ddf[TARGET_COLUMN].astype(int)

    # Train/test split sequential
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=False
    )

    # Final training and logging using fixed parameters from config
    results = {}
    for model_name, params in FIXED_PARAMS.items():
        logging.info("Training %s with fixed parameters...", model_name)
        run_name = f"{model_name}_{RUN_SUFFIX}"
        
        # Initialize wandb run
        if not DRY_RUN and WANDB_PROJECT:
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                entity=WANDB_ENTITY,
                config={
                    "model": model_name,
                    "parameters": params,
                    "dry_run": DRY_RUN,
                    "total_rows": TOTAL_ROWS,
                    "sample_fraction": SAMPLE_FRACTION if DRY_RUN else 1.0,
                    "timestamp": datetime.now().isoformat(),
                },
                reinit=True
            )
        
        try:
            if model_name == 'xgboost':
                # Use standard XGBoost Dask API
                model_path = os.path.join(ARTIFACT_DIR, f"xgboost_model_{RUN_SUFFIX}.json")
                
                # Create XGBoost params dict
                xgb_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    **params
                }
                
                # Create DaskDMatrix for training
                dtrain = DaskDMatrix(client, X_train, y_train)
                dtest = DaskDMatrix(client, X_test, y_test)
                
                # Log training progress to wandb
                evals_result = {}
                
                t_start = time.time()
                # Train the model with evaluation
                result = xgb_dask_train(
                    xgb_params, 
                    dtrain, 
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    evals_result=evals_result,
                    num_boost_round=params.get('n_estimators', 100)
                )
                booster = result['booster']
                train_time = time.time() - t_start
                
                # Log training metrics to wandb
                if not DRY_RUN and WANDB_PROJECT:
                    # Log training progress if available
                    for i, (train_metric, test_metric) in enumerate(zip(
                            evals_result.get('train', {}).get('auc', []),
                            evals_result.get('test', {}).get('auc', [])
                    )):
                        wandb.log({
                            "iteration": i,
                            "train_auc": train_metric,
                            "test_auc": test_metric
                        })
                
                # Evaluate on test set
                t_start = time.time()
                y_pred_proba = xgb_dask_predict(client, booster, dtest)
                predict_time = time.time() - t_start
                
                # Compute metrics
                y_test_np = y_test.compute()
                y_pred_np = client.compute(y_pred_proba).result()
                metrics = calculate_metrics(y_test_np, y_pred_np)
                
                # Save the model
                booster.save_model(model_path)
                
                # Log model to wandb
                if not DRY_RUN and WANDB_PROJECT:
                    # Log performance metrics
                    wandb.log({
                        "test_auc": metrics['auc'],
                        "test_accuracy": metrics['accuracy'],
                        "test_precision": metrics['precision'],
                        "test_recall": metrics['recall'],
                        "test_f1": metrics['f1'],
                        "train_time_seconds": train_time,
                        "predict_time_seconds": predict_time,
                        "total_time_seconds": train_time + predict_time
                    })
                    
                    # Log model artifact
                    model_artifact = wandb.Artifact(
                        name=f"xgboost-model-{run_name}",
                        type="model",
                        description=f"XGBoost model with AUC {metrics['auc']:.4f}"
                    )
                    model_artifact.add_file(model_path)
                    wandb_run.log_artifact(model_artifact)
                    
                    # Optional: Log feature importance
                    feature_importance = booster.get_score(importance_type='gain')
                    wandb.log({"feature_importance": wandb.Table(
                        data=[[k, v] for k, v in feature_importance.items()],
                        columns=["feature", "importance"]
                    )})
                
                results[model_name] = {
                    'model_path': model_path,
                    'test_auc': float(metrics['auc']),
                    'test_accuracy': float(metrics['accuracy']),
                    'test_precision': float(metrics['precision']),
                    'test_recall': float(metrics['recall']),
                    'test_f1': float(metrics['f1']),
                    'train_time_seconds': train_time,
                    'predict_time_seconds': predict_time,
                    'dry_run': DRY_RUN
                }
                logging.info("XGBoost test metrics - AUC: %.4f, Accuracy: %.4f, F1: %.4f (train: %.1fs, predict: %.1fs)", 
                            metrics['auc'], metrics['accuracy'], metrics['f1'], train_time, predict_time)
                
            elif model_name == 'lightgbm':
                model_path = os.path.join(ARTIFACT_DIR, f"lightgbm_model_{RUN_SUFFIX}.txt")
                
                # Prepare metric tracking for LightGBM
                metric_callback = []
                
                clf = DaskLGBMClassifier(
                    **params,
                    callbacks=[metric_callback]
                )
                
                t_start = time.time()
                clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
                train_time = time.time() - t_start
                
                # Evaluate on test set
                t_start = time.time()
                y_pred_proba = clf.predict_proba(X_test)
                predict_time = time.time() - t_start
                
                # For LightGBM with Dask, predictions are also Dask arrays
                y_test_np = y_test.compute()
                y_pred_np = y_pred_proba[:, 1].compute()
                metrics = calculate_metrics(y_test_np, y_pred_np)
                
                # Save the model
                clf.booster_.save_model(model_path)
                
                # Log to wandb
                if not DRY_RUN and WANDB_PROJECT:
                    # Log performance metrics
                    wandb.log({
                        "test_auc": metrics['auc'],
                        "test_accuracy": metrics['accuracy'],
                        "test_precision": metrics['precision'],
                        "test_recall": metrics['recall'],
                        "test_f1": metrics['f1'],
                        "train_time_seconds": train_time,
                        "predict_time_seconds": predict_time,
                        "total_time_seconds": train_time + predict_time
                    })
                    
                    # Log model artifact
                    model_artifact = wandb.Artifact(
                        name=f"lightgbm-model-{run_name}",
                        type="model",
                        description=f"LightGBM model with AUC {metrics['auc']:.4f}"
                    )
                    model_artifact.add_file(model_path)
                    wandb_run.log_artifact(model_artifact)
                    
                    # Log feature importance if available
                    if hasattr(clf, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': features,
                            'importance': clf.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        wandb.log({"feature_importance": wandb.Table(
                            dataframe=importance_df
                        )})
                
                results[model_name] = {
                    'model_path': model_path,
                    'test_auc': float(metrics['auc']),
                    'test_accuracy': float(metrics['accuracy']),
                    'test_precision': float(metrics['precision']),
                    'test_recall': float(metrics['recall']),
                    'test_f1': float(metrics['f1']),
                    'train_time_seconds': train_time,
                    'predict_time_seconds': predict_time,
                    'dry_run': DRY_RUN
                }
                logging.info("LightGBM test metrics - AUC: %.4f, Accuracy: %.4f, F1: %.4f (train: %.1fs, predict: %.1fs)", 
                            metrics['auc'], metrics['accuracy'], metrics['f1'], train_time, predict_time)
                
            else:  # catboost
                model_path = os.path.join(ARTIFACT_DIR, f"catboost_model_{RUN_SUFFIX}.cbm")
                model = CatBoostClassifier(**params)
                
                # CatBoost needs to compute the Dask DataFrame
                logging.info("Computing data for CatBoost (non-Dask)...")
                X_train_cb = X_train.compute()
                y_train_cb = y_train.compute()
                X_test_cb = X_test.compute()
                y_test_cb = y_test.compute()
                
                # Training metrics
                train_metrics = {}
                
                t_start = time.time()
                model.fit(
                    X_train_cb, y_train_cb, 
                    eval_set=[(X_test_cb, y_test_cb)],
                    use_best_model=True, 
                    verbose=DRY_RUN
                )
                train_time = time.time() - t_start
                
                # Evaluate on test set
                t_start = time.time()
                y_pred_proba = model.predict_proba(X_test_cb)
                predict_time = time.time() - t_start
                
                metrics = calculate_metrics(y_test_cb, y_pred_proba[:, 1])
                
                # Save the model
                model.save_model(model_path)
                
                # Log to wandb
                if not DRY_RUN and WANDB_PROJECT:
                    # Log performance metrics
                    wandb.log({
                        "test_auc": metrics['auc'],
                        "test_accuracy": metrics['accuracy'],
                        "test_precision": metrics['precision'],
                        "test_recall": metrics['recall'],
                        "test_f1": metrics['f1'],
                        "train_time_seconds": train_time,
                        "predict_time_seconds": predict_time,
                        "total_time_seconds": train_time + predict_time
                    })
                    
                    # Log model artifact
                    model_artifact = wandb.Artifact(
                        name=f"catboost-model-{run_name}",
                        type="model",
                        description=f"CatBoost model with AUC {metrics['auc']:.4f}"
                    )
                    model_artifact.add_file(model_path)
                    wandb_run.log_artifact(model_artifact)
                    
                    # Log feature importance
                    feature_importance = model.get_feature_importance()
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    wandb.log({"feature_importance": wandb.Table(
                        dataframe=importance_df
                    )})
                
                results[model_name] = {
                    'model_path': model_path,
                    'test_auc': float(metrics['auc']),
                    'test_accuracy': float(metrics['accuracy']),
                    'test_precision': float(metrics['precision']),
                    'test_recall': float(metrics['recall']),
                    'test_f1': float(metrics['f1']),
                    'train_time_seconds': train_time,
                    'predict_time_seconds': predict_time,
                    'dry_run': DRY_RUN
                }
                logging.info("CatBoost test metrics - AUC: %.4f, Accuracy: %.4f, F1: %.4f (train: %.1fs, predict: %.1fs)", 
                            metrics['auc'], metrics['accuracy'], metrics['f1'], train_time, predict_time)
        except Exception as e:
            logging.error("Error training %s: %s", model_name, e)
            results[model_name] = {
                'error': str(e),
                'dry_run': DRY_RUN
            }
        
        # Close wandb run if active
        if not DRY_RUN and WANDB_PROJECT:
            wandb.finish()

    # Save results
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    metrics_filename = f"{'dry_run_' if DRY_RUN else ''}{OUTPUT_METRICS}"
    metrics_path = os.path.join(ARTIFACT_DIR, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        # Add metadata to results
        full_results = {
            'metadata': {
                'dry_run': DRY_RUN,
                'sample_fraction': SAMPLE_FRACTION if DRY_RUN else 1.0,
                'total_rows': TOTAL_ROWS,
                'sample_rows': int(TOTAL_ROWS * SAMPLE_FRACTION) if DRY_RUN else TOTAL_ROWS,
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': time.time() - start_time
            },
            'models': results
        }
        json.dump(full_results, f, indent=4)
    
    logging.info("Saved metrics to %s", metrics_path)
    
    # Log total runtime
    total_runtime = time.time() - start_time
    logging.info("Total runtime: %.2f seconds (%.2f minutes)", 
                total_runtime, total_runtime / 60)
    
    # Close Dask client
    client.close()
    logging.info("Dask client closed. %s complete.", "Dry run" if DRY_RUN else "Job")

if __name__ == '__main__':
    main()
