import os
import time
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, confusion_matrix, fbeta_score, make_scorer
)

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import optuna
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import uuid
import argparse

# === CONFIGURATION ===
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
NOUNS = ["lion", "tiger", "eagle", "hawk", "wolf", "panther", "leopard", "shark", "falcon", "bear"]


def generate_run_name(model_type):
    """Generate a unique run name: <noun>-<4char_id>-<model_type>"""
    noun = random.choice(NOUNS)
    uid = uuid.uuid4().hex[:4]
    return f"{noun}-{uid}-{model_type}"


def load_config(config_path='config_dry_run.yaml'):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data_and_columns(config):
    """Load data from BigQuery and apply column filtering."""
    print("Loading columns to drop...")
    with open(config['data']['columns_to_drop_id'], 'r') as f:
        cols1 = [line.strip() for line in f]
    with open(config['data']['columns_to_drop_high_correlation'], 'r') as f:
        cols2 = [line.strip() for line in f]
    with open(config['data']['columns_to_drop_leakage'], 'r') as f:
        cols3 = [line.strip() for line in f]

    cols_to_drop = set(cols1 + cols2 + cols3 + config['data']['post_event_columns'])
    project_id = os.environ.get('GCP_PROJECT_ID')
    if not project_id:
        raise EnvironmentError("GCP_PROJECT_ID environment variable not set.")

    query = f"""
        SELECT * EXCEPT ({', '.join(f'{col}' for col in cols_to_drop)})
        FROM {project_id}.{config['data']['dataset']}.{config['data']['table']}
        LIMIT {config['data']['limit']}
    """
    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def prepare_data(df, target_col='Cancelled', date_col='FlightDate'):
    """Split data into train and test sets."""
    y = df[target_col]
    X = df.drop(columns=[target_col, date_col])
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def drop_categoricals(X):
    """Drop object-type columns and convert integer columns with NaNs to float."""
    X_clean = X.select_dtypes(exclude='object').copy()
    for col in X_clean.columns:
        if pd.api.types.is_integer_dtype(X_clean[col]) and X_clean[col].isna().any():
            X_clean[col] = X_clean[col].astype('float64')
    return X_clean


def build_pipeline(model_type, params=None, impute_missing=False):
    """Build a pipeline with the specified model and parameters."""
    steps = [('drop_cat', FunctionTransformer(drop_categoricals, validate=False))]
    
    if impute_missing:
        steps.append(('impute', SimpleImputer(strategy='mean')))
    
    if model_type == 'xgboost':
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'n_jobs': -1,
            'random_state': SEED
        }
        if params:
            default_params.update(params)
        steps.append(('model', xgb.XGBClassifier(**default_params)))
    
    elif model_type == 'lightgbm':
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_jobs': -1,
            'random_state': SEED,
            'verbose': -1
        }
        if params:
            default_params.update(params)
        steps.append(('model', lgb.LGBMClassifier(**default_params)))
    
    elif model_type == 'catboost':
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'thread_count': -1,
            'random_seed': SEED,
            'verbose': False
        }
        if params:
            default_params.update(params)
        steps.append(('model', ctb.CatBoostClassifier(**default_params)))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return Pipeline(steps)


def evaluate(y_true, y_scores, beta=1.0):
    """Evaluate model performance metrics, selecting threshold based on F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    # compute F-beta scores
    fbeta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    best_idx = np.nanargmax(fbeta_scores)
    threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': float(f1_scores[best_idx]),
        'fbeta': float(fbeta_scores[best_idx]),
        'roc_auc': roc_auc_score(y_true, y_scores),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def plot_feature_importance(model, model_type, out_file=None):
    """Plot feature importance for the given model."""
    plt.figure(figsize=(12, 8))
    
    if model_type == 'xgboost':
        importance = model.get_booster().get_score(importance_type='weight')
        if not importance:
            return None
        imp_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
    
    elif model_type == 'lightgbm':
        importance = model.booster_.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({
            'Feature': model.booster_.feature_name(),
            'Importance': importance
        })
    
    elif model_type == 'catboost':
        importance = model.get_feature_importance()
        imp_df = pd.DataFrame({
            'Feature': model.feature_names_,
            'Importance': importance
        })
    
    else:
        return None
    
    imp_df = imp_df.sort_values('Importance', ascending=True)
    plt.barh(imp_df['Feature'].iloc[-20:], imp_df['Importance'].iloc[-20:])
    plt.title(f"{model_type.capitalize()} Feature Importance")
    plt.tight_layout()
    
    if out_file:
        plt.savefig(out_file)
        plt.close()
    
    return imp_df


def optimize_hyperparameters(X_train, y_train, model_type, config, cv=5):
    """Use Optuna to find optimal hyperparameters maximizing F-beta, with separate optimize and fixed params."""
    beta = config['optimization'].get('beta', 1.0)
    param_block = config['hyperparameters'][model_type]
    search_space = param_block.get('optimize', {})
    fixed_params = param_block.get('fixed', {})

    def objective(trial):
        # Suggest only the parameters defined under 'optimize'
        if model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int(
                    'max_depth',
                    search_space['max_depth']['min'],
                    search_space['max_depth']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    search_space['learning_rate']['min'],
                    search_space['learning_rate']['max'],
                    log=True
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    search_space['n_estimators']['min'],
                    search_space['n_estimators']['max']
                )
            }
        elif model_type == 'lightgbm':
            params = {
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    search_space['num_leaves']['min'],
                    search_space['num_leaves']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    search_space['learning_rate']['min'],
                    search_space['learning_rate']['max'],
                    log=True
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    search_space['n_estimators']['min'],
                    search_space['n_estimators']['max']
                )
            }
        elif model_type == 'catboost':
            params = {
                'depth': trial.suggest_int(
                    'depth',
                    search_space['depth']['min'],
                    search_space['depth']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    search_space['learning_rate']['min'],
                    search_space['learning_rate']['max'],
                    log=True
                ),
                'iterations': trial.suggest_int(
                    'iterations',
                    search_space['iterations']['min'],
                    search_space['iterations']['max']
                )
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Merge suggested and fixed parameters
        trial_params = {**params, **fixed_params}

        # Build and evaluate pipeline
        pipeline = build_pipeline(model_type, trial_params, config['preprocessing']['impute_missing'])

        # Class balancing
        counts = y_train.value_counts()
        class_ratio = counts.get(0, 0) / counts.get(1, 1)
        if model_type == 'xgboost':
            pipeline.named_steps['model'].set_params(scale_pos_weight=class_ratio)
        elif model_type == 'lightgbm':
            pipeline.named_steps['model'].set_params(class_weight='balanced')
        elif model_type == 'catboost':
            pipeline.named_steps['model'].set_params(auto_class_weights='Balanced')

        # Cross-validate using F-beta
        scorer = make_scorer(fbeta_score, beta=beta)
        cv_results = cross_validate(
            pipeline,
            X_train, y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )
        return np.mean(cv_results['test_score'])

    # Run study
    study = optuna.create_study(
        direction='maximize',
        study_name=f"{model_type}_optimization",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=config['optimization']['n_trials'])

    # Retrieve and merge best parameters with fixed
    best_search_params = study.best_params
    best_params = {**best_search_params, **fixed_params}

    print(f"Best {model_type} parameters: {best_params}")
    print(f"Best {model_type} F{beta}-score: {study.best_value:.4f}")
    return best_params, study.best_value




def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type, best_params, config):
    impute_missing = config['preprocessing']['impute_missing']
    beta = config['optimization'].get('beta', 1.0)
    # MLflow setup unchanged
    mlflow_available = True
    try:
        mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    except Exception:
        mlflow_available = False
        print(f"Warning: MLflow is not available. Training {model_type} without tracking.")
    try:
        if mlflow_available:
            run_name = generate_run_name(model_type)
            mlflow_run = mlflow.start_run(run_name=run_name)
        else:
            from contextlib import nullcontext
            mlflow_run = nullcontext()
        with mlflow_run:
            pipeline = build_pipeline(model_type, best_params, impute_missing)
            # class balancing
            counts = y_train.value_counts()
            class_ratio = counts.loc[0] / counts.loc[1] if 0 in counts.index and 1 in counts.index else 1.0
            if model_type == 'xgboost':
                pipeline.named_steps['model'].set_params(scale_pos_weight=class_ratio)
            elif model_type == 'lightgbm':
                pipeline.named_steps['model'].set_params(class_weight='balanced')
            elif model_type == 'catboost':
                pipeline.named_steps['model'].set_params(auto_class_weights='Balanced')
            # training
            print(f"Training {model_type}...")
            start = time.time()
            pipeline.fit(X_train, y_train)
            duration = time.time() - start
            print(f"Done in {duration:.2f}s")
            # metrics
            y_train_p = pipeline.predict_proba(X_train)[:, 1]
            train_metrics = evaluate(y_train, y_train_p, beta=beta)
            y_test_p = pipeline.predict_proba(X_test)[:, 1]
            test_metrics = evaluate(y_test, y_test_p, beta=beta)
            print(f"{model_type} Test Metrics: F1={test_metrics['f1']:.4f}, F{beta}={test_metrics['fbeta']:.4f}, ROC_AUC={test_metrics['roc_auc']:.4f}")
            if mlflow_available:
                mlflow.log_params({
                    'model': model_type,
                    'run_name': run_name,
                    'duration': round(duration, 2),
                    **best_params
                })
                mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)})
                # artifacts unchanged
                cm = test_metrics['confusion_matrix']
                cm_file = f"{run_name}_cm.txt"
                with open(cm_file, 'w') as f:
                    f.write(f"TN:{cm[0][0]}, FP:{cm[0][1]}\nFN:{cm[1][0]}, TP:{cm[1][1]}")
                mlflow.log_artifact(cm_file)
                fi_file = f"{run_name}_fi.png"
                fi_df = plot_feature_importance(pipeline.named_steps['model'], model_type, fi_file)
                if fi_df is not None:
                    csv_file = f"{run_name}_fi.csv"
                    fi_df.to_csv(csv_file, index=False)
                    mlflow.log_artifact(fi_file)
                    mlflow.log_artifact(csv_file)
                mlflow.sklearn.log_model(pipeline, run_name)
                print(f"Logged run '{run_name}' to MLflow.")
            else:
                print("Artifacts saved locally.")
            return pipeline, test_metrics
    except Exception as e:
        print(f"Error during training {model_type}: {e}")
        return None, None

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """
    Ensure that if the tracking URI is a local file path, 
    its directory exists, then configure MLflow.
    """
    parsed = urlparse(tracking_uri)
    # Treat both "file://…" and bare paths as local
    if parsed.scheme in ("", "file"):
        # parsed.path covers both file://<path> and plain paths
        local_dir = parsed.path or parsed.netloc
        os.makedirs(local_dir, exist_ok=True)
        print(f"Ensured MLflow directory exists at: {local_dir}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to: {tracking_uri!r}")
    print(f"MLflow experiment set to: {experiment_name!r}")

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate flight-cancellation models based on a YAML config."
    )
    parser.add_argument(
        "-c", "--config",
        default="config_dry_run.yaml",
        help="Path to YAML config file (default: config_dry_run.yaml)"
    )
    args = parser.parse_args()

    # load the YAML from the path the user passed in
    config = load_config(args.config)

    df = load_data_and_columns(config)
    df[config['data']['target_column']] = df[config['data']['target_column']].astype(int)
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target_col=config['data']['target_column'],
        date_col=config['data']['date_column']
    )
    setup_mlflow(
        config['mlflow']['tracking_uri'],
        config['mlflow']['experiment_name']
    )
    results = {}
    for model in config['models']:
        print(f"--- {model} ---")
        best_params, _ = optimize_hyperparameters(
            X_train, y_train, model, config,
            cv=config['optimization']['cv_folds']
        )
        _, metrics = train_and_evaluate_model(
            X_train, X_test, y_train, y_test,
            model, best_params, config
        )
        if metrics:
            results[model] = metrics

    beta = config['optimization'].get('beta', 1.0)
    best = max(results, key=lambda m: results[m]['fbeta'])
    print(f"Best model: {best} with F{beta}={results[best]['fbeta']:.4f}")


if __name__ == "__main__":
    main()
