"""
Flight Cancellation Prediction Model
====================================
A machine learning pipeline for predicting flight cancellations using various boosting models.

The pipeline includes:
- Data loading from BigQuery
- Feature preprocessing
- Model training with XGBoost, LightGBM, or CatBoost
- Hyperparameter optimization with Optuna
- Model evaluation and visualization
- MLflow tracking

Usage:
    python flight_cancellation_predictor.py -c config.yaml
"""

import os
import time
import random
import uuid
import json
import argparse
from contextlib import nullcontext
from urllib.parse import urlparse

# Data processing
import yaml
import numpy as np
import pandas as pd
from google.cloud import bigquery

# Visualization
import matplotlib.pyplot as plt

# ML tools
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, confusion_matrix, fbeta_score, make_scorer
)

# ML models
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

# Hyperparameter optimization and tracking
import optuna
import mlflow
import mlflow.sklearn

from memory_profiler import profile


# === CONFIGURATION ===
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Constants
NOUNS = ["lion", "tiger", "eagle", "hawk", "wolf", "panther", "leopard", "shark", "falcon", "bear"]


# === UTILITIES ===
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


def setup_mlflow(tracking_uri, experiment_name):
    """Configure MLflow tracking URI and experiment."""
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


# === DATA PROCESSING ===
class DataLoader:
    """Handles loading data from BigQuery and initial processing."""
    
    def __init__(self, config):
        self.config = config
        self.project_id = os.environ.get('GCP_PROJECT_ID')
        if not self.project_id:
            raise EnvironmentError("GCP_PROJECT_ID environment variable not set.")
    
    def _load_columns_to_drop(self):
        """Load the list of columns to drop from configuration files."""
        print("Loading columns to drop...")
        cols_to_drop = []
        
        # Load from the three specified files
        file_paths = [
            self.config['data']['columns_to_drop_id'],
            self.config['data']['columns_to_drop_high_correlation'],
            self.config['data']['columns_to_drop_leakage']
        ]
        
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                cols_to_drop.extend([line.strip() for line in f])
        
        # Add post-event columns from config
        cols_to_drop.extend(self.config['data']['post_event_columns'])
        
        return set(cols_to_drop)
    
    def load_data(self):
        """Load data from BigQuery and apply column filtering."""
        cols_to_drop = self._load_columns_to_drop()
        
        query = f"""
            SELECT * EXCEPT ({', '.join(f'{col}' for col in cols_to_drop)})
            FROM {self.project_id}.{self.config['data']['dataset']}.{self.config['data']['table']}
            LIMIT {self.config['data']['limit']}
        """
        
        client = bigquery.Client(project=self.project_id)
        df = client.query(query).to_dataframe()
        print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # Convert target to int
        target_col = self.config['data']['target_column']
        df[target_col] = df[target_col].astype(int)
        
        return df


def prepare_data(df, target_col='Cancelled', date_col='FlightDate'):
    """Split data into train and test sets."""
    y = df[target_col]
    X = df.drop(columns=[target_col, date_col], inplace=False, errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    del df, X          # free memory early; add `gc.collect()` if needed
    return X_train, X_test, y_train, y_test


def drop_categoricals_to_numpy(X: pd.DataFrame) -> np.ndarray:
    X_num = X.select_dtypes(exclude='object')
    for col in X_num.columns:
        if pd.api.types.is_integer_dtype(X_num[col]) or pd.api.types.is_sparse(X_num[col]):
            X_num[col] = X_num[col].astype('float32', copy=False)
    return X_num.to_numpy(dtype=np.float32, copy=False)


# === MODEL BUILDING ===
class ModelFactory:
    """Factory for creating different types of boosting models."""
    
    @staticmethod
    def create_model(model_type, params=None):
        """Create a model of the specified type with given parameters."""
        if model_type == 'xgboost':
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'error',
                'tree_method': 'hist',
                'n_jobs': 4,
                'random_state': SEED,
                'verbosity': 2
            }
            if params:
                default_params.update(params)
            return xgb.XGBClassifier(**default_params)
            
        elif model_type == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'n_jobs': 4,
                'random_state': SEED,
                'verbose': -1
            }
            if params:
                default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
            
        elif model_type == 'catboost':
            default_params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Recall',
                'thread_count': 4,
                'random_seed': SEED,
                'verbose': False
            }
            if params:
                default_params.update(params)
            return ctb.CatBoostClassifier(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def apply_class_balancing(model, model_type, class_ratio):
        """Apply class balancing to the model based on model type."""
        if model_type == 'xgboost':
            model.set_params(scale_pos_weight=class_ratio)
        elif model_type == 'lightgbm':
            model.set_params(class_weight='balanced')
        elif model_type == 'catboost':
            model.set_params(auto_class_weights='Balanced')
        return model


def build_pipeline(model_type, params=None, impute_missing=False):
    """Build a pipeline with the specified model and parameters."""
    steps = [('drop_cat', FunctionTransformer(drop_categoricals_to_numpy, validate=False))]
    
    if impute_missing:
        steps.append(('impute', SimpleImputer(strategy='mean')))
    
    model = ModelFactory.create_model(model_type, params)
    steps.append(('model', model))
    
    return Pipeline(steps)


# === MODEL EVALUATION ===
class ModelEvaluator:
    """Handles model evaluation and metric calculation."""
    
    def __init__(self, beta=1.0):
        self.beta = beta
    
    def evaluate(self, y_true, y_scores):
        """Evaluate model performance metrics, selecting threshold based on F-beta."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Handle empty thresholds edge case
        if len(thresholds) == 0:
            return self._get_default_metrics(y_true, y_scores)
        
        # Compute F1 scores
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Compute F-beta scores
        fbeta_scores = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + 1e-10)
        
        # Find best threshold based on F-beta
        best_idx = np.nanargmax(fbeta_scores)
        threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        
        # Apply threshold and calculate metrics
        y_pred = (y_scores >= threshold).astype(int)
        
        return {
            'threshold': float(threshold),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': float(f1_scores[best_idx]),
            'fbeta': float(fbeta_scores[best_idx]),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _get_default_metrics(self, y_true, y_scores):
        """Return default metrics when thresholds is empty."""
        y_pred = (y_scores >= 0.5).astype(int)
        return {
            'threshold': 0.5,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': fbeta_score(y_true, y_pred, beta=1.0),
            'fbeta': fbeta_score(y_true, y_pred, beta=self.beta),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }


# === VISUALIZATION ===
class FeatureImportancePlotter:
    """Handles plotting and saving feature importance."""
    
    @staticmethod
    def plot_feature_importance(model, model_type, out_file=None):
        """Plot feature importance for the given model."""
        plt.figure(figsize=(12, 8))
        
        imp_df = FeatureImportancePlotter._get_importance_df(model, model_type)
        if imp_df is None:
            return None
        
        # Sort and plot top 20 features
        imp_df = imp_df.sort_values('Importance', ascending=True)
        plt.barh(imp_df['Feature'].iloc[-20:], imp_df['Importance'].iloc[-20:])
        plt.title(f"{model_type.capitalize()} Feature Importance")
        plt.tight_layout()
        
        if out_file:
            plt.savefig(out_file)
            plt.close()
        
        return imp_df
    
    @staticmethod
    def _get_importance_df(model, model_type):
        """Extract feature importance from model based on model type."""
        if model_type == 'xgboost':
            importance = model.get_booster().get_score(importance_type='weight')
            if not importance:
                return None
            return pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
        
        elif model_type == 'lightgbm':
            importance = model.booster_.feature_importance(importance_type='gain')
            return pd.DataFrame({
                'Feature': model.booster_.feature_name(),
                'Importance': importance
            })
        
        elif model_type == 'catboost':
            importance = model.get_feature_importance()
            return pd.DataFrame({
                'Feature': model.feature_names_,
                'Importance': importance
            })
        
        else:
            return None


# === HYPERPARAMETER OPTIMIZATION ===
@profile
class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna."""
    
    def __init__(self, X_train, y_train, model_type, config):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.config = config
        self.beta = config['optimization'].get('beta', 1.0)
        self.cv = config['optimization'].get('cv_folds', 5)
        
        # Extract search space and fixed parameters
        param_block = config['hyperparameters'][model_type]
        self.search_space = param_block.get('optimize', {})
        self.fixed_params = param_block.get('fixed', {})
    
    def _suggest_params(self, trial):
        """Suggest hyperparameters based on model type."""
        params = {}
        
        if self.model_type == 'xgboost':
            params = {
                'max_depth': trial.suggest_int(
                    'max_depth',
                    self.search_space['max_depth']['min'],
                    self.search_space['max_depth']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.search_space['learning_rate']['min'],
                    self.search_space['learning_rate']['max'],
                    log=True
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    self.search_space['n_estimators']['min'],
                    self.search_space['n_estimators']['max']
                )
            }
        elif self.model_type == 'lightgbm':
            params = {
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    self.search_space['num_leaves']['min'],
                    self.search_space['num_leaves']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.search_space['learning_rate']['min'],
                    self.search_space['learning_rate']['max'],
                    log=True
                ),
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    self.search_space['n_estimators']['min'],
                    self.search_space['n_estimators']['max']
                )
            }
        elif self.model_type == 'catboost':
            params = {
                'depth': trial.suggest_int(
                    'depth',
                    self.search_space['depth']['min'],
                    self.search_space['depth']['max']
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.search_space['learning_rate']['min'],
                    self.search_space['learning_rate']['max'],
                    log=True
                ),
                'iterations': trial.suggest_int(
                    'iterations',
                    self.search_space['iterations']['min'],
                    self.search_space['iterations']['max']
                )
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return params
    
    def _objective(self, trial):
        """Optuna objective function for hyperparameter optimization."""
        # Suggest parameters based on search space
        params = self._suggest_params(trial)
        
        # Merge suggested and fixed parameters
        trial_params = {**params, **self.fixed_params}
        
        # Build pipeline
        pipeline = build_pipeline(
            self.model_type, 
            trial_params, 
            self.config['preprocessing']['impute_missing']
        )
        
        # Apply class balancing
        counts = self.y_train.value_counts()
        class_ratio = counts.get(0, 0) / counts.get(1, 1)
        pipeline.named_steps['model'] = ModelFactory.apply_class_balancing(
            pipeline.named_steps['model'], 
            self.model_type, 
            class_ratio
        )
        
        # Cross-validate using F-beta
        scorer = make_scorer(fbeta_score, beta=self.beta)
        cv_results = cross_validate(
            pipeline,
            self.X_train, self.y_train,
            cv=self.cv,
            scoring=scorer,
            n_jobs=1,
            return_train_score=False,
            return_estimator=False
        )     
        
        return np.mean(cv_results['test_score'])
    
    def optimize(self):
        """Run the hyperparameter optimization study."""
        # Set up storage
        storage = optuna.storages.RDBStorage(
            url="sqlite:///optuna_study.db"
        )
        
        # Create and run study
        study = optuna.create_study(
            study_name=f"{self.model_type}_optimization",
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=SEED),
            storage=storage,
            load_if_exists=True
        )
        
        study.optimize(self._objective, n_trials=self.config['optimization']['n_trials'])
        
        # Retrieve and merge best parameters with fixed
        best_search_params = study.best_params
        best_params = {**best_search_params, **self.fixed_params}
        
        # Print results
        print(f"Best {self.model_type} parameters: {best_params}")
        print(f"Best {self.model_type} F{self.beta}-score: {study.best_value:.4f}")
        
        # Save best parameters
        run_name = generate_run_name(self.model_type)
        results_dir = os.path.join("train_results", run_name)
        os.makedirs(results_dir, exist_ok=True)
        
        best_params_path = os.path.join(results_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        return best_params, study.best_value, results_dir


# === MODEL TRAINING ===
class ModelTrainer:
    """Handles model training, evaluation, and artifact saving."""
    
    def __init__(self, X_train, X_test, y_train, y_test, model_type, best_params, config, results_dir=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        self.best_params = best_params
        self.config = config
        self.impute_missing = config['preprocessing']['impute_missing']
        self.beta = config['optimization'].get('beta', 1.0)
        
        # Set up results directory
        if results_dir is None:
            self.run_name = generate_run_name(model_type)
            self.results_dir = os.path.join("train_results", self.run_name)
        else:
            self.results_dir = results_dir
            self.run_name = os.path.basename(results_dir)
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Check MLflow availability
        self.mlflow_available = True
        try:
            mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
        except Exception:
            self.mlflow_available = False
            print(f"Warning: MLflow is not available. Training {model_type} without tracking.")
    @profile
    def train(self):
        """Train the model, evaluate, and save artifacts."""
        try:
            # Start MLflow run if available
            if self.mlflow_available:
                mlflow_run = mlflow.start_run(run_name=self.run_name)
            else:
                mlflow_run = nullcontext()
            
            with mlflow_run:
                # Build pipeline
                pipeline = build_pipeline(self.model_type, self.best_params, self.impute_missing)
                
                # Apply class balancing
                counts = self.y_train.value_counts()
                class_ratio = (counts.loc[0] / counts.loc[1]) if (0 in counts.index and 1 in counts.index) else 1.0
                pipeline.named_steps['model'] = ModelFactory.apply_class_balancing(
                    pipeline.named_steps['model'], 
                    self.model_type, 
                    class_ratio
                )
                
                # Train model
                print(f"Training {self.model_type}...")
                start = time.time()
                pipeline.fit(self.X_train, self.y_train)
                duration = time.time() - start
                print(f"Done in {duration:.2f}s")
                
                # Evaluate
                evaluator = ModelEvaluator(beta=self.beta)
                y_train_p = pipeline.predict_proba(self.X_train)[:, 1]
                train_metrics = evaluator.evaluate(self.y_train, y_train_p)
                
                y_test_p = pipeline.predict_proba(self.X_test)[:, 1]
                test_metrics = evaluator.evaluate(self.y_test, y_test_p)
                
                print(
                    f"{self.model_type} Test Metrics: "
                    f"F1={test_metrics['f1']:.4f}, "
                    f"F{self.beta}={test_metrics['fbeta']:.4f}, "
                    f"ROC_AUC={test_metrics['roc_auc']:.4f}"
                )
                
                # Save artifacts
                self._save_artifacts(pipeline, test_metrics, duration)
                
                return pipeline, test_metrics
        
        except Exception as e:
            print(f"Error during training {self.model_type}: {e}")
            return None, None
    
    def _save_artifacts(self, pipeline, test_metrics, duration):
        """Save model artifacts and log to MLflow if available."""
        # Save best parameters to JSON
        best_params_path = os.path.join(self.results_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save confusion matrix
        cm = test_metrics['confusion_matrix']
        cm_path = os.path.join(self.results_dir, "confusion_matrix.txt")
        with open(cm_path, 'w') as f:
            f.write(f"TN:{cm[0][0]}, FP:{cm[0][1]}\n")
            f.write(f"FN:{cm[1][0]}, TP:{cm[1][1]}")
        
        # Save feature importance
        fi_path = os.path.join(self.results_dir, "feature_importance.png")
        fi_df = FeatureImportancePlotter.plot_feature_importance(
            pipeline.named_steps['model'], 
            self.model_type, 
            fi_path
        )
        
        if fi_df is not None:
            csv_path = os.path.join(self.results_dir, "feature_importance.csv")
            fi_df.to_csv(csv_path, index=False)
        
        # —————— dump the full YAML config for reproducibility ——————
        config_out_path = os.path.join(self.results_dir, "config.yaml")
        with open(config_out_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
        
        # Log to MLflow if available
        if self.mlflow_available:
            # Log artifacts
            mlflow.log_artifact(best_params_path)
            mlflow.log_artifact(cm_path)
            
            if fi_df is not None:
                mlflow.log_artifact(fi_path)
                mlflow.log_artifact(csv_path)
            
            # Log the config.yaml as well
            mlflow.log_artifact(config_out_path)
            
            # Log parameters and metrics
            mlflow.log_params({
                'model': self.model_type,
                'run_name': self.run_name,
                'duration': round(duration, 2),
                **self.best_params
            })
            
            mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)})
            
            # Log model
            mlflow.sklearn.log_model(pipeline, self.run_name)
            print(f"Logged run '{self.run_name}' to MLflow.")
        else:
            print(f"Artifacts saved locally in '{self.results_dir}'.")


# === MAIN EXECUTION ===
def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train and evaluate flight-cancellation models based on a YAML config."
    )
    parser.add_argument(
        "-c", "--config",
        default="config_dry_run.yaml",
        help="Path to YAML config file (default: config_dry_run.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Ensure train_results directory exists
    os.makedirs("train_results", exist_ok=True)
    
    # Set up MLflow
    setup_mlflow(
        config['mlflow']['tracking_uri'],
        config['mlflow']['experiment_name']
    )
    
    # Load data
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target_col=config['data']['target_column'],
        date_col=config['data']['date_column']
    )
    
    # Train models
    results = {}
    for model_type in config['models']:
        print(f"\n=== Training {model_type.upper()} ===")
        
        # Optimize hyperparameters
        optimizer = HyperparameterOptimizer(X_train, y_train, model_type, config)
        best_params, _, _ = optimizer.optimize()
        
        # Create a dedicated directory for this run
        run_name = generate_run_name(model_type)
        results_dir = os.path.join("train_results", run_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Train and evaluate model
        trainer = ModelTrainer(
            X_train, X_test, y_train, y_test,
            model_type, best_params, config, results_dir
        )
        _, metrics = trainer.train()
        
        if metrics:
            results[model_type] = metrics
    
    # Print best model
    if results:
        beta = config['optimization'].get('beta', 1.0)
        best = max(results, key=lambda m: results[m]['fbeta'])
        print(f"\nBest model: {best} with F{beta}={results[best]['fbeta']:.4f}")


if __name__ == "__main__":
    main()
