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
import psutil

# Data processing
import yaml
import numpy as np
import pandas as pd
from google.cloud import bigquery

# Visualization
import matplotlib.pyplot as plt

# ML tools
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, confusion_matrix, fbeta_score
)

# ML models
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

# Hyperparameter optimization and tracking
import optuna
import mlflow
import mlflow.sklearn

import traceback
import gc 

# === CONFIGURATION ===
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import logging
from pathlib import Path
from urllib.parse import urlparse
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants
NOUNS = ["lion", "tiger", "eagle", "hawk", "wolf", "panther", "leopard", "shark", "falcon", "bear"]


# === UTILITIES ===
def force_gc():
    """Force a full garbage‐collection pass and log how many objects were freed."""
    collected = gc.collect()
    logger.info(f"Forced garbage collection: freed {collected} objects")
    return collected

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
    """
    Configure MLflow. If the URI refers to a local directory (either as
    file://<path> or a plain path), ensure that directory exists.
    """
    # 1) Detect local‐filesystem URIs:
    parsed = urlparse(tracking_uri)
    if parsed.scheme == "file":
        # file://<path>  → strip off the leading slash if present
        local_path = Path(parsed.path)
    elif parsed.scheme == "" and not tracking_uri.startswith(("http://", "https://")):
        # plain path with no scheme
        local_path = Path(tracking_uri)
    else:
        local_path = None

    # 2) Create the directory if it’s local
    if local_path is not None:
        local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured MLflow directory exists at: {local_path}")

    # 3) Hand off to MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI set to: {tracking_uri!r}, experiment is set to: {experiment_name!r}")

def memory_efficient_cv(pipeline, X_train, y_train, cv=5, beta=1.0):
    """
    Run memory-efficient cross validation by manually splitting data
    and cleaning up after each fold.
    
    Args:
        pipeline: The sklearn pipeline to evaluate
        X_train: Feature DataFrame
        y_train: Target Series
        cv: Number of cross-validation folds
        beta: F-beta parameter for scoring
        
    Returns:
        Mean F-beta score across folds
    """
    # Create validation indices
    n_samples = len(X_train)
    fold_size = n_samples // cv
    test_fold = np.repeat(-1, n_samples)
    
    for i in range(cv):
        start = i * fold_size
        end = (i + 1) * fold_size if i < cv - 1 else n_samples
        test_fold[start:end] = i
    
    ps = PredefinedSplit(test_fold)
    
    # Run CV with manual memory cleanup
    scores = []
    for train_idx, val_idx in ps.split():
        # Monitor memory before fold
        before_mem = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info(f"Memory before fold: {before_mem:.2f} GB")
        
        # Get this fold's data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        pipeline.fit(X_fold_train, y_fold_train)
        
        # Predict and score
        y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
        
        # Calculate optimal threshold using F-beta
        precision, recall, thresholds = precision_recall_curve(y_fold_val, y_proba)
        
        # Compute F-beta scores
        eps = 1e-10
        fbeta_scores = (
            (1 + beta ** 2) * precision * recall /
            (beta ** 2 * precision + recall + eps)
        )
        
        # Find best threshold
        best_idx = np.nanargmax(fbeta_scores)
        if best_idx < len(thresholds):
            threshold = float(thresholds[best_idx])
        else:
            threshold = 0.5
            
        # Calculate score with optimal threshold
        y_pred = (y_proba >= threshold).astype(int)
        fold_score = fbeta_score(y_fold_val, y_pred, beta=beta)
        scores.append(fold_score)
        
        # Clear memory
        del X_fold_train, X_fold_val, y_fold_train, y_fold_val
        gc.collect()
        
        # Monitor memory after fold
        after_mem = psutil.Process().memory_info().rss / (1024 ** 3)
        logger.info(f"Memory after fold: {after_mem:.2f} GB, " 
                    f"fold score: {fold_score:.4f}")
    
    mean_score = np.mean(scores)
    logger.info(f"CV complete. Mean F-beta score: {mean_score:.4f}")
    return mean_score

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
        logger.info("Loading columns to drop...")
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

        
        # Log the loaded columns
        unique_cols = sorted(set(cols_to_drop))
        logger.info(f"Dropped columns: {', '.join(unique_cols)}")
        
        return set(unique_cols)
    
    def load_data(self):
        """Load data from BigQuery and apply column filtering."""
        cols_to_drop = self._load_columns_to_drop()
        
        query = f"""
            SELECT * EXCEPT ({', '.join(f'{col}' for col in cols_to_drop)})
            FROM {self.project_id}.{self.config['data']['dataset']}.{self.config['data']['table']}
            LIMIT {self.config['data']['limit']}
        """
        
        client = bigquery.Client(project=self.project_id)
        proc = psutil.Process()
        df = client.query(query).to_dataframe(create_bqstorage_client=True)
        mem = proc.memory_info().rss / 1e9
        logger.info(f"After load_data: df.shape={df.shape}, RSS={mem:.2f} GB")
        df = df.convert_dtypes(dtype_backend="pyarrow")
        logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns. \
                    After convert_dtypes: memory={df.memory_usage(deep=True).sum()/1e9:.2f} GB;\
                      columns={list(df.columns)}…")
        
        # Convert target to int
        target_col = self.config['data']['target_column']
        df[target_col] = df[target_col].astype(int)
        
        return df


def prepare_data(df, target_col='Cancelled', date_col='FlightDate'):
    """Split data into train and test sets."""
    y = df[target_col]
    X = df.drop(columns=[target_col, date_col], inplace=False, errors='ignore')
    logger.info("Splitting data…")
    proc = psutil.Process()
    logger.info(f"Before split: RSS={proc.memory_info().rss/1e9:.2f} GB")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    logger.info(f"Split complete: X_train={X_train.shape}, X_test={X_test.shape}")
    logger.info(f"After split:  RSS={proc.memory_info().rss/1e9:.2f} GB")
    del df, X          # free memory early; add `gc.collect()` if needed
    gc.collect()       # garbage collection
    return X_train, X_test, y_train, y_test

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
                'verbose': 2
            }
            if params:
                default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
            
        elif model_type == 'catboost':
            default_params = {
                'loss_function': 'Logloss',  # Keep using 'Logloss'
                'eval_metric': 'Recall',     # Keep using 'Recall'
                'thread_count': 4,
                'random_seed': SEED,
                'verbose': True
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
            model.set_params(is_unbalance=True)  # More memory efficient than class_weight
        elif model_type == 'catboost':
            model.set_params(auto_class_weights='Balanced')
        return model

def build_pipeline(
    model_type,
    params=None,
    impute_missing=False,
    numeric_cols=None,
    categorical_cols=None
):
    """
    Build a preprocessing + model pipeline.
    
    numeric_cols: list of column-names to treat as numeric
    categorical_cols: list of column-names to treat as categorical
    """
    if numeric_cols is None or categorical_cols is None:
        raise ValueError(
            "build_pipeline requires both numeric_cols and categorical_cols"
        )
    
    # numeric preprocessing
    num_steps = []
    if impute_missing:
        num_steps.append(('impute', SimpleImputer(strategy='mean')))
    num_steps.append(('scale',   StandardScaler()))
    
    # categorical preprocessing
    cat_steps = [
        ('impute',  SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ]
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline(num_steps), numeric_cols),
        ('cat', Pipeline(cat_steps), categorical_cols),
    ])
    
    # model instantiation
    model = ModelFactory.create_model(model_type, params)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model',        model)
    ])


# === MODEL EVALUATION ===
class ModelEvaluator:
    """Handles model evaluation and metric calculation."""
    def __init__(self, beta=1.0):
        self.beta = beta

    def evaluate(self, y_true, y_scores):
        """Evaluate model performance metrics, selecting threshold based on F-beta."""
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Compute F-beta scores
        eps = 1e-10
        fbeta_scores = (
            (1 + self.beta ** 2) * precision * recall /
            (self.beta ** 2 * precision + recall + eps)
        )

        # Determine best threshold
        best_idx = np.nanargmax(fbeta_scores)
        if best_idx < len(thresholds):
            threshold = float(thresholds[best_idx])
        else:
            threshold = 0.5

        # Generate predictions
        y_pred = (y_scores >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': fbeta_score(y_true, y_pred, beta=1.0),
            'fbeta': float(fbeta_scores[best_idx]),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return metrics


# === VISUALIZATION ===
class FeatureImportancePlotter:
    """Handles plotting and saving feature importance."""
    
    @staticmethod
    def plot_feature_importance(pipeline, model_type, out_file=None):
        """
        Plot feature importance, using real feature names from the pipeline's
        preprocessor, not f0/f1.
        
        Args:
            pipeline: sklearn Pipeline with steps ['preprocessor', 'model']
            model_type: one of 'xgboost', 'lightgbm', 'catboost'
            out_file: if provided, path to save the figure
        Returns:
            DataFrame of features & importances, or None if unavailable.
        """
        # Extract model & preprocessor
        model = pipeline.named_steps['model']
        preproc = pipeline.named_steps['preprocessor']
        
        # Get transformed feature names
        try:
            feat_names = preproc.get_feature_names_out()
        except Exception:
            feat_names = None
        
        # Build importance DataFrame
        if model_type == 'xgboost':
            booster = model.get_booster()
            imp_dict = booster.get_score(importance_type='weight')  # {'f0': 12, 'f1': 3, ...}
            if not imp_dict:
                return None
            # Map 'f0' → feat_names[0], etc.
            rows = []
            for fn, imp in imp_dict.items():
                idx = int(fn[1:])  # strip leading 'f'
                name = feat_names[idx] if feat_names is not None else fn
                rows.append((name, imp))
            imp_df = pd.DataFrame(rows, columns=['Feature', 'Importance'])
        
        elif model_type == 'lightgbm':
            booster = model.booster_
            names = booster.feature_name()
            imps = booster.feature_importance(importance_type='gain')
            imp_df = pd.DataFrame({'Feature': names, 'Importance': imps})
        
        elif model_type == 'catboost':
            names = model.feature_names_
            imps = model.get_feature_importance()
            imp_df = pd.DataFrame({'Feature': names, 'Importance': imps})
        
        else:
            return None
        
        # Plot top 20
        imp_df = imp_df.sort_values('Importance', ascending=True)
        top = imp_df.iloc[-20:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(top['Feature'], top['Importance'])
        plt.title(f"{model_type.capitalize()} Feature Importance")
        plt.tight_layout()
        if out_file:
            plt.savefig(out_file)
            plt.close()
        return imp_df
    
    # @staticmethod
    # def _get_importance_df(model, model_type):
    #     """Extract feature importance from model based on model type."""
    #     if model_type == 'xgboost':
    #         importance = model.get_booster().get_score(importance_type='weight')
    #         if not importance:
    #             return None
    #         return pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
        
    #     elif model_type == 'lightgbm':
    #         importance = model.booster_.feature_importance(importance_type='gain')
    #         return pd.DataFrame({
    #             'Feature': model.booster_.feature_name(),
    #             'Importance': importance
    #         })
        
    #     elif model_type == 'catboost':
    #         importance = model.get_feature_importance()
    #         return pd.DataFrame({
    #             'Feature': model.feature_names_,
    #             'Importance': importance
    #         })
        
    #     else:
    #         return None

# === HYPERPARAMETER OPTIMIZATION ===
class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna with config-driven column lists."""
    def __init__(self, X_train, y_train, model_type, config):
        self.X_train    = X_train
        self.y_train    = y_train
        self.model_type = model_type
        self.config     = config
        
        # optimization settings
        self.beta = config['optimization'].get('beta', 1.0)
        self.cv   = config['optimization'].get('cv_folds', 5)
        
        # hyperparameter search space
        block = config['hyperparameters'][model_type]
        self.search_space = block.get('optimize', {})
        self.fixed_params = block.get('fixed', {})

    def _suggest_params(self, trial):
        params = {}
        for name, spec in self.search_space.items():
            low, high = spec['min'], spec['max']
            # decide int vs. float
            p_type = spec.get('type')
            if p_type not in ('int','float'):
                p_type = 'int' if isinstance(low, int) and isinstance(high, int) else 'float'
            if p_type == 'int':
                params[name] = trial.suggest_int(name, low, high)
            else:
                log = spec.get('log', False)
                params[name] = trial.suggest_float(name, low, high, log=log)
        return params

    def _merge_params(self, suggested):
        return {**suggested, **self.fixed_params}

    def _objective(self, trial):
        # 1) Suggest & merge hyperparameters
        trial_params = self._merge_params(self._suggest_params(trial))

        # 2) Read flag: include or exclude categorical columns
        include_cat = self.config['preprocessing'].get('include_categorical', True)

        # Determine categorical columns based on flag
        if include_cat:
            cfg_cat = self.config['preprocessing']['categorical_columns']
            cat_cols = [c for c in cfg_cat if c in self.X_train.columns]
            missing_cat = set(cfg_cat) - set(cat_cols)
            if missing_cat:
                logger.warning(f"Omitting missing categorical columns: {missing_cat}")
        else:
            cat_cols = []

        # 3) Determine numeric columns
        num_cols_cfg = self.config['preprocessing'].get('numeric_columns')
        if num_cols_cfg is None:
            all_num = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in all_num if c not in cat_cols]
        else:
            num_cols = [c for c in num_cols_cfg if c in self.X_train.columns]
            missing_num = set(num_cols_cfg) - set(num_cols)
            if missing_num:
                logger.warning(f"Omitting missing numeric columns: {missing_num}")

        # 4) Build the pipeline with the filtered column lists
        pipeline = build_pipeline(
            self.model_type,
            params            = trial_params,
            impute_missing    = self.config['preprocessing']['impute_missing'],
            numeric_cols      = num_cols,
            categorical_cols  = cat_cols
        )

        # 5) Class balancing
        counts      = self.y_train.value_counts()
        class_ratio = counts.get(0,0) / counts.get(1,1)
        pipeline.named_steps['model'] = ModelFactory.apply_class_balancing(
            pipeline.named_steps['model'],
            self.model_type,
            class_ratio
        )

        # 6) Cross-validation on F-beta
        mean_score = memory_efficient_cv(
                pipeline, 
                self.X_train, 
                self.y_train, 
                cv=self.cv, 
                beta=self.beta
            )
    
        # Force garbage collection after each trial
        force_gc()
        return float(mean_score)

    def optimize(self):
        # study naming
        if self.config['optimization']['if_continue']:
            name = self.config['optimization']['study_name']
        else:
            name = f"{self.model_type}_{generate_run_name(self.model_type)}"

        storage = optuna.storages.RDBStorage(
            url="sqlite:///optuna_study.db",
            engine_kwargs={"connect_args": {"check_same_thread": False}}
        )
        study = optuna.create_study(
            study_name     = name,
            direction      = "maximize",
            sampler        = optuna.samplers.TPESampler(seed=SEED),
            storage        = storage,
            load_if_exists = self.config['optimization']['if_continue']
        )
        study.optimize(self._objective, n_trials=self.config['optimization']['n_trials'])

        # save best params
        best = self._merge_params(study.best_params)
        results_dir = os.path.join("train_results", generate_run_name(self.model_type))
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "best_params.json"), 'w') as f:
            json.dump(best, f, indent=2)

        return best, study.best_value, results_dir


# === MODEL TRAINING ===
class ModelTrainer:
    """Handles model training, evaluation, and artifact saving."""
    def __init__(self, X_train, X_test, y_train, y_test, model_type, best_params, config,
                 results_dir=None, optimization_duration=None):
        self.config = config
        include_cat = config['preprocessing'].get('include_categorical', True)

        if include_cat:
            self.numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
            self.categorical_cols = X_train.select_dtypes(exclude=['int64','float64']).columns.tolist()
        else:
            self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = []  # drop all categoricals    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        self.best_params = best_params
        self.config = config
        self.impute_missing = config['preprocessing']['impute_missing']
        self.beta = config['optimization'].get('beta', 1.0)
        self.optimization_duration = optimization_duration or 0.0
        
        train_cfg = config.get('training', {})
        cv_folds = config['optimization']['cv_folds']
        self.val_frac = 1.0 / cv_folds        

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
            logger.warning(f"Warning: MLflow is not available. Training {model_type} without tracking.")

    def train(self):
        """Train the model, evaluate, and save artifacts (without early stopping)."""
        try:
            # 1) Start MLflow run if available
            if self.mlflow_available:
                mlflow_run = mlflow.start_run(run_name=self.run_name)
            else:
                mlflow_run = nullcontext()

            with mlflow_run:
                # 2) Log optimization duration
                mlflow.log_metric("optimization_duration", round(self.optimization_duration, 2))

                # 3) Build the full pipeline
                pipeline = build_pipeline(
                    self.model_type,
                    self.best_params,
                    self.impute_missing,
                    numeric_cols=self.numeric_cols,
                    categorical_cols=self.categorical_cols
                )

                # 4) Compute and apply class balancing
                counts = self.y_train.value_counts()
                class_ratio = (counts.get(0, 0) / counts.get(1, 1)) if 0 in counts.index and 1 in counts.index else 1.0
                # XGBoost uses scale_pos_weight, LightGBM uses class_weight, CatBoost uses auto_class_weights
                if self.model_type == 'xgboost':
                    pipeline.set_params(model__scale_pos_weight=class_ratio)
                elif self.model_type == 'lightgbm':
                    pipeline.set_params(model__class_weight='balanced')
                elif self.model_type == 'catboost':
                    pipeline.set_params(model__auto_class_weights='Balanced')

                # 5) Train the model on the full training data (no early stopping)
                logger.info(f"Training {self.model_type} without early stopping...")
                train_start = time.time()
                pipeline.fit(self.X_train, self.y_train)
                train_duration = time.time() - train_start
                logger.info(f"Training done in {train_duration:.2f}s")

                # 6) Log durations
                mlflow.log_metric("training_duration", round(train_duration, 2))
                total_duration = self.optimization_duration + train_duration
                mlflow.log_metric("total_duration", round(total_duration, 2))

                # 7) Evaluate on the test set
                evaluator = ModelEvaluator(beta=self.beta)
                y_test_proba = pipeline.predict_proba(self.X_test)[:, 1]
                test_metrics = evaluator.evaluate(self.y_test, y_test_proba)

                logger.info(
                    f"{self.model_type} Test Metrics: "
                    f"F1={test_metrics['f1']:.4f}, "
                    f"F{self.beta}={test_metrics['fbeta']:.4f}, "
                    f"ROC_AUC={test_metrics['roc_auc']:.4f}"
                )

                # 8) Save artifacts and return
                self._save_artifacts(pipeline, test_metrics, train_duration)
                return pipeline, test_metrics

        except Exception as e:
            logger.exception(f"Error during training {self.model_type}: {e}")
            traceback.print_exc()
            return None, None
    
    
    def _save_artifacts(self, pipeline, test_metrics, duration):
        """Save model artifacts and log to MLflow if available."""
        # 1) Save all artifacts to disk
        # — best params
        best_params_path = os.path.join(self.results_dir, "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)

        # — confusion matrix
        cm = test_metrics['confusion_matrix']
        cm_path = os.path.join(self.results_dir, "confusion_matrix.txt")
        with open(cm_path, 'w') as f:
            f.write(f"TN:{cm[0][0]}, FP:{cm[0][1]}\n")
            f.write(f"FN:{cm[1][0]}, TP:{cm[1][1]}")

        # — feature importance (and CSV if available)
        fi_path = os.path.join(self.results_dir, "feature_importance.png")
        fi_df = FeatureImportancePlotter.plot_feature_importance(
            pipeline,
            self.model_type,
            fi_path
        )
        if fi_df is not None:
            csv_path = os.path.join(self.results_dir, "feature_importance.csv")
            fi_df.to_csv(csv_path, index=False)

        # — dump full YAML config
        config_out_path = os.path.join(self.results_dir, "config.yaml")
        with open(config_out_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

        # 2) Log to MLflow (all artifacts in one go) or just note their location
        if self.mlflow_available:
            # Log the entire results_dir in one call
            mlflow.log_artifacts(self.results_dir)

            # Log params & metrics
            mlflow.log_params({
                'model': self.model_type,
                'run_name': self.run_name,
                'duration': round(duration, 2),
                **self.best_params
            })
            mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)})

            # Log the trained pipeline
            mlflow.sklearn.log_model(pipeline, self.run_name)
            logger.info(f"Logged run '{self.run_name}' to MLflow.")
        else:
            logger.info(f"Artifacts saved locally in '{self.results_dir}'.")


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

    logger.info(f"Training with {X_train.shape[1]} features, {X_train.shape[0]} samples. Testing with {X_test.shape[1]} features, {X_test.shape[0]} samples.")
    
    # Train models
    results = {}
    for model_type in config['models']:
        logger.info(f"\n=== Training {model_type.upper()} ===")
        
        # --- 1a. Run hyperparameter optimization and time it
        optimizer = HyperparameterOptimizer(X_train, y_train, model_type, config)
        opt_start = time.time()
        best_params, _, _ = optimizer.optimize()
        opt_duration = time.time() - opt_start
        
        # Create a dedicated directory for this run
        run_name = generate_run_name(model_type)
        results_dir = os.path.join("train_results", run_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Train and evaluate model
        trainer = ModelTrainer(
             X_train, X_test, y_train, y_test,
             model_type, best_params, config, results_dir,
             optimization_duration=opt_duration
        )
        _, metrics = trainer.train()
        
        if metrics:
            results[model_type] = metrics
    
    # Print best model
    if results:
        beta = config['optimization'].get('beta', 1.0)
        best = max(results, key=lambda m: results[m]['fbeta'])
        logger.info(f"\nBest model: {best} with F{beta}={results[best]['fbeta']:.4f}")


if __name__ == "__main__":
    main()
