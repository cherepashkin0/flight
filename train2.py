"""
Flight Cancellation Prediction Model (Simplified)
================================================
A streamlined machine learning pipeline for predicting flight cancellations.

Features:
- Data loading from BigQuery
- Automatic feature type detection
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
from pathlib import Path
from urllib.parse import urlparse
import logging
import traceback
import gc

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
import warnings
warnings.filterwarnings('ignore', 
                       message='X does not have valid feature names, but LGBMClassifier was fitted with feature names',
                       category=UserWarning)
# === CONFIGURATION ===
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

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
def generate_run_name(model_type):
    """Generate a unique run name: <noun>-<4char_id>-<model_type>"""
    noun = random.choice(NOUNS)
    uid = uuid.uuid4().hex[:4]
    return f"{noun}-{uid}-{model_type}"


def load_config(config_path='config_dry_run.yaml'):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_mlflow(tracking_uri, experiment_name):
    """Configure MLflow with local directory support."""
    # Handle local filesystem URIs
    parsed = urlparse(tracking_uri)
    if parsed.scheme == "file":
        local_path = Path(parsed.path)
    elif parsed.scheme == "" and not tracking_uri.startswith(("http://", "https://")):
        local_path = Path(tracking_uri)
    else:
        local_path = None

    if local_path is not None:
        local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured MLflow directory exists at: {local_path}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri}, experiment: {experiment_name}")


class ArtifactManager:
    """Handle all file I/O operations."""
    
    @staticmethod
    def save_json(data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def save_yaml(data, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def save_text(text, filepath):
        with open(filepath, 'w') as f:
            f.write(text)
    
    @staticmethod
    def load_lines(filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f]


# === COLUMN TYPE DETECTION ===
class ColumnTypeDetector:
    """Single source of truth for column type detection with manual YAML configuration."""
    
    NUMERIC_DTYPES = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'Int64', 'Float64']
    CATEGORICAL_DTYPES = ['object', 'string', 'category']
    
    @classmethod
    def detect_types(cls, df, config):
        manual_config = config['preprocessing'].get('manual_column_types')
        if manual_config:
            return cls._get_manual_column_types(df, manual_config)
        else:
            return cls._get_automatic_column_types(df, config)
    
    @classmethod
    def _get_manual_column_types(cls, df, manual_config):
        all_columns = set(df.columns)

        onehot_categorical = [col for col in manual_config.get('one_hot_categorical', []) if col in all_columns]
        not_onehot_categorical = [col for col in manual_config.get('not_one_hot_categorical', []) if col in all_columns]
        numerical = [col for col in manual_config.get('numerical', []) if col in all_columns]
        
        # Warn if columns are missing (optional, but helpful)
        for group_name, config_cols in [
            ('one_hot_categorical', manual_config.get('one_hot_categorical', [])),
            ('not_onehot_categorical', manual_config.get('not_one_hot_categorical', [])),
            ('numerical', manual_config.get('numerical', []))
        ]:
            missing = set(config_cols) - all_columns
            if missing:
                logger.warning(f"The following columns from '{group_name}' are not in the DataFrame and will be ignored: {missing}")

        if not numerical:
            used_columns = set(onehot_categorical + not_onehot_categorical)
            numerical = [col for col in df.columns if col not in used_columns]
            logger.info(f"No numerical columns specified, using remaining {len(numerical)} columns as numerical")
        logger.info(f"Manual column configuration:")
        logger.info(f"  - Numerical: {len(numerical)}")
        logger.info(f"  - One-hot categorical: {len(onehot_categorical)}")
        logger.info(f"  - Not-one-hot categorical: {len(not_onehot_categorical)}")
        return numerical, onehot_categorical, not_onehot_categorical

    @classmethod
    def _get_automatic_column_types(cls, df, config):
        actual_numeric = df.select_dtypes(include=cls.NUMERIC_DTYPES).columns.tolist()
        actual_categorical = df.select_dtypes(include=cls.CATEGORICAL_DTYPES).columns.tolist()
        include_cat = config['preprocessing'].get('include_categorical', True)
        if include_cat:
            cfg_cat = config['preprocessing'].get('categorical_columns', [])
            cat_cols = cls._filter_valid_columns(cfg_cat, actual_categorical, df.columns, 'categorical')
        else:
            cat_cols = []
        cfg_num = config['preprocessing'].get('numeric_columns')
        if cfg_num:
            num_cols = cls._filter_valid_columns(cfg_num, actual_numeric, df.columns, 'numeric')
        else:
            num_cols = [c for c in actual_numeric if c not in cat_cols]
        # By default, all cats as one-hot, and not_onehot_cat empty
        return num_cols, cat_cols, []
    
    @staticmethod
    def _filter_valid_columns(configured, actual_type, all_columns, column_type):
        """Filter configured columns to only valid ones."""
        if not configured:
            return actual_type
        
        valid = [c for c in configured if c in actual_type]
        missing = set(configured) - set(all_columns)
        wrong_type = set(configured) - set(actual_type) - missing
        
        if missing:
            logger.warning(f"Missing {column_type} columns: {missing}")
        if wrong_type:
            logger.warning(f"Wrong type for {column_type} columns: {wrong_type}")
        
        return valid


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
        
        # Load from specified files
        file_paths = [
            self.config['data']['columns_to_drop_id'],
            self.config['data']['columns_to_drop_high_correlation'],
            self.config['data']['columns_to_drop_leakage'],
            self.config['data']['columns_to_drop_low_importance']
        ]
        
        for file_path in file_paths:
            cols_to_drop.extend(ArtifactManager.load_lines(file_path))
        
        # Add post-event columns
        cols_to_drop.extend(self.config['data']['post_event_columns'])
        
        unique_cols = sorted(set(cols_to_drop))
        logger.info(f"Dropping {len(unique_cols)} columns")
        
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
        df = client.query(query).to_dataframe(create_bqstorage_client=True)
        df = df.convert_dtypes(dtype_backend="pyarrow")
        
        logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Convert target to int
        target_col = self.config['data']['target_column']
        df[target_col] = df[target_col].astype(int)
        
        return df


def prepare_data(df, target_col='Cancelled', date_col='FlightDate'):
    """Split data into train and test sets."""
    y = df[target_col]
    X = df.drop(columns=[target_col, date_col], errors='ignore')
    
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Clear memory
    del df, X
    gc.collect()
    
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
                'verbose': True,
                'one_hot_max_size': 10,
                'max_ctr_complexity': 1,
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
            model.set_params(is_unbalance=True)
        elif model_type == 'catboost':
            model.set_params(auto_class_weights='Balanced')
        return model


def build_pipeline(model_type, params=None, config=None, X_train=None):
    """Build preprocessing + model pipeline with manual or automatic column detection."""
    if X_train is None or config is None:
        raise ValueError("X_train and config are required")
    
    num_cols, onehot_cat_cols, not_onehot_cat_cols = ColumnTypeDetector.detect_types(X_train, config)
    
    logger.info(f"Building pipeline with {len(num_cols)} numeric columns")
    
    # Build transformers
    transformers = []
    
    # Numerical transformer
    if num_cols:
        num_steps = []
        if config['preprocessing']['impute_missing']:
            num_steps.append(('impute', SimpleImputer(strategy='constant', fill_value=0)))
        num_steps.append(('scale', StandardScaler()))
        
        if len(num_steps) == 1:
            transformers.append(('num', StandardScaler(), num_cols))
        else:
            transformers.append(('num', Pipeline(num_steps), num_cols))
    
    # One-hot categorical transformer
    if onehot_cat_cols:
        onehot_transformer = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        transformers.append(('onehot_cat', onehot_transformer, onehot_cat_cols))
    
    # Non-one-hot categorical transformer (for models that handle categorical natively)
    if not_onehot_cat_cols:
        # For models like CatBoost that handle categorical features natively
        # We'll just impute missing values but not one-hot encode
        not_onehot_transformer = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='unknown'))
        ])
        transformers.append(('not_onehot_cat', not_onehot_transformer, not_onehot_cat_cols))
    
    # Create model
    model = ModelFactory.create_model(model_type, params)
    
    # For CatBoost, specify categorical features
    if model_type == 'catboost' and not_onehot_cat_cols:
        # Get the indices of categorical features after preprocessing
        # This is a bit complex since we need to account for the column transformer
        cat_feature_indices = []
        current_idx = 0
        
        # Count numerical features
        if num_cols:
            current_idx += len(num_cols)
        
        # Count one-hot encoded features (this will expand)
        if onehot_cat_cols:
            # We can't know exact count without fitting, so we'll handle this in a callback
            # For now, we'll let CatBoost auto-detect
            pass
        
        # The not-one-hot categorical features come last
        if not_onehot_cat_cols:
            for i in range(len(not_onehot_cat_cols)):
                cat_feature_indices.append(current_idx + i)
            
            # Set categorical features for CatBoost
            if hasattr(model, 'set_params'):
                model.set_params(cat_features=cat_feature_indices)
    
    return Pipeline([
        ('preprocessor', ColumnTransformer(transformers)),
        ('model', model)
    ])


# === HYPERPARAMETER OPTIMIZATION ===
class HyperparameterOptimizer:
    """Handles hyperparameter optimization using Optuna."""
    
    def __init__(self, X_train, y_train, model_type, config):
        self.X_train = X_train
        self.y_train = y_train
        self.model_type = model_type
        self.config = config
        
        # Optimization settings
        self.beta = config['optimization'].get('beta', 1.0)
        self.cv = config['optimization'].get('cv_folds', 5)
        
        # Hyperparameter search space
        hp_config = config['hyperparameters'][model_type]
        self.search_space = hp_config.get('optimize', {})
        self.fixed_params = hp_config.get('fixed', {})
        
        # Calculate class ratio once
        counts = y_train.value_counts()
        self.class_ratio = counts.get(0, 0) / counts.get(1, 1)
    
    def _suggest_params(self, trial):
        """Suggest hyperparameters for the trial."""
        params = {}
        for name, spec in self.search_space.items():
            low, high = spec['min'], spec['max']
            p_type = spec.get('type', 'int' if isinstance(low, int) and isinstance(high, int) else 'float')
            
            if p_type == 'int':
                params[name] = trial.suggest_int(name, low, high)
            else:
                log = spec.get('log', False)
                params[name] = trial.suggest_float(name, low, high, log=log)
        
        return params
    
    def _objective(self, trial):
        """Objective function for Optuna optimization."""
        # Get hyperparameters
        trial_params = {**self._suggest_params(trial), **self.fixed_params}
        
        # Build pipeline
        pipeline = build_pipeline(
            self.model_type,
            params=trial_params,
            config=self.config,
            X_train=self.X_train
        )
        
        # Apply class balancing
        pipeline.named_steps['model'] = ModelFactory.apply_class_balancing(
            pipeline.named_steps['model'],
            self.model_type,
            self.class_ratio
        )
        
        # Run cross-validation
        return self._cross_validate(pipeline)
    
    def _cross_validate(self, pipeline):
        """Memory-efficient cross-validation."""
        n_samples = len(self.X_train)
        fold_size = n_samples // self.cv
        scores = []
        
        for i in range(self.cv):
            # Define fold indices
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.cv - 1 else n_samples
            
            # Create train/val split
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[start:end] = True
            
            X_fold_train = self.X_train[~val_mask]
            X_fold_val = self.X_train[val_mask]
            y_fold_train = self.y_train[~val_mask]
            y_fold_val = self.y_train[val_mask]
            
            # Train and evaluate
            pipeline.fit(X_fold_train, y_fold_train)
            y_proba = pipeline.predict_proba(X_fold_val)[:, 1]
            
            # Calculate F-beta score with optimal threshold
            score = self._calculate_fbeta_score(y_fold_val, y_proba)
            scores.append(score)
            
            # Clear memory
            del X_fold_train, X_fold_val, y_fold_train, y_fold_val
            gc.collect()
        
        return float(np.mean(scores))
    
    def _calculate_fbeta_score(self, y_true, y_proba):
        """Calculate F-beta score with optimal threshold."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        fbeta_scores = ((1 + self.beta**2) * precision * recall / 
                       (self.beta**2 * precision + recall + 1e-10))
        
        best_idx = np.nanargmax(fbeta_scores)
        threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        
        y_pred = (y_proba >= threshold).astype(int)
        return fbeta_score(y_true, y_pred, beta=self.beta)
    
    def optimize(self):
        """Run hyperparameter optimization."""
        # Setup study
        study_name = (f"{self.model_type}_{generate_run_name(self.model_type)}" 
                     if not self.config['optimization']['if_continue']
                     else self.config['optimization']['study_name'])
        
        storage = optuna.storages.RDBStorage(
            url="sqlite:///optuna_study.db",
            engine_kwargs={"connect_args": {"check_same_thread": False}}
        )
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            storage=storage,
            load_if_exists=self.config['optimization']['if_continue']
        )
        
        # Run optimization
        logger.info(f"Starting optimization for {self.model_type}...")
        study.optimize(self._objective, n_trials=self.config['optimization']['n_trials'])
        
        # Return best parameters
        best_params = {**study.best_params, **self.fixed_params}
        logger.info(f"Best F-beta score: {study.best_value:.4f}")
        
        return best_params, study.best_value


# === MODEL PIPELINE ===
class ModelPipeline:
    """Unified pipeline for optimization, training, and evaluation."""
    
    def __init__(self, model_type, config):
        self.model_type = model_type
        self.config = config
        self.beta = config['optimization'].get('beta', 1.0)
        self.run_name = generate_run_name(model_type)
        self.results_dir = os.path.join("train_results", self.run_name)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Check MLflow availability
        self.mlflow_available = self._check_mlflow()
    
    def _check_mlflow(self):
        """Check if MLflow is available."""
        try:
            mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
            return True
        except Exception:
            logger.warning("MLflow not available. Training without tracking.")
            return False
    
    def run(self, X_train, X_test, y_train, y_test):
        """Run complete pipeline: optimize, train, evaluate."""
        try:
            # Start MLflow run if available
            if self.mlflow_available:
                mlflow_run = mlflow.start_run(run_name=self.run_name)
            else:
                mlflow_run = nullcontext()
            
            with mlflow_run:
                # 1. Optimize hyperparameters
                optimizer = HyperparameterOptimizer(X_train, y_train, self.model_type, self.config)
                opt_start = time.time()
                best_params, best_score = optimizer.optimize()
                opt_duration = time.time() - opt_start
                
                # 2. Train final model
                logger.info(f"Training {self.model_type} with best parameters...")
                pipeline = build_pipeline(
                    self.model_type,
                    params=best_params,
                    config=self.config,
                    X_train=X_train
                )
                
                # Apply class balancing
                counts = y_train.value_counts()
                class_ratio = counts.get(0, 0) / counts.get(1, 1)
                pipeline.named_steps['model'] = ModelFactory.apply_class_balancing(
                    pipeline.named_steps['model'],
                    self.model_type,
                    class_ratio
                )
                
                # Train
                train_start = time.time()
                pipeline.fit(X_train, y_train)
                train_duration = time.time() - train_start
                
                # 3. Evaluate
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics = self._evaluate_model(y_test, y_proba)
                
                logger.info(
                    f"{self.model_type} Test Metrics: "
                    f"F1={metrics['f1']:.4f}, "
                    f"F{self.beta}={metrics['fbeta']:.4f}, "
                    f"ROC_AUC={metrics['roc_auc']:.4f}"
                )
                
                # 4. Save artifacts
                self._save_artifacts(
                    pipeline, best_params, metrics, 
                    opt_duration, train_duration
                )
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error in {self.model_type} pipeline: {e}")
            traceback.print_exc()
            return None
    
    def _evaluate_model(self, y_true, y_scores):
        """Evaluate model with F-beta threshold optimization."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Compute F-beta scores
        fbeta_scores = ((1 + self.beta**2) * precision * recall / 
                       (self.beta**2 * precision + recall + 1e-10))
        
        # Find best threshold
        best_idx = np.nanargmax(fbeta_scores)
        threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        
        # Generate predictions and metrics
        y_pred = (y_scores >= threshold).astype(int)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': fbeta_score(y_true, y_pred, beta=1.0),
            'fbeta': float(fbeta_scores[best_idx]),
            'roc_auc': roc_auc_score(y_true, y_scores),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _save_artifacts(self, pipeline, best_params, metrics, opt_duration, train_duration):
        """Save all artifacts and log to MLflow."""
        # Save parameters
        ArtifactManager.save_json(best_params, os.path.join(self.results_dir, "best_params.json"))
        
        # Save config
        ArtifactManager.save_yaml(self.config, os.path.join(self.results_dir, "config.yaml"))
        
        # Save metrics with durations
        all_metrics = {
            **metrics,
            'optimization_duration': opt_duration,
            'training_duration': train_duration,
            'total_duration': opt_duration + train_duration
        }
        ArtifactManager.save_json(all_metrics, os.path.join(self.results_dir, "metrics.json"))
        
        # Save confusion matrix separately
        cm = metrics['confusion_matrix']
        cm_text = f"TN:{cm[0][0]}, FP:{cm[0][1]}\nFN:{cm[1][0]}, TP:{cm[1][1]}"
        ArtifactManager.save_text(cm_text, os.path.join(self.results_dir, "confusion_matrix.txt"))
        
        # Plot feature importance
        self._plot_feature_importance(pipeline)
        
        # Log to MLflow if available
        if self.mlflow_available:
            mlflow.log_artifacts(self.results_dir)
            mlflow.log_params({
                'model': self.model_type,
                'run_name': self.run_name,
                **best_params
            })
            mlflow.log_metrics({
                k: v for k, v in all_metrics.items() 
                if isinstance(v, (int, float))
            })
            mlflow.sklearn.log_model(pipeline, self.run_name)
            logger.info(f"Logged run '{self.run_name}' to MLflow")
    
    def _plot_feature_importance(self, pipeline):
        """Plot and save feature importance."""
        try:
            model = pipeline.named_steps['model']
            preprocessor = pipeline.named_steps['preprocessor']
            
            # Get feature names
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except:
                feature_names = None
            
            # Get importances based on model type
            if self.model_type == 'xgboost':
                booster = model.get_booster()
                importance_dict = booster.get_score(importance_type='weight')
                if not importance_dict:
                    return
                
                # Map to feature names if available
                importances = []
                names = []
                for feat, imp in importance_dict.items():
                    if feature_names is not None and feat.startswith('f') and feat[1:].isdigit():
                        idx = int(feat[1:])
                        if idx < len(feature_names):
                            names.append(feature_names[idx])
                        else:
                            names.append(feat)
                    else:
                        names.append(feat)
                    importances.append(imp)
                    
            elif self.model_type == 'lightgbm':
                importances = model.booster_.feature_importance(importance_type='gain')
                if feature_names is not None and len(feature_names) == len(importances):
                    names = feature_names
                else:
                    names = model.booster_.feature_name()
                
            elif self.model_type == 'catboost':
                importances = model.get_feature_importance()
                if feature_names is not None and len(feature_names) == len(importances):
                    names = feature_names
                else:
                    names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create dataframe and plot top 100
            df_imp = pd.DataFrame({'Feature': names, 'Importance': importances})
            df_imp = df_imp.sort_values('Importance', ascending=True).tail(100)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(df_imp)), df_imp['Importance'])
            plt.yticks(range(len(df_imp)), df_imp['Feature'])
            plt.xlabel('Importance')
            plt.title(f'{self.model_type.capitalize()} - Top 100 Feature Importance')
            plt.tight_layout()
            
            output_path = os.path.join(self.results_dir, 'feature_importance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save CSV too
            df_imp.to_csv(os.path.join(self.results_dir, 'feature_importance.csv'), index=False)
            
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")


# === MAIN EXECUTION ===
def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train flight cancellation models")
    parser.add_argument("-c", "--config", default="config_dry_run.yaml", help="Config file path")
    parser.add_argument(
    "--quiet", "-q", action="store_true",
    help="Suppress info and warning logs (only show errors)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    # Load config and setup
    config = load_config(args.config)
    os.makedirs("train_results", exist_ok=True)
    setup_mlflow(config['mlflow']['tracking_uri'], config['mlflow']['experiment_name'])
    
    # Load and prepare data
    logger.info("Loading data from BigQuery...")
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        config['data']['target_column'],
        config['data']['date_column']
    )
    
    # Print data info
    logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    num_cols, onehot_cat_cols, not_onehot_cat_cols = ColumnTypeDetector.detect_types(X_train, config)
    logger.info(f"Building pipeline with {len(num_cols)} numeric columns, {len(onehot_cat_cols)} one-hot categorical columns, {len(not_onehot_cat_cols)} non-one-hot categorical columns")    
    # Train all models
    results = {}
    for model_type in config['models']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()}")
        logger.info(f"{'='*60}")
        
        pipeline = ModelPipeline(model_type, config)
        metrics = pipeline.run(X_train, X_test, y_train, y_test)
        
        if metrics:
            results[model_type] = metrics
    
    # Report best model
    if results:
        beta = config['optimization'].get('beta', 1.0)
        best_model = max(results, key=lambda m: results[m]['fbeta'])
        logger.info(f"\n{'='*60}")
        logger.info(f"Best model: {best_model} (F{beta}={results[best_model]['fbeta']:.4f})")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
