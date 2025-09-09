import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score, fbeta_score, precision_recall_curve, auc, classification_report
from google.cloud import bigquery
import optuna
import os
import mlflow
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier
import IPython
import json

import yaml
from pathlib import Path


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder  # or OrdinalEncoder for XGBoost
import joblib
from datetime import datetime, timezone
from sklearn.preprocessing import OrdinalEncoder

import logging, sys, time, warnings, traceback, faulthandler, signal
try:
    import psutil
except Exception:
    psutil = None

# Load config at the beginning
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


today_results = os.path.join(config['output_path'], datetime.today().strftime('%Y-%m-%d'))
Path(today_results).mkdir(exist_ok=True, parents=True)

# ---------- Logging setup ----------
def setup_logging(log_dir: str, level: str = None):
    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, level, logging.INFO)
    log_path = os.path.join(log_dir, "run.log")

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clean existing handlers (useful on re-runs)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = "[%(asctime)s] [%(levelname)s] [pid:%(process)d] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    # Console handler (pretty if rich is available)
    try:
        from rich.logging import RichHandler
        ch = RichHandler(rich_tracebacks=True, markup=False, show_time=False, show_level=False)
        ch.setLevel(numeric_level)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
    except Exception:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(ch)

    # Warnings to logs
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Enable faulthandler to dump tracebacks on fatal signals (not SIGKILL)
    faulthandler.enable(all_threads=True)

    logging.info("Logging initialized. Level=%s, file=%s", level, log_path)
    return logger

def log_context():
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Project: %s | Dataset: %s | Table: %s", config['project']['name'], config['project']['dataset'], config['project']['table'])
    logger.info("GCP_PROJECT_ID=%s", os.environ.get('GCP_PROJECT_ID'))
    try:
        tracking_uri = mlflow.get_tracking_uri()
        logger.info("MLflow tracking URI: %s", tracking_uri)
    except Exception as e:
        logger.warning("MLflow tracking URI not available: %s", e)
    if psutil:
        p = psutil.Process()
        vm = psutil.virtual_memory()
        logger.info("CPU cores: %s | RAM total: %.1f GB | RAM avail: %.1f GB", psutil.cpu_count(), vm.total/1e9, vm.available/1e9)
        logger.info("Proc RSS: %.1f GB", p.memory_info().rss/1e9)
    else:
        logger.info("psutil not installed; memory stats not available.")

logger = setup_logging(today_results, level=config.get("logging", {}).get("level"))
log_context()

class StageTimer:
    def __init__(self, name): 
        self.name = name
        self.t0 = time.time()
        if psutil:
            rss = psutil.Process().memory_info().rss/1e9
            logger.info("▶ Start: %s | RSS=%.2f GB", self.name, rss)
        else:
            logger.info("▶ Start: %s", self.name)
    def done(self):
        dt = time.time() - self.t0
        if psutil:
            rss = psutil.Process().memory_info().rss/1e9
            logger.info("✅ Done: %s in %.2fs | RSS=%.2f GB", self.name, dt, rss)
        else:
            logger.info("✅ Done: %s in %.2fs", self.name, dt)

def log_df(df: pd.DataFrame, name: str, head_n: int = 0):
    logger.info("%s shape=%s dtypes=%s", name, df.shape, dict(df.dtypes.astype(str)))
    nulls = df.isna().mean().sort_values(ascending=False).head(10)
    logger.info("%s top-10 null ratios: %s", name, nulls.to_dict())
    if head_n:
        logger.info("%s head:\n%s", name, df.head(head_n).to_string())



# from sklearn import set_config
# set_config(transform_output='pandas')

SAMPLE_SIZE = config['project']['sample_size']
project_id = os.environ.get('GCP_PROJECT_ID')
table_name = config['project']['table']
dataset_name = config['project']['dataset']
table_id = f'{project_id}.{dataset_name}.{table_name}'
describe_df = pd.read_csv(config['data']['describe_csv_path'])


scaler = StandardScaler()


def get_col_roles():
    dct_cols = {}
    for key in ['num', 'cat', 'hot', 'tgt', 'dat']:
        dct_cols[key] = describe_df.loc[describe_df['Role'] == key, 'Column_Name'].sort_values().tolist()
    return dct_cols

def _bq_dry_run(client: bigquery.Client, query: str):
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(query, job_config=job_config)
    bytes_proc = job.total_bytes_processed
    logger.info("BigQuery dry-run: will process ~%.2f MB", bytes_proc / 1e6)

def data_load_from_bigquery(dct_cols, sample_size=None):
    all_columns = [col for cols in dct_cols.values() for col in cols]
    client = bigquery.Client(project=project_id)

    if sample_size:
        pos = sample_size // 10
        neg = sample_size - pos
        cols = ", ".join(all_columns)

        query = f"""
        WITH balanced_sample AS (
            SELECT {cols}
            FROM `{table_id}`
            WHERE Cancelled IS TRUE
            ORDER BY RAND()
            LIMIT {pos}
        ),
        normal_sample AS (
            SELECT {cols}
            FROM `{table_id}`
            WHERE Cancelled IS FALSE
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


    logger.info("BQ query:\n%s", query.strip())
    try:
        _bq_dry_run(client, query)
    except Exception as e:
        logger.warning("Dry-run failed (continuing): %s", e)

    st = StageTimer("BigQuery to_dataframe")
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    st.done()

    log_df(df, "Loaded DF", head_n=0)
    return df

def get_skiped_cols(describe_df):
    skip_cols = describe_df.loc[
        (describe_df['Skip_reason_phik'] != 'unk') |
        (describe_df['Data_leakage'] != 'unk'),
        'Column_Name'
    ].tolist()
    skip_cols = list(set(skip_cols + config['columns']['skip_additional']))
    return skip_cols

def drop_skip_cols(skip_cols, dct_cols):
    filtered_dct_cols = {}
    for key, cols in dct_cols.items():
        filtered_dct_cols[key] = [col for col in cols if col not in skip_cols]
    return filtered_dct_cols

def ffit_all_models():
    target_col = 'Cancelled'
    dct_cols = get_col_roles()
    skip_cols = get_skiped_cols(describe_df)
    dct_cols = drop_skip_cols(skip_cols, dct_cols)
    logger.info("Column roles: %s", {k: len(v) for k,v in dct_cols.items()})
    logger.info("Skip cols (%d): %s", len(skip_cols), skip_cols[:50] + (["..."] if len(skip_cols) > 50 else []))
    df = data_load_from_bigquery(dct_cols, sample_size=SAMPLE_SIZE)  
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    inverted_roles = {
        column: role
        for role, columns in dct_cols.items()
        for column in columns
    }    
    with open(config['data']['roles_output_path'], "w") as f:
        json.dump(inverted_roles, f, indent=4)

    numeric_features = dct_cols['num']
    categorical_features = dct_cols['cat'] + dct_cols['hot']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    log_df(X_train, "X_train")
    logger.info("y_train distribution: %s", y_train.value_counts(normalize=True).to_dict())


    def run_study(objective_fn, model_name):
        logger.info("==== Optuna study: %s ====", model_name)
        study_name = config['optuna']['study_name'] + f"_{model_name}"
        storage_path = f"sqlite:///{config['optuna']['storage_path']}"
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def _cb(study, trial):
            logger.info("Trial %d finished: value=%.5f, params=%s",
                        trial.number, trial.value, trial.params)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_path,
            load_if_exists=True
        )
        st = StageTimer(f"Optuna optimize [{model_name}]")
        study.optimize(lambda t: objective_fn(t, X_train, y_train, categorical_features, numeric_features),
                    n_trials=config['optuna']['n_trials'],
                    callbacks=[_cb])
        st.done()
        logger.info("Best [%s]: value=%.5f params=%s", model_name, study.best_value, study.best_params)
        return study

    # Индивидуальные objective-функции
    study_lgb = run_study(objective_lightgbm, 'lightgbm')
    study_xgb = run_study(objective_xgboost, 'xgboost')
    study_cat = run_study(objective_catboost, 'catboost')

    study_dict = {"lightgbm": study_lgb, "xgboost": study_xgb, "catboost": study_cat}
    # study_dict = {"catboost": study_cat}
    return study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features


def objective_lightgbm(trial, X_train, y_train, categorical_features, numeric_features):
    k_threshold = trial.suggest_int("k_unique_threshold", 0, 8)
    hot_cols, cat_cols = split_categorical_by_uniqueness(X_train, categorical_features, k_threshold)
    logger.debug("k_unique_threshold=%s -> hot=%d, cat=%d", k_threshold, len(hot_cols), len(cat_cols))    
    preprocessor = preprocessor_lightgbm_make(numeric_features, hot_cols, cat_cols)

    # class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    counts = y_train.value_counts()
    n_pos = int(counts.get(True, 0))
    n_neg = int(counts.get(False, 0))
    spw = n_neg / max(n_pos, 1)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
        
        # КРИТИЧНО: ОБРАБОТКА ДИСБАЛАНСА
        'scale_pos_weight': spw,
        'is_unbalance': False,
        "class_weight": None,
        
        'verbosity': -1,
        'seed': 42
    }
    
    model = lgb.LGBMClassifier(**params)
    pipeline = build_pipeline(model, preprocessor)
    
    # СТРАТИФИЦИРОВАННАЯ КРОСС-ВАЛИДАЦИЯ
    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    logger.info("[%s] CV=%d folds", model.__class__.__name__, cv.get_n_splits())
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_t = StageTimer(f"{model.__class__.__name__} fold")        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
        fold_t.done()
        if psutil:
            logger.info("Post-fold RSS=%.2f GB", psutil.Process().memory_info().rss/1e9)         
    return np.mean(scores)

def objective_xgboost(trial, X_train, y_train, categorical_features, numeric_features):
    k_threshold = trial.suggest_int("k_unique_threshold", 0, 8)
    hot_cols, cat_cols = split_categorical_by_uniqueness(X_train, categorical_features, k_threshold)
    logger.debug("k_unique_threshold=%s -> hot=%d, cat=%d", k_threshold, len(hot_cols), len(cat_cols))    
    preprocessor = preprocessor_xgboost_make(numeric_features, hot_cols, cat_cols)
    
    # ОБРАБОТКА ДИСБАЛАНСА
    counts = y_train.value_counts()
    class_ratio = counts.get(False, 0) / max(counts.get(True, 0), 1)    
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-5, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
        'scale_pos_weight': class_ratio,
        'use_label_encoder': False,
        'verbosity': 0,
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    pipeline = build_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    logger.info("[%s] CV=%d folds", model.__class__.__name__, cv.get_n_splits())
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_t = StageTimer(f"{model.__class__.__name__} fold")        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
        fold_t.done()
        if psutil:
            logger.info("Post-fold RSS=%.2f GB", psutil.Process().memory_info().rss/1e9)         
    return np.mean(scores)

def objective_catboost(trial, X_train, y_train, categorical_features, numeric_features):
    preprocessor = preprocessor_catboost_make(numeric_features, categorical_features)    
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'loss_function': 'Logloss',
        'verbose': False,
        'random_seed': 42
    }

    model = CatBoostClassifier(**params)
    pipeline = build_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    logger.info("[%s] CV=%d folds", model.__class__.__name__, cv.get_n_splits())
    for train_idx, val_idx in cv.split(X_train, y_train):
        fold_t = StageTimer(f"{model.__class__.__name__} fold")        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
        fold_t.done()
        if psutil:
            logger.info("Post-fold RSS=%.2f GB", psutil.Process().memory_info().rss/1e9)        
    return np.mean(scores)

def build_final_model(model_name, study):
    best_params = study.best_params
    if model_name == "lightgbm":
        final_model = lgb.LGBMClassifier(**best_params, objective='binary', random_state=42)
    elif model_name == "xgboost":
        final_model = xgb.XGBClassifier(
                        **best_params,
                        tree_method="hist",
                        enable_categorical=True,
                        random_state=42
                    )
    elif model_name == "catboost":
        final_model = CatBoostClassifier(
            **best_params,
            loss_function="Logloss",
            random_seed=42,
            verbose=False
        )
    return final_model

def train_and_log_pipeline(model_name, final_pipeline, X_train, X_test, y_train, y_test):
    final_pipeline.fit(X_train, y_train)
    logger.info("Fitted final pipeline: %s", model_name)
    model = final_pipeline.named_steps["classifier"]
    logger.info("Final model params: %s", model.get_params())


    with mlflow.start_run() as run:
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()

        # ✅ Extract and log all hyperparameters from trained model
        model = final_pipeline.named_steps["classifier"]
        full_params = model.get_params()
        mlflow.log_params(full_params)

        # Predict and compute metrics
        y_pred = log_model_metrics(final_pipeline, X_test, y_test, model_name, artifact_uri, run_id)

        log_final_pipeline(final_pipeline, model_name)

        if hasattr(model, "feature_importances_"):
            log_feature_importances(final_pipeline, [], today_results)

    print(classification_report(y_test, y_pred))

def final_fit_track(study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features):
    final_pipeline_dict = {}
    full_params_dict = {}

    for model_name, study in study_dict.items():
        if study is None:
            continue

        best_params = study.best_params
        k_threshold = best_params.get("k_unique_threshold", 5)  # fallback на 5, если параметра нет
        hot_cols, cat_cols = split_categorical_by_uniqueness(X_train, categorical_features, k_threshold)

        # Построить нужный препроцессор
        if model_name == "lightgbm":
            k_threshold = best_params.get("k_unique_threshold", 5)
            hot_cols, cat_cols = split_categorical_by_uniqueness(X_train, categorical_features, k_threshold)
            preprocessor = preprocessor_lightgbm_make(numeric_features, hot_cols, cat_cols)
        elif model_name == "xgboost":
            k_threshold = best_params.get("k_unique_threshold", 5)
            hot_cols, cat_cols = split_categorical_by_uniqueness(X_train, categorical_features, k_threshold)
            preprocessor = preprocessor_xgboost_make(numeric_features, hot_cols, cat_cols)
        elif model_name == "catboost":
            preprocessor = preprocessor_catboost_make(numeric_features, categorical_features)
        else:
            continue

        # Построить модель и пайплайн
        final_model = build_final_model(model_name, study)
        final_pipeline = build_pipeline(final_model, preprocessor)

        final_pipeline_dict[model_name] = final_pipeline

    # Обучение и логгирование
    for model_name, pipeline in final_pipeline_dict.items():
        if pipeline:
            train_and_log_pipeline(model_name, pipeline, X_train, X_test, y_train, y_test)

def make_preprocessor(categorical_features, numeric_features):
    # Ensure categoricals are string to avoid mixed types
    def to_string(df):
        out = df.copy()
        for c in categorical_features:
            out[c] = out[c].astype("string")
        return out

    cat_cast = FunctionTransformer(to_string)

    numeric_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tr = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tr, numeric_features),
            ("cat", categorical_tr, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=None,
    )

    # Wrap with a small pipeline so the cast happens first
    return Pipeline([("cast", cat_cast), ("ct", pre)])
    
def preprocessor_lightgbm_make(numeric_features, hot_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    hot_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    cat_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('hot', hot_transformer, hot_features),
        ('cat', cat_transformer, cat_features)
    ])

def preprocessor_xgboost_make(numeric_features, hot_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    # One-hot bucket (explicit low-cardinality)
    hot_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(lambda X: X.astype('string'), validate=False)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # Native categorical for XGBoost: use OrdinalEncoder to convert to numeric codes
    cat_transformer = Pipeline(steps=[
        ('to_string', FunctionTransformer(lambda X: X.astype('string'), validate=False)),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('hot', hot_transformer, hot_features),
        ('cat', cat_transformer, cat_features),
    ], remainder='drop', sparse_threshold=0.0)

def preprocessor_catboost_make(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def build_pipeline(model, preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ], verbose=False)

def train_pipeline(pipeline, X_tr, y_tr, categorical_features, model_name):
    if model_name == 'catboost':
        X_tr[categorical_features] = X_tr[categorical_features].astype(str)
    pipeline.fit(X_tr, y_tr)
    return pipeline

def evaluate_pipeline(pipeline, X_val, y_val):
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    
    # НАЙТИ ОПТИМАЛЬНЫЙ ПОРОГ ДЛЯ F2
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f2_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if len(np.unique(y_pred)) > 1:  # Избежать деления на ноль
            f2 = fbeta_score(y_val, y_pred, beta=2.0)
            f2_scores.append(f2)
        else:
            f2_scores.append(0)
    
    return max(f2_scores) if f2_scores else 0

def extract_feature_names(preprocessor, numeric_features, categorical_features):
    """
    Extract feature names from the preprocessor after transformation.
    Handles OneHotEncoder and OrdinalEncoder properly.
    """
    try:
        # Try to get feature names directly from preprocessor
        return preprocessor.get_feature_names_out()
    except AttributeError:
        # If get_feature_names_out() is not available, construct manually
        output_features = []
        
        # Get the transformers from ColumnTransformer
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                # Numeric features keep their names after scaling
                output_features.extend(columns)
            elif name == 'hot':
                # OneHot encoded features - get expanded feature names
                if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                    encoder = transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        # Use the encoder's feature names
                        hot_feature_names = encoder.get_feature_names_out(columns)
                        output_features.extend(hot_feature_names)
                    else:
                        # Fallback: construct feature names manually
                        for col in columns:
                            unique_vals = encoder.categories_[columns.index(col)]
                            for val in unique_vals:
                                output_features.append(f"{col}_{val}")
            elif name == 'cat':
                # Ordinal encoded features keep their original names
                if hasattr(transformer, 'named_steps') and 'ordinal' in transformer.named_steps:
                    output_features.extend(columns)
                elif hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                    # If it's OneHotEncoder in cat transformer
                    encoder = transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_feature_names = encoder.get_feature_names_out(columns)
                        output_features.extend(cat_feature_names)
        
        return np.array(output_features)

def log_feature_importances(pipeline, feature_names, output_path):
    """
    Log feature importances with proper feature name extraction.
    """
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Get the actual feature names from the preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    actual_feature_names = extract_feature_names(preprocessor, [], [])
    
    if len(importances) != len(actual_feature_names):
        print(f"Warning: {len(importances)} importances vs {len(actual_feature_names)} features")
        # Use indices as fallback
        actual_feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    feat_imp_df = pd.DataFrame({
        'feature': actual_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    csv_path = os.path.join(output_path, "feature_importances.csv")
    feat_imp_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    plt.figure(figsize=(12, 6))
    top_30 = feat_imp_df.head(30)
    plt.barh(top_30['feature'][::-1], top_30['importance'][::-1])
    plt.xlabel("Importance")
    plt.title("Top 30 Feature Importances")
    plt.tight_layout()
    plot_path = os.path.join(output_path, "feature_importance_plot.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

def log_model_metrics(pipeline, X_test, y_test, model_name, artifact_uri, run_id):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f2 = fbeta_score(y_test, y_pred, beta=2.0)
    recall = recall_score(y_test, y_pred)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(recall_vals, precision_vals)

    mlflow.log_metric("f2_score", f2)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("pr_auc", pr_auc_val)

    # Push metrics to BigQuery using actual MLflow artifact URI
    pipeline_uri = f"{artifact_uri}/{model_name}_pipeline.pkl"

    log_metric_to_bigquery(model_name, "f2_score", f2, pipeline_uri)
    log_metric_to_bigquery(model_name, "recall", recall, pipeline_uri)
    log_metric_to_bigquery(model_name, "pr_auc", pr_auc_val, pipeline_uri)

    return y_pred


def log_final_pipeline(pipeline, model_name):
    model_path = os.path.join(today_results, model_name + "_pipeline")
    
    # Save locally
    joblib.dump(pipeline, model_path + ".pkl")
    
    # Log with MLflow
    mlflow.log_artifact(model_path + ".pkl")
    logger.info("Saved pipeline -> %s.pkl and logged to MLflow", model_path)    
    
    # Optional: Also log as MLflow model for direct serving later
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=model_name
    )

def log_metric_to_bigquery(model_name, metric_name, metric_value, pipeline_uri):
    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{config['bq_logging']['dataset']}.{config['bq_logging']['target_table']}"

    project_name = config['project']['name']

    timestamp = datetime.now(timezone.utc).isoformat()
    row = {
        "timestamp": timestamp,
        "project_name": project_name,
        "model_name": model_name,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "pipeline_uri": pipeline_uri  # <- now from actual MLflow URI
    }

    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"Error inserting row to BigQuery: {errors}")

def split_categorical_by_uniqueness(df, categorical_cols, k_threshold):
    hot_cols, cat_cols = [], []
    for col in categorical_cols:
        nunique = df[col].nunique()
        if nunique <= k_threshold:
            hot_cols.append(col)
        else:
            cat_cols.append(col)
    return hot_cols, cat_cols

def analyze_class_distribution(y_train, y_test):
    print("=== АНАЛИЗ ДИСБАЛАНСА КЛАССОВ ===")
    print(f"Train distribution:")
    print(y_train.value_counts(normalize=True))
    print(f"Test distribution:")
    print(y_test.value_counts(normalize=True))
    
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Recommended scale_pos_weight: {pos_weight:.2f}")
    return pos_weight

if __name__ == "__main__":
    try:
        logger.info("=== RUN START ===")
        study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features = ffit_all_models()
        final_fit_track(study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features)
        logger.info("=== RUN END (OK) ===")
    except MemoryError:
        logger.critical("MemoryError: process likely OOM-killed soon. Consider reducing sample size, folds, or threads.", exc_info=True)
        sys.exit(137)  # common OOM exit
    except SystemExit:
        raise
    except Exception as e:
        logger.critical("Fatal exception: %s", e, exc_info=True)
        # Extra dump of all threads
        try:
            faulthandler.dump_traceback_all()
        except Exception:
            pass
        sys.exit(1)
