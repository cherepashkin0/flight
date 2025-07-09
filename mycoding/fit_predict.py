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


# Load config at the beginning
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

today_results = os.path.join(config['output_path'], datetime.today().strftime('%Y-%m-%d'))
Path(today_results).mkdir(exist_ok=True, parents=True)

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

def data_load_from_bigquery(dct_cols):
    all_columns = [col for cols in dct_cols.values() for col in cols]    
    client = bigquery.Client(project=project_id)
    query = f"""
            SELECT {", ".join(all_columns)}
            FROM `{table_id}`
            LIMIT {SAMPLE_SIZE}
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
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
    df = data_load_from_bigquery(dct_cols)
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

    def run_study(objective_fn, preprocessor_fn, model_name):
        print(f"Running study for: {model_name}")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_fn(
            trial, X_train, y_train, categorical_features, numeric_features, preprocessor_fn
        ), n_trials=config['optuna']['n_trials'])
        return study

    preprocessors_dict = {"lightgbm": preprocessor_lightgbm_make(numeric_features, categorical_features), 
                          "xgboost": preprocessor_xgboost_make(numeric_features, categorical_features),
                          "catboost": preprocessor_catboost_make(numeric_features, categorical_features)}
    # Индивидуальные objective-функции
    study_lgb = run_study(objective_lightgbm, preprocessors_dict["lightgbm"], 'lightgbm')
    study_xgb = run_study(objective_xgboost, preprocessors_dict["xgboost"], 'xgboost')
    study_cat = run_study(objective_catboost, preprocessors_dict["catboost"], 'catboost')

    study_dict = {"lightgbm": study_lgb, "xgboost": study_xgb, "catboost": study_cat}
    # study_dict = {"catboost": study_cat}
    return study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features, preprocessors_dict


def objective_lightgbm(trial, X_train, y_train, categorical_features, numeric_features, preprocessor):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 32, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
        'verbosity': -1,
        'seed': 42
    }
    model = lgb.LGBMClassifier(**params)
    pipeline = build_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
    return np.mean(scores)

def objective_xgboost(trial, X_train, y_train, categorical_features, numeric_features, preprocessor):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'eta': trial.suggest_float('eta', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-5, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True),
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
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
    return np.mean(scores)

def objective_catboost(trial, X_train, y_train, categorical_features, numeric_features, preprocessor):
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'loss_function': 'Logloss',
        'verbose': False,
        'random_seed': 42,
        'cat_features': categorical_features  # directly use column names
    }

    model = CatBoostClassifier(**params)

    pipeline = build_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
    return np.mean(scores)


def objective_catboost(trial, X_train, y_train, categorical_features, numeric_features, preprocessor):
    """Simplified CatBoost objective that doesn't use cat_features parameter"""
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'loss_function': 'Logloss',
        'verbose': False,
        'random_seed': 42,
        # Don't specify cat_features - let it work with one-hot encoded features
    }

    model = CatBoostClassifier(**params)
    pipeline = build_pipeline(model, preprocessor)

    cv = StratifiedKFold(n_splits=config['cross_validation']['nfolds'], shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        scores.append(evaluate_pipeline(pipeline, X_val, y_val))
    return np.mean(scores)



def build_final_model(model_name, study):
    # for model_name, study in study_dict.items():
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

def train_and_log_pipeline(model_name, final_pipeline, X_train, X_test, y_train, y_test, full_params):
    final_pipeline.fit(X_train, y_train)
    # full_params = full_params_dict[model_name]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri()
        # Log best parameters
        mlflow.log_params(full_params)

        # Predict and compute metrics
        y_pred = log_model_metrics(final_pipeline, X_test, y_test, model_name, artifact_uri, run_id)
        # IPython.embed()
        log_final_pipeline(final_pipeline, model_name)


        if hasattr(final_pipeline.named_steps['classifier'], "feature_importances_"):
            preprocessor = final_pipeline.named_steps['preprocessor']
            feature_names = extract_feature_names(preprocessor, numeric_features, categorical_features)

            log_feature_importances(final_pipeline, feature_names, today_results)

    # Optionally print classification report
    print(classification_report(y_test, y_pred))

def final_fit_track(study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features, preprocessors_dict):
    final_pipeline_dict = {}
    full_params_dict = {}

    for model_name, study in study_dict.items():
        if study is None:
            continue
        best_params = study.best_params
        final_model = build_final_model(model_name, study)
        final_pipeline = build_pipeline(final_model, preprocessors_dict[model_name])
        
        final_pipeline_dict[model_name] = final_pipeline
        full_params_dict[model_name] = get_full_params(model_name, best_params)

    for model_name, pipeline in final_pipeline_dict.items():
        if pipeline:
            train_and_log_pipeline(model_name, pipeline, X_train, X_test, y_train, y_test, full_params_dict[model_name])
     

def get_full_params(model_name, best_params):
    common_params = {'random_state': 42}
    if model_name == 'lightgbm':
        return {**best_params, **common_params, 'objective': 'binary'}
    elif model_name == 'xgboost':
        return {**best_params, **common_params, 'tree_method': 'hist', 'enable_categorical': True}
    elif model_name == 'catboost':
        return {**best_params, 'loss_function': 'Logloss', 'random_seed': 42, 'verbose': False}
    return best_params

def preprocessor_lightgbm_make(numeric_features, categorical_features):
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

def preprocessor_xgboost_make(numeric_features, categorical_features):
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
    y_pred = pipeline.predict(X_val)
    return fbeta_score(y_val, y_pred, beta=2.0)

def extract_feature_names(preprocessor, numeric_features, categorical_features):
    num_feats = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
    cat_feats = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
    return np.concatenate([num_feats, cat_feats])

def log_feature_importances(pipeline, feature_names, output_path):
    importances = pipeline.named_steps['classifier'].feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(f"Mismatch: {len(importances)} importances vs {len(feature_names)} features")

    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    csv_path = os.path.join(output_path, "feature_importances.csv")
    feat_imp_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    plt.figure(figsize=(12, 6))
    plt.barh(feat_imp_df['feature'][:30][::-1], feat_imp_df['importance'][:30][::-1])
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
    
    # Optional: Also log as MLflow model for direct serving later
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=model_name + "_mlflow_model"
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


if __name__ == "__main__":
    study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features, preprocessors_dict = ffit_all_models()
    final_fit_track(study_dict, X_train, X_test, y_train, y_test, categorical_features, numeric_features, preprocessors_dict)
