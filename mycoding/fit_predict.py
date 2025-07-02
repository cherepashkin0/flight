import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, fbeta_score, precision_recall_curve, auc, classification_report
from google.cloud import bigquery
import optuna
import os
import mlflow
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
import IPython
import json
from datetime import datetime

import yaml
from pathlib import Path


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder  # or OrdinalEncoder for XGBoost

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

# class CategoryImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, fill_value="missing"):
#         self.fill_value = fill_value

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X_filled = pd.DataFrame(X).copy()
#         for col in X_filled.columns:
#             if not pd.api.types.is_categorical_dtype(X_filled[col]):
#                 X_filled[col] = X_filled[col].astype("category")

#             # Add 'missing' to the categories if not present
#             if self.fill_value not in X_filled[col].cat.categories:
#                 X_filled[col] = X_filled[col].cat.add_categories([self.fill_value])

#             # Fill NaNs with 'missing'
#             X_filled[col] = X_filled[col].fillna(self.fill_value)

#         return X_filled

def remove_prefix_param(best_params, prefix):
    parameters = {
            key.replace(prefix+'_', ''): value
            for key, value in best_params.items()
            if key.startswith(prefix+'_')
                }
    return parameters

def ffit():
    mlflow.set_experiment(config['mlflow']['experiment_name'])
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


    X[categorical_features] = X[categorical_features].astype('category')

    # Preprocessing for numeric features
    # numeric_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
    #     ('scaler', StandardScaler())
    # ])

    # Preprocessing for categorical features
    # categorical_transformer = Pipeline(steps=[
    #     ('cat_imputer', CategoryImputer())
    # ])
    # categorical_transformer = 'passthrough'

    # print('numeric: \n', numeric_features, 'categorical: \n', categorical_features, 'one_hotable: \n', one_hotable_features)
    # Combine preprocessing
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numeric_transformer, numeric_features),
    #         ('cat', categorical_transformer, categorical_features)
    #     ]
    #     )

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    def objective(trial):
        # classifier_name = trial.suggest_categorical('classifier', ['lightgbm', 'xgboost', 'catboost'])
        classifier_name = trial.suggest_categorical('classifier', ['xgboost'])

        if classifier_name == 'lightgbm':
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('lightgbm_num_leaves', 2, 32, log=True),
                'max_depth': trial.suggest_int('lightgbm_max_depth', 3, 12),
                'learning_rate': trial.suggest_float('lightgbm_learning_rate', 1e-4, 1e-1, log=True),
                'feature_fraction': trial.suggest_float('lightgbm_feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('lightgbm_bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('lightgbm_bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lightgbm_lambda_l1', 1e-5, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lightgbm_lambda_l2', 1e-5, 10.0, log=True),
                'verbosity': -1,
                'seed': 42
            }
            model = lgb.LGBMClassifier(**param)

        elif classifier_name == 'xgboost':
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'eta': trial.suggest_float('xgboost_eta', 1e-4, 1e-1, log=True),
                'max_depth': trial.suggest_int('xgboost_max_depth', 3, 12),
                'subsample': trial.suggest_float('xgboost_subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('xgboost_colsample_bytree', 0.5, 1.0),
                'lambda': trial.suggest_float('xgboost_lambda', 1e-5, 10.0, log=True),
                'alpha': trial.suggest_float('xgboost_alpha', 1e-5, 10.0, log=True),
                'use_label_encoder': False,
                'verbose': -1
            }
            model = xgb.XGBClassifier(
                                        **param,
                                        tree_method="hist",
                                        enable_categorical=True,
                                        random_state=42
                                    )

        elif classifier_name == 'catboost':
            param = {
                'iterations': 500,
                'learning_rate': trial.suggest_float('catboost_learning_rate', 1e-3, 1e-1, log=True),
                'depth': trial.suggest_int('catboost_depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 1e-2, 10.0, log=True),
                'random_strength': trial.suggest_float('catboost_random_strength', 0.1, 10.0),
                'loss_function': 'Logloss',
                'verbose': False,
                'random_seed': 42
            }
            model = CatBoostClassifier(**param)

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Scale numeric columns
            X_tr[numeric_features] = scaler.fit_transform(X_tr[numeric_features])
            X_val[numeric_features] = scaler.transform(X_val[numeric_features])

            if classifier_name == 'xgboost':
                # Convert categoricals to strings for LabelEncoder or use pd.get_dummies

                for col in categorical_features:
                    X_tr[col] = X_tr[col].astype("category")
                    X_val[col] = X_val[col].astype("category")

                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

                # X_tr = pd.get_dummies(X_tr, columns=categorical_features)
                # X_val = pd.get_dummies(X_val, columns=categorical_features)
                # # Align columns
                # X_tr, X_val = X_tr.align(X_val, join='left', axis=1, fill_value=0)
                # model.fit(X_tr, y_tr)

            elif classifier_name == 'catboost':
                # Ensure categorical columns are strings
                for col in categorical_features:
                    X_tr[col] = X_tr[col].astype(str)
                    X_val[col] = X_val[col].astype(str)
                model.fit(X_tr, y_tr, cat_features=categorical_features, eval_set=(X_val, y_val), verbose=False)

            elif classifier_name == 'lightgbm':  # lightgbm
                model.fit(X_tr, y_tr, categorical_feature=categorical_features)

            y_pred = model.predict(X_val)

            score = fbeta_score(y_val, y_pred, beta=2.0)
            scores.append(score)

        return np.mean(scores)

    study_name = config['optuna']['study_name']
    storage = config['optuna']['storage']
    n_trials = config['optuna']['n_trials']

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except:
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize")

    study.optimize(objective, n_trials=n_trials)
    
    return study, X_train, X_test, y_train, y_test, categorical_features, numeric_features


def best_model_fit(study, X_train, X_test, y_train, y_test, categorical_features, numeric_features):
    best_params = study.best_params
    print(best_params)
    best_model_type = best_params.pop('classifier')

    if best_model_type == 'lightgbm':
        lightgbm_params = remove_prefix_param(best_params, best_model_type)

        final_model = lgb.LGBMClassifier(**lightgbm_params, objective='binary', random_state=42)
        final_model.fit(X_train, y_train, categorical_feature=categorical_features)

    elif best_model_type == 'xgboost':
        xgboost_params = remove_prefix_param(best_params, best_model_type)

        for col in categorical_features:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        final_model = xgb.XGBClassifier(
            **xgboost_params,
            tree_method="hist",
            enable_categorical=True,
            random_state=42
        )
        final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        # X_test = X_test_encoded  # Обновляем X_test для предсказаний

    elif best_model_type == 'catboost':
        catboost_params = remove_prefix_param(best_params, best_model_type)

        for col in categorical_features:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

        final_model = CatBoostClassifier(
            **catboost_params,
            loss_function="Logloss",
            random_seed=42,
            verbose=False
        )
        final_model.fit(X_train, y_train, cat_features=categorical_features)

        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]

    with mlflow.start_run():
        # Log best parameters
        mlflow.log_params(best_params)

        # Predict and compute metrics
        y_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = final_model.predict(X_test)

        f2 = fbeta_score(y_test, y_pred, beta=2.0)
        recall = recall_score(y_test, y_pred)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc_val = auc(recall_vals, precision_vals)

        mlflow.log_metric("f2_score", f2)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("pr_auc", pr_auc_val)

        model_name = config['mlflow']['model_name']

        if best_model_type == "lightgbm":
            mlflow.lightgbm.log_model(final_model, artifact_path="model", registered_model_name=model_name)

        elif best_model_type == "xgboost":
            mlflow.xgboost.log_model(final_model, artifact_path="model", registered_model_name=model_name)

        elif best_model_type == "catboost":
            mlflow.catboost.log_model(final_model, artifact_path="model", registered_model_name=model_name)



        # Feature importance logging
        if hasattr(final_model, "feature_importances_"):
            importances = final_model.feature_importances_

            # Определить имена признаков в зависимости от типа модели
            if best_model_type == 'xgboost':
                feature_names = X_train.columns.tolist()
            else:
                feature_names = numeric_features + categorical_features

            if len(importances) != len(feature_names):
                raise ValueError(f"Mismatch: {len(importances)} importances vs {len(feature_names)} features")
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            # Save CSV
            feat_imp_path = os.path.join(today_results, "feature_importances.csv")
            feat_imp_df.to_csv(feat_imp_path, index=False)
            mlflow.log_artifact(feat_imp_path)

            # Save top 30 bar plot
            plt.figure(figsize=(12, 6))
            plt.barh(feat_imp_df['feature'][:30][::-1], feat_imp_df['importance'][:30][::-1])
            plt.xlabel("Importance")
            plt.title("Top 30 Feature Importances")
            plt.tight_layout()
            plot_path = os.path.join(today_results, "feature_importance_plot.png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)

    # Optionally print classification report
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    args = ffit()
    best_model_fit(*args)
