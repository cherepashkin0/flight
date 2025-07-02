import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, fbeta_score, precision_recall_curve, auc, classification_report
from sklearn.impute import SimpleImputer
from google.cloud import bigquery
import optuna
import os
import mlflow
import matplotlib.pyplot as plt

# from sklearn import set_config
# set_config(transform_output='pandas')

SAMPLE_SIZE = 100_000
project_id = os.environ.get('GCP_PROJECT_ID')
table_name = 'combined_flights'
dataset_name = 'flight_data'
table_id = f'{project_id}.{dataset_name}.{table_name}'
describe_df = pd.read_csv('results/flights_all_analysis_with_roles.csv')
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
    skip_cols = list(set(skip_cols + 
    ['DivAirportLandings', 'DepartureDelayGroups', 'AirTime', 'ArrivalDelayGroups', 'ArrDelay', 'ArrDelayMinutes',
      'DepDelay', 'DepDelayMinutes', 'ActualElapsedTime', 'TaxiOut', 'TaxiIn', 'Diverted', '__index_level_0__']))
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
    
def ffit():
    mlflow.set_experiment("flight_cancellation_prediction")    
    target_col = 'Cancelled'
    dct_cols = get_col_roles()
    skip_cols = get_skiped_cols(describe_df)
    dct_cols = drop_skip_cols(skip_cols, dct_cols)
    df = data_load_from_bigquery(dct_cols) 
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = dct_cols['num']
    categorical_features = dct_cols['cat'] + dct_cols['hot']

    X[categorical_features] = X[categorical_features].astype('category')

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    # categorical_transformer = Pipeline(steps=[
    #     ('cat_imputer', CategoryImputer())
    # ])
    categorical_transformer = 'passthrough'

    # print('numeric: \n', numeric_features, 'categorical: \n', categorical_features, 'one_hotable: \n', one_hotable_features)
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    def objective(trial):
        classifier_name = trial.suggest_categorical('classifier', ['lightgbm'])
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

            cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Scale numeric columns
                X_tr[numeric_features] = scaler.fit_transform(X_tr[numeric_features])
                X_val[numeric_features] = scaler.transform(X_val[numeric_features])

                model = lgb.LGBMClassifier(**param)
                model.fit(X_tr, y_tr, categorical_feature=categorical_features)

                y_pred = model.predict(X_val)
                score = fbeta_score(y_val, y_pred, beta=2.0)
                scores.append(score)
        return np.mean(scores)
    try:
        study = optuna.load_study(study_name="my_study", storage="sqlite:///my_study.db")
    except:
        study = optuna.create_study(study_name="my_study", storage="sqlite:///my_study.db", direction="maximize")
    
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    # Scale numeric features
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    # Train final model
    final_model = lgb.LGBMClassifier(**best_params, objective='binary', random_state=42)
    final_model.fit(X_train, y_train, categorical_feature=categorical_features)

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]


    # final_model = lgb.LGBMClassifier(**best_params, objective='binary', random_state=42)
    # pipeline = Pipeline(steps=[
    #             ('preprocessor', preprocessor),
    #             ('classifier', final_model)
    #             ])
    # pipeline.fit(X_train, y_train, categorical_feature=categorical_features)
    # y_pred = pipeline.predict(X_test)
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

        # Log model using native LightGBM saving
        model_path = "lgb_model.txt"
        # final_model.booster_.save_model(model_path)
        mlflow.log_artifact(model_path)

        # Feature importance logging
        if hasattr(final_model, "feature_importances_"):
            all_features = numeric_features + categorical_features
            importances = final_model.feature_importances_

            feat_imp_df = pd.DataFrame({
                'feature': all_features,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            # Save CSV
            feat_imp_path = "feature_importances.csv"
            # feat_imp_df.to_csv(feat_imp_path, index=False)
            mlflow.log_artifact(feat_imp_path)

            # Save top 30 bar plot
            plt.figure(figsize=(12, 6))
            plt.barh(feat_imp_df['feature'][:30][::-1], feat_imp_df['importance'][:30][::-1])
            plt.xlabel("Importance")
            plt.title("Top 30 Feature Importances")
            plt.tight_layout()
            plot_path = "feature_importance_plot.png"
            # plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)

    # Optionally print classification report
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    ffit()
