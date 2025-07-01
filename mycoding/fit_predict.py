import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, fbeta_score, precision_recall_curve, auc, make_scorer, classification_report
from sklearn.impute import SimpleImputer
from google.cloud import bigquery
import optuna
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# from sklearn import set_config
# set_config(transform_output='pandas')

SAMPLE_SIZE = 100_000
project_id = os.environ.get('GCP_PROJECT_ID')
table_name = 'combined_flights'
dataset_name = 'flight_data'
table_id = f'{project_id}.{dataset_name}.{table_name}'
describe_df = pd.read_csv('results/flights_all_analysis_with_roles.csv')

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
    ['ArrDelay', 'ArrDelayMinutes', 'DepDelay', 'DepDelayMinutes', 'ActualElapsedTime', 'TaxiOut', 'TaxiIn', 'Diverted', '__index_level_0__']))
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
            model = lgb.LGBMClassifier(**param)
            pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', model)
                        ])

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fbeta_scorer = make_scorer(fbeta_score, beta=2.0)
            result = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=fbeta_scorer).mean()
            # result = cross_val_score(model, X_train[filtered_dct_cols['num']], y_train.squeeze(), 
                                    #  cv=cv, scoring=make_scorer(fbeta_scorer, needs_proba=True)).mean()
            print('user_attrs', trial.user_attrs)
            # print(f"Mean F2: {trial.user_attrs['mean_f2']:.4f} ± {trial.user_attrs['std_f2']:.4f}")                                
        return result
    try:
        study = optuna.load_study(study_name="my_study", storage="sqlite:///my_study.db")
    except:
        study = optuna.create_study(study_name="my_study", storage="sqlite:///my_study.db", direction="maximize")
    
    study.optimize(objective, n_trials=3)

    best_params = study.best_params
    final_model = lgb.LGBMClassifier(**best_params, objective='binary', random_state=42)
    pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', final_model)
                ])
    pipeline.fit(X_train, y_train, categorical_feature=categorical_features)
    y_pred = pipeline.predict(X_test)
    with mlflow.start_run():
        # Log model params
        mlflow.log_params(best_params)

        # Log metrics
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        f2 = fbeta_score(y_test, y_pred, beta=2.0)
        recall = recall_score(y_test, y_pred)
        precision, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision)
        mlflow.log_metric("f2_score", f2)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Feature importance
        model = pipeline.named_steps["classifier"]
        if hasattr(model, "feature_importances_"):
            # Get feature names after preprocessing
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            cat_features = ohe.get_feature_names_out(categorical_features)
            all_features = numeric_features + list(cat_features)
            importances = model.feature_importances_

            # Create DataFrame
            feat_imp_df = pd.DataFrame({
                'feature': all_features,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            # Save CSV
            feat_imp_path = "feature_importances.csv"
            feat_imp_df.to_csv(feat_imp_path, index=False)
            mlflow.log_artifact(feat_imp_path)

            # Plot and save bar chart
            plt.figure(figsize=(12, 6))
            plt.barh(feat_imp_df['feature'][:30][::-1], feat_imp_df['importance'][:30][::-1])
            plt.xlabel("Importance")
            plt.title("Top 30 Feature Importances")
            plt.tight_layout()
            plot_path = "feature_importance_plot.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    ffit()
