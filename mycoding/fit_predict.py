import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from google.cloud import bigquery
import os

SAMPLE_SIZE = 1_000_000
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

def ffit():
    target_col = 'Cancelled'
    dct_cols = get_col_roles()
    skip_cols = get_skiped_cols(describe_df)
    dct_cols = drop_skip_cols(skip_cols, dct_cols)
    df = data_load_from_bigquery(dct_cols) 
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify column types
    numeric_features = dct_cols['num']
    categorical_features = dct_cols['cat']

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=np.nan)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # LightGBM classifier
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        # 'metric': 'auc',                
        'boosting_type': 'gbdt',
        'num_leaves': 20,
        'min_data_in_leaf': 1,        
        'min_gain_to_split': 0.0001,  
        'max_depth': 200, 
        'learning_rate': 1e-2,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 
        'bagging_freq': 5,
        'lambda_l1': 0.1, 
        'lambda_l2': 0.1,
        # 'scale_pos_weight': pos_weight,
        'verbosity': -1,
        'seed': 42
    }
    model = lgb.LGBMClassifier(**param)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    ffit()
