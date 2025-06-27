import os
from google.cloud import bigquery
import yaml
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from phik import phik_matrix
import numpy as np
from pathlib import Path
pd.options.display.max_rows = None
pd.set_option('display.max_rows', None)
import lightgbm as lgb
from sklearn.metrics import f1_score
import optuna
import IPython


project_id = os.environ.get('GCP_PROJECT_ID')
table_name = 'combined_flights'
dataset_name = 'flight_data'
table_id = f'{project_id}.{dataset_name}.{table_name}'
SAMPLE_SIZE = 100_000
PHIK_TARGET_HIGH_CORR = 0.85
PHIK_PAIR_HIGH_CORR = 0.9


# def load_all_columns():
#     client = bigquery.Client(project=project_id)
#     query = f"""
#                 SELECT column_name
#                 FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
#                 WHERE table_name = {table_name};
#             """
#     df = client.query(query).to_dataframe(create_bqstorage_client=True)
#     lst = [name[0] for name in df.values]
#     return lst

def plots2pdf(plots, fname):
    with PdfPages(fname) as pp:
        for plot in plots:
           pp.savefig(plot.figure)

def histogram_create(describe_df, dct_cols):
    print(describe_df['Role'].value_counts())
    client = bigquery.Client(project=project_id)
    

    print('Num cols:', dct_cols['num'])
    print('Cat not hot cols:', dct_cols['cat'])
    print('Cat hot cols:', dct_cols['hot'])

    query = f"""
            SELECT {", ".join(dct_cols['num'])+", Cancelled"}
            FROM `{table_id}`
            ORDER BY RAND()
            LIMIT {SAMPLE_SIZE}
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    lst_of_histograms = []
    for col_name in dct_cols['num']:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        sns.histplot(data=df[[col_name, 'Cancelled']], x=col_name, hue='Cancelled', alpha=0.4, stat='density', common_norm=False, ax=ax, bins=100)
        lst_of_histograms.append(ax)
        plt.close(fig)
    plots2pdf(lst_of_histograms, os.path.join(today_dir, 'histogram_nums.pdf'))


def train_test_split_create(dct_cols):
    all_columns = [col for cols in dct_cols.values() for col in cols]    
    client = bigquery.Client(project=project_id)
    query = f"""
            SELECT {", ".join(all_columns)}
            FROM `{table_id}`
            WHERE rand() < {SAMPLE_SIZE}/30000000
            LIMIT {SAMPLE_SIZE}
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)

    train_idx, test_idx = train_test_split(df.index, test_size=0.10, random_state=42, stratify=df['Cancelled'])

    X_train = df.drop(columns=['Cancelled']).iloc[train_idx]
    X_test = df.drop(columns=['Cancelled']).iloc[test_idx]
    y_train = df[['Cancelled']].iloc[train_idx]
    y_test = df[['Cancelled']].iloc[test_idx]
    return X_train, X_test, y_train, y_test

def drop_skip_cols(df, skip_cols, dct_cols):
    cols_existing_to_skip = [col for col in skip_cols if col in df.columns]
    df = df.drop(columns=cols_existing_to_skip)
    filtered_dct_cols = {}
    for key, cols in dct_cols.items():
        filtered_dct_cols[key] = [col for col in cols if col not in skip_cols]
    return df, filtered_dct_cols

def phik_create_matrix(X_train, y_train, dct_cols, describe_df):
    skip_cols = describe_df.loc[
        (describe_df['Skip_reason_phik'].notna()) & (describe_df['Skip_reason_phik'] != 'unk'),
        'Column_Name'
    ].tolist()
    df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    # print(89, 'df columns', df.columns)
    df[['Cancelled', 'Diverted']] = df[['Cancelled', 'Diverted']].astype('Int8')
    df, filtered_dct_cols = drop_skip_cols(df, skip_cols, dct_cols)
    # for col in filtered_dct_cols['num']:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    df[filtered_dct_cols['num']] = df[filtered_dct_cols['num']].astype(float)
    cat_cols = filtered_dct_cols['cat'] + filtered_dct_cols['hot']
    df[cat_cols] = df[cat_cols].astype('category')
    phik_corr = phik_matrix(df, interval_cols=filtered_dct_cols['num'], dropna=True, verbose=1)
    cancelled_corr = phik_corr['Cancelled'].dropna().copy()
    cancelled_corr = cancelled_corr[cancelled_corr.index != 'Cancelled']
    high_corr = cancelled_corr[abs(cancelled_corr) > PHIK_TARGET_HIGH_CORR].sort_values(ascending=False)
    high_corr_dict = {col: 'true' for col in high_corr}
    describe_df['Phik_high_target'] = describe_df['Column_Name'].map(high_corr_dict).fillna('unk')
    phik_corr.to_csv(os.path.join(today_dir, 'phik_matrix.csv'))

    # phik_long = phik_corr.stack().reset_index()
    # phik_long.columns = ['column_a', 'column_b', 'value']
    # phik_long = phik_long[phik_long['column_a'] != phik_long['column_b']]
    # phik_long['min_col'] = phik_long[['column_a', 'column_b']].min(axis=1)
    # phik_long['max_col'] = phik_long[['column_a', 'column_b']].max(axis=1)
    # phik_long_unique = phik_long[['min_col', 'max_col', 'value']].drop_duplicates()
    # phik_long_unique.columns = ['column_a', 'column_b', 'value']
    # phik_long_unique.sort_values('value', ascending=False).to_csv(os.path.join(today_dir, 'phik_long.csv'), index=False)

def f1_eval(y_pred, dataset):
    y_true = dataset.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred_binary), True

def train(X_train, X_test, y_train, y_test, dct_cols, describe_df):
    skip_cols = describe_df.loc[
        (describe_df['Skip_reason_phik'] != 'unk') |
        (describe_df['Phik_high_target'] != 'unk') |
        (describe_df['Data_leakage'] != 'unk'),
        'Column_Name'
    ].tolist()
    X_train, filtered_dct_cols = drop_skip_cols(X_train, skip_cols, dct_cols)
    X_test, filtered_dct_cols = drop_skip_cols(X_test, skip_cols, dct_cols)
    cat_cols = filtered_dct_cols['cat'] + filtered_dct_cols['hot']
    X_train[cat_cols] = X_train[cat_cols].astype('category')
    X_test[cat_cols] = X_test[cat_cols].astype('category')
    print("Remaining numeric features:", filtered_dct_cols['num'])
    print("Remaining categorical features:", cat_cols)
    print("X_train shape:", X_train.shape)
    # print("Non-null per column:\n", X_train.notna().sum())
    # zero_var = X_train.nunique(dropna=False) <= 1
    # print("Zero-variance features:", zero_var[zero_var].index.tolist())
    # Check class distribution
    class_counts = y_train.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    pos_weight = class_counts[0] / class_counts[1]
    print(f"Calculated pos_weight (for class imbalance): {pos_weight:.2f}")

    print("Final features shape:", X_train.shape)
    # print("Number of non-null values:\n", X_train.notnull().sum())
    # print("Unique values per categorical column:\n", X_train[cat_cols].nunique())

    train_data = lgb.Dataset(X_train,
                             label=y_train.squeeze(),
                             feature_name=filtered_dct_cols['num'] + cat_cols,
                             categorical_feature=cat_cols)

    def objective(trial):
        classifier_name = trial.suggest_categorical('classifier', ['lightgbm'])
        if classifier_name == 'lightgbm':
            param = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('lightgbm_num_leaves', 2, 15, log=True),
                'min_data_in_leaf': 1,        
                'min_gain_to_split': 0.0001,  
                'max_depth': trial.suggest_int('lightgbm_max_depth', 10, 250, log=True), 
                'learning_rate': trial.suggest_float('lightgbm_learning_rate', 1e-5, 1e-2, log=True),
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8, 
                'bagging_freq': 5,
                'lambda_l1': 0.1, 
                'lambda_l2': 0.1,
                'scale_pos_weight': pos_weight,
                'verbosity': 1,
                'seed': 42
            }
            cv_results = lgb.cv(param, train_data, num_boost_round=10, nfold=3, feval=f1_eval, stratified=True, seed=42, return_cvbooster=False)
            result_metric = np.mean(cv_results['valid f1-mean'])
        return result_metric
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    # num_round = 10
    # tmp_cols = filtered_dct_cols['num'][:5] + cat_cols[:3]
    # tmp_train = lgb.Dataset(X_train[tmp_cols], label=y_train.squeeze(),
    #                         categorical_feature=[c for c in tmp_cols if c in cat_cols])
    # print(model.dump_model(num_iteration=10)['tree_info'][:1])

    # cv_results = lgb.cv(
    #     param,
    #     train_data,
    #     num_boost_round=1000,
    #     nfold=3,
    #     feval=f1_eval,
    #     stratified=True,
    #     seed=42,
    #     return_cvbooster=False,
    #     callbacks=[
    #         lgb.early_stopping(stopping_rounds=50, verbose=True),
    #         lgb.log_evaluation(period=10)
    #     ]
    # )
    # print("Available metrics in cv_results:", list(cv_results.keys()))
    # best_round = len(cv_results['valid f1-mean'])
    # final_model = lgb.train(param, train_data, num_boost_round=best_round)
    # final_model.save_model(os.path.join(today_dir, 'model.txt'))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    # lgb.plot_importance(final_model, max_num_features=20, ax=ax)
    # fig.savefig(os.path.join(today_dir, 'feautre_importance_final.png'))


def main():
    describe_df = pd.read_csv('results/flights_all_analysis_with_roles.csv')
    dct_cols = {}
    for key in ['num', 'cat', 'hot', 'tgt', 'dat']:
        dct_cols[key] = describe_df.loc[describe_df['Role'] == key, 'Column_Name'].sort_values().tolist()   
    histogram_create(describe_df, dct_cols)
    X_train, X_test, y_train, y_test = train_test_split_create(dct_cols)
    phik_create_matrix(X_train, y_train, dct_cols, describe_df)
    train(X_train, X_test, y_train, y_test, dct_cols, describe_df)

if __name__ == "__main__":
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_dir = os.path.join('results', today_date)
    Path(today_dir).mkdir(exist_ok=True, parents=True)
    main()
