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


project_id = os.environ.get('GCP_PROJECT_ID')
table_id = f'{project_id}.flights.flights_all'
SAMPLE_SIZE = 100_000

def load_all_columns():
    client = bigquery.Client(project=project_id)
    query = f"""
                SELECT column_name
                FROM `{project_id}.flights.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = 'flights_all';
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    lst = [name[0] for name in df.values]
    return lst

def plots2pdf(plots, fname):
    with PdfPages(fname) as pp:
        for plot in plots:
           pp.savefig(plot.figure)

def histogram_create(dct_cols):
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

def phik_create_matrix(X_train, y_train, dct_cols):
    skip_cols = describe_df.loc[describe_df['Skip_reason_phik'].notna(), 'Column_Name'].tolist()
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
    phik_long = phik_corr.stack().reset_index()
    phik_long.columns = ['column_a', 'column_b', 'value']
    phik_long = phik_long[phik_long['column_a'] != phik_long['column_b']]
    phik_long['min_col'] = phik_long[['column_a', 'column_b']].min(axis=1)
    phik_long['max_col'] = phik_long[['column_a', 'column_b']].max(axis=1)
    phik_long_unique = phik_long[['min_col', 'max_col', 'value']].drop_duplicates()
    phik_long_unique.columns = ['column_a', 'column_b', 'value']
    phik_corr.to_csv(os.path.join(today_dir, 'phik_matrix.csv'))
    phik_long_unique.to_csv(os.path.join(today_dir, 'phik_long.csv'), index=False)

def f1_eval(y_pred, dataset):
    y_true = dataset.get_label()
    y_pred_binary = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred_binary), True

def train(X_train, X_test, y_train, y_test, dct_cols):
    skip_cols = describe_df.loc[
        (describe_df['Skip_reason_phik'].notna()) | (describe_df['Phik_high_target'].notna()),
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

    train_data = lgb.Dataset(X_train,
                             label=y_train.squeeze(),
                             feature_name=filtered_dct_cols['num'] + cat_cols,
                             categorical_feature=cat_cols)
    param = {
        'objective': 'binary',
        'metric': 'None',
        'num_leaves': 31,
        'min_data_in_leaf': 5,            # 🔽 меньше ограничение на количество наблюдений
        'min_gain_to_split': 0.0,         # 🔽 разрешаем сплиты с 0 приростом
        'max_depth': 10,                  # 🔽 ограничиваем глубину
        'verbosity': 1
    }
    num_round = 10
    cv_results = lgb.cv(param, train_data, num_round, nfold=2, feval=f1_eval, stratified=True, seed=42, return_cvbooster=False)
    best_round = len(cv_results['f1-mean']) if 'f1-mean' in cv_results else len(cv_results['binary_logloss-mean'])
    final_model = lgb.train(param, train_data, num_boost_round=best_round)
    final_model.save_model(os.path.join(today_dir, 'model.txt'))


def main():
    dct_cols = {}
    for key in ['num', 'cat', 'hot', 'tgt', 'dat']:
        dct_cols[key] = describe_df.loc[describe_df['Role'] == key, 'Column_Name'].sort_values().tolist()   
    # histogram_create(dct_cols)
    X_train, X_test, y_train, y_test = train_test_split_create(dct_cols)
    # phik_create_matrix(X_train, y_train, dct_cols)
    train(X_train, X_test, y_train, y_test, dct_cols)

if __name__ == "__main__":
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_dir = os.path.join('results', today_date)
    Path(today_dir).mkdir(exist_ok=True, parents=True)
    describe_df = pd.read_csv('results/flights_all_analysis.csv')
    main()
