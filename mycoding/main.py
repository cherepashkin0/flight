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
    existing_cols = [col for col in skip_cols if col in df.columns]
    df = df.drop(columns=existing_cols)
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
    # print('Columns before phik correlation \n', df.dtypes)
    print(df[filtered_dct_cols['num']].dtypes)
    phik_corr = phik_matrix(df, interval_cols=filtered_dct_cols['num'], dropna=True, verbose=1)
        # interval_cols=filtered_dct_cols['num'], 
    # phik_long = phik_corr.stack().reset_index()
    # phik_long.columns = ['column_a', 'column_b', 'value']
    # phik_long = phik_long[phik_long['column_a'] != phik_long['column_b']]
    # phik_long = 


def train(X_train, y_train, dct_cols):
    # skip_cols = describe_df.loc[describe_df['Skip_reason_phik'] or describe_df['Phik_high_target'], 'Column_Name'].tolist()

    # X_train, filtered_dct_cols = drop_skip_cols(X_train, skip_cols, dct_cols)
    # X_test, filtered_dct_cols = drop_skip_cols(X_test, skip_cols, dct_cols)
    # X_train[['Diverted']] = df[['Diverted']].astype('int8')
    # X_test[['Diverted']] = 
    pass



def main():
    dct_cols = {}
    for key in ['num', 'cat', 'hot', 'tgt', 'dat']:
        dct_cols[key] = describe_df.loc[describe_df['Role'] == key, 'Column_Name'].sort_values().tolist()   
    # histogram_create(dct_cols)
    X_train, X_test, y_train, y_test = train_test_split_create(dct_cols)
    phik_create_matrix(X_train, y_train, dct_cols)
    train(X_train, y_train, dct_cols)

if __name__ == "__main__":
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_dir = os.path.join('results', today_date)
    Path(today_dir).mkdir(exist_ok=True, parents=True)
    describe_df = pd.read_csv('results/flights_all_analysis.csv')
    main()
