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

today_date = datetime.today().strftime('%Y-%m-%d')


project_id = os.environ.get('GCP_PROJECT_ID')
table_id = f'{project_id}.flights.flights_all'

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

def histogram_create():
    describe_df = pd.read_csv('results/flights_all_analysis.csv')
    print(describe_df['Role'].value_counts())
    client = bigquery.Client(project=project_id)
    num_cols = [name[0] for name in describe_df[describe_df['Role']=='num'].values.tolist()]
    num_cols += ["Cancelled"]
    print(num_cols)
    query = f"""
            SELECT {", ".join(num_cols)}
            FROM `{table_id}`
            ORDER BY RAND()
            LIMIT 100000
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    lst_of_histograms = []
    for col_name in df.columns:
        if col_name != 'Cancelled':
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            
            sns.histplot(data=df[[col_name, 'Cancelled']], x=col_name, hue='Cancelled', alpha=0.4, stat='density', common_norm=False, ax=ax, bins=100)
            lst_of_histograms.append(ax)
            plt.close(fig)

    plots2pdf(lst_of_histograms, os.path.join('results', today_date, 'histogram_nums.pdf'))
    return num_cols


def train_test_split_create(num_cols):
    client = bigquery.Client(project=project_id)
    query = f"""
            SELECT {", ".join(num_cols)}
            FROM `{table_id}`
            LIMIT 100000
            """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Cancelled']), df[['Cancelled']], test_size=0.10, random_state=42)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def phik_create_matrix(X_train, y_train, num_cols):
    df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df.columns = num_cols + ['Cancelled']
    phik_corr = phik_matrix(df, interval_cols=df.columns, dropna=True, verbose=1)
    print(phik_corr)


def main():   
    num_cols = histogram_create()
    X_train, X_test, y_train, y_test = train_test_split_create(num_cols)
    phik_create_matrix(X_train, y_train, num_cols)

if __name__ == "__main__":
    main()
