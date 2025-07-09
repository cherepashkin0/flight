import os
import pandas as pd
import joblib
import json
from google.cloud import bigquery
from pathlib import Path
import gcsfs
import numpy as np

# Configuration
project_id = os.environ.get('GCP_PROJECT_ID')
dataset_name = 'flight_data'
table_name = 'combined_flights'
table_id = f'{project_id}.{dataset_name}.{table_name}'

PIPELINE_URI = "gs://mlflow_flight_cancellation/artifacts/0/addea66c90a04e6dbc9c09412ac8df26/artifacts/catboost_pipeline.pkl"
ROLE_PATH = "training_taken_roles.json"
SAMPLES_PER_CLASS = 5
output_dir = 'sampled_rows'
Path(output_dir).mkdir(exist_ok=True, parents=True)

# Load pipeline from GCS using gcsfs
def load_pipeline(uri):
    fs = gcsfs.GCSFileSystem()
    with fs.open(uri, 'rb') as f:
        pipeline = joblib.load(f)
    return pipeline

# Fetch data from BigQuery and return a DataFrame
def fetch_data(load_cols, limit=1000):
    client = bigquery.Client(project=project_id)
    load_cols_str = ", ".join(f"`{col}`" for col in load_cols)
    query = f"""
        SELECT {load_cols_str}
        FROM `{table_id}`
        WHERE Cancelled IS NOT NULL
        ORDER BY RAND()
        LIMIT {limit}
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)
    return df

# Save individual samples as CSVs
def export_samples(df, label):
    for i, row in df.iterrows():
        row_df = pd.DataFrame([row])
        filename = f"row_{i+1}_cancelled_{label}.csv"
        filepath = os.path.join(output_dir, filename)
        row_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")

def main():
    print("Loading pipeline...")
    pipeline = load_pipeline(PIPELINE_URI)

    print("Loading role mapping...")
    with open(ROLE_PATH, "r") as f:
        role_mapping = json.load(f)
    load_cols = list(role_mapping.keys())

    print("Fetching raw data from BigQuery...")
    df_raw = fetch_data(load_cols, limit=1000)

    print("Applying pipeline...")
    X = df_raw.drop(columns=["Cancelled"], errors='ignore')
    y_pred = pipeline.predict(X)

    df_raw["prediction"] = y_pred

    pos_samples = df_raw[df_raw["prediction"] == 1].drop(columns=["prediction"]).head(SAMPLES_PER_CLASS)
    neg_samples = df_raw[df_raw["prediction"] == 0].drop(columns=["prediction"]).head(SAMPLES_PER_CLASS)

    print("Saving predicted positive samples...")
    export_samples(pos_samples, "true")

    print("Saving predicted negative samples...")
    export_samples(neg_samples, "false")

if __name__ == "__main__":
    main()
