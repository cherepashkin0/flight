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
PIPELINE_URI = "gs://mlflow_flight_cancellation3/artifacts/1/models/m-8e0862b1e83942509092f57c3b1e013f/artifacts/model.pkl"
ROLE_PATH = "columns_roles.json"
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
    if df.empty:
        print(f"Warning: No samples found for label '{label}'")
        return
    
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
    
    print(f"Fetched {len(df_raw)} rows")
    print(f"Actual cancellation distribution:")
    print(df_raw["Cancelled"].value_counts())
    
    print("Applying pipeline...")
    X = df_raw.drop(columns=["Cancelled"], errors='ignore')
    y_pred = pipeline.predict(X)
    
    df_raw["prediction"] = y_pred
    
    # Debug: Check prediction distribution
    print(f"\nPrediction distribution:")
    print(pd.Series(y_pred).value_counts())
    
    # Debug: Check unique prediction values
    print(f"Unique prediction values: {np.unique(y_pred)}")
    
    pos_samples = df_raw[df_raw["prediction"] == 1].drop(columns=["prediction"]).head(SAMPLES_PER_CLASS)
    neg_samples = df_raw[df_raw["prediction"] == 0].drop(columns=["prediction"]).head(SAMPLES_PER_CLASS)
    
    print(f"\nFound {len(pos_samples)} positive samples")
    print(f"Found {len(neg_samples)} negative samples")
    
    print("Saving predicted positive samples...")
    export_samples(pos_samples, "true")
    
    print("Saving predicted negative samples...")
    export_samples(neg_samples, "false")
    
    # Alternative approach: If no positive predictions, try using probability scores
    if len(pos_samples) == 0:
        print("\nNo positive predictions found. Trying probability-based approach...")
        try:
            # Get prediction probabilities if available
            y_proba = pipeline.predict_proba(X)
            if y_proba.shape[1] > 1:  # Binary classification
                df_raw["prob_positive"] = y_proba[:, 1]
                
                # Get samples with highest probability of being positive
                high_prob_samples = df_raw.nlargest(SAMPLES_PER_CLASS, "prob_positive").drop(columns=["prediction", "prob_positive"])
                
                print("Saving high-probability positive samples...")
                export_samples(high_prob_samples, "high_prob_true")
                
        except Exception as e:
            print(f"Could not get probabilities: {e}")
            
            # Final fallback: use actual cancelled flights
            print("Using actual cancelled flights as positive samples...")
            actual_cancelled = df_raw[df_raw["Cancelled"] == 1].drop(columns=["prediction"]).head(SAMPLES_PER_CLASS)
            export_samples(actual_cancelled, "actual_true")

if __name__ == "__main__":
    main()
