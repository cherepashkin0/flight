import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
import pandas as pd
from google.cloud import bigquery

warnings.filterwarnings('ignore')

# Get GCP project ID from environment variable
project_id = os.environ.get('GCP_PROJECT_ID')
if not project_id:
    raise ValueError("GCP_PROJECT_ID environment variable not set. Please set it before running this script.")

# Create output directory
output_dir = "flight_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Read list of columns to drop
drop_file = 'columns_to_drop.txt'
if not os.path.isfile(drop_file):
    raise FileNotFoundError(f"Drop-list file '{drop_file}' not found.")
with open(drop_file, 'r') as f:
    columns_to_drop = [line.strip() for line in f if line.strip()]

print(f"Connecting to BigQuery using project ID: {project_id}")
client = bigquery.Client(project=project_id)

# Query to perform stratified sampling (equal size from Cancelled=0 and Cancelled=1)
query = f"""
    WITH counts AS (
    SELECT Cancelled, COUNT(*) AS total
    FROM `{project_id}.flights.flights_all`
    GROUP BY Cancelled
    ),
    sample_sizes AS (
    SELECT
        Cancelled,
        total,
        SAFE_DIVIDE(total, SUM(total) OVER ()) AS proportion,
        ROUND(1000000 * SAFE_DIVIDE(total, SUM(total) OVER ())) AS n_sample
    FROM counts
    ),
    stratified AS (
    SELECT a.*
    FROM `{project_id}.flights.flights_all` a
    JOIN sample_sizes b
    USING (Cancelled)
    QUALIFY ROW_NUMBER() OVER (PARTITION BY Cancelled ORDER BY RAND()) <= b.n_sample
    )

    SELECT * FROM stratified
"""

print("Loading stratified sample from BigQuery...")
arrow_table = client.query(query).to_arrow()
df = arrow_table.to_pandas()

# Set up Seaborn theme
sns.set_theme(style="whitegrid")

# === Inspect time-related fields for Cancelled == 1 ===
time_columns = ['DepTime', 'WheelsOff', 'WheelsOn', 'ArrTime']
print("\n=== Time column check for Cancelled == 1 ===")
for col in time_columns:
    if col in df.columns:
        cancelled_vals = df[df['Cancelled'] == 1][col]
        print(f"{col}:")
        print(f"  NaNs: {cancelled_vals.isna().sum()}")
        print(f"  Unique values: {cancelled_vals.nunique()}")
        if not cancelled_vals.dropna().empty:
            print(f"  Most frequent: {cancelled_vals.value_counts(dropna=False).head(1).to_dict()}")
        print()

# === Plot cancellation rate by DepTime_Hour ===
if 'DepTime_Hour' in df.columns:
    hour_cancel = (
        df.groupby('DepTime_Hour')['Cancelled']
        .agg(['count', 'mean'])
        .rename(columns={'mean': 'Cancellation Rate'})
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=hour_cancel, x='DepTime_Hour', y='Cancellation Rate', palette='coolwarm')
    plt.title('Cancellation Rate by Scheduled Departure Hour')
    plt.xlabel('Scheduled Departure Hour')
    plt.ylabel('Cancellation Rate')
    plt.tight_layout()
    hour_plot_path = f"{output_dir}/cancel_rate_by_hour.png"
    plt.savefig(hour_plot_path)
    plt.close()
    print(f"Saved: {hour_plot_path}")

# === Plot cancellation rate by DepTime_Minute ===
if 'DepTime_Minute' in df.columns:
    minute_cancel = (
        df.groupby('DepTime_Minute')['Cancelled']
        .agg(['count', 'mean'])
        .rename(columns={'mean': 'Cancellation Rate'})
        .reset_index()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=minute_cancel, x='DepTime_Minute', y='Cancellation Rate')
    plt.title('Cancellation Rate by Scheduled Departure Minute')
    plt.xlabel('Scheduled Departure Minute')
    plt.ylabel('Cancellation Rate')
    plt.tight_layout()
    minute_plot_path = f"{output_dir}/cancel_rate_by_minute.png"
    plt.savefig(minute_plot_path)
    plt.close()
    print(f"Saved: {minute_plot_path}")
