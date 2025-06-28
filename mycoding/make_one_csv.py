import os
import pandas as pd
from google.cloud import bigquery
from pathlib import Path

# Setup
project_id = os.environ.get('GCP_PROJECT_ID')
dataset_name = 'flight_data'
table_name = 'combined_flights'
table_id = f'{project_id}.{dataset_name}.{table_name}'

# Output directory
output_dir = 'sampled_rows'
Path(output_dir).mkdir(exist_ok=True, parents=True)

# Connect to BigQuery
client = bigquery.Client(project=project_id)

# Number of samples per class
SAMPLES_PER_CLASS = 5

def fetch_and_export(cancelled_value: bool, label: str, describe_df):
    # skip_cols = describe_df.loc[
    #     (describe_df['Skip_reason_phik'] != 'unk') |
    #     (describe_df['Data_leakage'] != 'unk'),
    #     'Column_Name'
    # ].tolist()
    load_cols = ['AirTime', 'ArrivalDelayGroups', 'CRSElapsedTime', 'DOT_ID_Marketing_Airline', 'DOT_ID_Operating_Airline', 'DayofMonth', 'DepartureDelayGroups', 'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID', 'DestStateFips', 'Distance', 'DivAirportLandings', 'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'OriginStateFips', 'OriginWac']
    # num_cols = describe_df.loc[describe_df['Role'] == 'num', 'Column_Name'].tolist()
    load_cols_str = ", ".join(f"`{col}`" for col in load_cols)  # wrap in backticks for safety
    query = f"""
        SELECT {load_cols_str}
        FROM `{table_id}`
        WHERE Cancelled IS {str(cancelled_value).upper()}
        ORDER BY RAND()
        LIMIT {SAMPLES_PER_CLASS}
    """
    df = client.query(query).to_dataframe(create_bqstorage_client=True)

    for i, row in df.iterrows():
        row_df = pd.DataFrame([row])
        row_df = row_df.drop(columns=["Cancelled"], errors='ignore')  # Drop label column
        filename = f"row_{i+1}_cancelled_{label}.csv"
        filepath = os.path.join(output_dir, filename)
        row_df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")

def main():
    describe_df = pd.read_csv('results/flights_all_analysis_with_roles.csv')
    print("Fetching Cancelled = True samples...")
    fetch_and_export(True, 'true', describe_df)

    print("Fetching Cancelled = False samples...")
    fetch_and_export(False, 'false', describe_df)
if __name__ == "__main__":
    main()
