import os
import sys
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import List

def find_parquet_files(directory: str) -> List[str]:
    """Find all Combined_Flights_YYYY.parquet files in the given directory (YYYY between 2018 and 2022)."""
    parquet_files = []
    for file in os.listdir(directory):
        if file.startswith('Combined_Flights_') and file.endswith('.parquet'):
            # Extract year part between last underscore and .parquet
            year_part = file[len('Combined_Flights_'):-len('.parquet')]
            if year_part in {"2018", "2019", "2020", "2021", "2022"}:
                parquet_files.append(os.path.join(directory, file))
    return parquet_files

def determine_bigquery_type(pandas_dtype: str) -> str:
    """Map pandas data types to BigQuery data types."""
    if 'int' in pandas_dtype:
        return 'INT64'
    elif 'float' in pandas_dtype:
        return 'FLOAT64'
    elif 'bool' in pandas_dtype:
        return 'BOOL'
    elif 'datetime' in pandas_dtype:
        return 'TIMESTAMP'
    elif 'date' in pandas_dtype:
        return 'DATE'
    else:
        return 'STRING'

def generate_schema_from_parquet(parquet_file: str) -> List[bigquery.SchemaField]:
    """Generate BigQuery schema from a single parquet file."""
    df = pd.read_parquet(parquet_file)
    schema = []
    for column_name, dtype in df.dtypes.items():
        bq_type = determine_bigquery_type(str(dtype))
        schema.append(bigquery.SchemaField(column_name, bq_type))
    return schema

def ensure_dataset_exists(client: bigquery.Client, dataset_id: str, location: str = "US") -> None:
    """Create dataset if it doesn't exist."""
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset.description = f"Dataset created for parquet files import on {pd.Timestamp.now().strftime('%Y-%m-%d')}"
        dataset.default_table_expiration_ms = None
        client.create_dataset(dataset)
        print(f"Dataset {dataset_id} created in {location}.")

def load_to_bigquery(
    file_path: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str = "US"
) -> None:
    """Load a single parquet file to BigQuery."""
    client = bigquery.Client(project=project_id)
    ensure_dataset_exists(client, dataset_id, location)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    # Generate schema
    schema = generate_schema_from_parquet(file_path)
    if not schema:
        print(f"No schema generated for {file_path}. Skipping.")
        return

    # Delete existing table if present
    try:
        client.get_table(table_ref)
        client.delete_table(table_ref)
        print(f"Deleted existing table {table_id}.")
    except NotFound:
        pass

    # Create table
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    print(f"Created table {table_id} with schema.")

    # Load the file
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        schema=schema,
    )
    with open(file_path, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    job.result()
    table = client.get_table(table_ref)
    print(f"Loaded {table.num_rows} rows into {project_id}.{dataset_id}.{table_id} from {file_path}.")

def get_env_var(var_name: str, required: bool = True) -> str:
    """Get environment variable and exit if missing."""
    value = os.environ.get(var_name)
    if required and not value:
        print(f"Error: Required environment variable '{var_name}' is not set.")
        sys.exit(1)
    return value

def main():
    project_id = get_env_var("GCP_PROJECT_ID")
    dataset_id = "flights"
    location = "US"

    directory = 'flight_delay'
    print(f"Searching for Combined_Flights_YYYY.parquet files in {directory}...")
    parquet_files = find_parquet_files(directory)
    print(f"Found {len(parquet_files)} matching parquet files.")

    if not parquet_files:
        print("No matching parquet files found. Exiting.")
        return

    # Process each parquet file individually
    for file_path in parquet_files:
        filename = os.path.basename(file_path)
        year_part = filename[len('Combined_Flights_'):-len('.parquet')]
        table_id = f"flight_{year_part}"
        print(f"Loading {filename} into table {table_id}...")
        load_to_bigquery(file_path, project_id, dataset_id, table_id, location)

    print("All files processed.")

if __name__ == "__main__":
    main()
