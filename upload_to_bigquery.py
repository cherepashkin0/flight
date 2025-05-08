import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from typing import List
from tqdm import tqdm
from pandas.api.types import is_bool_dtype


def find_parquet_files(directory: str) -> List[str]:
    """Find all Combined_Flights_YYYY.parquet files in the given directory (YYYY between 2018 and 2022)."""
    parquet_files = []
    for file in os.listdir(directory):
        if file.startswith('Combined_Flights_') and file.endswith('.parquet'):
            year_part = file[len('Combined_Flights_'):-len('.parquet')]
            if year_part in {"2018", "2019", "2020", "2021", "2022"}:
                parquet_files.append(os.path.join(directory, file))
    return parquet_files


def map_pa_type_to_bq(pa_type: pa.DataType) -> str:
    """Map a pyarrow DataType to a BigQuery field type."""
    if pa.types.is_int8(pa_type) or pa.types.is_int16(pa_type) or pa.types.is_int32(pa_type) or pa.types.is_int64(pa_type):
        return 'INT64'
    if pa.types.is_float16(pa_type) or pa.types.is_float32(pa_type) or pa.types.is_float64(pa_type):
        return 'FLOAT64'
    if pa.types.is_boolean(pa_type):
        return 'BOOL'
    if pa.types.is_timestamp(pa_type):
        return 'TIMESTAMP'
    if pa.types.is_date(pa_type):
        return 'DATE'
    if pa.types.is_decimal(pa_type):
        return 'NUMERIC'
    # fallback to string for binary, fixed_size_binary, list, etc.
    return 'STRING'

def determine_bigquery_type(pandas_dtype: str) -> str:
    """
    Fallback mapping when pa schema not available. Converts pandas dtype string to BQ type.
    """
    if pandas_dtype in ('int64', 'Int64'):
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
    # First get the DataFrame with all optimizations applied
    df = pd.read_parquet(parquet_file)
    df = optimize_float_columns(df)
    df = preprocess_time_columns(df)
    
    # Now create schema based on the optimized DataFrame
    schema = []
    
    # Get column info from parquet file as reference
    parquet_schema = pq.read_schema(parquet_file)
    
    # Process each column in the optimized DataFrame
    for col in df.columns:
        # Check if it's a timestamp column
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            bq_type = 'TIMESTAMP'
        # Handle delay columns specifically - ALWAYS use INT64 for these
        elif col in ['DepDelayMinutes', 'ArrDelayMinutes', 'TaxiIn', 'TaxiOut', 'ActualElapsedTime', 'CRSElapsedTime']:
            # Convert column to integer if not already
            if df[col].dtype != 'Int64':
                try:
                    # Only convert if all values can be converted without loss
                    if df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                        df[col] = df[col].astype('Int64')
                        bq_type = 'INT64'
                    else:
                        bq_type = 'FLOAT64'
                except:
                    bq_type = 'FLOAT64'
            else:
                bq_type = 'INT64'
        elif is_bool_dtype(df[col]):        # <‑‑ add this block
            bq_type = 'BOOL'                
        # Handle integer columns
        elif str(df[col].dtype) in ('int64', 'Int64'):
            bq_type = 'INT64'
        # Handle float columns
        elif 'float' in str(df[col].dtype):
            bq_type = 'FLOAT64'
        # Default to String for other types
        else:
            bq_type = 'STRING'
            
        schema.append(bigquery.SchemaField(col, bq_type))
    
    return schema, df


def ensure_dataset_exists(client: bigquery.Client, dataset_id: str, location: str = "US") -> None:
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset.description = (
            f"Dataset created for parquet files import on "
            f"{pd.Timestamp.now().strftime('%Y-%m-%d')}"
        )
        client.create_dataset(dataset)
        print(f"Dataset {dataset_id} created in {location}.")


def load_to_bigquery(
    file_path: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str = "US"
) -> None:
    client = bigquery.Client(project=project_id)
    ensure_dataset_exists(client, dataset_id, location)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    # Generate schema and get optimized dataframe
    schema, df = generate_schema_from_parquet(file_path)
    if not schema:
        print(f"No schema generated for {file_path}. Skipping.")
        return

    # Delete existing table
    try:
        client.get_table(table_ref)
        client.delete_table(table_ref)
        print(f"Deleted existing table {table_id}.")
    except NotFound:
        pass

    # Create table
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    print(f"Created table {table_id} with updated schema.")

    # Instead of loading from file, load from the optimized DataFrame
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    # Option 1: Load directly from DataFrame
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    
    # OR Option 2: Save optimized DataFrame to temporary parquet and load from that
    # temp_file = f"{file_path}_optimized.parquet"
    # df.to_parquet(temp_file, index=False)
    # job_config = bigquery.LoadJobConfig(
    #     source_format=bigquery.SourceFormat.PARQUET,
    # )
    # with open(temp_file, 'rb') as source_file:
    #     job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
    # os.remove(temp_file)  # Clean up
    
    job.result()
    print(f"Loaded data into {project_id}.{dataset_id}.{table_id}.")


def get_env_var(var_name: str, required: bool = True) -> str:
    value = os.environ.get(var_name)
    if required and not value:
        print(f"Error: Required environment variable '{var_name}' is not set.")
        sys.exit(1)
    return value


def preprocess_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["CRSDepTime", "DepTime", "WheelsOff", "WheelsOn", "CRSArrTime", "ArrTime"]
    interval_cols = ["DepTimeBlk", "ArrTimeBlk"]

    timestamp_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    for col in time_cols:
        if col in df.columns and col not in timestamp_columns:
            df[col] = df[col].fillna(0).astype(int)
            df[f"{col}_Hour"] = df[col] // 100
            df[f"{col}_Minute"] = df[col] % 100

    for col in interval_cols:
        if col in df.columns and col not in timestamp_columns:
            parts = df[col].str.extract(r"(?P<start>\d{4})-(?P<end>\d{4})")
            df[f"{col}_Hour_start"] = parts['start'].astype(int) // 100
            df[f"{col}_Minute_start"] = parts['start'].astype(int) % 100
            df[f"{col}_Hour_finish"] = parts['end'].astype(int) // 100
            df[f"{col}_Minute_finish"] = parts['end'].astype(int) % 100

    # Drop raw time/interval cols
    to_drop = [c for c in time_cols + interval_cols if c in df.columns and c not in timestamp_columns]
    df.drop(columns=to_drop, inplace=True)
    return df


def optimize_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include='float'):
        series = df[col]
        if (series.dropna() == series.dropna().astype(int)).all():
            df[col] = series.astype('Int64')
    return df


def main():
    project_id = get_env_var('GCP_PROJECT_ID')
    dataset_id = 'flights'
    directory = 'flight_delay'

    parquet_files = find_parquet_files(directory)
    if not parquet_files:
        print("No matching parquet files found. Exiting.")
        return

    for path in tqdm(parquet_files, desc="Uploading Parquet files to BigQuery"):
        year = os.path.basename(path)[len('Combined_Flights_'):-len('.parquet')]
        load_to_bigquery(path, project_id, dataset_id, f"flight_{year}")

    print("All files processed.")


if __name__ == '__main__':
    main()
