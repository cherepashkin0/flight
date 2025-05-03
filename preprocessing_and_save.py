import os
import pandas as pd
import numpy as np
import json
import clickhouse_connect
from sklearn.preprocessing import PowerTransformer

# -------------------------------
# Config
# -------------------------------
DB_NAME = 'flights_data'
INPUT_TABLE = f'{DB_NAME}.flight_all'
OUTPUT_TABLE = f'{DB_NAME}.flight_all_preprocessed'
TARGET_COLUMN = 'Cancelled'
CONFIG_FILE = 'flight_figs/columns_info.json'

# -------------------------------
# Helpers
# -------------------------------
def connect_to_clickhouse():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST'),
        port=os.getenv('CLICKHOUSE_PORT'),
        username=os.getenv('CLICKHOUSE_USER'),
        password=os.getenv('CLICKHOUSE_PASSWORD')
    )

def read_column_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config["columns_to_load"], config.get("columns_to_drop", [])

def load_selected_columns(client, table, use_columns):
    query = f"SELECT {', '.join(use_columns)} FROM {table} WHERE {TARGET_COLUMN} IS NOT NULL"
    return client.query_df(query)

def convert_booleans_and_categories(df):
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            df[col] = df[col].map({True: 1, False: 0}) if df[col].dtype == 'bool' else pd.Categorical(df[col]).codes
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.Categorical(df[col]).codes
    return df

def apply_yeo_johnson_chunked(df, target_column, chunk_size=10000):
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]

    transformer = PowerTransformer(method='yeo-johnson')
    
    # Fit on a sample (or full numeric part if small enough)
    sample = df[numeric_cols].dropna().sample(min(100000, len(df)), random_state=42)
    transformer.fit(sample)

    # Apply in chunks
    transformed_chunks = []
    for start in range(0, len(df), chunk_size):
        end = start + chunk_size
        chunk = df.iloc[start:end].copy()
        chunk[numeric_cols] = transformer.transform(chunk[numeric_cols])
        transformed_chunks.append(chunk)

    return pd.concat(transformed_chunks, ignore_index=True)


def process_datetime_columns(df):
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[s, Etc/UTC]']).columns
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_dayofyear"] = df[col].dt.dayofyear
        df.drop(columns=[col], inplace=True)
    return df

def save_to_clickhouse(df, table_name, client):
    # Build CREATE TABLE SQL from DataFrame dtypes
    type_mapping = {
        'int8': 'Int8',
        'int16': 'Int16',
        'int32': 'Int32',
        'int64': 'Int64',
        'uint8': 'UInt8',
        'uint16': 'UInt16',
        'uint32': 'UInt32',
        'uint64': 'UInt64',
        'float32': 'Float32',
        'float64': 'Float64',
        'object': 'String',
        'bool': 'UInt8',
        'category': 'UInt8'
    }

    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        ch_type = type_mapping.get(dtype, 'String')  # Default to String
        columns.append(f"`{col}` {ch_type}")
    
    schema_sql = ", ".join(columns)
    create_sql = f"CREATE TABLE {table_name} ({schema_sql}) ENGINE = MergeTree() ORDER BY tuple()"

    print(f"📐 Creating table {table_name}...")
    client.command(f"DROP TABLE IF EXISTS {table_name}")
    client.command(create_sql)

    print("📤 Inserting data...")
    client.insert_df(table_name, df)

# -------------------------------
# Main
# -------------------------------
def main():
    print("🔌 Connecting to ClickHouse...")
    client = connect_to_clickhouse()

    print("📖 Reading column config...")
    cols_to_load, cols_to_drop = read_column_config(CONFIG_FILE)
    cols_to_load = [col for col in cols_to_load if col not in cols_to_drop]

    print("📥 Loading data from DB...")
    df = load_selected_columns(client, INPUT_TABLE, cols_to_load)

    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    df = df.dropna(subset=[TARGET_COLUMN])

    print("🔄 Converting columns...")
    df = convert_booleans_and_categories(df)
    df = process_datetime_columns(df)
    print("🧼 Applying Yeo-Johnson transform...")
    df = apply_yeo_johnson_chunked(df, TARGET_COLUMN)
    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Target column contains invalid values after transform!"

    print(f"💾 Saving preprocessed data to ClickHouse table: {OUTPUT_TABLE}")
    save_to_clickhouse(df, OUTPUT_TABLE, client)

if __name__ == "__main__":
    main()