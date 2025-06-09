import os
import argparse
import pandas as pd
from google.cloud import bigquery

def load_columns_from_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        columns = [line.strip() for line in f if line.strip()]
    return columns

def load_data(project_id, columns):
    client = bigquery.Client(project=project_id)
    cols = ['Cancelled'] + columns
    column_str = ", ".join(cols)
    table = f"{project_id}.flights.flights_all"
    query = f"SELECT {column_str} FROM `{table}`"
    print(f"Loading data for columns: {cols}")
    df = client.query(query).to_arrow().to_pandas()
    print(f"Loaded {df.shape[0]} rows")
    return df

def summarize_missing_zero(df, columns):
    """
    Returns a DataFrame with one row per column and six summary columns:
      Present_0, Missing_0, Zero_0, Present_1, Missing_1, Zero_1
    """
    missing_strings = {'nan', 'NaN', '-', 'None'}
    records = []

    for col in columns:
        if col not in df.columns:
            print(f"Column {col} not found, skipping.")
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        rec = {'column': col}

        for status in (0, 1):
            grp = df[df['Cancelled'] == status]
            total = len(grp)

            if is_numeric:
                m = grp[col].isna().sum()
                z = (grp[col] == 0).sum()
            else:
                m = grp[col].isna().sum() + grp[col].isin(missing_strings).sum()
                z = (grp[col] == '').sum()

            p = total - m - z

            rec[f'Missing_{status}'] = m
            rec[f'Zero_{status}']    = z
            rec[f'Present_{status}'] = p

        records.append(rec)

    summary_df = pd.DataFrame.from_records(records)
    # reorder columns
    summary_df = summary_df.set_index('column')[
        ['Present_0','Missing_0','Zero_0','Present_1','Missing_1','Zero_1']
    ]
    return summary_df

def main():
    parser = argparse.ArgumentParser(
        description='Generate missing/zero/present summary CSV for specified columns.'
    )
    parser.add_argument('--columns_file', type=str, default='columns_to_visualize.txt',
                        help='File listing column names to analyze')
    parser.add_argument('--output_csv', type=str, default='missing_zero_summary.csv',
                        help='Path to write the summary CSV')
    args = parser.parse_args()

    project_id = os.environ.get('GCP_PROJECT_ID')
    if not project_id:
        raise ValueError('GCP_PROJECT_ID env var not set')

    columns = load_columns_from_file(args.columns_file)
    df = load_data(project_id, columns)
    summary_df = summarize_missing_zero(df, columns)

    summary_df.to_csv(args.output_csv, index_label='Column')
    print(f"Summary written to {args.output_csv}")

if __name__ == "__main__":
    main()
