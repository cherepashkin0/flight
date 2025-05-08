import os
import polars as pl
from google.cloud import bigquery

def analyze_bigquery_table(
    table_id: str,
    project: str = None,
    output_path: str = None
):
    """
    Analyze a BigQuery table and save detailed column information to CSV,
    processing all columns in a single query.
    
    Args:
        table_id: Full table identifier 'dataset.table' or 'project.dataset.table'
        project: GCP project ID (if not included in table_id)
        output_path: Where to save the CSV (defaults to './<table>_analysis.csv')
    """
    if output_path is None:
        table_name = table_id.split('.')[-1]
        output_path = f"{table_name}_analysis.csv"
    
    # Initialize client
    bq_client = bigquery.Client(project=project)
    
    # Get table schema first
    table = bq_client.get_table(table_id)
    columns = [field.name for field in table.schema]
    field_types = {field.name: field.field_type for field in table.schema}
    
    print(f"Analyzing {len(columns)} columns...")
    
    # Generate a query that creates one row per column with all statistics
    column_stats = []
    for col_name in columns:
        col_type = field_types[col_name]
        
        if col_type in ['INTEGER', 'FLOAT', 'NUMERIC', 'BIGNUMERIC']:
            # For numeric columns, include mean and std_dev
            column_stats.append(f"""
            SELECT 
                '{col_name}' AS Column_Name,
                '{col_type}' AS Data_Type,
                CAST(MIN({col_name}) AS STRING) AS Min_Value,
                CAST(MAX({col_name}) AS STRING) AS Max_Value,
                COUNT(DISTINCT {col_name}) AS Unique_Values_Count,
                COUNT(*) - COUNT({col_name}) AS Missing_Values_Count,
                AVG({col_name}) AS Mean_Value,
                STDDEV({col_name}) AS Std_Dev,
                (SELECT STRING_AGG(CAST(sample AS STRING), ', ')
                 FROM (
                     SELECT DISTINCT {col_name} AS sample
                     FROM `{table_id}`
                     WHERE {col_name} IS NOT NULL
                     LIMIT 5
                 )) AS Sample_Values
            FROM `{table_id}`
            """)
        else:
            # For non-numeric columns, use null for mean and std_dev
            column_stats.append(f"""
            SELECT 
                '{col_name}' AS Column_Name,
                '{col_type}' AS Data_Type,
                CAST(MIN({col_name}) AS STRING) AS Min_Value,
                CAST(MAX({col_name}) AS STRING) AS Max_Value,
                COUNT(DISTINCT {col_name}) AS Unique_Values_Count,
                COUNT(*) - COUNT({col_name}) AS Missing_Values_Count,
                CAST(NULL AS FLOAT64) AS Mean_Value,
                CAST(NULL AS FLOAT64) AS Std_Dev,
                (SELECT STRING_AGG(CAST(sample AS STRING), ', ')
                 FROM (
                     SELECT DISTINCT {col_name} AS sample
                     FROM `{table_id}`
                     WHERE {col_name} IS NOT NULL
                     LIMIT 5
                 )) AS Sample_Values
            FROM `{table_id}`
            """)
    
    # Union all the per-column queries
    full_query = " UNION ALL ".join(column_stats)
    
    print("Executing analysis query...")
    results_df = bq_client.query(full_query).to_dataframe()
    
    # Write results to CSV
    pl_df = pl.DataFrame(results_df)
    pl_df.write_csv(output_path)
    print(f"Analysis saved to: {output_path}")

if __name__ == "__main__":
    # Read your project ID from the environment
    project_id = os.environ["GCP_PROJECT_ID"]
    
    # Compose the full table identifier
    table_id = f"{project_id}.flights.flights_all"
    analyze_bigquery_table(table_id, project=project_id)
