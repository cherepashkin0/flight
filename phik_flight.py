import os
import warnings
import json
import pandas as pd
from google.cloud import bigquery
from phik import phik_matrix
warnings.filterwarnings('ignore')

ROW_LIMIT = 1000000  # Limit for BigQuery sampling
# Get GCP project ID from environment variable
project_id = os.environ.get('GCP_PROJECT_ID')
if not project_id:
    raise ValueError("GCP_PROJECT_ID environment variable not set. Please set it before running this script.")

# Create output directory
output_dir = "flight_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Read list of columns to drop
# drop_file = 'columns_to_drop.txt'
# if not os.path.isfile(drop_file):
#     raise FileNotFoundError(f"Drop-list file '{drop_file}' not found.")
# with open(drop_file, 'r') as f:
#     columns_to_drop = [line.strip() for line in f if line.strip()]

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
            ROUND({ROW_LIMIT} * SAFE_DIVIDE(total, SUM(total) OVER ())) AS n_sample
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
print(f"Sampled data loaded into pandas DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop unwanted columns
# drop_present = [c for c in columns_to_drop if c in df.columns]
# if not drop_present:
#     print("No columns from drop list found in dataset.")
# else:
#     df = df.drop(columns=drop_present)
#     print(f"Dropped columns: {drop_present}")

# Generate metadata for columns to determine phik types
print("Analyzing column metadata for phik type classification...")
metadata = []
for col in df.columns:
    # Convert datetime columns to string to avoid phik conversion issues
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        print(f"Converting datetime column '{col}' to string format")
        df[col] = df[col].astype(str)
        
    col_data = {
        'Column_Name': col,
        'Data_Type': str(df[col].dtype),
        'Unique_Values_Count': df[col].nunique()
    }
    metadata.append(col_data)

df_meta = pd.DataFrame(metadata)

# Classify columns based on heuristics
def classify_column(row):
    if 'bool' in row['Data_Type'].lower():
        return 'nominal'
    elif 'object' in row['Data_Type'].lower() or 'string' in row['Data_Type'].lower():
        return 'nominal'
    elif 'datetime' in row['Data_Type'].lower() or 'timestamp' in row['Data_Type'].lower():
        return 'nominal'  # Changed to nominal to avoid type conversion issues
    elif 'int' in row['Data_Type'].lower() or 'float' in row['Data_Type'].lower():
        if row['Unique_Values_Count'] <= 11:
            return 'ordinal'
        else:
            return 'interval'
    else:
        return 'nominal'  # Default to nominal for unknown types

def find_correlated_features(correlation_matrix, target_column, top_n=100):
    """
    Analyze correlation matrix to find highly correlated feature pairs and determine candidates for dropping.
    
    Parameters:
    correlation_matrix (pd.DataFrame): The correlation matrix (features and target)
    target_column (str): The name of the target variable in the correlation matrix
    top_n (int): Number of top correlated pairs to return
    
    Returns:
    tuple: (top_pairs_df, drop_candidates)
    """
    # Create a copy to avoid modifying the original
    corr_matrix = correlation_matrix.copy()
    
    # Get feature columns (all columns except target)
    feature_columns = [col for col in corr_matrix.columns if col != target_column]
    
    # Create a list to store pairs and their correlations
    pairs_data = []
    
    # Find all pairs of features and their correlations
    for i, col1 in enumerate(feature_columns):
        for col2 in feature_columns[i+1:]:  # Start from i+1 to avoid duplicates
            correlation = abs(corr_matrix.loc[col1, col2])  # Use absolute correlation
            
            # Get correlations with target
            corr1_with_target = abs(corr_matrix.loc[col1, target_column])
            corr2_with_target = abs(corr_matrix.loc[col2, target_column])
            
            # Create a row for this pair
            pairs_data.append({
                'Feature1': col1,
                'Feature2': col2,
                'Correlation': correlation,
                'Feature1_Target_Corr': corr1_with_target,
                'Feature2_Target_Corr': corr2_with_target
            })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_data)
    
    # Sort by correlation value (descending)
    pairs_df = pairs_df.sort_values('Correlation', ascending=False).reset_index(drop=True)
    
    # Get top N pairs
    top_pairs_df = pairs_df.head(top_n)
    
    # Determine candidates for dropping
    # For each pair, the feature with lower correlation with target is a candidate for dropping
    drop_candidates = []
    
    for _, row in top_pairs_df.iterrows():
        if row['Correlation'] > 0.7:  # Only consider highly correlated features (threshold can be adjusted)
            if row['Feature1_Target_Corr'] < row['Feature2_Target_Corr']:
                drop_candidates.append(row['Feature1'])
            else:
                drop_candidates.append(row['Feature2'])
    
    # Remove duplicates from drop_candidates
    drop_candidates = list(set(drop_candidates))
    
    return top_pairs_df, drop_candidates

def main():
    df_meta['Phik_Type'] = df_meta.apply(classify_column, axis=1)
    # Extract interval and ordinal columns based on classification
    interval_columns = df_meta[df_meta['Phik_Type'] == 'interval']['Column_Name'].tolist()
    ordinal_columns = df_meta[df_meta['Phik_Type'] == 'ordinal']['Column_Name'].tolist()

    print(f"Automatically classified {len(interval_columns)} interval columns and {len(ordinal_columns)} ordinal columns")
    print(f"Interval columns: {interval_columns[:5]}{'...' if len(interval_columns) > 5 else ''}")
    print(f"Ordinal columns: {ordinal_columns[:5]}{'...' if len(ordinal_columns) > 5 else ''}")

    # Save column classification
    df_meta.to_csv(f"{output_dir}/column_classifications.csv", index=False)

    # Extra safeguard: check that interval columns can be converted to float
    safe_interval_columns = []
    for col in interval_columns:
        try:
            # Just test conversion on first few values
            test_values = df[col].head(5)
            test_values.astype(float)
            safe_interval_columns.append(col)
        except (ValueError, TypeError):
            print(f"Warning: Column '{col}' cannot be converted to float, treating as nominal instead")
            # Update the metadata to match
            df_meta.loc[df_meta['Column_Name'] == col, 'Phik_Type'] = 'nominal'

    interval_columns = safe_interval_columns

    # Ensure required column exists
    if 'Cancelled' not in df.columns:
        raise KeyError("'Cancelled' column not found in the dataset after dropping.")

    # Fix any potential datetime columns that might have been missed
    for col in interval_columns:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Converting interval column '{col}' from datetime to string")
            df[col] = df[col].astype(str)
            # Move from interval to nominal columns
            interval_columns.remove(col)
            if col not in df_meta[df_meta['Phik_Type'] == 'nominal']['Column_Name'].tolist():
                df_meta.loc[df_meta['Column_Name'] == col, 'Phik_Type'] = 'nominal'

    # Print the final column type counts
    nominal_columns = df_meta[df_meta['Phik_Type'] == 'nominal']['Column_Name'].tolist()
    print(f"Final column type counts: {len(interval_columns)} interval, {len(ordinal_columns)} ordinal, {len(nominal_columns)} nominal")

    # Compute phik correlation matrix
    print("Calculating phik correlation matrix on sampled dataset...")
    phik_corr = phik_matrix(
        df,
        interval_cols=interval_columns,
        dropna=True,
        verbose=1
    )
    print("Phik correlation matrix calculation complete.")

    # Extract correlations with 'Cancelled'
    cancelled_corr = phik_corr['Cancelled'].sort_values(ascending=False)

    # Save outputs
    phik_corr.to_csv(f"{output_dir}/sampled_phik_correlation_matrix.csv")
    cancelled_corr.to_csv(f"{output_dir}/cancelled_correlations_sampled.csv")

    # Categorize correlations
    extreme_high = cancelled_corr[(cancelled_corr > 0.9) & (cancelled_corr < 1.0)]
    good = cancelled_corr[(cancelled_corr >= 0.3) & (cancelled_corr <= 0.9)]
    extreme_low = cancelled_corr[cancelled_corr < 0.05]

    categories = {
        "extreme_high_correlation": extreme_high.index.tolist(),
        "good_correlation": good.index.tolist(),
        "extreme_low_correlation": extreme_low.index.tolist()
    }

    with open(f"{output_dir}/correlation_categories_sampled.json", 'w') as jf:
        json.dump(categories, jf, indent=4)

    # Save low correlation columns to a separate file
    low_corr_file = os.path.join("columns_to_drop2.txt")
    with open(low_corr_file, 'w') as f:
        for col in categories["extreme_low_correlation"]:
            f.write(f"{col}\n")

    print(f"Low correlation columns saved to: {low_corr_file}")
    print("Results saved to output directory.")

if __name__ == "__main__":
    main()

