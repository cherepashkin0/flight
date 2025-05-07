import polars as pl
import os

def analyze_parquet_file(file_path, output_path=None):
    """
    Analyze a single parquet file and save detailed column information to CSV
    
    Args:
        file_path: Path to the parquet file to analyze
        output_path: Path where to save the CSV file (defaults to same directory as input)
    """
    print(f"Analyzing file: {file_path}")
    
    # Read the parquet file
    df = pl.read_parquet(file_path)
    
    # Create output path if not provided
    if output_path is None:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).split('.')[0]
        output_path = os.path.join(file_dir, f"{file_name}_analysis.csv")
    
    # Store results for each column
    results = []
    
    # Analyze each column
    for col_name in df.columns:
        # Get column data type
        col_type = str(df.schema[col_name])
        
        # Calculate statistics
        min_val = df.select(pl.col(col_name).min()).item()
        max_val = df.select(pl.col(col_name).max()).item()
        unique_count = df.select(pl.col(col_name).n_unique()).item()
        
        # Get sample of unique non-null values (up to 5)
        sample_expr = (
            pl.col(col_name)
            .filter(pl.col(col_name).is_not_null())
            .unique()
            .limit(5)
        )
        
        sample_values = df.select(sample_expr).to_series().to_list()
        sample_str = str(sample_values)
        
        # Add results to table
        results.append({
            "Column_Name": col_name,
            "Data_Type": col_type,
            "Min_Value": str(min_val),  # Convert to string to handle various types
            "Max_Value": str(max_val),
            "Unique_Values_Count": unique_count,
            "Sample_Values": sample_str
        })
    
    # Convert results to DataFrame
    results_df = pl.DataFrame(results)
    
    # Save to CSV
    results_df.write_csv(output_path)
    print(f"Analysis saved to: {output_path}")

def describe2():
    # Specify the file path - update this to your file path
    file_path = 'flight_delay/Combined_Flights_2022.parquet'
    
    # Analyze the file and save results to CSV
    analyze_parquet_file(file_path)

if __name__ == '__main__':
    describe2()
