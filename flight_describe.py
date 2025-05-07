import polars as pl
import os
import glob

def analyze_flight_data_2022():
    """Analyze 2022 flight data to identify columns to drop and dtype optimizations"""
    print("Analyzing 2022 flight data...")
    
    input_file = 'flight_delay/Combined_Flights_2022.parquet'
    df = pl.read_parquet(input_file)
    
    print("\nDataFrame Schema:")
    print(df.schema)
    
    # Get column dtypes
    dtypes = {name: str(dtype) for name, dtype in zip(df.columns, df.dtypes)}
    
    # Identify non-numeric columns
    numeric_dtypes = ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", 
                      "Float32", "Float64", "Decimal"]
    temporal_dtypes = ["Date", "Datetime", "Time"]
    
    non_numeric_columns = [
        col for col, dtype_str in dtypes.items() 
        if not any(num_type in dtype_str for num_type in numeric_dtypes) and 
           not any(temp_type in dtype_str for temp_type in temporal_dtypes)
    ]
    
    print("\nNon-numeric columns:")
    print(sorted(non_numeric_columns))
    
    # Try to convert non-numeric columns to numeric
    numeric_convertible = []
    for col in non_numeric_columns:
        try:
            # Test if column can be converted to numeric
            df.select(pl.col(col).cast(pl.Float64, strict=False))
            numeric_convertible.append(col)
        except Exception as e:
            print(f"Error converting column {col}: {e}")
    
    # Identify binary columns among remaining non-numeric columns
    binary_columns = []
    for col in [c for c in non_numeric_columns if c not in numeric_convertible]:
        unique_count = df.select(pl.col(col).n_unique()).item()
        if unique_count <= 2:
            binary_columns.append(col)
    
    # Columns to drop: non-numeric, non-binary, and not convertible to numeric
    columns_to_drop = [
        col for col in non_numeric_columns 
        if col not in binary_columns and col not in numeric_convertible
    ]
    
    print("\nBinary columns (kept):")
    print(sorted(binary_columns))
    
    print("\nColumns to drop (non-numeric, non-binary, not convertible):")
    print(sorted(columns_to_drop))
    
    # Drop identified columns to analyze the remaining ones
    df_filtered = df.drop(columns_to_drop)
    
    # Analyze optimal dtypes for numeric columns
    dtype_optimizations = {}
    
    for col in df_filtered.columns:
        col_dtype = str(df_filtered.schema[col])
        
        # For integer columns
        if "Int64" in col_dtype:
            unique_count = df_filtered.select(pl.col(col).n_unique()).item()
            min_val = df_filtered.select(pl.col(col).min()).item()
            max_val = df_filtered.select(pl.col(col).max()).item()
            
            if min_val is not None and max_val is not None:
                if min_val >= -32768 and max_val <= 32767 and unique_count < 2**16:
                    dtype_optimizations[col] = pl.Int16
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    dtype_optimizations[col] = pl.Int32
        
        # For floating point columns
        elif "Float64" in col_dtype:
            unique_count = df_filtered.select(pl.col(col).n_unique()).item()
            if unique_count < 2**16:
                dtype_optimizations[col] = pl.Float32
    
    print("\nDtype optimizations:")
    print(dtype_optimizations)
    
    # Print min and max for numerical columns
    numerical_stats = {}
    for col in df_filtered.columns:
        col_dtype = str(df_filtered.schema[col])
        if any(num_type in col_dtype for num_type in numeric_dtypes):
            min_val = df_filtered.select(pl.col(col).min()).item()
            max_val = df_filtered.select(pl.col(col).max()).item()
            distinct = df_filtered.select(pl.col(col).n_unique()).item()
            numerical_stats[col] = {"min": min_val, "max": max_val, "n_unique": distinct}
    
    print("\nMin and Max for numerical columns:")
    for col, stats in sorted(numerical_stats.items()):
        print(f"{col}: min={stats['min']}, max={stats['max']}, unique values={stats['n_unique']}")
    
    return columns_to_drop, dtype_optimizations

def optimize_all_flight_data(columns_to_drop, dtype_optimizations):
    """Apply transformations to all flight data from 2018 to 2022"""
    print("\n\nApplying transformations to all flight data (2018-2022)...\n")
    
    # Define base directory for flight data
    base_dir = 'flight_delay/'
    
    # Process each year
    for year in range(2018, 2023):
        input_file = f"{base_dir}Combined_Flights_{year}.parquet"
        output_file = f"{base_dir}Combined_Flights_{year}_optimized.parquet"
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue
        
        print(f"\nProcessing flights data for {year}...")
        
        # Load the parquet file using all available cores
        df = pl.scan_parquet(input_file)
        
        # Get initial column count
        initial_df = df.collect()
        initial_columns = initial_df.columns
        initial_memory = initial_df.estimated_size() / (1024 * 1024)  # MB
        print(f"Initial columns: {len(initial_columns)}, Memory usage: {initial_memory:.2f} MB")
        
        # Drop non-numeric, non-binary columns
        # First filter to only drop columns that actually exist in this year's data
        cols_to_drop = [col for col in columns_to_drop if col in initial_columns]
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns: {sorted(cols_to_drop)}")
        
        # Apply optimized dtypes
        cast_expressions = []
        for col, dtype in dtype_optimizations.items():
            if col in initial_columns:
                cast_expressions.append(pl.col(col).cast(dtype))
        
        if cast_expressions:
            df = df.with_columns(cast_expressions)
            print(f"Applied type optimizations to {len(cast_expressions)} columns")
        
        # Materialize the lazy DataFrame to execute the transformations
        df_result = df.collect()
        
        # Get final stats
        final_columns = df_result.columns
        final_memory = df_result.estimated_size() / (1024 * 1024)  # MB
        print(f"Final columns: {len(final_columns)}, Memory usage: {final_memory:.2f} MB")
        print(f"Memory reduction: {initial_memory/final_memory:.2f}x")
        
        # Save optimized file using all available cores for parallel writing
        df_result.write_parquet(output_file)
        print(f"Saved optimized file to {output_file}")
        
        # Print final columns in alphabetical order
        print("Final columns:")
        print(sorted(final_columns))

def main():
    # First analyze 2022 data to identify transformations
    columns_to_drop, dtype_optimizations = analyze_flight_data_2022()
    
    # Then apply those transformations to all years
    optimize_all_flight_data(columns_to_drop, dtype_optimizations)

if __name__ == '__main__':
    main()
