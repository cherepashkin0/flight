import polars as pl
import os
from tabulate import tabulate

# Add this function to standardize schemas across DataFrames
def standardize_schema(dfs):
    """Ensure all DataFrames have compatible schema for concatenation."""
    # Get a list of all columns across all DataFrames
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    # Determine common data types based on the majority
    column_types = {}
    for col in all_columns:
        types = []
        for df in dfs:
            if col in df.columns:
                types.append(str(df.schema[col]))
        
        # Default to more flexible types (Float32 over Int8, etc.)
        if any("Float" in t for t in types):
            column_types[col] = pl.Float32
        elif any("Int" in t for t in types):
            column_types[col] = pl.Int32
        else:
            # For string, datetime, etc. - use as is from first occurrence
            for df in dfs:
                if col in df.columns:
                    column_types[col] = df.schema[col]
                    break
    
    # Standardize each DataFrame
    standardized_dfs = []
    for df in dfs:
        # Prepare expressions for casting columns
        cast_expressions = []
        for col, dtype in column_types.items():
            if col in df.columns and str(df.schema[col]) != str(dtype):
                cast_expressions.append(pl.col(col).cast(dtype))
        
        # Apply casting if needed
        if cast_expressions:
            df = df.with_columns(cast_expressions)
        
        standardized_dfs.append(df)
    
    return standardized_dfs

def optimize_flight_data():
    """Process flight data for years 2018-2022 and merge into a single optimized parquet file"""
    # Define columns to drop (instead of columns to keep)
    columns_to_drop = [
        "__index_level_0__",
        "Year",
        "Tail_Number",
        "Flight_Number_Operating_Airline",
        "Flight_Number_Marketing_Airline",
        "DOT_ID_Marketing_Airline",
        "DOT_ID_Operating_Airline",
        "IATA_Code_Marketing_Airline",
        "IATA_Code_Operating_Airline",
        "OriginAirportID",
        "OriginAirportSeqID",
        "OriginCityMarketID",
        "DestAirportID",
        "DestAirportSeqID",
        "DestCityMarketID",
        "CRSDepTime",
        "DepTime",
        "WheelsOff",
        "WheelsOn",
        "CRSArrTime",
        "ArrTime"]   
                           
    # Define base directory for flight data
    base_dir = 'flight_delay/'
    
    # List to store DataFrames for each year
    all_dfs = []
    
    # Summary results for final report
    summary_results = []
    
    print("Processing flight data for years 2018-2022...")
    
    # Process each year
    for year in range(2018, 2023):
        input_file = f"{base_dir}Combined_Flights_{year}.parquet"
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue
        
        print(f"\nProcessing flights data for {year}...")
        
        # Load the parquet file
        df = pl.read_parquet(input_file)
        
        # Get initial stats
        initial_columns = df.columns
        initial_memory = df.estimated_size() / (1024 * 1024)  # MB
        print(f"Initial columns: {len(initial_columns)}, Memory usage: {initial_memory:.2f} MB")
        
        # Drop specified columns if they exist in the dataset
        columns_to_remove = [col for col in columns_to_drop if col in df.columns]
        if columns_to_remove:
            df = df.drop(columns_to_remove)
            print(f"Dropped {len(columns_to_remove)} columns")
        else:
            print("No columns were dropped (empty drop list)")
        
        # Analyze and optimize data types
        dtype_optimizations = {}
        
        for col in df.columns:
            col_dtype = str(df.schema[col])
            
            # For integer columns
            if "Int64" in col_dtype:
                min_val = df.select(pl.col(col).min()).item()
                max_val = df.select(pl.col(col).max()).item()
                
                if min_val is not None and max_val is not None:
                    if min_val >= -128 and max_val <= 127:
                        dtype_optimizations[col] = pl.Int8
                    elif min_val >= -32768 and max_val <= 32767:
                        dtype_optimizations[col] = pl.Int16
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        dtype_optimizations[col] = pl.Int32
            
            # For floating point columns
            elif "Float64" in col_dtype:
                # Check if it can be represented as Float32 without loss of precision
                dtype_optimizations[col] = pl.Float32
            
            # For binary columns (Cancelled, Diverted, DepDel15)
            elif col in ["Cancelled", "Diverted", "DepDel15"]:
                # Check if these are already boolean or can be converted
                unique_values = df.select(pl.col(col).unique()).to_series().to_list()
                if len(unique_values) <= 2:
                    dtype_optimizations[col] = pl.Boolean
        
        # Apply optimized dtypes
        cast_expressions = []
        for col, dtype in dtype_optimizations.items():
            cast_expressions.append(pl.col(col).cast(dtype))
        
        if cast_expressions:
            df = df.with_columns(cast_expressions)
            print(f"Applied type optimizations to {len(cast_expressions)} columns")
        
        # Get final stats
        final_memory = df.estimated_size() / (1024 * 1024)  # MB
        memory_reduction = initial_memory/final_memory if final_memory > 0 else 0
        
        print(f"Final columns: {len(df.columns)}, Memory usage: {final_memory:.2f} MB")
        print(f"Memory reduction: {memory_reduction:.2f}x")
        
        # Add year column if not present
        if "Year" not in df.columns:
            df = df.with_columns(pl.lit(year).alias("Year"))
        
        # Print optimized data types
        print("\nOptimized data types:")
        for col in df.columns:
            print(f"{col}: {df.schema[col]}")
        
        # Store summary data
        summary_results.append([
            year,
            len(initial_columns),
            len(df.columns),
            f"{initial_memory:.2f} MB",
            f"{final_memory:.2f} MB",
            f"{memory_reduction:.2f}x"
        ])
        
        # Add to list of DataFrames
        all_dfs.append(df)
    
    # Concatenate all DataFrames
 
    if all_dfs:
        print("\nStandardizing schemas across all years...")
        all_dfs = standardize_schema(all_dfs)

        print("\nMerging data from all years...")
        merged_df = pl.concat(all_dfs)
        
        # Save merged data
        output_file = f"{base_dir}flights_optimized.parquet"  # Fixed typo in filename
        merged_df.write_parquet(output_file)
        
        merged_size = merged_df.estimated_size() / (1024 * 1024)  # MB
        print(f"Saved merged file to {output_file}")
        print(f"Total merged size: {merged_size:.2f} MB")
        print(f"Total rows: {merged_df.height}")
    else:
        print("No data to merge!")
    
    # Print summary table for all years
    print("\n\nSummary of optimizations:")
    print(tabulate(summary_results,
                  headers=["Year", "Initial Columns", "Final Columns", "Initial Memory", "Final Memory", "Reduction Factor"],
                  tablefmt="grid"))

if __name__ == '__main__':
    optimize_flight_data()
