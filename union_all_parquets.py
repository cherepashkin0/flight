import pandas as pd
import os
import gc
from tqdm import tqdm

def main():
    output_file = 'flight_delay/flights_all.parquet'
    
    # Remove output file if it exists to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Create an empty dataframe with the correct schema first
    df_full = None
    
    # Process each year's data
    for year in tqdm(range(2018, 2023), desc="Processing years"):
        parquet_path = f'flight_delay/Combined_Flights_{year}.parquet'
        print(f"Processing {parquet_path}...")
        
        try:
            # Read the current year's data
            df = pd.read_parquet(parquet_path)
            
            # For the first file, just save it directly
            if df_full is None:
                df.to_parquet(output_file, index=False)
                df_full = True  # Just a flag, not actually storing data
            else:
                # For subsequent files, use fastparquet's append option
                df.to_parquet(
                    output_file,
                    engine='fastparquet',  # Use fastparquet for append support
                    index=False,
                    compression='snappy',
                    append=True  # Use append instead of mode
                )
            
            # Free memory explicitly
            del df
            gc.collect()
            print(f"Successfully processed and appended data for {year}")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
    
    print(f"All data has been processed and saved to {output_file}")

if __name__ == '__main__':
    main()
