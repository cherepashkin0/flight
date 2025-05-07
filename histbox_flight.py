import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Read drop-list files as before
def read_columns(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

columns_to_drop = []
for fname in (
    'flight_visualizations/ultra_high_corr.txt',
    'flight_visualizations/ultra_low_corr.txt'
):
    if os.path.exists(fname):
        columns_to_drop.extend(read_columns(fname))
    else:
        print(f"Warning: {fname} not found, skipping.")

# Manually exclude these columns from visualization
manual_exclude = [
    # e.g. 'flight_date', 'tail_number', 'carrier_code'
]

# Inspect parquet schema to get all column names WITHOUT loading data
parquet_path = 'flight_delay/flights_all.parquet'
pf = pq.ParquetFile(parquet_path)
all_columns = pf.schema.names

# Build the list of columns to keep (drop highly correlated or low-correlated)
columns_to_keep = [c for c in all_columns if c not in columns_to_drop]
print(f"Loading only these columns: {columns_to_keep!r}")

# Now read only the needed columns
df = pd.read_parquet(
    parquet_path,
    engine='pyarrow',
    columns=columns_to_keep
)
print(f"Dataframe shape after selective load: {df.shape}")

# Drop the specified columns (ignoring any that aren’t present)
df = df.drop(
    columns=[col for col in columns_to_drop if col in df.columns],
    errors='ignore'
)
print(f"Dataframe shape after dropping columns: {df.shape}")

# Create a PDF to save all plots
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)
pdf_path = os.path.join(output_dir, 'all_plots.pdf')
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# …

with PdfPages(pdf_path) as pdf:
    print("Generating plots for each column...")
    for column in df.columns:

        # Skip non-numeric columns
        if not is_numeric_dtype(df[column]):
            print(f"Skipping non-numeric column: {column}")
            continue

        # Skip boolean dtype columns
        if is_bool_dtype(df[column]):
            print(f"Skipping boolean column: {column}")
            continue

        # Skip numeric columns that only take 2 distinct values (e.g. Cancelled)
        # (dropna() so that NaNs don't count as an extra category)
        if df[column].dropna().nunique() == 2:
            print(f"Skipping binary column: {column}")
            continue

        # Skip any manually excluded columns
        if column in manual_exclude:
            print(f"Skipping manually excluded column: {column}")
            continue

        # ——— Proceed with plotting ———

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Analysis of {column}')

        sns.histplot(data=df, x=column, kde=True, ax=ax1)
        ax1.set_title(f'Histogram of {column}')

        sns.boxplot(x=df[column], ax=ax2)
        ax2.set_title(f'Box plot of {column}')

        plt.tight_layout()

        png_path = os.path.join(output_dir, f'{column}_plots.png')
        plt.savefig(png_path)
        print(f"Saved plot to {png_path}")

        pdf.savefig(fig)
        plt.close(fig)


print(f"All plots saved to {pdf_path}")
print("Data processing and visualization complete!")
