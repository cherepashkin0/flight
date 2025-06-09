import os
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from google.cloud import bigquery
from tqdm import tqdm  # NEW: progress bar

ROW_LIMIT = 1000000  # Limit for BigQuery sampling
ROW_FRACTION = 0.01  # Fraction of rows to sample from BigQuery

def plot_object_column_pairs(df, output_pdf_path, max_unique=20, max_plots=50):
    """
    Generate contingency heatmaps for every pair of object columns in the DataFrame.
    Saves all plots to a single PDF, with progress bar.

    Parameters:
    - df: pd.DataFrame, your data
    - output_pdf_path: str, where to save the PDF
    - max_unique: int, skip columns with too many unique values for plotting
    - max_plots: int, maximum number of plots to generate
    """
    obj_cols = [col for col in df.columns if df[col].dtype == 'object']
    pairs = list(itertools.combinations(obj_cols, 2))
    plots_created = 0

    if not pairs:
        print("No pairs of object columns to plot.")
        return

    print(f"Generating contingency plots for {len(pairs)} object column pairs...")

    with PdfPages(output_pdf_path) as pdf:
        for col1, col2 in tqdm(pairs[:max_plots], desc="Plotting", unit="pair"):
            unique1 = df[col1].nunique()
            unique2 = df[col2].nunique()
            if unique1 > max_unique or unique2 > max_unique:
                continue  # Skip if too many unique categories

            contingency = pd.crosstab(df[col1], df[col2])

            plt.figure(figsize=(min(12, 1.5 + 0.5*unique2), min(8, 1.5 + 0.4*unique1)))
            sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar=True)
            plt.title(f'Contingency Table: {col1} vs {col2}')
            plt.xlabel(col2)
            plt.ylabel(col1)
            plt.tight_layout()

            pdf.savefig()
            plt.close()

            plots_created += 1

    print(f"Saved {plots_created} object column pair contingency plots to {output_pdf_path}")

def main():
    # Get GCP project ID from environment variable
    project_id = os.environ.get('GCP_PROJECT_ID')
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set. Please set it before running this script.")

    output_dir = "flight_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Connecting to BigQuery using project ID: {project_id}")
    client = bigquery.Client(project=project_id)

    # Query to perform stratified sampling (equal size from Cancelled=0 and Cancelled=1)
    query = f"""
        SELECT *
        FROM `{project_id}.flights.flights_all`
        WHERE RAND() < {ROW_FRACTION} -- adjust fraction for desired sample size (e.g. 1% sample)
        LIMIT 1000000        -- optional, to cap max rows
    """

    print("Loading stratified sample from BigQuery...")
    arrow_table = client.query(query).to_arrow()
    df = arrow_table.to_pandas()
    print(f"Sampled data loaded into pandas DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # Convert datetime columns to string (object), if any, for consistency
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Converting datetime column '{col}' to string format for plotting")
            df[col] = df[col].astype(str)

    pdf_file = os.path.join(output_dir, "object_pairs_contingency_plots.pdf")
    plot_object_column_pairs(df, pdf_file, max_unique=200, max_plots=1000)

if __name__ == "__main__":
    main()
