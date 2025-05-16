import os
from typing import List, Set
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from google.cloud import bigquery
from tqdm import tqdm
from scipy.stats import iqr

def load_columns_to_drop(filepaths: List[str]) -> Set[str]:
    """Load column names to drop from a list of file paths."""
    columns = set()
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            columns.update(line.strip() for line in f if line.strip())
    return columns

def get_columns_to_load(
    client: bigquery.Client,
    table_ref: str,
    columns_to_drop: Set[str]
) -> List[str]:
    """Get column names to load from BigQuery, excluding those to drop."""
    table = client.get_table(table_ref)
    all_columns = [field.name for field in table.schema]
    return [col for col in all_columns if col not in columns_to_drop]

def query_column_sample(
    client: bigquery.Client,
    table_ref: str,
    column: str,
    max_rows_percent: float
) -> pd.Series:
    """Query a random sample of a single column from BigQuery."""
    query = (
        f"SELECT `{column}` FROM `{table_ref}` "
        f"TABLESAMPLE SYSTEM ({max_rows_percent} PERCENT)"
    )
    df = client.query(query).to_dataframe()
    return df[column].dropna()

def is_low_cardinality_numeric(series: pd.Series, threshold: int = 12) -> bool:
    """Returns True if series is numeric and has few unique values (should be visualized as categories)."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() <= threshold

def plot_numeric_column(series: pd.Series, column: str, bins: int = None) -> plt.Figure:
    """Plot histogram and boxplot for a numeric or boolean column."""
    is_bool = pd.api.types.is_bool_dtype(series)
    if bins is not None:
        # Forced bin count (e.g., for _Total columns)
        bins_ = bins
    elif is_bool:
        bins_ = 2
    else:
        data_iqr = iqr(series)
        n = len(series)
        if data_iqr == 0 or n < 2:
            bins_ = 10
        else:
            bin_width = 2 * data_iqr / (n ** (1/3))
            bins_ = max(5, int((series.max() - series.min()) / bin_width))
    bins_ = min(bins_, 250)
    print(f"{column}: {bins_} bins{' (bool)' if is_bool else ''}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(series, kde=not is_bool, bins=bins_, ax=axes[0])
    axes[0].set_title(f'Histogram of {column} ({bins_} bins)')
    sns.boxplot(x=series, ax=axes[1])
    axes[1].set_title(f'Boxplot of {column}')
    fig.suptitle(column)
    fig.tight_layout()
    return fig

def plot_categorical_column(series: pd.Series, column: str, top_n: int = 20) -> plt.Figure:
    """Plot barplot for the top N categories of a categorical column, with value labels."""
    value_counts = series.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(x=value_counts.values, y=value_counts.index, orient='h', ax=ax)
    ax.set_title(f'Barplot of top {top_n} categories: {column}')
    ax.set_xlabel('Count')
    ax.set_ylabel(column)
    fig.tight_layout()

    # Add count labels to bars
    for i, (count, label) in enumerate(zip(value_counts.values, value_counts.index)):
        ax.text(
            count, i,                # x position is count, y position is index (i)
            f'{count:,}',            # label text, with thousands separator
            va='center',
            ha='left',               # align left of the bar
            fontsize=10,
            color='black'
        )
    return fig


def save_figure(fig: plt.Figure, png_path: str, pdf: PdfPages):
    """Save the figure to both PNG and the provided PDF."""
    fig.savefig(png_path)
    pdf.savefig(fig)
    plt.close(fig)

def visualize_bigquery_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    columns_to_drop: Set[str],
    max_rows_percent: float = 0.1,
    output_dir: str = "flight_visualizations",
    top_n_categories: int = 20
):
    """Main visualization function: loads columns, samples data, and saves plots."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    columns_to_load = get_columns_to_load(client, table_ref, columns_to_drop)

    os.makedirs(f"{output_dir}/png", exist_ok=True)
    pdf_path = f"{output_dir}/all_histbox.pdf"
    pdf = PdfPages(pdf_path)

    for column in tqdm(columns_to_load, desc="Processing columns"):
        series = query_column_sample(client, table_ref, column, max_rows_percent)
        if len(series) == 0:
            continue

        if is_low_cardinality_numeric(series):
            fig = plot_categorical_column(series.astype(str), column, top_n=series.nunique())
        elif pd.api.types.is_numeric_dtype(series):
            # --- Here is the new logic ---
            if column.endswith("_Total"):
                fig = plot_numeric_column(series, column, bins=96)
            else:
                fig = plot_numeric_column(series, column)
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            fig = plot_categorical_column(series, column, top_n=top_n_categories)
        else:
            continue

        png_path = f"{output_dir}/png/{column}.png"
        save_figure(fig, png_path, pdf)
        del series

    pdf.close()
    print(f"Saved all plots to PNG files and combined PDF: {pdf_path}")

def main():
    # Example configuration
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = "flights"
    table_id = "flights_all"
    columns_to_drop = set()  # Fill in or use load_columns_to_drop(['columns_to_drop.txt', ...])
    max_rows_percent = 1

    visualize_bigquery_table(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        columns_to_drop=columns_to_drop,
        max_rows_percent=max_rows_percent
    )

if __name__ == "__main__":
    main()
