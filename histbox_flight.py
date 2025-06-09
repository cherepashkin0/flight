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
    table = client.get_table(table_ref)
    all_columns = [field.name for field in table.schema]
    return [col for col in all_columns if col not in columns_to_drop]

def query_column_sample(
    client: bigquery.Client,
    table_ref: str,
    columns: List[str],
    max_rows_percent: float
) -> pd.DataFrame:
    # Always query the Cancelled column
    select_cols = ', '.join(columns)
    query = (
        f"SELECT {select_cols} FROM {table_ref} "
        f"TABLESAMPLE SYSTEM ({max_rows_percent} PERCENT)"
    )
    return client.query(query).to_dataframe()

def is_low_cardinality_numeric(series: pd.Series, threshold: int = 12) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.nunique() <= threshold

def plot_numeric_column_with_hue(
    df: pd.DataFrame, column: str, cancelled_column: str, bins: int = None
) -> plt.Figure:
    # Prepare data
    is_bool = pd.api.types.is_bool_dtype(df[column])
    if bins is None:
        data_iqr = iqr(df[column].dropna())
        n = len(df[column].dropna())
        if data_iqr == 0 or n < 2:
            bins_ = 10
        else:
            bin_width = 2 * data_iqr / (n ** (1/3))
            bins_ = max(5, int((df[column].max() - df[column].min()) / bin_width))
    else:
        bins_ = bins
    bins_ = min(bins_, 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#e74c3c"]

    # Density histograms for Cancelled=0 and Cancelled=1
    for idx, val in enumerate(sorted(df[cancelled_column].dropna().unique())):
        subset = df[df[cancelled_column] == val][column].dropna()
        if len(subset) == 0:
            continue
        sns.histplot(
            subset,
            bins=bins_,
            ax=ax,
            stat="density",
            label=f"{cancelled_column}={val}",
            color=colors[idx % 2],
            alpha=0.3,
            kde=False,
            element="step"
        )

    ax.set_title(f"{column}: Density Histogram by {cancelled_column} ({bins_} bins)")
    ax.set_xlabel(column)
    ax.set_ylabel("Density (relative frequency)")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_categorical_column_with_hue(
    df: pd.DataFrame, column: str, cancelled_column: str, top_n: int = 20
) -> plt.Figure:
    # Focus on top_n categories by overall count
    top_categories = df[column].value_counts().nlargest(top_n).index
    df = df[df[column].isin(top_categories)].copy()
    df[column] = df[column].astype(str)

    # Compute counts for each Cancelled value
    grouped = (
        df.groupby([column, cancelled_column])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to densities (row sum per Cancelled value)
    densities = grouped.divide(grouped.sum(axis=0), axis=1).fillna(0)
    densities = densities.sort_values(by=list(densities.columns)[0], ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = range(len(densities.index))
    colors = ["#3498db", "#e74c3c"]
    bar_width = 0.4

    # Plot bars and add counts as text labels
    for idx, val in enumerate(densities.columns):
        bars = ax.barh(
            y=[y + (bar_width if idx else 0) for y in y_positions],
            width=densities[val].values,
            height=bar_width,
            color=colors[idx % 2],
            alpha=0.6,
            label=f"{cancelled_column}={val}"
        )
        # Add counts as labels on bars
        for bar, count in zip(bars, grouped[val].values):
            width = bar.get_width()
            ax.text(
                width + 0.01,  # Slightly offset from bar end
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                va="center",
                fontsize=9,
                color=colors[idx % 2]
            )

    ax.set_yticks([y + bar_width / 2 for y in y_positions])
    ax.set_yticklabels(densities.index)
    ax.set_xlabel("Density (relative frequency)")
    ax.set_title(f"{column}: Density by {cancelled_column} (Top {top_n})")
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, png_path: str, pdf: PdfPages):
    fig.savefig(png_path)
    pdf.savefig(fig)
    plt.close(fig)

def visualize_bigquery_table_with_hue(
    project_id: str,
    dataset_id: str,
    table_id: str,
    columns_to_drop: Set[str],
    cancelled_column: str = "Cancelled",
    max_rows_percent: float = 0.1,
    output_dir: str = "flight_visualizations",
    top_n_categories: int = 20
):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    columns_to_load = get_columns_to_load(client, table_ref, columns_to_drop)
    if cancelled_column not in columns_to_load:
        columns_to_load.append(cancelled_column)

    os.makedirs(f"{output_dir}/png", exist_ok=True)
    pdf_path = f"{output_dir}/all_histbox.pdf"
    pdf = PdfPages(pdf_path)

    for column in tqdm(columns_to_load, desc="Processing columns"):
        if column == cancelled_column:
            continue

        # Always query both the column and the Cancelled status
        df = query_column_sample(client, table_ref, [column, cancelled_column], max_rows_percent)
        # Drop missing or empty Cancelled values
        df = df.dropna(subset=[cancelled_column])
        if len(df) == 0:
            continue

        # Skip columns with no data
        if len(df[column].dropna()) == 0:
            continue

        # Plotting logic
        if is_low_cardinality_numeric(df[column]):
            fig = plot_categorical_column_with_hue(df, column, cancelled_column, top_n=df[column].nunique())
        elif pd.api.types.is_numeric_dtype(df[column]):
            fig = plot_numeric_column_with_hue(df, column, cancelled_column)
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            fig = plot_categorical_column_with_hue(df, column, cancelled_column, top_n=top_n_categories)
        else:
            continue

        png_path = f"{output_dir}/png/{column}.png"
        save_figure(fig, png_path, pdf)
        del df

    pdf.close()
    print(f"Saved all plots to PNG files and combined PDF: {pdf_path}")

def main():
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = "flights"
    table_id = "flights_all"
    columns_to_drop = set()  # Optionally: load_columns_to_drop(['columns_to_drop.txt'])
    max_rows_percent = 100

    visualize_bigquery_table_with_hue(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        columns_to_drop=columns_to_drop,
        cancelled_column="Cancelled",
        max_rows_percent=max_rows_percent
    )

if __name__ == "__main__":
    main()
