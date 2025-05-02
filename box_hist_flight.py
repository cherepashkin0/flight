import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import clickhouse_connect
from sklearn.preprocessing import PowerTransformer
from phik import phik_matrix
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json

# Configuration
DB_NAME = 'flights_data'
MERGED_TABLE = f'{DB_NAME}.flight_2022'
OUTPUT_DIR = 'flight_figs'

# Columns to load
COLUMN_SUBSET = [
    "FlightDate", "Airline", "Flight_Number_Marketing_Airline",
    "Origin", "Dest", "Cancelled", "Diverted", "CRSDepTime",
    "DepTime", "DepDelayMinutes", "OriginAirportID", "OriginCityName",
    "OriginStateName", "DestAirportID", "DestCityName", "DestStateName",
    "TaxiOut", "TaxiIn", "CRSArrTime", "ArrTime", "ArrDelayMinutes",
]

CATEGORICAL_COLS = [
    "Airline", "Origin", "Dest", "OriginStateName", "DestStateName"
]

# ----------------------------------
# Database
# ----------------------------------

def connect_to_clickhouse():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST'),
        port=os.getenv('CLICKHOUSE_PORT'),
        username=os.getenv('CLICKHOUSE_USER'),
        password=os.getenv('CLICKHOUSE_PASSWORD')
    )


def load_flight_data(client, table, columns):
    columns_str = ', '.join(columns)
    query = f"SELECT {columns_str} FROM {table}"
    df = client.query_df(query)
    return df.reset_index(drop=True)


# ----------------------------------
# Preprocessing
# ----------------------------------

def preprocess_data(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].astype('category')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    return df, numeric_cols


# ----------------------------------
# Plotting
# ----------------------------------

def plot_2x2_hist_box(df, numeric_columns, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_numeric = df[numeric_columns].dropna()
    
    pt = PowerTransformer(method="yeo-johnson")
    df_transformed = pd.DataFrame(pt.fit_transform(df_numeric), columns=numeric_columns)

    for col in numeric_columns:
        raw = df_numeric[col]
        transformed = df_transformed[col]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Histogram before
        axes[0, 0].hist(raw, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f"Histogram (Raw) - {col}")

        # Boxplot before
        axes[1, 0].boxplot(raw.dropna())
        axes[1, 0].set_title(f"Boxplot (Raw) - {col}")

        # Histogram after
        axes[0, 1].hist(transformed, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title(f"Histogram (Yeo-Johnson) - {col}")

        # Boxplot after
        axes[1, 1].boxplot(transformed)
        axes[1, 1].set_title(f"Boxplot (Yeo-Johnson) - {col}")

        plt.tight_layout()
        filename = f"{output_dir}/2x2_{col}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved 2x2 plot for: {col}")

    # Save all figures into one PDF
    save_all_figures_to_pdf(output_dir, f"{output_dir}/all_figures.pdf")


def save_all_figures_to_pdf(figs_dir, output_pdf_path):
    pdf = PdfPages(output_pdf_path)
    for filename in sorted(os.listdir(figs_dir)):
        if filename.endswith(".png"):
            img_path = os.path.join(figs_dir, filename)
            img = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(11.7, 8.3))  # A4 landscape size
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()
    print(f"📄 All figures saved into one PDF: {output_pdf_path}")


def compute_signed_phik(df_numeric):
    """
    Computes Phi-k correlation matrix and applies Spearman sign to each value.
    Drops columns with only one unique value.
    """
    df_filtered = df_numeric.loc[:, df_numeric.nunique() > 1]

    # Phi-k matrix
    phik_corr = df_filtered.phik_matrix(interval_cols=df_filtered.columns.tolist())

    # Spearman sign matrix
    spearman_corr = df_filtered.corr(method='spearman')
    spearman_sign = np.sign(spearman_corr)

    # Align indexes and columns in case phik dropped columns
    spearman_sign = spearman_sign.reindex_like(phik_corr)

    # Apply sign
    signed_phik = phik_corr * spearman_sign
    return signed_phik



def save_phik_heatmap(corr_matrix, output_path):
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, cbar=True)
    plt.title("Phi-K Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Correlation heatmap saved to {output_path}")


def save_top_signed_correlations(corr_matrix, output_path, top_n=20):
    """
    Saves the top N positive and negative correlations to a JSON file.
    """
    pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            corr_value = corr_matrix.iloc[i, j]

            if pd.notna(corr_value):
                pairs.append({
                    "column_1": col1,
                    "column_2": col2,
                    "correlation": round(float(corr_value), 4)
                })

    # Sort by absolute correlation descending
    pairs_sorted = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)

    # Split into top positives and negatives
    top_positive = [p for p in pairs_sorted if p['correlation'] > 0][:top_n]
    top_negative = [p for p in pairs_sorted if p['correlation'] < 0][:top_n]

    output = {
        "top_positive_correlations": top_positive,
        "top_negative_correlations": top_negative
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"📄 Top correlations saved to {output_path}")

# ----------------------------------
# Main
# ----------------------------------

def main():
    print("🔌 Connecting to ClickHouse...")
    client = connect_to_clickhouse()

    print(f"📥 Loading data from {MERGED_TABLE}...")
    df = load_flight_data(client, MERGED_TABLE, COLUMN_SUBSET)

    print("🧼 Preprocessing data...")
    df, numeric_cols = preprocess_data(df, CATEGORICAL_COLS)

    print("📊 Generating 2x2 histograms and boxplots...")
    plot_2x2_hist_box(df, numeric_cols, OUTPUT_DIR)

    # Compute and save the Phi-k correlation matrix
    print("🔍 Calculating phi-k correlation matrix...")
    df_numeric = df[numeric_cols].dropna()  # Ensure no NaN values for correlation
    phik_corr = compute_signed_phik(df_numeric)
    save_phik_heatmap(phik_corr, f"{OUTPUT_DIR}/phik_correlation_heatmap.png")
    save_top_signed_correlations(phik_corr, f"{OUTPUT_DIR}/top_signed_phik_correlations.json", top_n=10)
    print("✅ All tasks completed successfully.")


if __name__ == "__main__":
    main()
