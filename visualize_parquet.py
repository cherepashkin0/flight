import os
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Create output directory if it doesn't exist
output_dir = "flight_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Load the parquet file
print("Loading parquet file...")
df = pl.read_parquet("flight_delay/flights_optimized.parquet")

# Convert to pandas for seaborn compatibility
pdf = df.to_pandas()

# List of columns to exclude from visualizations
# Add any column names you want to exclude here
# columns_to_exclude = ['DepDel15', 'CRSDepTime', 'Year'] 
columns_to_exclude = [ ]
# Get numerical columns
numerical_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()

# Filter out excluded columns
numerical_cols = [col for col in numerical_cols if col not in columns_to_exclude]

print(f"Found {len(numerical_cols)} numerical columns after exclusions: {numerical_cols}")

# Create a PDF to save all plots
pdf_path = os.path.join(output_dir, "all_visualizations.pdf")
with PdfPages(pdf_path) as pdf_pages:
    # For each numerical column, create boxplot and histogram
    for col in numerical_cols:
        print(f"Creating visualizations for {col}")
        
        # Create a figure with 2 subplots (boxplot and histogram)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Visualizations for {col}')
        
        # Boxplot
        sns.boxplot(y=pdf[col], ax=ax1)
        ax1.set_title(f'Boxplot for {col}')
        
        # Histogram
        sns.histplot(pdf[col], kde=True, ax=ax2)
        ax2.set_title(f'Histogram for {col}')
        
        # Save individual plot as PNG
        png_path = os.path.join(output_dir, f"{col}_visualization.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        
        # Save to PDF
        pdf_pages.savefig(fig)
        
        # Close the figure to free memory
        plt.close(fig)

print(f"All visualizations saved to {output_dir}/")
print(f"Combined PDF saved to {pdf_path}")
