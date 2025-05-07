import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

# Create output directory
output_dir = "flight_visualizations"
os.makedirs(output_dir, exist_ok=True)

print("Loading parquet file...")
df = pd.read_parquet("flight_delay/flights_optimized.parquet")
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display column names
print("Columns in dataset:")
print(df.columns.tolist())

# For large datasets, sampling can speed up calculation
sample_size = min(500000, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Use the correct column name 'Cancelled' throughout the code
# Check if 'Cancelled' column exists
if 'Cancelled' not in df_sample.columns:
    print("Error: 'Cancelled' column not found in the dataset")
    exit(1)
else:
    print("Found 'Cancelled' column")

# ===== MANUAL DEFINITION OF INTERVAL COLUMNS =====
# Define your interval columns here - these are continuous numerical variables
interval_columns = [
    'DepDelayMinutes', 
    'DepDelay', 
    'ArrDelayMinutes', 
    'AirTime', 
    'CRSElapsedTime', 
    'ActualElapsedTime', 
    'Distance',
    'TaxiOut', 
    'TaxiIn', 
    'ArrDelay'
]

# Define your ordinal columns here - these are ordered categorical variables
ordinal_columns = [
    'Quarter',
    'Month', 
    'DayofMonth', 
    'DayOfWeek',
    'DepDel15',
    'DepartureDelayGroups',
    'ArrDel15',
    'ArrivalDelayGroups',
    'DistanceGroup', 
    'DivAirportLandings'
]

# All other columns will be treated as nominal (unordered categorical)
nominal_columns = [col for col in df_sample.columns 
                  if col not in interval_columns and col not in ordinal_columns]

print("\nManually defined column types:")
print(f"Interval columns (continuous): {len(interval_columns)}")
print(f"Ordinal columns (ordered categorical): {len(ordinal_columns)}")
print(f"Nominal columns (unordered categorical): {len(nominal_columns)}")

print("\nCalculating phik correlation matrix...")
try:
    import phik
    from phik import phik_matrix
    
    # Calculate the Phik correlation matrix with manually defined column types
    phik_corr = phik_matrix(
        df_sample,
        interval_cols=interval_columns,
        dropna=True,  # Drop NA values
        verbose=1     # Show some progress info
    )
    
    print("Phik correlation matrix calculation complete.")
    
except ImportError as e:
    print(f"Error importing phik package: {e}")
    print("Please install phik with: pip install phik")
    exit(1)
except Exception as e:
    print(f"Error calculating phik correlation: {e}")
    print("Trying alternative approach...")
    
    try:
        # Alternative approach: Calculate correlations for numerical columns first
        numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
        phik_corr = phik_matrix(
            df_sample[numeric_cols], 
            interval_cols=interval_columns,
            dropna=True
        )
        
        # Then calculate correlations with categorical columns one by one
        remaining_cols = [col for col in df_sample.columns if col not in numeric_cols]
        for col in remaining_cols:
            try:
                # Calculate phik correlation between this column and Cancelled
                subset_df = df_sample[[col, 'Cancelled']]
                subset_phik = phik_matrix(subset_df, dropna=True)
                phik_corr.loc[col, 'Cancelled'] = subset_phik.loc[col, 'Cancelled']
                phik_corr.loc['Cancelled', col] = subset_phik.loc['Cancelled', col]
            except Exception as inner_e:
                print(f"  Skipping column {col}: {inner_e}")
    except Exception as alt_e:
        print(f"Alternative approach failed: {alt_e}")
        exit(1)

# Get correlations with 'Cancelled' column
cancelled_corr = phik_corr['Cancelled'].sort_values(ascending=False)

# Save full correlation matrix
phik_corr.to_csv(f"{output_dir}/full_phik_correlation_matrix.csv")
print(f"Full correlation matrix saved to {output_dir}/full_phik_correlation_matrix.csv")

mask = np.triu(np.ones_like(phik_corr, dtype=bool))
plt.figure(figsize=(30,30))
sns.heatmap(
    phik_corr,
    mask=mask,
    cmap='coolwarm',
    cbar_kws={'shrink':.5},
    linewidths=0.2
)
plt.title("Phik Correlation Matrix (upper triangle only)")
plt.tight_layout()
plt.savefig("flight_visualizations/full_phik_heatmap_masked.png", dpi=300)

# Identify highly correlated columns (>0.9)
high_corr = cancelled_corr[(cancelled_corr > 0.9) & (cancelled_corr < 1.0)]  # Exclude self-correlation

# Identify weakly correlated columns (<0.05)
weak_corr = cancelled_corr[cancelled_corr < 0.05]

# Prepare a dict with ultra-high and ultra-low correlations
extreme_corrs = {
    "ultra_high_corr (>0.9)": high_corr.index.tolist(),
    "ultra_low_corr (<0.05)": weak_corr.index.tolist()
}

# Save ultra-high correlations to a text file
high_txt_path = os.path.join(output_dir, "ultra_high_corr.txt")
with open(high_txt_path, "w") as f_high:
    for col in high_corr.index:
        f_high.write(f"{col}\n")
print(f"Ultra-high correlation columns saved to {high_txt_path}")

# Save ultra-low correlations to a text file
low_txt_path = os.path.join(output_dir, "ultra_low_corr.txt")
with open(low_txt_path, "w") as f_low:
    for col in weak_corr.index:
        f_low.write(f"{col}\n")
print(f"Ultra-low correlation columns saved to {low_txt_path}")

print("\n===== RESULTS =====")
print("\nColumns highly correlated with 'Cancelled' (>0.9):")
if len(high_corr) > 0:
    for col, corr_val in high_corr.items():
        print(f"{col}: {corr_val:.4f}")
else:
    print("No columns with correlation > 0.9")

print("\nColumns weakly correlated with 'Cancelled' (<0.05):")
if len(weak_corr) > 0:
    for col, corr_val in weak_corr.items():
        print(f"{col}: {corr_val:.4f}")
else:
    print("No columns with correlation < 0.05")

# Save correlation results to CSV
cancelled_corr.to_csv(f"{output_dir}/cancelled_correlations.csv")
print(f"\nAll correlations saved to {output_dir}/cancelled_correlations.csv")

# Plot correlation heatmap for top correlations
plt.figure(figsize=(12, 10))

# Get top 15 correlated features (positive or negative)
top_corr = abs(cancelled_corr).sort_values(ascending=False).head(16)
top_corr_features = top_corr.index.tolist()

# Make sure 'Cancelled' is the first column
if 'Cancelled' in top_corr_features:
    top_corr_features.remove('Cancelled')
top_corr_features = ['Cancelled'] + top_corr_features[:15]  # Limit to 15 other features

# Create subset correlation matrix for visualization
subset_phik = phik_corr.loc[top_corr_features, top_corr_features]

# Plot heatmap
sns.heatmap(subset_phik, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Top Correlations with Cancelled Column (Phik correlation)')
plt.tight_layout()
plt.savefig(f"{output_dir}/cancelled_correlation_heatmap.png", dpi=300)
print(f"Correlation heatmap saved to {output_dir}/cancelled_correlation_heatmap.png")

# Plot bar chart of correlations with 'Cancelled'
plt.figure(figsize=(14, 8))
top_corrs_excluding_self = cancelled_corr[cancelled_corr.index != 'Cancelled'].sort_values(ascending=False).head(15)
sns.barplot(x=top_corrs_excluding_self.index, y=top_corrs_excluding_self.values)
plt.title('Features Most Correlated with Flight Cancellation (Phik correlation)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{output_dir}/cancelled_correlation_barplot.png", dpi=300)
print(f"Correlation barplot saved to {output_dir}/cancelled_correlation_barplot.png")

print("\nAnalysis complete!")
