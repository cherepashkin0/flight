#file: preprocess_parquet
#author: Seva with copilot

import pandas as pd
import os
import json
import phik
from phik import phik_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_parquet('flight_delay/Combined_Flights_2018.parquet')
    # calculate phi-k correlation matrix
    phi_corr = df.phik_matrix().round(2)
    # visualize the correlation matrix with seaborn
    plt.figure(figsize=(20, 20))
    sns.heatmap(phi_corr, annot=True, cmap='coolwarm', square=True, cbar=True)
    plt.title("Phi-K Correlation Heatmap")
    plt.tight_layout()
    # show figure
    plt.show()
    plt.savefig('artifacts/flight_corr.png')
    # save the correlation matrix
    # calculate correlation with respect to the target column 'Cancelled'
    target_corr = phi_corr['Cancelled'].sort_values(ascending=False)
    # save the correlation matrix to a json file
    with open('artifacts/flight_corr.json', 'w') as f:
        json.dump(target_corr.to_dict(), f, indent=4)
    # identify first 10 columns with highest correlation and with correlation closer to 0
    target_corr = target_corr[target_corr.abs() < 0.1]
    target_corr = target_corr[target_corr.abs() > 0.9]
    # drop these columns
    columns_to_drop = target_corr.index.tolist()
    
    for year in range(2018, 2023):
        parquet_path = f'flight_delay/Combined_Flights_{year}.parquet'
        df = pd.read_parquet(parquet_path)
        # append current df to df_full
        df_full = pd.concat([df_full, df], ignore_index=True) if 'df_full' in locals() else df
        


if __name__ == '__main__':
    main()

