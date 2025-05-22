import pandas as pd
import numpy as np

def find_correlated_features(correlation_matrix, target_column, top_n=100):
    """
    Analyze correlation matrix to find highly correlated feature pairs and determine candidates for dropping.
    
    Parameters:
    correlation_matrix (pd.DataFrame): The correlation matrix (features and target)
    target_column (str): The name of the target variable in the correlation matrix
    top_n (int): Number of top correlated pairs to return
    
    Returns:
    tuple: (top_pairs_df, drop_candidates)
    """
    # Create a copy to avoid modifying the original
    corr_matrix = correlation_matrix.copy()
    
    # Get feature columns (all columns except target)
    feature_columns = [col for col in corr_matrix.columns if col != target_column]
    
    # Create a list to store pairs and their correlations
    pairs_data = []
    
    # Find all pairs of features and their correlations
    for i, col1 in enumerate(feature_columns):
        for col2 in feature_columns[i+1:]:  # Start from i+1 to avoid duplicates
            correlation = abs(corr_matrix.loc[col1, col2])  # Use absolute correlation
            
            # Get correlations with target
            corr1_with_target = abs(corr_matrix.loc[col1, target_column])
            corr2_with_target = abs(corr_matrix.loc[col2, target_column])
            
            # Create a row for this pair
            pairs_data.append({
                'Feature1': col1,
                'Feature2': col2,
                'Correlation': correlation,
                'Feature1_Target_Corr': corr1_with_target,
                'Feature2_Target_Corr': corr2_with_target
            })
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_data)
    
    # Sort by correlation value (descending)
    pairs_df = pairs_df.sort_values('Correlation', ascending=False).reset_index(drop=True)
    
    # Get top N pairs
    top_pairs_df = pairs_df.head(top_n)
    
    # Determine candidates for dropping
    # For each pair, the feature with lower correlation with target is a candidate for dropping
    drop_candidates = []
    
    for _, row in top_pairs_df.iterrows():
        if row['Correlation'] > 0.7:  # Only consider highly correlated features (threshold can be adjusted)
            if row['Feature1_Target_Corr'] < row['Feature2_Target_Corr']:
                drop_candidates.append(row['Feature1'])
            else:
                drop_candidates.append(row['Feature2'])
    
    # Remove duplicates from drop_candidates
    drop_candidates = list(set(drop_candidates))
    
    return top_pairs_df, drop_candidates

def main():
    corr_matrix = pd.read_csv('flight_visualizations/sampled_phik_correlation_matrix.csv', index_col=0)
    top_pairs, drop_candidates = find_correlated_features(corr_matrix, 'Cancelled', top_n=100)
    
    print("Top Correlated Feature Pairs:")
    print(top_pairs)
    
    print("\nCandidates for Dropping:")
    for feature in drop_candidates[:100]:  # Limit to 100
        print(f"- {feature}")
    
    # Save top pairs table to CSV
    top_pairs.to_csv('flight_visualizations/top_100_correlated_pairs.csv', index=False)
    
    # Save up to 100 drop candidates to a CSV file (with a second column for correlation with target)
    corr_with_target = [
        abs(corr_matrix.loc[feature, 'Cancelled']) if feature in corr_matrix.index else np.nan
        for feature in drop_candidates[:100]
    ]
    drop_candidates_df = pd.DataFrame({
        'Drop_Candidate': drop_candidates[:100],
        'Correlation_with_Target': corr_with_target
    })

    drop_candidates_df = drop_candidates_df.sort_values('Correlation_with_Target', ascending=False).reset_index(drop=True)
    drop_candidates_df.to_csv('flight_visualizations/drop_candidates_top100.csv', index=False)
    
    return top_pairs, drop_candidates


if __name__ == "__main__":
    main()
