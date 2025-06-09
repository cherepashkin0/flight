import pandas as pd
import numpy as np
import networkx as nx

def find_correlated_features_graph(correlation_matrix, target_column, corr_threshold=0.7, top_n=100):
    """
    Анализирует корреляционную матрицу с помощью графового жадного метода отбора.
    
    Parameters:
    correlation_matrix (pd.DataFrame): Матрица корреляций (features и target)
    target_column (str): Имя таргет-колонки
    corr_threshold (float): Порог для сильной корреляции между фичами
    top_n (int): Сколько самых скоррелированных пар сохранить для отчета
    
    Returns:
    tuple: (top_pairs_df, selected_features, drop_candidates)
    """
    corr_matrix = correlation_matrix.copy()
    feature_columns = [col for col in corr_matrix.columns if col != target_column]
    
    # Формируем melt таблицу для пар признаков
    pairs_data = []
    for i, col1 in enumerate(feature_columns):
        for col2 in feature_columns[i+1:]:
            correlation = abs(corr_matrix.loc[col1, col2])
            corr1_with_target = abs(corr_matrix.loc[col1, target_column])
            corr2_with_target = abs(corr_matrix.loc[col2, target_column])
            pairs_data.append({
                'Feature1': col1,
                'Feature2': col2,
                'Correlation': correlation,
                'Feature1_Target_Corr': corr1_with_target,
                'Feature2_Target_Corr': corr2_with_target
            })
    pairs_df = pd.DataFrame(pairs_data)
    pairs_df = pairs_df.sort_values('Correlation', ascending=False).reset_index(drop=True)
    top_pairs_df = pairs_df.head(top_n)
    
    # --- GRAPH-BASED SELECTION ---
    # 1. Строим граф: ребро между парами, где корреляция выше порога
    G = nx.Graph()
    G.add_nodes_from(feature_columns)
    for _, row in pairs_df.iterrows():
        if row['Correlation'] > corr_threshold:
            G.add_edge(row['Feature1'], row['Feature2'])
    
    # 2. Веса - абсолютная корреляция с таргетом
    target_corr = corr_matrix[target_column][feature_columns].abs()
    nx.set_node_attributes(G, target_corr.to_dict(), name='weight')
    
    # 3. Жадный отбор максимального независимого множества
    features = set(G.nodes)
    selected = set()
    while features:
        best = max(features, key=lambda x: G.nodes[x]['weight'])
        selected.add(best)
        features.remove(best)
        neighbors = set(G.neighbors(best))
        features -= neighbors

    # Drop candidates — это все фичи, которые не попали в selected
    drop_candidates = list(set(feature_columns) - selected)
    selected_features = list(selected)

    return top_pairs_df, selected_features, drop_candidates

def main():
    corr_matrix = pd.read_csv('flight_visualizations/sampled_phik_correlation_matrix.csv', index_col=0)
    target_column = 'Cancelled'
    top_n = 1000
    corr_threshold = 0.85
    
    top_pairs, selected_features, drop_candidates = find_correlated_features_graph(
        corr_matrix, target_column, corr_threshold=corr_threshold, top_n=top_n)
    
    print("Top Correlated Feature Pairs:")
    print(top_pairs)
    
    print("\nSelected Features (keep):")
    for feature in selected_features[:100]:
        print(f"+ {feature}")

    print("\nCandidates for Dropping (graph-based):")
    for feature in drop_candidates[:100]:
        print(f"- {feature}")

    # Save tables
    top_pairs.to_csv('flight_visualizations/top_100_correlated_pairs.csv', index=False)

    # Сохраняем selected_features и drop_candidates с корреляцией к таргету
    selected_df = pd.DataFrame({
        'Selected_Feature': selected_features,
        'Correlation_with_Target': [abs(corr_matrix.loc[feature, target_column]) if feature in corr_matrix.index else np.nan for feature in selected_features]
    }).sort_values('Correlation_with_Target', ascending=False).reset_index(drop=True)
    selected_df.to_csv('flight_visualizations/selected_features_graph.csv', index=False)

    drop_candidates_df = pd.DataFrame({
        'Drop_Candidate': drop_candidates,
        'Correlation_with_Target': [abs(corr_matrix.loc[feature, target_column]) if feature in corr_matrix.index else np.nan for feature in drop_candidates]
    }).sort_values('Correlation_with_Target', ascending=False).reset_index(drop=True)
    drop_candidates_df.to_csv('flight_visualizations/drop_candidates_graph.csv', index=False)

    return top_pairs, selected_features, drop_candidates

if __name__ == "__main__":
    main()
