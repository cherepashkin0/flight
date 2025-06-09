import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression  # Using logistic regression for classification
from tqdm import tqdm
import time
from fastparquet import ParquetFile
import sys
from phik import phik_matrix
from pandas.api.types import is_numeric_dtype, is_object_dtype

POST_EVENT = [
    'TaxiOut', 'TaxiIn', 'DepDelay', 'DepDelayMinutes',
    'ArrDelay', 'ActualElapsedTime', 'AirTime',  # etc.
]


def load_data_and_columns():
    # 1. Load the two drop‐lists
    print("Loading columns to drop from file1.txt and file2.txt...")
    with open('flight_visualizations/ultra_high_corr.txt', 'r') as f:
        cols_to_drop_1 = [line.strip() for line in f]
    with open('flight_visualizations/ultra_low_corr.txt', 'r') as f:
        cols_to_drop_2 = [line.strip() for line in f]
    with open('flight_visualizations/drop_it.txt', 'r') as f:
        drop_it = [line.strip() for line in f]    
    cols_to_drop = set(cols_to_drop_1 + cols_to_drop_2 + drop_it + POST_EVENT)

    # 2. Open the parquet with fastparquet to get all column names WITHOUT reading full data
    print("Inspecting parquet schema to get all columns…")
    pf = ParquetFile('flight_delay/flights_optimized.parquet')
    all_columns = pf.columns

    # 3. Compute the columns we actually want to load
    cols_to_load = [c for c in all_columns if c not in cols_to_drop]
    print(f"Will load {len(cols_to_load)}/{len(all_columns)} columns.")

    # 4. Read only those columns
    print("Loading data from file.parquet with fastparquet (selected columns)…")
    df = pf.to_pandas(columns=cols_to_load)
    # take only first 1 million rows
    # df = df[:1_000_000]
    print(f"Loaded {len(df)} rows.")
    print(f"Loaded column names and their data types:\n{df.dtypes}")
    print(f"Data shape after selective load: {df.shape}")
    return df

def prepare_data(df):
    # Target
    y = df['Cancelled']
    # Drop target and any other columns you don't want as features
    X = df.drop(columns=['Cancelled', 'FlightDate'])

    # Stratified split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test

# Add this function to preprocess features before training
def preprocess_features(X_train, X_test):
    """
    Preprocess features by extracting datetime parts, dropping categorical columns,
    and handling missing values.
    """
    print("Preprocessing features...")

    # Create copies to avoid modifying original data
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()

    # Identify datetime and categorical columns
    datetime_columns = [col for col in X_train.columns if X_train[col].dtype == 'datetime64[ns]']
    categorical_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # Process datetime columns
    for col in datetime_columns:
        for df in (X_train_processed, X_test_processed):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek

        # Drop the original datetime column
        X_train_processed = X_train_processed.drop(columns=[col])
        X_test_processed = X_test_processed.drop(columns=[col])

    # Drop all categorical columns (no one-hot encoding)
    if categorical_columns:
        print(f"Dropping {len(categorical_columns)} categorical columns: {categorical_columns}")
        X_train_processed = X_train_processed.drop(columns=categorical_columns)
        X_test_processed = X_test_processed.drop(columns=categorical_columns)
    
    # Check for missing values
    missing_train = X_train_processed.isna().sum()
    missing_cols = missing_train[missing_train > 0].index.tolist()
    
    if missing_cols:
        print(f"Found {len(missing_cols)} columns with missing values: {missing_cols}")
        print("Missing value counts:")
        for col in missing_cols:
            print(f"  {col}: {missing_train[col]} missing values")
        
        # Handle missing values using mean imputation for numeric columns
        from sklearn.impute import SimpleImputer
        
        print("Applying mean imputation to handle missing values...")
        imputer = SimpleImputer(strategy='mean')
        
        # Convert to numpy arrays for imputation
        X_train_array = X_train_processed.values
        X_test_array = X_test_processed.values
        
        # Fit imputer on training data and transform both datasets
        X_train_imputed = imputer.fit_transform(X_train_array)
        X_test_imputed = imputer.transform(X_test_array)
        
        # Convert back to pandas DataFrames
        X_train_processed = pd.DataFrame(X_train_imputed, columns=X_train_processed.columns, index=X_train_processed.index)
        X_test_processed = pd.DataFrame(X_test_imputed, columns=X_test_processed.columns, index=X_test_processed.index)
        
        # Verify no more missing values
        missing_after = X_train_processed.isna().sum().sum()
        print(f"Missing values after imputation: {missing_after}")

    print(f"Processed training set shape: {X_train_processed.shape}")
    print(f"Processed test set shape: {X_test_processed.shape}")

    return X_train_processed, X_test_processed

# Function to train a Logistic Regression model (replacing XGBoost)
def train_logistic_regression_model(X_train, y_train):
    print("Training Logistic Regression model...")
    
    # Check class imbalance
    class_counts = np.bincount(y_train)
    print(f"Class distribution in training data: {class_counts}")
    imbalance_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 0
    print(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
    
    # Use a simpler, more efficient model as the default approach
    print("Using efficient model configuration for large dataset")
    model = LogisticRegression(
        max_iter=200,            # Lower max_iter to reduce training time
        solver='liblinear',      # More efficient solver for large datasets
        C=1.0,                   # Standard regularization strength
        penalty='l2',            # L2 penalty is more computationally efficient
        random_state=42,
        class_weight='balanced', # Handle class imbalance
        n_jobs=-1,               # Use all available cores
        tol=1e-3                 # Slightly relax tolerance for faster convergence
    )
    
    # Train model with timing
    start_time = time.time()
    print("Training progress:")
    with tqdm(total=1) as pbar:
        try:
            model.fit(X_train, y_train)
            pbar.update(1)
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
        except Exception as e:
            print(f"Error during model training: {e}")
            print("Falling back to even simpler model configuration...")
            
            # Fallback to even simpler model if needed
            model = LogisticRegression(
                max_iter=100,
                solver='lbfgs',     # Fast solver for small-medium datasets
                C=1.0,
                random_state=42,
                class_weight=None,  # Skip class weights to speed up training
                n_jobs=1            # Sometimes more efficient for this solver
            )
            
            model.fit(X_train, y_train)
            pbar.update(1)
            training_time = time.time() - start_time
            print(f"Fallback training completed in {training_time:.2f} seconds")
    
    # Print model configuration actually used
    print(f"Final model parameters: {model.get_params()}")
    
    return model

# Function to do cross-validation
def perform_cross_validation(X, y):
    print("Performing 3-fold cross-validation...")
    # Create Logistic Regression classifier
    lr_clf = LogisticRegression(
        max_iter=1000,
        solver='liblinear',
        random_state=42
    )
    
    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Metrics to evaluate
    metrics = {
        'roc_auc': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }
    
    # Perform cross-validation with tqdm
    fold = 0
    for train_idx, val_idx in tqdm(cv.split(X, y), total=3, desc="CV Folds"):
        fold += 1
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train the model
        lr_clf.fit(X_train_cv, y_train_cv)
        
        # Make predictions
        y_pred_proba = lr_clf.predict_proba(X_val_cv)
        y_pred = lr_clf.predict(X_val_cv)
        
        # Binary classification metrics
        if len(np.unique(y)) == 2:
            metrics['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba[:, 1]))
            metrics['precision'].append(precision_score(y_val_cv, y_pred))
            metrics['recall'].append(recall_score(y_val_cv, y_pred))
        else:
            # For multiclass, use weighted average
            metrics['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba, multi_class='ovr'))
            metrics['precision'].append(precision_score(y_val_cv, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_val_cv, y_pred, average='weighted'))
        
        metrics['accuracy'].append(accuracy_score(y_val_cv, y_pred))
        
        print(f"Fold {fold} - ROC-AUC: {metrics['roc_auc'][-1]:.4f}, Precision: {metrics['precision'][-1]:.4f}, "
              f"Recall: {metrics['recall'][-1]:.4f}, Accuracy: {metrics['accuracy'][-1]:.4f}")
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean(values) for metric, values in metrics.items()}
    
    print("\nAverage CV Metrics:")
    print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    
    return avg_metrics

# Function to calculate naive baseline metrics
def calculate_naive_baseline(X_test, y_test):
    print("\nCalculating naive baseline metrics...")
    
    if len(np.unique(y_test)) == 2:
        # For binary classification, predict the majority class
        majority_class = np.argmax(np.bincount(y_test))
        baseline_pred = np.full_like(y_test, majority_class)
        
        # Calculate metrics
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
        baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
        
        # For ROC-AUC, we need probabilities - use constant probability for majority class
        baseline_proba = np.zeros((len(y_test), 2))
        baseline_proba[:, majority_class] = 1
        baseline_roc_auc = roc_auc_score(y_test, baseline_proba[:, 1])
        
    else:
        # For multiclass, predict the majority class
        majority_class = np.argmax(np.bincount(y_test))
        baseline_pred = np.full_like(y_test, majority_class)
        
        # Calculate metrics
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        baseline_precision = precision_score(y_test, baseline_pred, average='weighted', zero_division=0)
        baseline_recall = recall_score(y_test, baseline_pred, average='weighted', zero_division=0)
        
        # For multiclass ROC-AUC, create one-hot encoded probabilities for majority class
        n_classes = len(np.unique(y_test))
        baseline_proba = np.zeros((len(y_test), n_classes))
        baseline_proba[:, majority_class] = 1
        baseline_roc_auc = roc_auc_score(y_test, baseline_proba, multi_class='ovr')
    
    print("Naive Baseline Metrics:")
    print(f"ROC-AUC: {baseline_roc_auc:.4f}")
    print(f"Precision: {baseline_precision:.4f}")
    print(f"Recall: {baseline_recall:.4f}")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    
    return {
        'roc_auc': baseline_roc_auc,
        'precision': baseline_precision,
        'recall': baseline_recall,
        'accuracy': baseline_accuracy
    }

# Function to evaluate model and compare with baseline
def evaluate_model(model, X_test, y_test, cv_metrics, baseline_metrics):
    print("\nEvaluating model on test set...")
    y_pred_proba = model.predict_proba(X_test)
    
    # For imbalanced datasets, the decision threshold can be adjusted
    # Default threshold is 0.5, but with high imbalance, we might need to lower it
    # to improve recall of the minority class
    
    # First, evaluate with default threshold (0.5)
    default_threshold = 0.5
    y_pred_default = (y_pred_proba[:, 1] >= default_threshold).astype(int)
    
    # Calculate metrics with default threshold
    if len(np.unique(y_test)) == 2:
        test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        test_precision_default = precision_score(y_test, y_pred_default)
        test_recall_default = recall_score(y_test, y_pred_default)
        test_f1_default = 2 * (test_precision_default * test_recall_default) / (test_precision_default + test_recall_default + 1e-10)
    else:
        test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        test_precision_default = precision_score(y_test, y_pred_default, average='weighted')
        test_recall_default = recall_score(y_test, y_pred_default, average='weighted')
        test_f1_default = 2 * (test_precision_default * test_recall_default) / (test_precision_default + test_recall_default + 1e-10)
    
    test_accuracy_default = accuracy_score(y_test, y_pred_default)
    
    print("Test Set Metrics (Default Threshold = 0.5):")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Precision: {test_precision_default:.4f}")
    print(f"Recall: {test_recall_default:.4f}")
    print(f"F1 Score: {test_f1_default:.4f}")
    print(f"Accuracy: {test_accuracy_default:.4f}")
    
    # Try different thresholds to find the best F1 score
    print("\nFinding optimal threshold for F1 score...")
    thresholds = np.linspace(0.1, 0.9, 9)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    # Calculate precision, recall, and F1 for different thresholds
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_threshold)
        recall = recall_score(y_test, y_pred_threshold)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Find best threshold for F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precision_scores[best_idx]
    best_recall = recall_scores[best_idx]
    
    print(f"Optimal threshold: {best_threshold:.2f}, F1: {best_f1:.4f}, Precision: {best_precision:.4f}, Recall: {best_recall:.4f}")
    
    # Get predictions with best threshold
    y_pred_best = (y_pred_proba[:, 1] >= best_threshold).astype(int)
    test_accuracy_best = accuracy_score(y_test, y_pred_best)
    
    # Output confusion matrix for best threshold
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix (Best Threshold):")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # Calculate metrics based on confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"False Positive Rate: {fp / (fp + tn):.4f}")
    print(f"False Negative Rate: {fn / (fn + tp):.4f}")
    
    # Calculate final metrics
    test_metrics = {
        'roc_auc': test_roc_auc,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'accuracy': test_accuracy_best
    }
    
    # Compare with cross-validation and baseline
    print("\nComparison Summary (with optimal threshold):")
    print("Metric      | Test Set  | CV Average | Naive Baseline | Improvement over Baseline")
    print("------------|-----------|------------|----------------|-------------------------")
    print(f"ROC-AUC     | {test_roc_auc:.4f}    | {cv_metrics['roc_auc']:.4f}     | {baseline_metrics['roc_auc']:.4f}           | {(test_roc_auc - baseline_metrics['roc_auc']) * 100:.2f}%")
    print(f"Precision   | {best_precision:.4f}    | {cv_metrics['precision']:.4f}     | {baseline_metrics['precision']:.4f}           | {(best_precision - baseline_metrics['precision']) * 100:.2f}%")
    print(f"Recall      | {best_recall:.4f}    | {cv_metrics['recall']:.4f}     | {baseline_metrics['recall']:.4f}           | {(best_recall - baseline_metrics['recall']) * 100:.2f}%")
    print(f"Accuracy    | {test_accuracy_best:.4f}    | {cv_metrics['accuracy']:.4f}     | {baseline_metrics['accuracy']:.4f}           | {(test_accuracy_best - baseline_metrics['accuracy']) * 100:.2f}%")
    
    return test_metrics, best_threshold

def phik_assert(df, high_card_threshold=100):
    df = df[:10_000_000]
    # --- PHIK correlation check (with exclusions) ---
    print("Computing Phik correlations with target 'Cancelled'...")

    # Hard-coded list of columns to skip
    PHIK_EXCLUDE = {
        'FlightDate'
    }

    # 1) Initial filter: dtype + explicit excludes
    cols_for_phik = [
        c for c in df.columns
        if c not in PHIK_EXCLUDE
           and (is_numeric_dtype(df[c]) or is_object_dtype(df[c]))
    ]
    # Ensure the target is present
    if 'Cancelled' not in cols_for_phik:
        cols_for_phik.append('Cancelled')

    # 2) Identify and drop high-cardinality columns
    high_card_cols = [
        c for c in cols_for_phik
        if df[c].nunique(dropna=False) > high_card_threshold and c != 'Cancelled'
    ]
    if high_card_cols:
        print(f"Dropping {len(high_card_cols)} high-cardinality columns (>{high_card_threshold} uniques):")
        print("  " + ", ".join(high_card_cols))
        cols_for_phik = [c for c in cols_for_phik if c not in high_card_cols]

    # 3) Identify numeric interval_cols
    interval_cols = [c for c in cols_for_phik if is_numeric_dtype(df[c])]

    # 4) Compute the Phik matrix
    phik_corr_matrix = df[cols_for_phik].phik_matrix(
        interval_cols=interval_cols,
        verbose=False
    )

    # 5) Extract correlations with the target
    corr_with_target = (
        phik_corr_matrix['Cancelled']
        .drop(labels=['Cancelled'], errors='ignore')
    )

    # 6) Assert none exceed 0.9
    high_corr = corr_with_target[abs(corr_with_target) > 0.9]
    if not high_corr.empty:
        feats = ", ".join(high_corr.index)
        corrs = ", ".join(f"{v:.3f}" for v in high_corr.values)
        print(f"✋ Stopping: found feature(s) [{feats}] with Phik > 0.9 ({corrs})")
        sys.exit(1)

    print("All features have Phik ≤ 0.9 with 'Cancelled'; proceeding…")

def main():
    # Load data and columns to drop
    df = load_data_and_columns()

    # Check if target variable exists
    if 'Cancelled' not in df.columns:
        print(f"Error: Target variable 'Cancelled' not found in the dataset.")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Compute Phik correlations
    # phik_assert(df)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Preprocess features
    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
    
    # Train Logistic Regression model (instead of XGBoost)
    model = train_logistic_regression_model(X_train_processed, y_train)
    
    # Perform cross-validation
    cv_metrics = perform_cross_validation(X_train_processed, y_train)
    
    # Calculate naive baseline
    baseline_metrics = calculate_naive_baseline(X_test_processed, y_test)
    
    # Evaluate model and compare with baseline
    test_metrics, best_threshold = evaluate_model(model, X_test_processed, y_test, cv_metrics, baseline_metrics)

    # Print feature importance (coefficients) for interpretability
    if hasattr(model, 'coef_'):
        print("\nModel Coefficients (Feature Importance):")
        feature_names = X_train_processed.columns
        coefficients = model.coef_[0]
        
        # Sort features by absolute coefficient value
        sorted_idx = np.argsort(np.abs(coefficients))[::-1]
        print("Top 15 Most Important Features:")
        for i in range(min(15, len(feature_names))):
            idx = sorted_idx[i]
            print(f"{feature_names[idx]}: {coefficients[idx]:.6f}")
            
        # Create a simple visualization of feature importances
        import matplotlib.pyplot as plt
        
        try:
            plt.figure(figsize=(10, 6))
            # Get the top 10 features
            top_n = min(10, len(feature_names))
            top_indices = sorted_idx[:top_n]
            top_features = [feature_names[i] for i in top_indices]
            top_coeffs = [abs(coefficients[i]) for i in top_indices]
            
            # Create bar chart of absolute coefficient values
            plt.barh(range(top_n), top_coeffs, align='center')
            plt.yticks(range(top_n), top_features)
            plt.xlabel('Absolute Coefficient Value')
            plt.title('Top 10 Feature Importances')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print("Feature importance visualization saved to 'feature_importance.png'")
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    # Save the trained model if needed
    from joblib import dump
    try:
        dump(model, 'flight_cancellation_model.joblib')
        print("Model saved to 'flight_cancellation_model.joblib'")
        
        # Save optimal threshold
        with open('optimal_threshold.txt', 'w') as f:
            f.write(f"{best_threshold}")
        print(f"Optimal threshold ({best_threshold:.2f}) saved to 'optimal_threshold.txt'")
        
        # Save summary results to a text file
        with open('model_performance_summary.txt', 'w') as f:
            f.write("=== FLIGHT CANCELLATION PREDICTION MODEL SUMMARY ===\n\n")
            f.write(f"Model: Logistic Regression\n")
            f.write(f"Dataset size: {len(X_train) + len(X_test)} flights\n")
            f.write(f"Class imbalance: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.2f}:1\n\n")
            
            f.write("=== PERFORMANCE METRICS ===\n")
            f.write(f"ROC-AUC: {test_metrics['roc_auc']:.4f}\n")
            f.write(f"Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {test_metrics['f1']:.4f}\n")
            f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n\n")
            
            f.write(f"Optimal threshold: {best_threshold:.2f}\n\n")
            
            f.write("=== TOP PREDICTIVE FEATURES ===\n")
            for i in range(min(10, len(feature_names))):
                idx = sorted_idx[i]
                f.write(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.6f}\n")
        
        print("Performance summary saved to 'model_performance_summary.txt'")
    except Exception as e:
        print(f"Could not save model or results: {e}")

    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
