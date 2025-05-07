import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import xgboost as xgb
from tqdm import tqdm
import time
from fastparquet import ParquetFile

# Function to load data and columns to drop

def load_data_and_columns():
    # 1. Load the two drop‐lists
    print("Loading columns to drop from file1.txt and file2.txt...")
    with open('flight_visualizations/ultra_high_corr.txt', 'r') as f:
        cols_to_drop_1 = [line.strip() for line in f]
    with open('flight_visualizations/ultra_low_corr.txt', 'r') as f:
        cols_to_drop_2 = [line.strip() for line in f]
    cols_to_drop = set(cols_to_drop_1 + cols_to_drop_2)

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

    print(f"Data shape after selective load: {df.shape}")
    return df

# Function to prepare data for training
def prepare_data(df):
    # Get target variable 'Cancelled'
    if 'Cancelled' not in df.columns:
        raise ValueError("Target variable 'Cancelled' not found in the dataset")
    
    # Separate features and target
    y = df['Cancelled']
    X = df.drop(columns=['Cancelled'])
    
    # Split data with random shuffle
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Function to train XGBoost model with tqdm progress
class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, rounds):
        self.pbar = tqdm(total=rounds)
        
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False
    
    def after_training(self, model):
        self.pbar.close()
        return model  # Return the model here

# Add this function to preprocess features before training
def preprocess_features(X_train, X_test):
    print("Preprocessing features...")
    
    # Create copies to avoid modifying original data
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    categorical_columns = []
    datetime_columns = []
    
    # Identify column types
    for col in X_train.columns:
        if X_train[col].dtype == 'datetime64[ns]':
            datetime_columns.append(col)
        elif X_train[col].dtype == 'object':
            categorical_columns.append(col)
    
    # Process datetime columns
    for col in datetime_columns:
        # Extract useful features from datetime
        X_train_processed[f"{col}_year"] = X_train[col].dt.year
        X_train_processed[f"{col}_month"] = X_train[col].dt.month
        X_train_processed[f"{col}_day"] = X_train[col].dt.day
        X_train_processed[f"{col}_dayofweek"] = X_train[col].dt.dayofweek
        
        X_test_processed[f"{col}_year"] = X_test[col].dt.year
        X_test_processed[f"{col}_month"] = X_test[col].dt.month
        X_test_processed[f"{col}_day"] = X_test[col].dt.day
        X_test_processed[f"{col}_dayofweek"] = X_test[col].dt.dayofweek
        
        # Drop the original datetime column
        X_train_processed = X_train_processed.drop(columns=[col])
        X_test_processed = X_test_processed.drop(columns=[col])
    
    # Process categorical columns with one-hot encoding
    for col in categorical_columns:
        # Get unique values from both train and test
        unique_values = pd.concat([X_train[col], X_test[col]]).unique()
        
        # Create dummy variables for each unique value
        for value in unique_values:
            if pd.notna(value):  # Skip NaN values
                X_train_processed[f"{col}_{value}"] = (X_train[col] == value).astype(int)
                X_test_processed[f"{col}_{value}"] = (X_test[col] == value).astype(int)
        
        # Drop the original categorical column
        X_train_processed = X_train_processed.drop(columns=[col])
        X_test_processed = X_test_processed.drop(columns=[col])
    
    print(f"Processed training set shape: {X_train_processed.shape}")
    print(f"Processed test set shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed

# Update the train_xgboost_model function
def train_xgboost_model(X_train, y_train):
    print("Training XGBoost model...")
    # Set XGBoost parameters
    params = {
        'objective': 'binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob',
        'eval_metric': 'auc' if len(np.unique(y_train)) == 2 else 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100
    }
    
    # Number of training rounds
    num_rounds = 100
    
    # Create DMatrix objects without enable_categorical
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train model with tqdm progress bar
    print("Training progress:")
    start_time = time.time()
    tqdm_callback = TqdmCallback(num_rounds)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        callbacks=[tqdm_callback]
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model

# Function to do cross-validation
def perform_cross_validation(X, y):
    print("Performing 3-fold cross-validation...")
    # Create XGBoost classifier with enable_categorical parameter
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic' if len(np.unique(y)) == 2 else 'multi:softprob',
        eval_metric='auc' if len(np.unique(y)) == 2 else 'mlogloss',
        eta=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        enable_categorical=True  # Add this parameter
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
        xgb_clf.fit(X_train_cv, y_train_cv)
        
        # Make predictions
        y_pred_proba = xgb_clf.predict_proba(X_val_cv)
        y_pred = xgb_clf.predict(X_val_cv)
        
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
    dtest = xgb.DMatrix(X_test)  # Remove enable_categorical parameter
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int) if len(np.unique(y_test)) == 2 else np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    if len(np.unique(y_test)) == 2:
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
    else:
        test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("Test Set Metrics:")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    # Compare with cross-validation and baseline
    print("\nComparison Summary:")
    print("Metric      | Test Set  | CV Average | Naive Baseline | Improvement over Baseline")
    print("------------|-----------|------------|----------------|-------------------------")
    print(f"ROC-AUC     | {test_roc_auc:.4f}    | {cv_metrics['roc_auc']:.4f}     | {baseline_metrics['roc_auc']:.4f}           | {(test_roc_auc - baseline_metrics['roc_auc']) * 100:.2f}%")
    print(f"Precision   | {test_precision:.4f}    | {cv_metrics['precision']:.4f}     | {baseline_metrics['precision']:.4f}           | {(test_precision - baseline_metrics['precision']) * 100:.2f}%")
    print(f"Recall      | {test_recall:.4f}    | {cv_metrics['recall']:.4f}     | {baseline_metrics['recall']:.4f}           | {(test_recall - baseline_metrics['recall']) * 100:.2f}%")
    print(f"Accuracy    | {test_accuracy:.4f}    | {cv_metrics['accuracy']:.4f}     | {baseline_metrics['accuracy']:.4f}           | {(test_accuracy - baseline_metrics['accuracy']) * 100:.2f}%")

def main():
    # Load data and columns to drop
    df = load_data_and_columns()
    
    # Check if target variable exists
    if 'Cancelled' not in df.columns:
        print(f"Error: Target variable 'Cancelled' not found in the dataset.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Preprocess features
    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
    
    # Train XGBoost model
    model = train_xgboost_model(X_train_processed, y_train)
    
    # Perform cross-validation (update this to use processed features)
    cv_metrics = perform_cross_validation(X_train_processed, y_train)
    
    # Calculate naive baseline
    baseline_metrics = calculate_naive_baseline(X_test_processed, y_test)
    
    # Evaluate model and compare with baseline (update this to use processed features)
    evaluate_model(model, X_test_processed, y_test, cv_metrics, baseline_metrics)
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
