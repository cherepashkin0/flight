# load parquet file from local
import pandas as pd
import os
import json
# import yaml to read from file
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import time
import wandb
# print all columns in df.head pandas
pd.set_option('display.max_columns', None)



def main():
    # load parameters from yaml file
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    # set wandb in disabled mode for local testing
    
    wandb.init(
        mode=params['wandb']['mode'],
        project=params['wandb']['project'],
        entity=params['wandb']['entity'],
        config=params
    )

    ts = int(time.time())
    df = pd.read_parquet('processed_data/flight_all_preprocessed.parquet')
    # print(df.head())
    # print(df.columns)
    # print(df.describe())

    # split data into train and test using random shuffle
    train_df, test_df = train_test_split(df[:1_000_000], test_size=0.2, random_state=42)
    # drop target column from train and add it to test data
    X_train = train_df.drop(columns=['Cancelled'])
    y_train = train_df['Cancelled']
    X_test = test_df.drop(columns=['Cancelled'])
    y_test = test_df['Cancelled']

    # testing data leakage
    y_baseline = [0]*len(y_test)
    print("Baseline accuracy:", accuracy_score(y_test, y_baseline))

    corrs = X_train.join(y_train.rename("Cancelled")) \
               .corr()["Cancelled"] \
               .abs() \
               .sort_values(ascending=False)
    print("Top 10 feature correlations:\n", corrs.head(10))




    # train xgboost model, hyperparameters are set in params.yaml
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=params['xgboost']['max_depth'],
        learning_rate=params['xgboost']['learning_rate'],
        n_estimators=params['xgboost']['n_estimators'],
        subsample=params['xgboost']['subsample'],
        colsample_bytree=params['xgboost']['colsample_bytree'],
        gamma=params['xgboost']['gamma'],
        min_child_weight=params['xgboost']['min_child_weight'],
        scale_pos_weight=params['xgboost']['scale_pos_weight']
    )
    model.fit(X_train, y_train)
    # predict on test data
    y_pred = model.predict(X_test)

    n_test   = len(y_test)
    n_errors = (y_test != y_pred).sum()
    print(f"errors: {n_errors} / {n_test}   ({n_errors/n_test:.6%})")

    # use metrics to evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    tf = int(time.time())
    print(f"Time taken: {tf - ts} seconds") 


if __name__ == "__main__":
    main()
