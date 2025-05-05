import os
import json
import time
import uuid
import random
import logging
import warnings
import glob

import yaml
import numpy as np
import pandas as pd
import clickhouse_connect
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------
# Suppress Specific Warnings
# -------------------------------
warnings.filterwarnings(
    "ignore",
    message=".*No further splits with positive gain.*",
    category=UserWarning
)

# -------------------------------
# Configure Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------------------
# Load Config
# -------------------------------
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

DB_NAME        = cfg['db']['name']
TABLE_NAME     = f"{DB_NAME}.{cfg['db']['table']}"
TARGET_COLUMN  = cfg['target_column']

OUTPUT_METRICS = cfg['output']['metrics_file']
ARTIFACT_DIR   = cfg['output']['artifact_dir']
# create artifact dir if missing
os.makedirs(ARTIFACT_DIR, exist_ok=True)

WANDB_PROJECT  = cfg['wandb']['project']
WANDB_ENTITY   = cfg['wandb']['entity'] or os.getenv('WANDB_ENTITY')

PARAM_GRIDS    = cfg['param_grids']
CV_FOLDS       = cfg.get('cv_folds', 3)
DRY_RUN        = cfg.get('dry_run', False)
SAMPLE_FRAC    = cfg.get('sample_fraction', 0.1)

# Generate a single suffix for all models
RUN_SUFFIX = uuid.uuid4().hex[:4]

# -------------------------------
# ClickHouse Helpers
# -------------------------------
def connect_to_clickhouse():
    logging.info("Connecting to ClickHouse at %s:%s", os.getenv('CLICKHOUSE_HOST'), os.getenv('CLICKHOUSE_PORT'))
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST'),
        port=os.getenv('CLICKHOUSE_PORT'),
        username=os.getenv('CLICKHOUSE_USER'),
        password=os.getenv('CLICKHOUSE_PASSWORD')
    )


def load_data(client, table):
    logging.info("Loading data from table %s", table)
    query = f"SELECT * FROM {table} WHERE {TARGET_COLUMN} IS NOT NULL"
    df = client.query_df(query)
    logging.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
    if DRY_RUN:
        sample_n = int(len(df) * SAMPLE_FRAC)
        logging.info("Dry run: sampling %d rows (%.1f%%)", sample_n, SAMPLE_FRAC*100)
        df = df.sample(n=sample_n, random_state=42)
        logging.info("Post-sample size: %d rows", len(df))
    return df

# -------------------------------
# Readable Run Names
# -------------------------------
def generate_readable_name():
    adjectives = ["brave", "curious", "gentle", "bold", "quiet", "lucky", "fierce",
                  "bright", "silly", "wise", "wild", "calm", "cheerful", "kind"]
    nouns = ["lion", "panda", "eagle", "otter", "dolphin", "tiger", "koala",
             "owl", "fox", "penguin", "bear", "falcon", "shark", "whale"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{RUN_SUFFIX}"

# -------------------------------
# Model Mapping & Training
# -------------------------------
MODEL_MAPPING = {
    'xgboost': XGBClassifier(eval_metric='logloss', random_state=42),
    'lightgbm': LGBMClassifier(random_state=42, verbose=-1),
    'catboost': CatBoostClassifier(verbose=0, random_seed=42)
}

def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, grid_search):
    logging.info("Training final %s model", model_name)
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cv_score_mean': grid_search.best_score_  #  Mean cross-validated score of the best_estimator
    }
    logging.info("%s metrics: %s", model_name, metrics)
    return grid_search.best_params_, metrics

# -------------------------------
# Main
# -------------------------------
def main():
    client = connect_to_clickhouse()
    df = load_data(client, TABLE_NAME)

    y = df[TARGET_COLUMN].astype(int)
    features = ['CRSDepTime', 'CRSArrTime', 'Distance', 'Year', 'Quarter', 'Month', 'DayOfWeek',
                'DOT_ID_Marketing_Airline', 'Flight_Number_Marketing_Airline', 'Flight_Number_Operating_Airline',
                'OriginAirportID', 'OriginCityMarketID', 'OriginStateFips', 'OriginWac',
                'DestAirportID', 'DestCityMarketID', 'DestStateFips',
                'FlightDate_year', 'FlightDate_month', 'FlightDate_day', 'FlightDate_weekday', 'FlightDate_dayofyear']
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    logging.info("Train/test split: %d/%d", len(y_train), len(y_test))

    results = {}
    logging.info("Begin training/evaluation: %d-fold CV", CV_FOLDS)

    for model_name in tqdm(MODEL_MAPPING, desc="Models"):
        logging.info("--- GridSearchCV: %s ---", model_name)
        run_name = f"{model_name}_{generate_readable_name()}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            entity=WANDB_ENTITY,
            config=PARAM_GRIDS[model_name],
            reinit=True
        )

        gs = GridSearchCV(
            estimator=MODEL_MAPPING[model_name],
            param_grid=PARAM_GRIDS[model_name],
            cv=CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
        try:
            start = time.time()
            gs.fit(X_train, y_train)
            elapsed = time.time() - start
            logging.info("Completed %s in %.1f sec", model_name, elapsed)
            logging.info("Best CV score: %.4f params: %s", gs.best_score_, gs.best_params_)

            # Heatmaps per learning_rate
            df_cv = pd.DataFrame(gs.cv_results_)
            # identify hyperparams
            keys = list(PARAM_GRIDS[model_name].keys())
            p_fixed, p_vary = keys[:2]
            if 'learning_rate' in keys:
                for lr in sorted(df_cv['param_learning_rate'].unique()):
                    df_lr = df_cv[df_cv['param_learning_rate'] == lr]
                    pivot = df_lr.groupby([f'param_{p_fixed}', f'param_{p_vary}'])['mean_test_score'].mean().unstack()
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(pivot, annot=True, fmt='.3f')
                    plt.title(f"{model_name} lr={lr} CV Mean ROC_AUC")
                    fname = os.path.join(ARTIFACT_DIR, f"{model_name}_heatmap_lr{lr}_{RUN_SUFFIX}.png")
                    plt.savefig(fname, bbox_inches='tight')
                    plt.close()
                    logging.info("Saved heatmap: %s", fname)
                    # wandb.log({f"{model_name}_heatmap_lr_{lr}": wandb.Image(fname)})
        except Exception:
            logging.exception("GridSearchCV error: %s", model_name)
            continue

        best_params, metrics = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, gs)
        wandb.log({'grid_search_time_sec': elapsed, **best_params, **metrics})
        wandb.finish()
        results[model_name] = {'best_params': best_params, 'metrics': metrics}

    # Save metrics JSON
    out = os.path.join(ARTIFACT_DIR, OUTPUT_METRICS)
    with open(out, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info("Results saved: %s", out)

    # Compose all heatmaps into a PDF
    heatmaps = sorted(glob.glob(os.path.join(ARTIFACT_DIR, f"*_heatmap_lr*_{RUN_SUFFIX}.png")))
    if heatmaps:
        pdf_path = os.path.join(ARTIFACT_DIR, f"all_heatmaps_{RUN_SUFFIX}.pdf")
        with PdfPages(pdf_path) as pdf:
            for img_file in heatmaps:
                img = plt.imread(img_file)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        logging.info("All heatmaps PDF saved: %s", pdf_path)

if __name__ == '__main__':
    main()
