import os
import uuid
import json
import time
import yaml
import random
import logging

import clickhouse_connect
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import wandb
from joblib import dump

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------
# Config Loading
# -------------------------------
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

DB_NAME        = CONFIG["db"]["name"]
TABLE_NAME     = f"{DB_NAME}.{CONFIG['db']['table']}"
TARGET_COLUMN  = CONFIG["target_column"]
WANDB_PROJECT  = CONFIG["wandb"]["project"]
WANDB_ENTITY   = os.getenv("WANDB_ENTITY", CONFIG["wandb"].get("entity", "")).strip()
ARTIFACT_DIR   = CONFIG["artifact_dir"]
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Sanitize any literal "${...}" in entity
if WANDB_ENTITY.startswith("${") and WANDB_ENTITY.endswith("}"):
    WANDB_ENTITY = ""

# -------------------------------
# Utility Functions
# -------------------------------
def generate_readable_name() -> str:
    adjectives = ["brave", "curious", "gentle", "bold", "quiet", "lucky", "fierce", "bright"]
    nouns      = ["lion", "panda", "eagle", "otter", "tiger", "koala", "owl", "fox"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{uuid.uuid4().hex[:4]}"

def connect_to_clickhouse():
    logger.info("🔌 Connecting to ClickHouse...")
    client = clickhouse_connect.get_client(
        host     = os.getenv("CLICKHOUSE_HOST"),
        port     = os.getenv("CLICKHOUSE_PORT"),
        username = os.getenv("CLICKHOUSE_USER"),
        password = os.getenv("CLICKHOUSE_PASSWORD"),
    )
    logger.info("✅ ClickHouse connection established.")
    return client

# -------------------------------
# True Streaming Trainer
# -------------------------------
def train_streaming_sgd(
    client,
    table: str,
    target: str,
    chunk_size: int = 100_000,
    max_rows: int = None,
    test_fraction: float = 0.1,
):
    # 1) Sample a held-out test set
    logger.info("🧪 Sampling held-out test set (~%.0f%%)", test_fraction*100)
    test_query = (
        f"SELECT * FROM {table} "
        f"WHERE {target} IS NOT NULL "
        f"AND rand()%100 < {int(test_fraction*100)} "
        f"LIMIT {int(chunk_size)}"
    )
    df_test = client.query_df(test_query)
    # Drop any rows with missing values
    before = len(df_test)
    df_test = df_test.dropna()
    logger.info("ℹ️ Dropped %d/%d rows with NaNs from test set", before - len(df_test), before)
    if df_test.empty:
        raise RuntimeError("Test sample was empty after dropping NaNs—adjust test_fraction or inspect your data.")
    X_test = df_test.drop(columns=[target]).values
    y_test = df_test[target].astype(int).values
    # Use fixed binary classes
    classes = [0, 1]

    # 2) Incremental learner
    clf = SGDClassifier(
        loss="log_loss",   # sklearn>=1.0 name for logistic loss
        max_iter=1,
        tol=None,
        random_state=CONFIG["train_test_split"]["random_state"],
        learning_rate="optimal",
    )

    offset    = 0
    total     = 0
    first_fit = True

    # Prepare wandb.init kwargs
    init_kwargs = {
        "project": WANDB_PROJECT,
        "name":    f"streaming-sgd-{generate_readable_name()}",
        "config":  {},
        "reinit":  True,
    }
    if WANDB_ENTITY:
        init_kwargs["entity"] = WANDB_ENTITY

    try:
        wandb.init(**init_kwargs)
    except Exception as e:
        logger.warning("⚠️ W&B init failed: %s", e)

    start_time = time.time()

    # 3) Streaming training loop
    while True:
        query = (
            f"SELECT * FROM {table} "
            f"WHERE {target} IS NOT NULL "
            f"AND rand()%100 >= {int(test_fraction*100)} "
            f"LIMIT {chunk_size} OFFSET {offset}"
        )
        df_chunk = client.query_df(query)
        if df_chunk.empty:
            logger.info("🛑 No more training data to fetch.")
            break

        # Drop any rows with NaNs before training
        before = len(df_chunk)
        df_chunk = df_chunk.dropna()
        dropped = before - len(df_chunk)
        if dropped:
            logger.debug("ℹ️ Dropped %d/%d rows with NaNs from this chunk", dropped, before)
        if df_chunk.empty:
            offset += chunk_size
            continue

        X_batch = df_chunk.drop(columns=[target]).values
        y_batch = df_chunk[target].astype(int).values

        if first_fit:
            clf.partial_fit(X_batch, y_batch, classes=classes)
            first_fit = False
        else:
            clf.partial_fit(X_batch, y_batch)

        offset += chunk_size
        total  += len(df_chunk)
        logger.debug("✅ Trained on chunk %d → %d total rows", offset//chunk_size, total)

        if max_rows and total >= max_rows:
            logger.info("🔒 Reached max_rows=%d", max_rows)
            break

    # 4) Final evaluation on hold-out
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred       = clf.predict(X_test)
    metrics = {
        "accuracy":           accuracy_score(y_test, y_pred),
        "f1_score":           f1_score(y_test, y_pred),
        "roc_auc":            roc_auc_score(y_test, y_pred_proba),
        "training_time_sec":  round(time.time() - start_time, 2),
        "n_samples_trained":  total,
    }
    for k, v in metrics.items():
        try:
            wandb.log({k: v})
        except Exception:
            pass

    logger.info("📊 Final metrics: %s", metrics)

    # 5) Save the model
    model_path = os.path.join(ARTIFACT_DIR, "sgd_streaming_model.pkl")
    dump(clf, model_path)
    logger.info("💾 Saved model to %s", model_path)
    try:
        wandb.save(model_path)
    except Exception:
        pass

    wandb.finish()
    return clf, metrics

# -------------------------------
# Main
# -------------------------------
def main():
    client = connect_to_clickhouse()

    _, metrics = train_streaming_sgd(
        client,
        TABLE_NAME,
        TARGET_COLUMN,
        chunk_size   = CONFIG.get("stream_chunk_size", 100_000),
        max_rows     = CONFIG.get("max_rows", None),
        test_fraction= CONFIG.get("test_fraction", 0.1),
    )

    out_path = os.path.join(ARTIFACT_DIR, CONFIG["output_metrics"])
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("📁 Results saved to %s", out_path)

if __name__ == "__main__":
    main()
