# Flight Cancellation Prediction Project

This project aims to predict flight cancellations using machine learning models (LightGBM) and provides scripts to preprocess data, analyze it, train models, finetune hyperparameters using optuna, and deploy a prediction API and UI (streamlit or gradio). Deployment is supported either on Google Kubernetes Engine (GKE), or via Google Cloud Build, triggered from GitLab CI (containers pushed to Artifact Registry).

The training pipeline in this repository is not a notebook - it's a clean, production-style script.

It:

- reads config + environment variables
- dynamically pulls only the needed columns directly from BigQuery
- filters feature roles based on a describe CSV (-> reproducible feature selection)
- runs Optuna hyperparameter search for LightGBM (with custom composite metric)
- trains the final model and logs everything into MLflow
- finally writes model metadata + metrics back into BigQuery for lineage

So the overall message is:

model training is automated, parameterized, traceable in MLflow, and directly integrated with the data warehouse (BQ).

This design makes retraining re-entrant (you can re-run it tomorrow and get comparable results), and it already "thinks" like MLOps: tracking, metadata, reproducibility, BQ -> MLflow -> BQ feedback loop - not experimental Jupyter.


## Project Structure

```
.
├── my_app                 # FastAPI Application
│   ├── app.py             # Prediction API
│   ├── generate_pydentic_model.py # Script to generate Pydantic models
│   └── pydantic_module.py # Generated Pydantic models
│
├── my_streamlit           # Streamlit application for UI
│   └── steamlit_app.py
│
└── mycoding               # Main ML scripts and utilities
    ├── config.yaml        # Project configuration
    ├── describe.py        # BigQuery table analysis
    ├── fit_predict.py     # ML pipeline with model training and evaluation
    ├── parquet_to_bigquery.py # Load Parquet files into BigQuery
    ├── role_assign.py     # Assign roles to dataset columns
    ├── make_one_csv.py    # Prepare sample data for uploading to the app    
    └── results            # Output data and analysis results
```