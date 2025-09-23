# Flight Cancellation Prediction Project

This project aims to predict flight cancellations using machine learning models (LightGBM) and provides scripts to preprocess data, analyze it, train models, finetune hyperparameters using optuna, and deploy a prediction API and UI (streamlit and gradio). Deployment is supported on Google Kubernetes Engine (GKE) and via Google Cloud Build, triggered from GitLab CI (containers pushed to Artifact Registry).

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
├── my_gradio           # Streamlit application for UI
│   └── gradio_app.py
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

## Setup

### Requirements
- Python 3.11
- requirements.txt
- Google Cloud Platform account with BigQuery enabled
- MLflow server for experiment tracking

### Installation

```bash
pip install -r mycoding/requirements.txt
```

### Environment Variables
Set the following environment variables:
```bash
export GCP_PROJECT_ID='your-google-cloud-project-id'
```

## Data Loading

### Parquet to BigQuery
Load yearly Parquet files into BigQuery tables:
```bash
python mycoding/parquet_to_bigquery.py
```

### Analyze BigQuery Table
Generate descriptive statistics and save to CSV:
```bash
python mycoding/describe.py
```

### Assign Column Roles
Assign roles and preprocess dataset:
```bash
python mycoding/role_assign.py
```

## Model Training and Prediction

### Configuration
Adjust settings in `mycoding/config.yaml`:
- dataset name, table name, MLflow experiment name, sample size, etc.

### Training and Evaluation
Run the main training pipeline:
```bash
python mycoding/fit_predict.py
```
- This will perform hyperparameter tuning with Optuna and log results to MLflow.

## FastAPI Prediction API

### Generate Pydantic Models
```bash
python my_app/generate_pydentic_model.py
```

### Run API
```bash
uvicorn my_app.app:app --reload
```

Access API at:
```
http://127.0.0.1:8001/predict/
```

## Streamlit UI
Navigate to `my_streamlit/streamlit_predictor` and run:
```bash
streamlit run your_streamlit_script.py
```

## Notes
- Ensure paths and configurations in YAML and Python files are updated according to your local setup.
- MLflow and Optuna configurations are optional but recommended for optimal tracking and reproducibility.

