from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
# import mlflow.sklearn
import io
import os
from pydantic_module import FlightFeatures
from google.cloud import bigquery
import gcsfs
import joblib

fs = gcsfs.GCSFileSystem()

app = FastAPI()

BQ_PROJECT = "flight-cancellation-pred"
BQ_DATASET = "mlflow_tracking"
BQ_TABLE = "mlflow_metrics_log"
TARGET_PROJECT_NAME = os.getenv("TARGET_PROJECT_NAME", "flight_cancellation_pred")

def get_best_model_uri():
    client = bigquery.Client(project=BQ_PROJECT)
    query = f"""
        SELECT pipeline_uri
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE project_name = @project_name
        AND metric_name = 'f2_score'
        ORDER BY metric_value DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("project_name", "STRING", TARGET_PROJECT_NAME)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    result = query_job.result()
    for row in result:
        return row.pipeline_uri
    raise RuntimeError("No model found for the specified project.")

pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline
    
    try:
        model_uri = get_best_model_uri()
        with fs.open(model_uri.replace('gs://', ''), "rb") as f:
            pipeline = joblib.load(f)
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Pipeline loading failed: {str(e)}. URI of best model is {model_uri}")

@app.post("/predict/xlsx/")
async def predict_from_xlsx(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported.")
    
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    
    p = load_pipeline()
    predictions = p.predict(df).tolist()
    df["prediction"] = predictions
    
    return JSONResponse(content={"predictions": df.to_dict(orient="records")})

@app.get("/predict/")
def explain_usage():
    return {"message": "Send POST /predict/ with JSON body containing list of records."}

@app.post("/predict/")
def predict(data: List[FlightFeatures]):
    try:
        p = load_pipeline()
        df = pd.DataFrame([record.model_dump() for record in data])
        preds = p.predict(df).tolist()
        df["prediction"] = preds
        return JSONResponse(content={"predictions": df.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction failed: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": bool(pipeline)}
