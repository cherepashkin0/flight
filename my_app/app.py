from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import joblib
import gcsfs
import io
import os

from pydantic_module import FlightFeatures
from google.cloud import bigquery

app = FastAPI()

BQ_PROJECT = "flight-cancellation-prediction"
BQ_DATASET = "mlflow_tracking"
BQ_TABLE = "mlflow_metrics_log"
TARGET_PROJECT_NAME = "flight_cancellation_project"



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

def load_pipeline():
    global pipeline, preprocessor
    if pipeline is None:
        GCS_URI = get_best_model_uri()
        fs = gcsfs.GCSFileSystem()
        with fs.open(GCS_URI, "rb") as f:
            buffer = io.BytesIO(f.read())
            pipeline = joblib.load(buffer)
            preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
    return pipeline

try:
    pipeline = None
    preprocessor = None
    pipeline = load_pipeline()

    # preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)

except Exception as e:
    raise RuntimeError(f"Pipeline could not be loaded: {e}")


@app.post("/predict/xlsx/")
async def predict_from_xlsx(file: UploadFile = File(...)):
    try:
        # Проверяем расширение
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="Только файлы .xlsx поддерживаются.")
        
        # Читаем Excel-файл в pandas DataFrame
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Проверка/приведение формата колонок под ожидания модели (по желанию!)
        # df = df[ОЖИДАЕМЫЕ_КОЛОНКИ]  # если требуется
        
        # Делаем предсказание
        predictions = pipeline.predict(df).tolist()
        df["prediction"] = predictions
        
        # Возвращаем результат как JSON
        return JSONResponse(content={"predictions": df.to_dict(orient="records")})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/predict/")
def explain_usage():
    return {
        "message": "Please send POST /predict/ with a JSON body containing a list of records."
    }

@app.post("/predict/")
def predict(data: List[FlightFeatures]):
    try:
        # Convert incoming JSON list to pandas DataFrame
        df = pd.DataFrame([record.model_dump() for record in data])
        
        # Make predictions
        predictions = pipeline.predict(df).tolist()
        
        # Add predictions to the DataFrame
        df["prediction"] = predictions
        
        # Convert back to list of dicts for JSON response
        return JSONResponse(content={"predictions": df.to_dict(orient="records")})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health_check():
    return {"status": "ok"}
