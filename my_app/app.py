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
TARGET_PROJECT_NAME = os.getenv("TARGET_PROJECT_NAME", "flight_cancellation_project")


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
preprocessor = None

def load_pipeline():
    global pipeline, preprocessor
    if pipeline is not None:
        return pipeline
    GCS_URI = get_best_model_uri()
    fs = gcsfs.GCSFileSystem()
    with fs.open(GCS_URI, "rb") as f:
        buffer = io.BytesIO(f.read())
        p = joblib.load(buffer)
    # cache
    pipeline = p
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
    return pipeline


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
        p = load_pipeline()  # load on demand
        df = pd.DataFrame([record.model_dump() for record in data])
        preds = p.predict(df).tolist()
        df["prediction"] = preds
        return JSONResponse(content={"predictions": df.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction temporarily unavailable: {e}")


@app.get("/health")
def health_check():
    # App is alive even if model isn't loaded yet
    # Optionally try a non-fatal model touch:
    try:
        _ = pipeline  # do not force load here
        return {"status": "ok", "model_loaded": bool(pipeline)}
    except Exception:
        return {"status": "ok", "model_loaded": False}
