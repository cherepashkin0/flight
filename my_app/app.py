from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import joblib
import gcsfs  # Google Cloud filesystem
import io

from pydantic_module import FlightFeatures

app = FastAPI()

GCS_URI = "gs://mlflow_flight_cancellation/artifacts/0/addea66c90a04e6dbc9c09412ac8df26/artifacts/catboost_pipeline.pkl"

try:
    # Initialize GCS file system
    fs = gcsfs.GCSFileSystem()

    # Open the file from GCS as a stream
    with fs.open(GCS_URI, "rb") as f:
        buffer = io.BytesIO(f.read())
        model = joblib.load(buffer)

except Exception as e:
    raise RuntimeError(f"Model could not be loaded from GCS: {e}")

@app.get("/predict/")
def explain_usage():
    return {
        "message": "Please send POST /predict/ with a JSON body containing a list of records."
    }

@app.post("/predict/")
def predict(data: List[FlightFeatures]):
    try:
        df = pd.DataFrame([record.dict() for record in data])
        preds_df = model.predict(df)
        print('predictoin for this file is', preds_df)

        # If it's a DataFrame, extract only the prediction column
        if isinstance(preds_df, pd.DataFrame) and "prediction" in preds_df.columns:
            predictions = preds_df["prediction"].tolist()
        else:
            predictions = preds_df.tolist()
        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
