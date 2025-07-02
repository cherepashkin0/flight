from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import pandas as pd
import lightgbm as lgb

from pydantic_module import FlightFeatures 

app = FastAPI()

try:
    model = lgb.Booster(model_file="/home/vsiewolod/projects/flight/mycoding/results/2025-06-28/lightgbm_model.lgb")
except Exception as e:
    raise RuntimeError(f"Model could not be loaded: {e}")

@app.get("/predict/")
def explain_usage():
    return {
        "message": "Please send POST /predict/ with a JSON body containing a list of records."
    }

@app.post("/predict/")
def predict(data: List[FlightFeatures]):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        # Run prediction
        preds = model.predict(df)
        return JSONResponse(content={"predictions": preds.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
