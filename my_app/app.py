from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import lightgbm as lgb
import joblib
from io import BytesIO

app = FastAPI()

try:
    model = lgb.Booster(model_file="/home/vsiewolod/projects/flight/mycoding/results/2025-06-28/lightgbm_model.lgb")
except Exception as e:
    raise RuntimeError(f"Model could not be loaded: {e}")

@app.get("/predict/")
def explain_usage():
    return {
        "message": "Please use POST /predict/ to upload a CSV file for prediction."
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)): # request body as a file upload
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read() # other commands to the app might be read during the file load
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    
    try:
        preds = model.predict(df)
        return JSONResponse(content={"predictions": preds.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


