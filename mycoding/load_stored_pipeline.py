import joblib
import gcsfs

fs = gcsfs.GCSFileSystem()

with fs.open("mlflow_flight_cancellation3/artifacts/1/eef46dafe7b2483386c780a855a0dbf4/artifacts/model.joblib", "rb") as f:
    pipeline = joblib.load(f)

print(pipeline)
