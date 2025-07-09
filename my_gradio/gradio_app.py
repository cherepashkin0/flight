import gradio as gr
import pandas as pd
import requests

API_URL = "http://localhost:8001/predict/"

def predict_from_csv(file):
    try:
        df = pd.read_csv(file.name)
        records = df.to_dict(orient="records")

        response = requests.post(API_URL, json=records)
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            return pd.DataFrame({"Prediction": predictions})
        else:
            return f"Request failed: {response.status_code}\n{response.text}"
    except Exception as e:
        return f"Error reading file or predicting: {e}"

demo = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="Upload your CSV file"),
    outputs=gr.Dataframe(label="Predictions"),
    title="Flight Delay Predictor",
    description="Upload a CSV file to get predictions from the FastAPI backend."
)

if __name__ == "__main__":
    demo.launch()
