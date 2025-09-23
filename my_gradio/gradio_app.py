import gradio as gr
import pandas as pd
import requests
import tempfile
import os

# For Cloud Run deployment - replace with actual service URL
API_URL = "https://flight-api-330145081433.europe-west4.run.app/predict/"
# to run over k8s
# API_URL = "http://api-flight/predict/"
# to run over docker
# API_URL = "http://api:8001/predict/"

def predict_from_file(file):
    try:
        _, ext = os.path.splitext(file.name)
        ext = ext.lower()

        if ext == ".csv":
            df = pd.read_csv(file.name)
        elif ext == ".xlsx":
            df = pd.read_excel(file.name)
        else:
            return pd.DataFrame(), None, None, f"Поддерживаются только .csv и .xlsx файлы, получено: {ext}"

        records = df.to_dict(orient="records")
        response = requests.post(API_URL, json=records)
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            result_df = pd.DataFrame(predictions)
            visible_df = result_df[["prediction"]]

            base_name = os.path.splitext(os.path.basename(file.name))[0]
            tmpdir = tempfile.gettempdir()

            csv_path = os.path.join(tmpdir, f"{base_name}_predictions.csv")
            xlsx_path = os.path.join(tmpdir, f"{base_name}_predictions.xlsx")

            result_df.to_csv(csv_path, index=False)
            result_df.to_excel(xlsx_path, index=False)

            return visible_df, csv_path, xlsx_path, ""
        else:
            return pd.DataFrame(), None, None, f"Request failed: {response.status_code}\n{response.text}"

    except Exception as e:
        return pd.DataFrame(), None, None, f"Error reading file or predicting: {e}"


with gr.Blocks(title="Flight Delay Predictor") as demo:
    gr.Markdown("## Flight Delay Predictor\nUpload a CSV or XLSX file to get predictions (True/False). Updated for Cloud Run.")

    with gr.Row():
        file_input = gr.File(label="Upload your CSV or XLSX file")
    
    with gr.Row():
        output_table = gr.Dataframe(label="Prediction", interactive=False)
    
    with gr.Row():
        csv_download = gr.File(label="Download CSV with predictions")
        xlsx_download = gr.File(label="Download XLSX with predictions")
    
    with gr.Row():
        error_box = gr.Textbox(label="Error message", interactive=False)

    file_input.change(
        fn=predict_from_file,
        inputs=[file_input],
        outputs=[output_table, csv_download, xlsx_download, error_box]
    )

demo.launch(server_name="0.0.0.0", server_port=8005)
