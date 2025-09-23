import streamlit as st
import pandas as pd
import requests
import json

# Title of the app
st.title("Flight Cancellation Predictor")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Define FastAPI backend URL
# to run over docker
# API_URL = "http://api-flight:8001/predict/"
# to run over kubernetes
# API_URL = "http://api-flight/predict/"
# API_URL = "http://api:8001/predict/"
# For Cloud Run deployment - replace with actual service URL
API_URL = "https://flight-api-330145081433.europe-west4.run.app/predict/"

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Convert dataframe to list of dicts
        records = df.to_dict(orient="records")

        # Send POST request to FastAPI
        if st.button("Get Predictions"):
            with st.spinner("Getting predictions..."):
                response = requests.post(API_URL, json=records)
                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    result_df = pd.DataFrame(predictions)
                    st.success("Predictions received:")
                    st.dataframe(result_df)
                else:
                    st.error(f"Request failed: {response.status_code}\n{response.text}")
    except Exception as e:
        st.error(f"Error reading file or predicting: {e}")
