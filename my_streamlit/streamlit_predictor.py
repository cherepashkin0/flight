import streamlit as st
import pandas as pd
import requests
import json

# Title of the app
st.title("Flight Delay Predictor")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Define FastAPI backend URL
API_URL = "http://localhost:8001/predict/"

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
                    # df["Prediction"] = predictions
                    st.success("Predictions received:")
                    st.dataframe(pd.DataFrame({"Prediction": predictions}))
                else:
                    st.error(f"Request failed: {response.status_code}\n{response.text}")
    except Exception as e:
        st.error(f"Error reading file or predicting: {e}")
