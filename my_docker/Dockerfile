# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1s
ENV GCP_PROJECT_ID="flight-cancellation-prediction"
ENV MLFLOW_TRACKING_URI=http://104.197.51.75:8000

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY ../mycoding/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY /mycoding/fit_predict.py .
COPY /mycoding/my_app/generate_pydantic.py /app/
COPY /mycoding/app.py /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
