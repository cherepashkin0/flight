FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_app.py .
EXPOSE 8005

CMD ["python", "gradio_app.py"]
