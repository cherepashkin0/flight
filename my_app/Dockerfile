FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY pydantic_module.py .

CMD ["uvicorn", "app:app", "--port", "8001", "--host", "0.0.0.0"]
