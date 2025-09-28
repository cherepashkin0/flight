FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .
# EXPOSE 8006

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["sh", "-c", "streamlit run streamlit_app.py \
#   --server.address=0.0.0.0 \
#   --server.port=${PORT:-8501} \
#   --browser.gatherUsageStats=false \
#   --server.enableCORS=false \
#   --server.enableXsrfProtection=false"]
