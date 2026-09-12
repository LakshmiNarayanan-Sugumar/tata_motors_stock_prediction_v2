FROM python:3.12-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY predict_api.py .
COPY tata_motors_bilstm.keras .
COPY data/scaler_y.pkl ./data/scaler_y.pkl

EXPOSE 8000

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]