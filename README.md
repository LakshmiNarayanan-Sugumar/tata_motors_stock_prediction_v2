# Tata Motors Stock Price Prediction — V2

Predicting the next-day closing price of Tata Motors Passenger Vehicles (TMPV.NS) using a Bidirectional LSTM model trained on 7 years of historical price data.

---

## What's different from V1

V1 used news sentiment and linear regression. V2 uses only historical closing prices and a BiLSTM — a deep learning model that learns temporal patterns in price sequences. No sentiment, no same-day features, no data leakage.

V2 also corrects two methodological issues from an earlier iteration:

- **Scaler leakage fixed** — MinMaxScaler is now fit only on training data and applied to test data separately, preventing future price statistics from influencing the model during training.
- **Input shape fixed** — `input_shape` now correctly uses `X_train` dimensions instead of `X_test`.

---

## Data

| Field | Value |
|-------|-------|
| Ticker | TMPV.NS (Tata Motors Passenger Vehicles, NSE — post-demerger entity) |
| Period | January 2019 – June 2026 |
| Trading days | 1,849 |
| Input feature | Closing price only |
| Lookback window | 60 days |

---

## Data Pipeline

- Raw closing prices fetched via yFinance
- Sliding window sequences built on unscaled prices
- Strict time-based train/test split (80/20) — no random shuffling, preserves chronological order
- MinMaxScaler fit only on training data, applied separately to test set
- Test set scaled values range 0.213–0.771 (outside 0–1, confirming scaler was fit on train only)

---

## Model Architecture

| Layer | Output Shape | Params |
|-------|-------------|--------|
| BiLSTM (64 units, return_sequences=True) | (None, 60, 128) | 33,792 |
| Dropout (0.2) | (None, 60, 128) | 0 |
| BiLSTM (32 units, return_sequences=False) | (None, 64) | 41,216 |
| Dropout (0.2) | (None, 64) | 0 |
| Dense (1) | (None, 1) | 65 |

Total params: 75,073 (293.25 KB)

Optimizer: Adam | Loss: MSE | Early stopping: patience=10, best weights restored

---

## Results

| Metric | Value |
|--------|-------|
| MAE | ₹20.43 |
| RMSE | ₹34.74 |

Evaluated on a strictly held-out test set (last 20% of trading days, ~358 days). The test period includes a significant price shock (sharp drop from ~₹680 to ~₹400), making this a harder evaluation than a stable trending stock. The model correctly tracks the crash direction with a short lag — expected behavior for a 60-day lookback model encountering a sudden shock event.

---

## Visualizations

![Closing Price](closing_price_plot.png)
![Predicted vs Actual](predicted_vs_actual.png)

---

## API Endpoint

Built a REST API using FastAPI to serve model predictions.

```bash
uvicorn predict_api:app --reload
```

**POST /predict** — returns next-day predicted closing price

```json
{
  "ticker": "TMPV.NS",
  "current_price": 352.20,
  "predicted_price": 340.60,
  "currency": "INR"
}
```

Interactive docs available at `http://127.0.0.1:8000/docs`

---

## Docker Deployment

The FastAPI serving layer (`predict_api.py`) is containerized for consistent, portable deployment.

**Build:**
```bash
docker build -t tata-motors-api .
```

**Run:**
```bash
docker run -p 8000:8000 tata-motors-api
```

**Test:**
```bash
curl http://localhost:8000/
curl -X POST http://localhost:8000/predict
```

### A real debugging note

The initial build used a `python:3.10-slim` base image, which failed at model load time with a Keras deserialization error (`Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`). The cause wasn't a package version mismatch — it was that Keras 3.15+ requires Python ≥3.11, so the container's Python 3.10 runtime couldn't satisfy the dependency at all, regardless of how precisely `tensorflow`/`keras` were pinned in `requirements-serving.txt`. Switching the base image to `python:3.12-slim` (matching the local development environment) resolved it.

**Lesson:** pinning application-level package versions doesn't help if the base image's language runtime itself is out of range for what those packages require — check the runtime version first, not just the package versions.

---

## How to Run

**Locally:**
```bash
pip install -r requirements.txt
python data_preprocessing.py
python model.py
uvicorn predict_api:app --reload
```

**Via Docker (serving only):**
```bash
docker build -t tata-motors-api .
docker run -p 8000:8000 tata-motors-api
```
---

## Stack

Python, TensorFlow, Keras, yFinance, scikit-learn, NumPy, pandas, matplotlib, FastAPI

---

**V1 of this project used news sentiment and linear regression.**
[View V1 here](https://github.com/LakshmiNarayanan-Sugumar/tata_motors_stock_prediction_v1)