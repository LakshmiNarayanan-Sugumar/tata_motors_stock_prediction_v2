import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# loading saved data
X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
scaler_y = joblib.load("data/scaler_y.pkl")


# build BiLSTM model
# FIX: both dimensions of input_shape use X_train, not X_test
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True),
        input_shape=(X_train.shape[1], X_train.shape[2])),  # was X_test.shape[2]
    Dropout(0.2),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# early stopping on validation loss
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,   # takes last 10% of X_train as val — fine since split is time-based
    callbacks=[early_stop],
    verbose=1
)
scaler_y = joblib.load("data/scaler_y.pkl")

# inverse transform using scaler_y
predictions   = scaler_y.inverse_transform(model.predict(X_test))
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# compute MAE and RMSE in actual rupee values
mae  = mean_absolute_error(y_test_actual, predictions)
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))

print(f"\nMAE:  ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")

# plot predicted vs actual
plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, color="steelblue", label="Actual Price")
plt.plot(predictions,   color="red",       label="Predicted Price")
plt.title("TMPV.NS — Predicted vs Actual Closing Price")
plt.xlabel("Days")
plt.ylabel("Price (₹)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()

# save model
os.makedirs("model", exist_ok=True)
model.save("model/tata_motors_bilstm.keras")
print("model saved")