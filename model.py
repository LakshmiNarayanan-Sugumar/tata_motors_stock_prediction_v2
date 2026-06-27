import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import math
import os

# loading the saved data from preprocessing step

X_train=np.load("data/X_train.npy")
X_test=np.load("data/X_test.npy")
y_train=np.load("data/y_train.npy")
y_test=np.load("data/y_test.npy")
scaler=joblib.load("data/scaler.pkl")

# building the bilstm model

model= Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_test.shape[2])),
    Dropout(0.2),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.2),
    Dense(1)
])

# compiling

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# training with early stopping

early_stop=EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history=model.fit(
    X_train,y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

#plotting training loss

plt.figure(figsize=(10,4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("model loss during training")
plt.xlabel("epoch")
plt.ylabel("loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show() 

# prdiction and inverse scale

predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)
Y_test_actual=scaler.inverse_transform(y_test.reshape(-1,1))

# evaluavtion using mae and rmse

mae=mean_absolute_error(Y_test_actual, predictions)
rmse=math.sqrt(mean_squared_error(Y_test_actual, predictions))

print(f"\n MAE:  {mae:.2f}")
print(f"\n RMSE: {rmse:.2f}")

# plot for predicted vs actual

plt.figure(figsize=(14,5))
plt.plot(Y_test_actual, color="steelblue", label="actual price")
plt.plot(predictions, color="red", label="predicted price")
plt.title("TATA MOTORS - predicted vs actual closing price")
plt.xlabel("days")
plt.ylabel("price")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.show()

# saving the model

os.makedirs("model", exist_ok=True)
model.save("model/tata_motors_bilstm.keras")
print("\ncompleted")