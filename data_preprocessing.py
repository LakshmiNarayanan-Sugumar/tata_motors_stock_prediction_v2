import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import os

#fetching data from yfinance

print("fetching tmpv.ns data")
df=yf.download("TMPV.NS", start="2019-01-01", end="2026-06-25")
print("shape:", df.shape)

#exploratory data analysis

print("\nmissing values:")
print(df.isnull().sum())

plt.figure(figsize=(14, 5))
plt.plot(df["Close"], color="steelblue", linewidth=1.3)
plt.title("tmpv.ns - closing price(2019-2026)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("closing_price_plot.png")



#selecting close and normalization

close_prices=df[["Close"]].values
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(close_prices)
print("scaled data shape:", scaled_data.shape)
print(f"min: {scaled_data.min():.3f}, max: {scaled_data.max():.3f}")

#sliding window

lookback=60

X,y=[],[]
for i in range(lookback,len(scaled_data)):
    X.append(scaled_data[i-lookback:i,0])
    y.append(scaled_data[i,0])

X, y= np.array(X), np.array(y)
print("\n X shape:", X.shape)
print("\n y shape:", y.shape)    

#train and test split of data

split=int(len(X) * 0.80)
X_train, X_test=X[:split], X[split:]
y_train, y_test=y[:split], y[split:]
print("\n train samples:", len(X_train))
print("\n test samples:", len(X_test))

# reshaping for lstm 2d to 3d

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
print("test reshaped:", X_train.shape)

# saving files as .npy for model training

os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)
np.save("data/close_prices.npy", close_prices)


import joblib
joblib.dump(scaler, "data/scaler.pkl")
print("\nAll arrays and scaler saved to data")
