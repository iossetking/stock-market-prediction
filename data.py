import sys

import numpy as np

import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker, period):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval="1d", auto_adjust=True)
    return df[["Close"]].dropna().values.reshape(-1, 1)

def data_preprocessing(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def make_windows(scaled, look_back):
    # X = [t-60, t-59, ..., t-1], y = t
    # X recieves a window of 60 days to predict the next day
    # y is the next datay
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i : i + look_back])
        y.append(scaled[i + look_back, 0])  # target: Close (col 0)
    return np.array(X), np.array(y)

# Test data
if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    data = load_data(ticker, "2y")
    scaled, scaler = data_preprocessing(data)
    X, y = make_windows(scaled, 60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
