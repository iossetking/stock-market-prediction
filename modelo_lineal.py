from sklearn.linear_model import LinearRegression
import numpy as np


def build_lr():
    return LinearRegression()

def train_lr(lr, X_train, y_train):
    lr.fit(X_train.reshape(len(X_train), -1), y_train)
    return lr

def predict_future_lr(lr, last_window, scaler, n_days):
    window = last_window.copy()
    preds = []
    for _ in range(n_days):
        pred = lr.predict(window.reshape(1, -1))[0]
        preds.append(pred)
        window = np.append(window[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
