import numpy as np
from sklearn.tree import DecisionTreeRegressor


def build_dt(dt_depth):
    model = DecisionTreeRegressor(max_depth=dt_depth, random_state=42)
    return model

def train_dt(dt, X_train, y_train):
    dt.fit(X_train.reshape(len(X_train), -1), y_train)
    return dt

def predict_future_dt(dt, last_window, scaler, n_days):
    window = last_window.copy()
    preds = []
    for _ in range(n_days):
        pred = dt.predict(window.reshape(1, -1))[0]
        preds.append(pred)
        window = np.append(window[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
