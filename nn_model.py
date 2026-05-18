import numpy as np
import tensorflow as tf

def build_model(look_back):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'],
    )
    return model

def predict_future_nn(model, last_window, scaler, n_days):
    window = last_window.copy()
    preds = []
    for _ in range(n_days):
        # window is shape (60, 1) — 60 days, 1 column.
        # The model expects shape (batch, timesteps, features), so reshape to (1, 60, 1)
        # batch of 1 sample, 60 timesteps, 1 feature.
        pred = model.predict(window.reshape(1, len(window), 1), verbose=0)[0, 0]
        preds.append(pred)
        # Append the prediction to the window
        window = np.append(window[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
