import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import load_data, data_preprocessing, make_windows
from nn_model import build_model, predict_future_nn
from dt_model import build_dt, train_dt, predict_future_dt
from modelo_lineal import build_lr, train_lr, predict_future_lr

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("Stock Market Prediction Dashboard")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Parameters")
    ticker      = st.text_input("Ticker",        value="AAPL").upper().strip()
    period      = st.selectbox("Period",         ["6mo", "1y", "2y", "3y", "5y"], index=2)
    look_back   = st.slider("Look Back (days)",  min_value=10, max_value=120, value=60, step=5)
    forecast    = st.slider("Forecast Days",     min_value=1,  max_value=60,  value=10)

    st.divider()
    st.header("Models")

    run_lstm = st.checkbox("LSTM (Neural Net)", value=True)
    if run_lstm:
        with st.expander("LSTM parameters"):
            epochs     = st.slider("Epochs",      10, 200, 50, step=10)
            batch_size = st.slider("Batch Size",  8,  128, 32, step=8)
    else:
        epochs = 50
        batch_size = 32

    run_dt = st.checkbox("Decision Tree", value=True)
    if run_dt:
        with st.expander("Decision Tree parameters"):
            dt_max_depth = st.slider("Max Depth", 1, 100, 40)
    else:
        dt_max_depth = 40

    run_lr = st.checkbox("Linear Regression", value=True)

    st.divider()
    run_btn = st.button("Run", type="primary", use_container_width=True)

# ── Keras callback that writes epoch progress into a Streamlit progress bar ───
class _EpochBar(tf.keras.callbacks.Callback):
    def __init__(self, bar, total_epochs, base_frac, alloc_frac, label):
        self._bar   = bar
        self._total = total_epochs
        self._base  = base_frac
        self._alloc = alloc_frac
        self._label = label

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        frac = (epoch + 1) / self._total
        self._bar.progress(
            min(self._base + frac * self._alloc, 1.0),
            text=f"{self._label} — epoch {epoch + 1}/{self._total}  (loss {loss:.4f})",
        )

# ── Helpers ───────────────────────────────────────────────────────────────────
def mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def make_individual_fig(ticker, actual, forecast_arr, label, color, forecast_days):
    fore_x = list(range(len(actual) - 1, len(actual) + forecast_days))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual,
                             name="Actual", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=fore_x,
                             y=np.concatenate([[actual[-1]], forecast_arr]),
                             name=label, line=dict(color=color, dash="dash")))
    fig.add_vline(x=len(actual) - 1, line_dash="dot", line_color="gray")
    fig.update_layout(title=f"{ticker} — {label} ({forecast_days}-Day Forecast)",
                      xaxis_title="Trading Days", yaxis_title="Close Price (USD)",
                      legend=dict(orientation="h"), height=400)
    return fig

# ── State ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not any([run_lstm, run_dt, run_lr]):
        st.error("Select at least one model.")
    else:
        progress = st.progress(0, text="Loading data…")

        data = load_data(ticker, period)
        scaled, scaler = data_preprocessing(data)
        X, y = make_windows(scaled, look_back)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        results = {"actual": actual, "scaler": scaler, "forecasts": {}, "metrics": {}}
        n_models = sum([run_lstm, run_dt, run_lr])
        step = 1 / (n_models + 1)
        prog = step

        if run_lstm:
            progress.progress(prog, text="Training LSTM — initializing…")
            nn = build_model(look_back)
            nn.fit(
                X_train, y_train,
                epochs=epochs, batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[_EpochBar(progress, epochs, prog, step, "Training LSTM")],
            )
            nn_pred_scaled = nn.predict(X_test.reshape(len(X_test), look_back, 1), verbose=0).flatten()
            nn_pred = scaler.inverse_transform(nn_pred_scaled.reshape(-1, 1)).flatten()
            nn_fc   = predict_future_nn(nn, X_test[-1], scaler, forecast)
            results["forecasts"]["LSTM"] = nn_fc
            results["metrics"]["LSTM"] = {
                "MAE":  mean_absolute_error(actual, nn_pred),
                "RMSE": np.sqrt(mean_squared_error(actual, nn_pred)),
                "MAPE": mape(actual, nn_pred),
            }
            prog += step

        if run_dt:
            progress.progress(prog, text="Training Decision Tree…")
            dt = build_dt(dt_max_depth)
            dt = train_dt(dt, X_train, y_train)
            dt_pred_scaled = dt.predict(X_test.reshape(len(X_test), -1))
            dt_pred = scaler.inverse_transform(dt_pred_scaled.reshape(-1, 1)).flatten()
            dt_fc   = predict_future_dt(dt, X_test[-1], scaler, forecast)
            results["forecasts"]["Decision Tree"] = dt_fc
            results["metrics"]["Decision Tree"] = {
                "MAE":  mean_absolute_error(actual, dt_pred),
                "RMSE": np.sqrt(mean_squared_error(actual, dt_pred)),
                "MAPE": mape(actual, dt_pred),
            }
            prog += step

        if run_lr:
            progress.progress(prog, text="Training Linear Regression…")
            lr = build_lr()
            lr = train_lr(lr, X_train, y_train)
            lr_pred_scaled = lr.predict(X_test.reshape(len(X_test), -1))
            lr_pred = scaler.inverse_transform(lr_pred_scaled.reshape(-1, 1)).flatten()
            lr_fc   = predict_future_lr(lr, X_test[-1], scaler, forecast)
            results["forecasts"]["Linear Regression"] = lr_fc
            results["metrics"]["Linear Regression"] = {
                "MAE":  mean_absolute_error(actual, lr_pred),
                "RMSE": np.sqrt(mean_squared_error(actual, lr_pred)),
                "MAPE": mape(actual, lr_pred),
            }

        results["ticker"]   = ticker
        results["forecast"] = forecast
        progress.progress(1.0, text="Done!")
        st.session_state.results = results

# ── Display results ────────────────────────────────────────────────────────────
if st.session_state.results:
    r       = st.session_state.results
    actual  = r["actual"]
    fcs     = r["forecasts"]
    metrics = r["metrics"]
    t       = r["ticker"]
    fd      = r["forecast"]

    COLORS = {
        "LSTM":              "tomato",
        "Decision Tree":     "seagreen",
        "Linear Regression": "darkorange",
    }

    # ── Combined chart ────────────────────────────────────────────────────────
    st.subheader(f"{t} — All Models")
    fig_all = go.Figure()
    fore_x  = list(range(len(actual) - 1, len(actual) + fd))
    fig_all.add_trace(go.Scatter(x=list(range(len(actual))), y=actual,
                                 name="Actual", line=dict(color="steelblue", width=2)))
    for name, fc in fcs.items():
        fig_all.add_trace(go.Scatter(x=fore_x,
                                     y=np.concatenate([[actual[-1]], fc]),
                                     name=name,
                                     line=dict(color=COLORS[name], dash="dash")))
    fig_all.add_vline(x=len(actual) - 1, line_dash="dot", line_color="gray")
    fig_all.update_layout(xaxis_title="Trading Days", yaxis_title="Close Price (USD)",
                          legend=dict(orientation="h"), height=450)
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("Model Metrics")
    import pandas as pd
    rows = []
    for name, m in metrics.items():
        rows.append({"Model": name, "MAE": f"${m['MAE']:.2f}",
                     "RMSE": f"${m['RMSE']:.2f}", "MAPE": f"{m['MAPE']:.2f}%"})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # ── Individual charts ─────────────────────────────────────────────────────
    st.subheader("Individual Models")
    tabs = st.tabs(list(fcs.keys()))
    for tab, (name, fc) in zip(tabs, fcs.items()):
        with tab:
            st.plotly_chart(
                make_individual_fig(t, actual, fc, name, COLORS[name], fd),
                use_container_width=True,
            )
            m = metrics[name]
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"${m['MAE']:.2f}")
            c2.metric("RMSE", f"${m['RMSE']:.2f}")
            c3.metric("MAPE", f"{m['MAPE']:.2f}%")

            st.markdown("**Forecast prices**")
            fc_df = pd.DataFrame({"Day": range(1, fd + 1), "Predicted Close ($)": np.round(fc, 2)})
            st.dataframe(fc_df.set_index("Day"), use_container_width=True)
else:
    st.info("Configure parameters in the sidebar and press **Run** to start.")
