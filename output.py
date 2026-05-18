import numpy as np
import matplotlib.pyplot as plt


def save_chart(ticker, y_test, nn_forecast, scaler, dt_forecast=None, lr_forecast=None, filename=None):
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    hist_x = range(len(actual))
    fore_x = range(len(actual) - 1, len(actual) + len(nn_forecast))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(hist_x, actual, label='Actual', color='steelblue')
    ax.plot(fore_x, np.concatenate([[actual[-1]], nn_forecast]),
            label='LSTM Forecast', color='tomato', linestyle='--')
    if dt_forecast is not None:
        ax.plot(fore_x, np.concatenate([[actual[-1]], dt_forecast]),
                label='Decision Tree Forecast', color='seagreen', linestyle='--')
    if lr_forecast is not None:
        ax.plot(fore_x, np.concatenate([[actual[-1]], lr_forecast]),
                label='Linear Regression Forecast', color='darkorange', linestyle='--')
    ax.axvline(len(actual) - 1, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f'{ticker} — {len(nn_forecast)}-Day Price Forecast')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Close Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if filename is None:
        filename = f'{ticker}_forecast.png'

    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Chart saved: {filename}")
    return filename
