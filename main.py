# Parameters
TICKER        = "AAPL"
FORECAST_DAYS = 10
LOOK_BACK     = 60
PERIOD        = "2y"
EPOCHS        = 50
BATCH_SIZE    = 32
DT_MAX_DEPTH  = 40

from sklearn.model_selection import train_test_split

from data import load_data, data_preprocessing, make_windows

from nn_model import build_model, predict_future_nn
from dt_model import build_dt, train_dt, predict_future_dt
from modelo_lineal import build_lr, train_lr, predict_future_lr

from output import save_chart

if __name__ == "__main__":
    #
    # Data
    #
    
    # Load data
    data = load_data(TICKER, PERIOD)

    # Scale data
    scaled, scaler = data_preprocessing(data)

    # Make windowws
    X, y = make_windows(scaled, LOOK_BACK)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    print("-" * 50)

    #
    # Models
    #

    # Neural Network
    # Build nn model
    nn = build_model(LOOK_BACK)

    # Train nn model
    nn.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Make nn moel forecast
    nn_forecast = predict_future_nn(
        nn,
        # The last 60 days (window)
        X_test[-1],
        scaler,
        FORECAST_DAYS
    )

    print(f"\nLSTM forecast for {TICKER} — next {FORECAST_DAYS} days:")
    for i, price in enumerate(nn_forecast, 1):
        print(f"  Day {i}: ${price:.2f}")
    print("-" * 50)

    # Decision Tree
    # Build dt model
    dt = build_dt(DT_MAX_DEPTH)

    # Train dt model
    dt = train_dt(
        dt,
        X_train,
        y_train
    )

    # Make dt model forecast
    dt_forecast = predict_future_dt(
        dt,
        X_test[-1],
        scaler,
        FORECAST_DAYS
    )

    print(f"DT forecast for {TICKER} — next {FORECAST_DAYS} days:")
    for i, price in enumerate(dt_forecast, 1):
        print(f"  Day {i}: ${price:.2f}")
    print("-" * 50)

    # Linear Regression
    lr = build_lr()
    lr = train_lr(lr, X_train, y_train)
    lr_forecast = predict_future_lr(lr, X_test[-1], scaler, FORECAST_DAYS)

    print(f"LR forecast for {TICKER} — next {FORECAST_DAYS} days:")
    for i, price in enumerate(lr_forecast, 1):
        print(f"  Day {i}: ${price:.2f}")
    print("-" * 50)

    save_chart(
        TICKER,
        y_test,
        nn_forecast,
        scaler,
        dt_forecast,
        lr_forecast
    )

