import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
import os

# ==================================
# CONFIG
# ==================================
DATA_DIR = "/data/quant14/EBX/"   # change this
N_DAYS = 50  # run on first 20 days

# ==================================
# Helper function: process one day
# ==================================
def run_lr_on_day(filepath):
    df = pd.read_parquet(filepath)
    df = df.sort_values("Time")

    # Compute log returns
    df["log_price"] = np.log(df["Price"])
    df["ret"] = df["log_price"].diff()

    # Feature engineering (lags + rolling)
    for lag in [1, 5, 20, 100]:
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)

    df["ret_ma_20"] = df["ret"].rolling(20).mean()
    df["ret_ma_100"] = df["ret"].rolling(100).mean()
    df["vol_200"] = df["ret"].rolling(30).std()

    # Rolling slope (trend)
    window = 50
    idx = np.arange(window)
    df["slope_50"] = df["log_price"].rolling(window).apply(
        lambda x: np.polyfit(idx, x, 1)[0], raw=True
    )

    df = df.dropna()

    # Target = next-second return
    df["target"] = df["ret"].shift(-1)
    df = df.dropna()

    # Features
    features = [
        "ret_lag_1", "ret_lag_5", "ret_lag_20", "ret_lag_100",
        "ret_ma_20", "ret_ma_100",
        "vol_200",
        "slope_50"
    ]

    X = df[features]
    y = df["target"]

    # Time-series split (80/20)
    split = int(len(X) * 0.6)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    direction_acc = np.mean((y_pred > 0) == (y_test > 0))

    return {
        "day": os.path.basename(filepath),
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "direction_accuracy": direction_acc
    }

# ==================================
# MAIN LOOP FOR MULTIPLE DAYS
# ==================================
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))[:N_DAYS]

results = []

for f in files:
    print("Running:", f)
    metrics = run_lr_on_day(f)
    results.append(metrics)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("linear_regression_daywise_results.csv", index=False)

print("\nDONE!")
print(results_df)
