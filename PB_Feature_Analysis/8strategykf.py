import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import cudf
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/8temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/8kf_trading_signals_EBY.csv"
COOLDOWN_PERIOD_SECONDS = 15
REQUIRED_COLUMNS = ['Time', 'Price']

# --- Strategy Config Example ---
# These can be replaced dynamically for different strategies
STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def extract_day_num(filepath):
    """Extracts the day number 'n' from a filepath like '.../day{n}.parquet'."""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


# -------------------------------------------------------------------
# Rolling Feature Preparation
# -------------------------------------------------------------------
def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    # transition_matrix = [1] 
    # observation_matrix = [1]
    # transition_covariance = 0.49
    # observation_covariance = 0.7
    # initial_state_covariance = 100
    # initial_state_mean = df["Price"].iloc[0]
    # kf = KalmanFilter(
    #     transition_matrices=transition_matrix,
    #     observation_matrices=observation_matrix,
    #     transition_covariance=transition_covariance,
    #     observation_covariance=observation_covariance,
    #     initial_state_mean=initial_state_mean,
    #     initial_state_covariance=initial_state_covariance
    # )
    # filtered_state_means, _ = kf.filter(df["PB9_T1"].dropna().values)
    # smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df["PB9_T1"].dropna().index)
    # df["KFTrendFast"] = smoothed_prices

    # df["KFTrendFast_Lagged"]=df["KFTrendFast"].shift(1)


    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.01
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = 100
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(df["PB9_T1"].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df["PB9_T1"].dropna().index)
    df["KFTrendSlow"] = smoothed_prices

    df["KFTrendSlow_Lagged"]=df["KFTrendSlow"].shift(1)

    df["SMA_25"]=df["PB9_T1"].rolling(window=90, min_periods=1).mean()
    df["SMA_25_Lagged"]=df["SMA_25"].shift(1)

    # window = 14
    # delta = df["PB9_T1"].diff()
    # gain = delta.clip(lower=0)
    # loss = -delta.clip(upper=0)
    # avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean().shift(1)
    # avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean().shift(1)
    # rs = avg_gain / avg_loss
    # df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    price = df["PB9_T1"].to_numpy(dtype=float)

    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)

    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(2)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    price = df["PB9_T1"].to_numpy(dtype=float)

    window = 10
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (10 + 1)
    sc_slow = 2 / (15 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)

    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["KAMA_Trend"] = kama
    df["KAMA_Trend_Lagged"] = df["KAMA_Trend"].shift(1)
    df["KAMA_Trend_Slope"] = df["KAMA_Trend"].diff()
    series = df["PB9_T1"]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("PB9_T1 has no valid values")

    alpha = 0.1
    beta = 0.02
    n = len(series)

    mu = np.zeros(n)         # level
    beta_slope = np.zeros(n)  # trend
    filtered = np.zeros(n)

    # initialize at first valid value
    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]

    # start recursion from next valid index
    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            # if NaN, carry forward previous filtered value
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue

        # prediction
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]

        # update
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]

    df["UCM"] = filtered
    df["UCM_Lagged"] = df["UCM"].shift(1)
    # series = df["PB9_T1"].values
    # alpha = 0.05
    # beta = 0.01

    # L = np.zeros(len(series))
    # T = np.zeros(len(series))
    # TES = np.full(len(series), np.nan)

    # first_valid = np.argmax(~np.isnan(series))
    # L[first_valid] = series[first_valid]
    # T[first_valid] = 0
    # TES[first_valid] = L[first_valid] + T[first_valid]

    # for t in range(first_valid + 1, len(series)):
    #     if np.isnan(series[t]):
    #         TES[t] = TES[t-1]
    #         L[t] = L[t-1]
    #         T[t] = T[t-1]
    #     else:
    #         L[t] = alpha * series[t] + (1 - alpha) * (L[t-1] + T[t-1])
    #         T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
    #         TES[t] = L[t] + T[t]

    # df['TES'] = TES
    # df['TES']=df['TES'].shift(1)
    # df["TES_Lagged"]=df["TES"].shift(1)
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=30).mean()
    df["ATR"]=atr
    df["ATR_High"]=df["ATR"].expanding(min_periods=1).max()
    df["PB9_T1_MA"]=df["PB9_T1"].rolling(window=20, min_periods=1).mean()
    df["STD"]=df["PB9_T1_MA"].rolling(window=20, min_periods=1).std()
    df["STD_High"]=df["STD"].expanding(min_periods=1).max()
    return df


# -------------------------------------------------------------------
# Strategy Template (Pluggable)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params):

    signal = 0
    # if row["Time_sec"]>=16200 and row["Time_sec"]<=18000:
    #     return signal
    if cooldown_over:
        if position == 0:
            if row["UCM"]>=row["KFTrendSlow"] and row["ATR_High"]>0.01 and row["STD_High"]>0.06 and row["KAMA_Slope_abs"]>0.001:
                    signal = 1
            elif row["UCM"]<=row["KFTrendSlow"] and row["ATR_High"]>0.01 and row["STD_High"]>0.06 and row["KAMA_Slope_abs"]>0.001:
                    signal = -1
        elif position == 1:
            if (row["UCM_Lagged"]>row["KFTrendSlow_Lagged"] and row["UCM"]<row["KFTrendSlow"]) and row["KAMA_Slope_abs"]>0.0007:
                signal = -1
        elif position == -1:
            if (row["UCM_Lagged"]<row["KFTrendSlow_Lagged"] and row["UCM"]>row["KFTrendSlow"]) and row["KAMA_Slope_abs"]>0.0007:
                signal = 1
    return signal


# -------------------------------------------------------------------
# Core Processing
# -------------------------------------------------------------------
def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    """Processes a single day's parquet file and applies strategy logic."""
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())

        # --- Read Data ---
        ddf = dask_cudf.read_parquet(file_path, columns=columns)
        gdf = ddf.compute()
        df = gdf.to_pandas()

        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None

        # --- Prepare Data ---
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
        df = prepare_derived_features(df, strategy_params)

        # --- Initialize State ---
        position = 0
        entry_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        # --- Process Row by Row ---
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['Time_sec']
            price = row['Price']
            is_last_tick = (i == len(df) - 1)

            signal = 0

            # --- EOD Square Off ---
            if is_last_tick and position != 0:
                signal = -position

            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                signal = generate_signal(row, position, cooldown_over, strategy_params)

                # --- Entry Logic ---
                if position == 0 and signal != 0:
                    entry_price = price  # record entry price
                    trailing_sl = False

                # --- Exit Logic (Price-based confirmation) ---
                elif cooldown_over and position != 0 :
                    if position == 1:
                        price_diff = price - entry_price
                        if price_diff >= 0.3:
                            entry_price = price  # trailing stop adjustment
                            trailing_sl = True
                        # elif price_diff <= -0.05:
                        #     signal = -1
                        if trailing_sl:
                            if price_diff > 0:
                                entry_price = price  # adjust trailing stop
                            elif price_diff <= -0.05:
                                signal = -1
                                trailing_sl = False
                    elif position == -1:
                        price_diff = entry_price - price
                        if price_diff >= 0.3:
                            entry_price = price  # trailing stop adjustment
                            trailing_sl = True
                        # elif price_diff <= -0.05:
                        #     signal = 1 
                        if trailing_sl:
                            if price_diff > 0:
                                entry_price = price  # adjust trailing stop
                            elif price_diff <= -0.05:
                                signal = 1
                                trailing_sl = False

            # --- Apply Signal ---
            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0  # redundant signal
                else:
                    position += signal
                    last_signal_time = current_time
                    if position == 0:  # position closed
                        entry_price = None

            signals[i] = signal
            positions[i] = position

        # --- Store Results ---
        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + ['Signal', 'Position',"KFTrendSlow","SMA_25","KAMA_Slope_abs","ATR_High","STD_High","UCM"]
        df[final_columns].to_csv(output_path, index=False)

        return str(output_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(directory: str, max_workers: int, strategy_params):
    start_time = time.time()

    temp_dir_path = pathlib.Path(TEMP_DIR)
    if temp_dir_path.exists():
        print(f"Removing existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")

    # --- Find Input Files ---
    files_pattern = os.path.join(directory, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)

    if not sorted_files:
        print(f"No parquet files found in {directory}")
        shutil.rmtree(temp_dir_path)
        return

    print(f"Found {len(sorted_files)} day files.")

    processed_files = []
    try:
        # --- Parallel Execution ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params): f
                for f in sorted_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                        print(f"Processed: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error in {file_path}: {e}")

        # --- Combine Results ---
        if not processed_files:
            print("No files processed successfully.")
            return

        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        with open(OUTPUT_FILE, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)

        print(f"Created combined output: {OUTPUT_FILE}")

    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)

    print(f"Total time: {time.time() - start_time:.2f}s")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily trading parquet files into signals.")
    parser.add_argument("directory", type=str, help="Directory containing 'day{n}.parquet' files.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="Parallel workers.")
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)
