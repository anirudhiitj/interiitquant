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
import rmm
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, get_context
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/11temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/11kf_trading_signals_EBX.csv"
COOLDOWN_PERIOD_SECONDS = 5
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

def extract_day_num_csv(filepath):
    """Extracts the day number 'n' from a filepath like '.../day{n}.csv'."""
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1


# -------------------------------------------------------------------
# Rolling Feature Preparation
# -------------------------------------------------------------------
def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    # transition_matrix = [1] 
    # observation_matrix = [1]
    # transition_covariance = 0.01
    # observation_covariance = 1.0
    # initial_state_covariance = 1.0
    # initial_state_mean = 100
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
    # df["KFTrend"] = smoothed_prices

    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.24
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = df["Price"].iloc[0]
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
    df["KFTrendFast"] = smoothed_prices.shift(1)

    df["KFTrendFast_Lagged"]=df["KFTrendFast"].shift(1)

    # transition_matrix = [1] 
    # observation_matrix = [1]
    # transition_covariance = 0.005
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
    # df["KFTrendSlow"] = smoothed_prices.shift(1)

    # df["KFTrendSlow_Lagged"]=df["KFTrendSlow"].shift(1)

    # df["SMA_20"]=df["PB9_T1"].rolling(window=20, min_periods=1).mean()
    # df["STD_SMA"]=df["SMA_20"].rolling(window=20, min_periods=1).std()
    # print(df["STD_SMA"])
    df["Take_Profit"]=0.4

    # window = 14
    # delta = df["PB9_T1"].diff()
    # gain = delta.clip(lower=0)
    # loss = -delta.clip(upper=0)
    # avg_gain = gain.ewm(com=window - 1, min_periods=window, adjust=False).mean().shift(1)
    # avg_loss = loss.ewm(com=window - 1, min_periods=window, adjust=False).mean().shift(1)
    # rs = avg_gain / avg_loss
    # df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    price = df["PB9_T1"].to_numpy(dtype=float)
    # price[0:60] = np.nan
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
    df["KAMA_Slope_abs_Lagged_2"] = df["KAMA_Slope_abs"].shift(2)
    df["KAMA_Sum"] = df["KAMA_Slope_abs"].rolling(window=5, min_periods=1).sum()

    # price = df["PB9_T1"].to_numpy(dtype=float)

    # window = 10
    # kama_signal = np.abs(pd.Series(price).diff(window))
    # kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    # er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    # sc_fast = 2 / (10 + 1)
    # sc_slow = 2 / (15 + 1)
    # sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    # kama = np.full(len(price), np.nan)

    # valid_start = np.where(~np.isnan(price))[0]
    # if len(valid_start) == 0:
    #     raise ValueError("No valid price values found.")
    # start = valid_start[0]
    # kama[start] = price[start]

    # for i in range(start + 1, len(price)):
    #     if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
    #         kama[i] = kama[i - 1]
    #     else:
    #         kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    # df["KAMA_Trend"] = kama
    # df["KAMA_Trend_Lagged"] = df["KAMA_Trend"].shift(1)
    # df["KAMA_Trend_Slope"] = df["KAMA_Trend"].diff()
    series = df["PB9_T1"]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("PB9_T1 has no valid values")
    
    alpha = 0.8
    beta = 0.08
    n = len(series)

    mu = np.zeros(n)          # level
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
    df.loc[0, "UCM"] = 100
    df["UCM_Lagged"] = df["UCM"].shift(1)
    
    # df["SMA_10"] = df["Price"].rolling(window=10, min_periods=1).mean()
    # df["SMA_45"] = df["Price"].rolling(window=45, min_periods=1).mean()
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=30).mean()
    df["ATR"]=atr
    df["ATR_High"]=df["ATR"].expanding(min_periods=1).max()
    return df


# -------------------------------------------------------------------
# Strategy Template (Pluggable)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params, state):
    position=round(position,2)
    signal = 0
    # if row["Time_sec"]>=16200 and row["Time_sec"]<=18000:
    #     return signal
    if cooldown_over:
        if position == 0:
            if row["KFTrendFast"] >= row["UCM"] and row["KAMA_Slope_abs"]>0.001 and row["ATR_High"]>0.01:
                state["applyTP"] = True
                signal = -1
            # elif row["KFTrendFast"]>=row["UCM"] and row["KAMA_Slope_abs"]>0.001:
            #     # state["applyTP"] = False
            #     if row["SMA_10"]>row["SMA_45"]:
            #         signal = 1
            #     else:
            #         signal = -1
            elif row["KFTrendFast"] <= row["UCM"] and row["KAMA_Slope_abs"]>0.001 and row["ATR_High"]>0.01:
                state["applyTP"] = True
                signal = 1
            # elif row["KFTrendFast"]<=row["UCM"] and row["KAMA_Slope_abs"]>0.001:
            #     # state["applyTP"] = False
            #     if row["SMA_10"]<row["SMA_45"]:
            #         signal = -1
            #     else:
            #         signal = 1
        # elif position > 0:
        #     if (row["KFTrendFast_Lagged"]<row["UCM_Lagged"] and row["KFTrendFast"]>row["UCM"] and row["KAMA_Slope_abs"]>0.0005):
        #         signal = position
        # elif position < 0:
        #     if (row["KFTrendFast_Lagged"]>row["UCM_Lagged"] and row["KFTrendFast"]<row["UCM"] and row["KAMA_Slope_abs"]>0.0005):
        #         signal = position
    return signal, state


# -------------------------------------------------------------------
# Core Processing
# -------------------------------------------------------------------
def process_day_wrapper(args):
    """Wrapper function to unpack arguments for Pool map."""
    return process_day(*args)
def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    """Processes a single day's parquet file and applies strategy logic."""
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())

        # --- Read Data ---
        df = pd.read_parquet(file_path, columns=columns)
        # gdf = ddf.compute()
        # df = gdf.to_pandas()

        # del ddf
        # del gdf
        # gc.collect()

        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None

        # --- Prepare Data ---
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
        df = prepare_derived_features(df, strategy_params)
        state={"applyTP":True}

        # --- Initialize State ---
        position = 0
        entry_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        # --- Process Row by Row ---
        for i in range(len(df)):
            position = round(position,2)
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
                signal, state = generate_signal(row, position, cooldown_over, strategy_params, state)

                # --- Entry Logic ---   
                if cooldown_over and position == 0 and signal != 0:
                    entry_price = price
                    # if state["applyTP"]:
                    tp1 = row["Take_Profit"]*1
                    tp2 = row["Take_Profit"]*1.75
                    tp3 = row["Take_Profit"]*2

                    tp_stage = 0
                    applySL = False
                    sl = 0.5
                    

                # --- Exit Logic ---
                elif cooldown_over and position != 0 and signal == 0:
                    if state["applyTP"]:
                        if position > 0:
                            price_diff = price - entry_price
                            tp1pos = 0.7
                            tp2pos = 0.5
                        elif position < 0:
                            price_diff = entry_price - price
                            tp1pos = -0.7
                            tp2pos = -0.5

                        # if price_diff >= tp3 and tp_stage < 4:
                        #     applySL = False
                        #     signal = -position
                        if price_diff >= tp3 and tp_stage < 3:
                            tp_stage = 3
                            applySL = False
                            signal = -position
                            # if row["KAMA_Slope_abs"]<0.001:
                            #     signal = -position
                            # else:
                            #     signal = 0
                        elif price_diff >= tp2 and tp_stage < 2:
                            tp_stage = 2
                            applySL = True
                            # signal = -position + tp2pos
                            if row["KAMA_Slope_abs"]<0.001:
                                signal = -position + tp2pos
                            else:
                                signal = 0
                        elif price_diff >= tp1 and tp_stage < 1:
                            tp_stage = 1
                            applySL = True
                            # signal = -position + tp1pos
                            if row["KAMA_Slope_abs"]<0.001:
                                signal = -position + tp1pos
                            else:
                                signal = 0
                        
                        if applySL:
                            if tp_stage == 1:
                                if price_diff - tp1 <-sl:
                                    signal = -position
                            elif tp_stage == 2:
                                if price_diff - tp2 <-sl:
                                    signal = -position

            # --- Apply Signal ---
            if signal != 0:
                if (position + signal > 1):
                    signal = 1 - position
                    position = 1
                elif (position + signal < -1):
                    signal = -1 - position
                    position = -1
                else:
                    position += signal
                    last_signal_time = current_time
                    if round(position,2) == 0:  # position closed
                        entry_price = None

            signals[i] = signal
            positions[i] = position

        # --- Store Results ---
        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ['Signal', 'Position', 'KFTrendFast', 'UCM', 'KAMA_Slope_abs']
        df[final_columns].to_csv(output_path, index=False)

        return str(output_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


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

    exclude_start = 60   # start excluding this day
    exclude_end = 79  # end excluding this day (inclusive)

    # Filter out unwanted range
    filtered_files = [
        f for f in all_files
        if not (exclude_start <= extract_day_num(f) <= exclude_end)
    ]

    # Sort remaining files
    sorted_files = sorted(filtered_files, key=extract_day_num)
    if not sorted_files:
        print(f"No parquet files found in {directory}")
        shutil.rmtree(temp_dir_path)
        return

    print(f"Found {len(sorted_files)} day files.")

    processed_files = []
    
    # Prepare arguments for multiprocessing
    # Each tuple corresponds to arguments for process_day_wrapper
    tasks = [
        (f, extract_day_num(f), temp_dir_path, strategy_params)
        for f in sorted_files
    ]

    try:
        # --- Parallel Execution with multiprocessing.Pool ---
        # Using 'spawn' context is generally safer when using GPU/CUDA libraries inside workers
        ctx = get_context("spawn")
        
        # maxtasksperchild=10 restarts worker processes periodically to clear memory/GPU context
        print(f"Starting Pool with {max_workers} workers...")
        with ctx.Pool(processes=max_workers, maxtasksperchild=10) as pool:
            
            # imap_unordered is usually efficient for this type of file processing
            results = pool.imap_unordered(process_day_wrapper, tasks)
            
            for result in results:
                if result:
                    processed_files.append(result)
                    print(f"Processed: {os.path.basename(result)}")
        
        # --- Combine Results ---
        if not processed_files:
            print("No files processed successfully.")
            return

        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        
        print("Merging output files...")
        with open(OUTPUT_FILE, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline() # Skip header
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
