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
import math
import time
from multiprocessing import Pool, cpu_count
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/trading_signals_EBY_combined.csv"
COOLDOWN_PERIOD_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

# --- Strategy Config Example ---
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
    # 1. Kalman Filter
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
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)
    df["Take_Profit"] = 0.4

    # 2. KAMA
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
    df["KAMA_Slope_abs_Lagged_1"] = df["KAMA_Slope_abs"].shift(1)
    df["KAMA_Slope_abs_Lagged_2"] = df["KAMA_Slope_abs"].shift(2)
    df["KAMA_Sum"] = df["KAMA_Slope_abs"].rolling(window=5).sum()

    # 3. UCM
    series = df["PB9_T1"]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("PB9_T1 has no valid values")
    
    alpha = 0.1
    beta = 0.02
    n = len(series)
    mu = np.zeros(n)          # level
    beta_slope = np.zeros(n)  # trend
    filtered = np.zeros(n)

    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]

    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]

    df["UCM"] = filtered
    df.loc[0, "UCM"] = 100
    df["UCM_Lagged"] = df["UCM"].shift(1)

    # 4. Heikin Ashi & HAMA
    price_vals = df["Price"].values
    n = len(price_vals)
    close_vals = df["Price"].values
    open_vals = np.roll(close_vals, 1)
    open_vals[0] = close_vals[0]
    
    if "High" in df.columns and "Low" in df.columns:
        high_vals = df["High"].values
        low_vals = df["Low"].values
    else:
        high_vals = df["Price"].rolling(5, min_periods=1).max().values
        low_vals = df["Price"].rolling(5, min_periods=1).min().values

    ha_open = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    ha_open[0] = open_vals[0]
    ha_close[0] = (open_vals[0] + high_vals[0] + low_vals[0] + close_vals[0]) / 4
    ha_high[0] = high_vals[0]
    ha_low[0] = low_vals[0]

    for i in range(1, n):
        ha_close[i] = (open_vals[i] + high_vals[i] + low_vals[i] + close_vals[i]) / 4
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(high_vals[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_vals[i], ha_open[i], ha_close[i])

    df["HA_Open"] = ha_open
    df["HA_Close"] = ha_close
    df["HA_High"] = ha_high
    df["HA_Low"] = ha_low
    
    # [FIX] Removed .bfill() to prevent lookahead bias. Used ffill() or fillna().
    df["HAMA"] = pd.Series(ha_close).rolling(180).mean().ffill()
    df["HAMA_Slope"] = df["HAMA"].diff().ffill()

    # 5. Ehlers Super Smoother
    period = 25
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    ss = np.zeros(n)
    if n > 0: ss[0] = price_vals[0]
    if n > 1: ss[1] = price_vals[1]
    
    c2 = b1
    c3 = -a1*a1
    for i in range(2, n):
        ss[i] = c1*price_vals[i] + c2*ss[i-1] + c3*ss[i-2]
    df["EhlersSuperSmoother"] = ss
    df["EhlersSuperSmoother_Slope"] = df["EhlersSuperSmoother"].diff()
    
    # 6. EAMA on Super Smoother
    eama_period = 15 
    eama_fast = 30 
    eama_slow = 120 
    ss_series = pd.Series(ss) 
    direction_ss = ss_series.diff(eama_period).abs() 
    volatility_ss = ss_series.diff(1).abs().rolling(eama_period).sum() 
    er_ss = (direction_ss / volatility_ss).fillna(0).values 

    fast_sc = 2 / (eama_fast + 1) 
    slow_sc = 2 / (eama_slow + 1) 
    sc_ss = ((er_ss * (fast_sc - slow_sc)) + slow_sc) ** 2 

    eama = np.zeros(n) 
    if n > 0: eama[0] = ss[0] 
    for i in range(1, n): 
        c = sc_ss[i] if not np.isnan(sc_ss[i]) else 0 
        eama[i] = eama[i-1] + c * (ss[i] - eama[i-1]) 

    df["EAMA"] = eama 
    df["EAMA_Slope"] = pd.Series(eama).diff().fillna(0)

    eama_period = 30
    eama_fast = 60 
    eama_slow = 300 
    ss_series = pd.Series(ss) 
    direction_ss = ss_series.diff(eama_period).abs() 
    volatility_ss = ss_series.diff(1).abs().rolling(eama_period).sum() 
    er_ss = (direction_ss / volatility_ss).fillna(0).values 

    fast_sc = 2 / (eama_fast + 1) 
    slow_sc = 2 / (eama_slow + 1) 
    sc_ss = ((er_ss * (fast_sc - slow_sc)) + slow_sc) ** 2 

    eama = np.zeros(n) 
    if n > 0: eama[0] = ss[0] 
    for i in range(1, n): 
        c = sc_ss[i] if not np.isnan(sc_ss[i]) else 0 
        eama[i] = eama[i-1] + c * (ss[i] - eama[i-1]) 

    df["EAMA_Slow"] = eama 
    df["EAMA_Slow_Slope"] = pd.Series(eama).diff().fillna(0)

    # 7. Recursive Hawkes
    alpha = 1.2
    beta = 1.8
    threshold = 0.01
    time_delta = 1
    decay = np.exp(-beta * time_delta)
    H_buy = np.zeros(n)
    H_sell = np.zeros(n)
    H_sig = np.zeros(n)
    for i in range(1, n):
        H_buy[i] = H_buy[i-1] * decay
        H_sell[i] = H_sell[i-1] * decay

        if df["EhlersSuperSmoother_Slope"][i] > threshold:
            H_buy[i] += alpha
            H_sig[i] += alpha
        if df["EhlersSuperSmoother_Slope"][i] < -threshold:
            H_sell[i] += alpha
            H_sig[i] += alpha

    df["H_Buy"] = H_buy
    df["H_Sell"] = H_sell
    df["H_Sig"] = H_sig

    return df


# -------------------------------------------------------------------
# Strategy Logic
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params, state):
    position = round(position, 2)
    signal = 0
    if cooldown_over:
        if position == 0:
            # if row["KAMA_Sum"] > 0.005:
                if row["KFTrendFast"] >= row["UCM"] and row["KAMA_Slope_abs_Lagged_2"] > 0.001:
                    state["applyTP"] = False
                    state["trade_kalman"] = True
                    signal = -1
                elif row["KFTrendFast"] <= row["UCM"] and row["KAMA_Slope_abs_Lagged_2"] > 0.001:
                    state["applyTP"] = False
                    state["trade_kalman"] = True
                    signal = 1
            # else:
            #     if row['H_Sig'] > 1 and row['KAMA_Slope_abs'] > 0.0005 and row["KAMA_Slope_abs"] < 0.0007:
            #         if row["EAMA"] - row["EAMA_Slow"] > 0.5:
            #             state["trade_kalman"] = False
            #             signal = 1
            #         elif row["EAMA_Slow"] - row["EAMA"] > 0.5:
            #             state["trade_kalman"] = False
            #             signal = -1
            #     #     state["trade_kalman"] = False
            #     #     signal = 1
            #     # elif row['H_Sig'] > 1 and row['KAMA_Slope_abs'] > 0.0005 and row["KAMA_Slope_abs"] < 0.0007:
            #     #     state["trade_kalman"] = False
            #     #     signal = -1
            
        elif position > 0:
            if not state["trade_kalman"]:
                if row["Price"] < row["EAMA"]:
                    signal = -position
                # if row['H_Sell'] > 1 and row['KAMA_Slope_abs'] > 0.0005 and row["KAMA_Slope_abs"] < 0.0007:
                #     signal = -position
                # if row["EhlersSuperSmoother_Slope"] < -0.01:
                #     signal = -position -0.5
                # elif row["EhlersSuperSmoother_Slope"] < -0.05:
                #     signal = -position
        elif position < 0:
            if not state["trade_kalman"]:
                if row["Price"] > row["EAMA"]:
                    signal = -position
                # if row['H_Buy'] > 1 and row['KAMA_Slope_abs'] > 0.0005 and row["KAMA_Slope_abs"] < 0.0007:
                #     signal = 1
                # if row["EhlersSuperSmoother_Slope"] > 0.01:
                #     signal = 1

    return signal, state


# -------------------------------------------------------------------
# Wrapper needed for multiprocessing.Pool
# -------------------------------------------------------------------
def process_day_wrapper(args):
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
        
        state = {"applyTP": True, "trade_kalman": False}

        position = 0.0
        entry_price = 0.0
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)
        
        tp = 0.4
        # --- Process Row by Row ---
        for i in range(len(df)):
            position = round(position, 2)
            row = df.iloc[i].copy()
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
                    
                    if state["trade_kalman"]:
                        tp1 = row["Take_Profit"] * 1
                        tp2 = row["Take_Profit"] * 1.75
                        tp3 = row["Take_Profit"] * 2
                        tp_stage = 0
                        applySL = False
                        sl = 0.5
                    else:
                        tp = 0.2
                        sl = 0.5
                    
                # --- Exit Logic ---
                elif cooldown_over and position != 0 and signal == 0:
                    
                    if state["trade_kalman"]:
                        if state["applyTP"]:
                            price_diff = 0
                            tp1pos, tp2pos = 0, 0
                            
                            if position > 0:
                                price_diff = price - entry_price
                                tp1pos = 0.7
                                tp2pos = 0.5
                            elif position < 0:
                                price_diff = entry_price - price
                                tp1pos = -0.7
                                tp2pos = -0.5

                            if price_diff >= tp3 and tp_stage < 3:
                                applySL = False
                                signal = -position
                            elif price_diff >= tp2 and tp_stage < 2:
                                tp_stage = 2
                                applySL = True
                                if row["KAMA_Slope_abs"] < 0.001:
                                    signal = -position + tp2pos
                                else:
                                    signal = 0
                            elif price_diff >= tp1 and tp_stage < 1:
                                tp_stage = 1
                                applySL = True
                                if row["KAMA_Slope_abs"] < 0.001:
                                    signal = -position + tp1pos
                                else:
                                    signal = 0
                            
                            if applySL:
                                if tp_stage == 1:
                                    if price_diff - tp1 < -sl:
                                        signal = -position
                                elif tp_stage == 2:
                                    if price_diff - tp2 < -sl:
                                        signal = -position
                        else:
                            if position > 0:
                                price_diff = price - entry_price
                            elif position < 0:
                                price_diff = entry_price - price
                            if price_diff >= tp1:
                                signal = -position
                    
                    # else:
                    #     price_diff = 0
                    #     if position > 0:
                    #         price_diff = price - entry_price
                    #     elif position < 0:
                    #         price_diff = entry_price - price

                    #     if price_diff >= tp:
                    #         signal = -position
                        
                    #     # elif price_diff <= -sl:
                    #     #     signal = -position

            # --- Apply Signal ---
            if signal != 0:
                new_pos = position + signal
                if new_pos > 1:
                    new_pos = 1
                    signal = 1 - position
                elif new_pos < -1:
                    new_pos = -1
                    signal = -1 - position
                
                position += signal
                position = round(position, 2)
                
                last_signal_time = current_time

                if position == 0:
                    entry_price = 0.0

            signals[i] = signal
            positions[i] = position

        # --- Store Results ---
        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        # Ensure all columns exist before saving
        save_cols = ['Time', 'Price', 'Signal', 'Position', 'KFTrendFast', 'UCM', 
                     'KAMA_Slope_abs', 'KAMA_Sum', 'H_Buy', 'H_Sell', 
                     'EhlersSuperSmoother', 'EhlersSuperSmoother_Slope',"EAMA","EAMA_Slow"]
        # Filter only existing columns
        final_columns = [c for c in save_cols if c in df.columns]
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

    exclude_start = 500  
    exclude_end = 501

    filtered_files = [
        f for f in all_files
        if not (exclude_start <= extract_day_num(f) <= exclude_end)
    ]

    sorted_files = sorted(filtered_files, key=extract_day_num)

    if not sorted_files:
        print(f"No parquet files found in {directory}")
        shutil.rmtree(temp_dir_path)
        return
    print(f"Found {len(sorted_files)} day files.")

    processed_files = []
    
    # Prepare arguments
    task_args = [
        (f, extract_day_num(f), temp_dir_path, strategy_params) 
        for f in sorted_files
    ]

    try:
        # --- Parallel Execution with multiprocessing.Pool ---
        print(f"Starting Pool with {max_workers} workers (maxtasksperchild=1)...")
        with Pool(processes=max_workers, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(process_day_wrapper, task_args):
                if result:
                    processed_files.append(result)
                    print(f"Processed: {os.path.basename(result)}")

        # --- Combine Results ---
        if not processed_files:
            print("No files processed successfully.")
            return

        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily trading parquet files into signals.")
    parser.add_argument("directory", type=str, help="Directory containing 'day{n}.parquet' files.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count(), help="Parallel workers.")
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)