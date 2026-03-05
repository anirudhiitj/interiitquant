import math
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
import gc
from multiprocessing import Pool, get_context
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter
from numba import jit

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/hawkes_trading_signals_EBX.csv"
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
    price_vals = df["Price"].values
    n = len(price_vals)

    # Ehlers Super Smoother
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
    
    ss_series = df["EhlersSuperSmoother"]
    
    eama_period = 10
    eama_fast = 20
    eama_slow = 100
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
    df["EAMA_Slope"] = df["EAMA"].diff().fillna(0)

    eama_period = 15
    eama_fast = 30 
    eama_slow = 150
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
    df["EAMA_Slow_Slope"] = df["EAMA_Slow"].diff().fillna(0)

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

    df["EAMA_Slowest"] = eama
    df["EAMA_Slowest_Slope"] = df["EAMA_Slowest"].diff().fillna(0)

    df["Mean"] = df["Price"].expanding().mean()
    df["STD"] = df["Price"].expanding().std()
    df["Upper"] = df["Mean"] + 2*df["STD"]
    df["Lower"] = df["Mean"] - 2*df["STD"]
    df["Upper_Distance"] = df["Upper"] - df["Price"]
    df["Lower_Distance"] = df["Price"] - df["Lower"]
    alpha = 1.2
    beta = 1.8
    threshold = 0.005
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
            H_sig[i] +=alpha

    df["H_Buy"] = H_buy
    df["H_Sell"] = H_sell
    df["H_Sig"] = H_sig
    df["H_Long"] = df["H_Buy"] * df["Lower_Distance"]
    df["H_Short"] = df["H_Sell"] * df["Upper_Distance"]

    price = df["PB9_T1"].to_numpy(dtype=float)
    price[0:60] = np.nan
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
    
    if cooldown_over:
        if position == 0:
            if row['H_Long'] > 1 and row["ATR_High"] < 0.01:
                signal = 1
            elif row['H_Short'] > 1 and row["ATR_High"] < 0.01:
                signal = -1
        # elif position > 0:
        #     if row["H_Sell"] > 1 or row["EAMA_Slow"] < row["EAMA"]:
        #         signal = -position
        # elif position < 0:
        #     if row["H_Buy"] > 1 or row["EAMA_Slow"] > row["EAMA"]:
        #         signal = -position
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
        # Note: Using dask_cudf inside a worker might create a new cluster context.
        # Ideally, use direct cudf if file fits in GPU memory, or standard pandas if CPU bound.
        # Proceeding with original logic as requested.
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
        state={}

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
                    tp1 = 0.3
                    tp2 = 0.5
                    sl = 0.5
                    applySL = False
                    applyTrailingSL = False
                    trailingSL = 0.05
                    trailingSLRef = entry_price

                # --- Exit Logic ---
                elif cooldown_over and position != 0 and signal == 0:
                    if position > 0:
                        price_diff = price - entry_price
                    elif position < 0:
                        price_diff = entry_price - price
                    
                    if price_diff >= tp1:
                        applyTrailingSL = True
                        # signal = -position
                        # tp = 0.1
                        # entry_price = price
                        # applySL = False
                    if applySL:
                        if price_diff < -sl:
                            signal = -position
                    elif price_diff >= tp2:
                        signal = -position
                    if applyTrailingSL:
                        if position > 0:
                            if price > trailingSLRef:
                                trailingSLRef = price
                            tsl_price_diff = price - trailingSLRef
                        elif position < 0:
                            if price < trailingSLRef:
                                trailingSLRef = price
                            tsl_price_diff = trailingSLRef - price

                        if tsl_price_diff < -trailingSL:
                            signal = -position 
                # elif cooldown_over and position != 0 and signal != 0:
                #     if position > 0:
                #         price_diff = price - entry_price
                #     elif position < 0:
                #         price_diff = entry_price - price
                    
                #     if price_diff < 0:
                #         signal = 0

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
        final_columns = REQUIRED_COLUMNS + ['Signal', 'Position','EhlersSuperSmoother','H_Buy','H_Sell','KAMA_Slope_abs','EhlersSuperSmoother_Slope','EAMA','EAMA_Slow']
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