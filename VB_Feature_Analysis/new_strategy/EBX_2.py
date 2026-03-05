import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import cudf
import time
import math
import gc
import multiprocessing as mp
from multiprocessing import Pool

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBX_Hydra.csv"
COOLDOWN_PERIOD_SECONDS = 5
MIN_TRADE_DURATION_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

# --- Indicator Helpers (Added for EAMA/Hydra) ---
def get_super_smoother_array(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    
    ss = np.zeros(n, dtype=np.float64)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss

def apply_recursive_filter(series: np.ndarray, sc: np.ndarray) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
        else:
            return out 
            
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        prev_out = out[i-1] if not np.isnan(out[i-1]) else target
        
        if not np.isnan(target):
            out[i] = prev_out + c * (target - prev_out)
        else:
            out[i] = prev_out
            
    return out

def calculate_eama_numpy(prices, ss_period, er_period, fast, slow):
    ss = get_super_smoother_array(prices, ss_period)
    ss_series = pd.Series(ss)
    direction = ss_series.diff(er_period).abs()
    volatility = ss_series.diff(1).abs().rolling(er_period).sum()
    er = (direction / volatility).fillna(0).values
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    return apply_recursive_filter(ss, sc)

# --- Main Logic ---

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    if strategy_params["pb9t1"] not in df.columns:
        raise KeyError(f"Required feature {strategy_params['pb9t1']} not found.")

    price = df[strategy_params["pb9t1"]].to_numpy(dtype=float)
    
    # 1. KAMA Calculation (Kept from your code)
    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) > 0:
        start = valid_start[0]
        kama[start] = price[start]
        for i in range(start + 1, len(price)):
            if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    
    # 2. Volatility Features (Kept from your code)
    tr = df[strategy_params["pb9t1"]].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    std = df[strategy_params["pb9t1"]].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    # 3. REPLACED: Removed HMA, EMA, SMA
    # 4. ADDED: EAMA and HYDRA Logic
    
    # EAMA (Fast)
    df["EAMA"] = calculate_eama_numpy(price, ss_period=25, er_period=15, fast=30, slow=120)
    
    # EAMA (Slow)
    eama_slow = calculate_eama_numpy(price, ss_period=25, er_period=15, fast=60, slow=180)
    
    # Hydra Calculation
    diff_std = df[strategy_params["pb9t1"]].diff().rolling(120).std()
    diff_std_std = diff_std.rolling(120).std()
    
    # Calculate Threshold (using first hour approx or simple expanding for streaming logic)
    # To keep it simple and efficient in this structure, we use the expanding 99th percentile 
    # (similar to how you used ATR_High) or a fixed calculation on the fly.
    # Here we replicate the logic: Threshold = 99th percentile of Diff_STD_STD / 2.5
    
    # Note: For strict causality in a loop, calculating a global percentile is technically 
    # forward looking if done on the whole day at once. However, your previous code 
    # calculated columns first. We will use expanding percentile to be safe, 
    # or just calculate it on the dataframe as is common in vectorization.
    
    # Using the logic from previous prompts (First hour calc):
    if len(df) > 3600: # Assuming 1 sec data, 3600 rows = 1 hour
        first_hour = diff_std_std.iloc[:3600].dropna()
        if not first_hour.empty:
            threshold = np.percentile(first_hour, 99) / 2.5
        else:
            threshold = 0.0 # Fallback
    else:
        threshold = 0.0

    raw_sig = np.where(diff_std_std > threshold, 1, 0)
    weight = pd.Series(raw_sig).rolling(300).sum().fillna(0) / 300
    
    df["HYDRA_MA"] = (weight * df["EAMA"]) + ((1 - weight) * eama_slow)
    
    # Shift indicators by 1 to prevent lookahead in the row-by-row loop
    df["EAMA"] = df["EAMA"].shift(1)
    df["HYDRA_MA"] = df["HYDRA_MA"].shift(1)

    return df

def generate_signal(row, position, cooldown_over, strategy_params):
    signal = 0
    ATR_THRESHOLD = 0.0
    STD_THRESHOLD = 0.08
    KAMA_SLOPE_ENTRY = 0.0008

    if cooldown_over:
        # Volatility check (same as before)
        volatility_confirmed = (row["ATR_High"] > ATR_THRESHOLD and row["STD_High"] > STD_THRESHOLD)
        
        if position == 0 and volatility_confirmed:
            # --- REPLACED LOGIC ---
            # Old: HMA/EMA/SMA trend logic
            # New: EAMA vs HYDRA crossover logic
            
            # Note: Since we shifted the columns in prepare_derived_features, 
            # row['EAMA'] represents the value at t-1 (closed candle).
            
            trend_long = (row["EAMA"] > row["HYDRA_MA"])
            trend_short = (row["EAMA"] < row["HYDRA_MA"])

            # Keep KAMA Slope Confirmation
            if trend_long and row["KAMA_Slope"] > KAMA_SLOPE_ENTRY:
                signal = 1
            elif trend_short and row["KAMA_Slope"] < -KAMA_SLOPE_ENTRY:
                signal = -1

    return signal

def process_day(args):
    """Worker function for multiprocessing.Pool"""
    file_path, day_num, temp_dir, strategy_params = args
    
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        df = pd.read_parquet(file_path, columns=columns)

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        df = prepare_derived_features(df, strategy_params)

        # Trading Loop
        position = 0
        entry_price = None
        entry_time_sec = None
        trailing_active = False
        trailing_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        TAKE_PROFIT = 0.35
        TRAIL_STOP = 0.15

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = int(row["Time_sec"])
            price = row["Price"]
            is_last_tick = (i == len(df) - 1)
            signal = 0

            if is_last_tick and position != 0:
                signal = -position
            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                desired_signal = generate_signal(row, position, cooldown_over, strategy_params)

                if position == 0 and desired_signal != 0:
                    signal = desired_signal
                elif position != 0:
                    price_diff = price - entry_price if position == 1 else entry_price - price

                    if not trailing_active and price_diff >= TAKE_PROFIT:
                        trailing_active = True
                        trailing_price = price

                    if trailing_active:
                        if position == 1:
                            if price > trailing_price:
                                trailing_price = price
                            elif price <= trailing_price - TRAIL_STOP:
                                signal = -1
                                trailing_active = False
                        elif position == -1:
                            if price < trailing_price:
                                trailing_price = price
                            elif price >= trailing_price + TRAIL_STOP:
                                signal = 1
                                trailing_active = False
                    
                    if signal == 0 and desired_signal != 0 and desired_signal == -position:
                        signal = -position

                if signal != 0 and position != 0:
                    time_in_trade = current_time - (entry_time_sec if entry_time_sec is not None else -999999)
                    if time_in_trade < MIN_TRADE_DURATION_SECONDS:
                        signal = 0

            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0
                else:
                    if position == 0 and signal != 0:
                        entry_price = price
                        entry_time_sec = int(current_time)
                        trailing_active = False
                        trailing_price = None

                    if position != 0 and signal == -position:
                        entry_price = None
                        entry_time_sec = None
                        trailing_price = None
                        trailing_active = False

                    position += signal
                    last_signal_time = int(current_time)

                    if position != 0 and entry_time_sec is None:
                        entry_time_sec = int(current_time)
                        entry_price = price

                    if position == 0:
                        entry_time_sec = None
                        entry_price = None
                        trailing_price = None
                        trailing_active = False

            signals[i] = signal
            positions[i] = position

        df["Signal"] = signals
        df["Position"] = positions
        df["Day"] = day_num

        output_path = temp_dir / f"day{day_num}.csv"
        # Updated columns list
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Day", "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD", "EAMA", "HYDRA_MA"
        ]

        df[final_columns].to_csv(output_path, index=False)
        print(f"✅ Processed Day {day_num}")
        
        del df
        gc.collect()
        
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_trade_reports_csv(output_file):
    import pandas as pd
    import os

    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)

    signals_file = output_file

    try:
        df = pd.read_csv(signals_file)
    except FileNotFoundError:
        print("Error: Combined signals file not found:", signals_file)
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found in final signal file.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]

        if position is None:  # expecting entry
            if signal in [1, -1]:
                position = signal
                entry = row
        else:                 # expecting exit
            if signal == -position:
                trades.append({
                    "day": entry["Day"],   # include day column
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": (float(row["Price"]) - float(entry["Price"])) * position
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades found.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)

    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

def main(directory: str, max_workers: int, strategy_params):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files:
        print("No parquet files found.")
        return

    tasks = [
        (f, extract_day_num(f), temp_dir_path, strategy_params)
        for f in files
    ]

    processed_files = []
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_day, tasks)
        processed_files = [res for res in results if res is not None]

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline()
                    shutil.copyfileobj(inp, out)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")

    print("\nGenerating trade_reports.csv ...")
    generate_trade_reports_csv(OUTPUT_FILE)

    print(f"⏱ Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)