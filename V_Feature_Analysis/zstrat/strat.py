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
from scipy.signal import savgol_filter

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/temp_signal_processing1"
OUTPUT_FILE = "/data/quant14/signals/z_trading_signals_EBX1.csv"
COOLDOWN_PERIOD_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

# --- SMA Parameters ---
SMA_5M = 5 * 60      # 300 bars
SMA_22M = 22 * 60    # 1320 bars
SMA_60M = 60 * 60    # 3600 bars

# --- Z-Score Parameters ---
ROLLING_WINDOW = 30 * 60     # 30min rolling z-window
CALIB_WINDOW = 30 * 60       # 30min baseline window

# --- Smoothing Parameters ---
SMOOTH_WINDOW = 50
SAVGOL_WINDOW = 51
SAVGOL_POLY = 3

# --- Trading Parameters ---
STOP_LOSS_PCT = 0.04  # 0.04% stop loss

# --- Strategy Config ---
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
    """
    Calculate:
    1. SMAs (5M, 22M, 60M)
    2. Z-Score with rolling window
    3. Smoothed Z-Score (Savitzky-Golay filter)
    """
    
    # Calculate SMAs
    df["SMA_5M"] = df["Price"].rolling(window=SMA_5M, min_periods=1).mean()
    df["SMA_22M"] = df["Price"].rolling(window=SMA_22M, min_periods=1).mean()
    df["SMA_60M"] = df["Price"].rolling(window=SMA_60M, min_periods=1).mean()
    
    # Calculate returns (percentage change)
    df["Returns"] = df["Price"].pct_change()
    
    # Calculate rolling mean and std on returns
    roll_mean = df["Returns"].rolling(window=ROLLING_WINDOW, min_periods=CALIB_WINDOW).mean()
    roll_std = df["Returns"].rolling(window=ROLLING_WINDOW, min_periods=CALIB_WINDOW).std(ddof=1)

    # Calculate Z-Score using returns
    df["Z-Score"] = (df["Returns"] - roll_mean) / roll_std
    df.loc[df.index[:CALIB_WINDOW], "Z-Score"] = np.nan
    
    # Apply Savitzky-Golay smoothing to Z-Score
    z_values = df["Z-Score"].fillna(0).values
    if len(z_values) >= SAVGOL_WINDOW:
        smoothed = savgol_filter(z_values, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
        df["Z-Savgol"] = smoothed
    else:
        df["Z-Savgol"] = df["Z-Score"]
    
    # Additional smoothing with rolling mean
    df["Z-Savgol"] = df["Z-Savgol"].rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
    
    return df


# -------------------------------------------------------------------
# Strategy Logic
# -------------------------------------------------------------------
def check_sma_trend(row):
    """
    Check SMA trend:
    - Returns 'short' if SMA_60M > SMA_22M > SMA_5M (downtrend)
    - Returns 'long' if SMA_5M > SMA_22M > SMA_60M (uptrend)
    - Returns 'neutral' otherwise
    """
    sma_5 = row["SMA_5M"]
    sma_22 = row["SMA_22M"]
    sma_60 = row["SMA_60M"]
    
    if pd.isna(sma_5) or pd.isna(sma_22) or pd.isna(sma_60):
        return 'neutral'
    
    if sma_60 > sma_22 > sma_5:
        return 'short'
    elif sma_5 > sma_22 > sma_60:
        return 'long'
    else:
        return 'neutral'


def generate_signal(row, position, cooldown_over, trend_at_entry, entry_price, stop_loss_triggered):
    """
    Signal generation logic:
    1. Entry: Z-Savgol crosses +0.75 or -0.75
    2. Direction: Based on SMA trend at signal generation
    3. Exit: When SMA trend reverses, apply 0.04% stop loss
    """
    signal = 0
    new_stop_loss_triggered = stop_loss_triggered
    
    if not cooldown_over:
        return signal, new_stop_loss_triggered
    
    z_savgol = row["Z-Savgol"]
    current_trend = check_sma_trend(row)
    price = row["Price"]
    
    # Entry Logic (position == 0)
    if position == 0:
        if not pd.isna(z_savgol):
            # Signal to go SHORT
            if z_savgol >= 0.99 or z_savgol<= -0.99 and current_trend == 'short':
                signal = -1
            # Signal to go LONG
            elif z_savgol <= -0.99 or z_savgol >= 0.99 and current_trend == 'long':
                signal = 1
    
    # Exit Logic (position != 0)
    elif position != 0:
        # Check if trend has changed
        trend_changed = (trend_at_entry != current_trend)
        
        if trend_changed and not stop_loss_triggered:
            # Trend changed, now apply stop loss
            new_stop_loss_triggered = True
        
        # Apply stop loss after trend change
        if stop_loss_triggered:
            if position == 1:  # Long position
                price_diff_pct = ((price - entry_price) / entry_price) * 100
                if price_diff_pct <= -STOP_LOSS_PCT:
                    signal = -1  # Close long
                    new_stop_loss_triggered = False
            
            elif position == -1:  # Short position
                price_diff_pct = ((entry_price - price) / entry_price) * 100
                if price_diff_pct <= -STOP_LOSS_PCT:
                    signal = 1  # Close short
                    new_stop_loss_triggered = False
    
    return signal, new_stop_loss_triggered


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
        trend_at_entry = None
        stop_loss_triggered = False
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
                stop_loss_triggered = False

            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                signal, stop_loss_triggered = generate_signal(
                    row, position, cooldown_over, trend_at_entry, entry_price, stop_loss_triggered
                )

            # --- Apply Signal ---
            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0  # redundant signal
                else:
                    position += signal
                    last_signal_time = current_time
                    
                    # Track entry details
                    if position != 0:  # Just entered a position
                        entry_price = price
                        trend_at_entry = check_sma_trend(row)
                        stop_loss_triggered = False
                    else:  # Position closed
                        entry_price = None
                        trend_at_entry = None
                        stop_loss_triggered = False

            signals[i] = signal
            positions[i] = position

        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ['SMA_5M', 'SMA_22M', 'SMA_60M', 'Z-Score', 'Z-Savgol'] + list(strategy_params.values()) + ['Signal', 'Position']
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