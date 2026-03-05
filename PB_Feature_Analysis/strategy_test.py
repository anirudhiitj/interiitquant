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
OUTPUT_FILE = "/data/quant14/signals/EBX_test.csv"
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
        if row["Time_sec"] == 0:
            signal = 1
        elif row["Time_sec"] == 23300:
            signal = -1
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
        final_columns = REQUIRED_COLUMNS + ['Signal', 'Position']
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

    # List of (start, end) tuples to exclude
    exclude_ranges = [
        (0, 11),
        (15, 510)
    ]

    def is_in_any_range(day, ranges):
        """Return True if day is within ANY of the given (start, end) ranges."""
        return any(start <= day <= end for start, end in ranges)

    filtered_files = [
        f for f in all_files
        if not is_in_any_range(extract_day_num(f), exclude_ranges)
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
