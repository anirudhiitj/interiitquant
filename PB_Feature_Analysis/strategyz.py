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
TEMP_DIR = "/data/quant14/signals/temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/z_trading_signals_EBX.csv"
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


# -------------------------------------------------------------------
# Rolling Feature Preparation
# -------------------------------------------------------------------
def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    
    roll_mean = df["Price"].rolling(window=1800).mean()
    roll_std  = df["Price"].rolling(window=1800).std(ddof=1)

    df["Z-Score"] = (df["Price"] - roll_mean) / roll_std
    df.loc[df.index[:1800], "Z-Score"] = np.nan

    sig = 1 / (1 + np.exp(-df["Z-Score"]))
    df["Z-Sigmoid"] = (sig - 0.5) * 2
    df["Z-Sigmoid"] = df["Z-Sigmoid"].rolling(window=10).mean()
    df["Z-Sigmoid_MA"] = df["Z-Sigmoid"].rolling(window=600).mean()
    df["Z-Sigmoid_Slope"] = df["Z-Sigmoid"].diff(5)
    df["Breakout"] = 0
    return df


# -------------------------------------------------------------------
# Strategy Template (Pluggable)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params, state):

    signal = 0
    if cooldown_over:
        if position == 0:
            if (row["Z-Sigmoid"]>=0.92 and row["Z-Sigmoid"]<=0.95 and row["Z-Sigmoid_Slope"]<0.1):
                    signal = -1
            elif (row["Z-Sigmoid"]<=-0.97):
                    signal = -1
                    state["Breakout"] = 1
            elif (row["Z-Sigmoid"]<=-0.92 and row["Z-Sigmoid"]>=-0.95 and row["Z-Sigmoid_Slope"]>-0.1):
                    signal = 1
            elif (row["Z-Sigmoid"]>=0.97):
                    signal = 1
                    state["Breakout"] = 1
        elif position == 1:
            if row["Z-Sigmoid"]>=0.875 and state["Breakout"]==0:
                signal = -1
        elif position == -1:
            if row["Z-Sigmoid"]<=-0.875 and state["Breakout"]==0:
                signal = 1
    row["Breakout"] = state["Breakout"]
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
        state = {"Breakout": 0}

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
                signal = generate_signal(row, position, cooldown_over, strategy_params, state)

                # # --- Entry Logic ---
                # if position == 0 and signal != 0:
                #     entry_price = price  # record entry price
                #     trailing_sl = False
                #     take_profit = 0.1
                #     stop_loss = 0.2

                # # --- Exit Logic (Price-based confirmation) ---
                # elif cooldown_over and position != 0 :
                #     if position == 1:
                #         price_diff = price - entry_price
                #         if price_diff >= take_profit:
                #             signal = -1
                #             # entry_price = price  # trailing stop adjustment
                #             # trailing_sl = True
                #         # elif price_diff <= -0.2:
                #         #     signal = -1
                #         # if trailing_sl:
                #         #     if price_diff > 0:
                #         #         entry_price = price  # adjust trailing stop
                #         #     elif price_diff <= -0.01:
                #         #         signal = -1
                #         #         trailing_sl = False
                #     elif position == -1:
                #         price_diff = entry_price - price
                #         if price_diff >= take_profit:
                #             signal = 1
                #             # entry_price = price  # trailing stop adjustment
                #             # trailing_sl = True
                #         # elif price_diff <= -0.2:
                #         #     signal = 1 
                #         # if trailing_sl:
                #         #     if price_diff > 0:
                #         #         entry_price = price  # adjust trailing stop
                #         #     elif price_diff <= -0.01:
                #         #         signal = 1
                #         #         trailing_sl = False

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

        df['Signal'] = signals
        df['Position'] = positions

        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + ['Signal','Position','Z-Score','Z-Sigmoid','Z-Sigmoid_MA','Z-Sigmoid_Slope', 'Breakout']
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
