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

# --- Configuration ---
TEMP_DIR = "temp_signal_processing"
OUTPUT_FILE = "trading_signals.csv"
COOLDOWN_PERIOD_SECONDS = 15
REQUIRED_COLUMNS = ['Time', 'Price']

# --- Strategy Config Example ---
# These can be replaced dynamically for different strategies
STRATEGY_PARAMS = {
    "pb7t12": "PB7_T12",
    "pb9t1": "PB9_T1",
    "pb14t3": "PB14_T3",
    "pb6t8": "PB6_T8"
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
    # df['EWMTrend'] = df['PB9_T1'].ewm(span=120, adjust=False).mean()
    # df['EWMResidual'] = df['PB9_T1'] - df['EWMTrend']
    # df['EWMResidual_Abs']= df['EWMResidual'].abs()
    # df['EWMResidual_High'] = df["EWMResidual_Abs"].rolling(window=1200, min_periods=1).max()

    df["PB7_T12_MA"]=df["PB7_T12"].rolling(window=30, min_periods=1).mean()
    df["PB7_T12_diff"]=df["PB7_T12"]-df["PB7_T12_MA"]
    df["PB7_T12_Signal"]=0
    df.loc[df["PB7_T12_diff"] > 0.25, "PB7_T12_Signal"] = 1
    df.loc[df["PB7_T12_diff"] < -0.25, "PB7_T12_Signal"] = 1
    df["PB7_T12_Reduced_Signal"] = df["PB7_T12_Signal"].copy()
    # df.loc[(df["PB6_T8"] < 0.0000006) & (df["EWMResidual_High"] < 0.1),"PB7_T12_Reduced_Signal"] = 0

    df["PB14_T3_MA_5"]=df["PB14_T3"].rolling(window=5, min_periods=1).mean()
    df["PB14_T3_Slope"]=df["PB14_T3_MA_5"]-df["PB14_T3_MA_5"].shift(1)
    df["PB14_T3_Slope"] = df["PB14_T3_Slope"].fillna(0)
    df["PB14_T3_Direction"] = 0
    df.loc[df["PB14_T3_Slope"] > 0, "PB14_T3_Direction"] = 1
    df.loc[df["PB14_T3_Slope"] < 0, "PB14_T3_Direction"] = -1
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=20).mean()
    df["ATR"]=atr

    df["STD"]=df["PB9_T1"].rolling(window=20, min_periods=1).std()

    kama_signal = df["PB9_T1"].diff(30).abs()
    kama_noise = df["PB9_T1"].diff(1).abs().rolling(window=30).sum()
    er = (kama_signal / kama_noise).fillna(0)

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    
    sc = (er * (sc_fast - sc_slow) + sc_slow).pow(2)

    kama = pd.Series(np.nan, index=df["PB9_T1"].index, dtype=float)

    first_valid_index = sc.first_valid_index()
    if first_valid_index is None:
        return kama

    kama.loc[first_valid_index] = df["PB9_T1"].loc[first_valid_index]

    sc_values = sc.to_numpy()
    price_values = df["PB9_T1"].to_numpy()
    kama_values = kama.to_numpy()

    for i in range(first_valid_index + 1, len(df["PB9_T1"])):
        if np.isnan(kama_values[i-1]):
             kama_values[i] = price_values[i]
        else:
             kama_values[i] = kama_values[i-1] + sc_values[i] * (price_values[i] - kama_values[i-1])
    df["KAMA"] = pd.Series(kama_values, index=df["PB9_T1"].index)
    return df


# -------------------------------------------------------------------
# Strategy Template (Pluggable)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params):

    signal = 0

    for col in ["PB7_T12_Signal", "PB14_T3_Direction"]:
        if col not in row or pd.isna(row[col]) or isinstance(row[col], str):
            return signal
    
    if cooldown_over:
        if position == 0:
            if row["PB7_T12_Signal"] != 0 and row["ATR"] > 0.009:
                if row["PB14_T3_Direction"] == 1:
                    signal = 1
                else:
                    signal = -1
        else:
            if row["PB7_T12_Signal"] != 0:
                signal = -position
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

                # # --- Entry Logic ---
                # if position == 0 and signal != 0:
                #     entry_price = price  # record entry price

                # # --- Exit Logic (Price-based confirmation) ---
                # elif position != 0 and signal == -position:
                #     if position == 1:
                #         price_diff = price - entry_price
                #         if price_diff < 0.2:
                #             # hold until profit >= 0.25
                #             signal = 0
                #         elif price_diff >= 0.25:
                #             signal = -1  # exit long
                #     elif position == -1:
                #         price_diff = entry_price - price
                #         if price_diff < 0.2:
                #             # hold until profit >= 0.25
                #             signal = 0
                #         elif price_diff >= 0.25:
                #             signal = 1  # exit short

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
        final_columns = REQUIRED_COLUMNS + list(strategy_params.values()) + ['Signal', 'Position']
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
