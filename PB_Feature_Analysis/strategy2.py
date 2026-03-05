import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
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
    "momentum": "PB5_T1",
    "last_close": "PB9_T1",
    "rolling_high": "PB10_T4",
    "rolling_low": "PB11_T4",
    "dpo_fast": "PB7_T2",
    "dpo_slow": "PB7_T4",
    "stoch": "PB15_T6"
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
    
    mom_col = strategy_params["momentum"]
    df["Momentum_avg5"] = df[mom_col].rolling(window=5, min_periods=1).mean()

    return df


# -------------------------------------------------------------------
# Strategy Template (Pluggable)
# -------------------------------------------------------------------
def generate_signal(row, position, cooldown_over, strategy_params):

    Momentum = row[strategy_params["momentum"]]
    MomentumStrength = abs(Momentum)
    MomentumMA = row["Momentum_avg5"]
    LastClose = row[strategy_params["last_close"]]
    High = row[strategy_params["rolling_high"]]
    Low = row[strategy_params["rolling_low"]]
    DPO_Fast = row[strategy_params["dpo_fast"]]
    DPO_Slow = row[strategy_params["dpo_slow"]]

    AtrRatio = (LastClose - Low) / (High - Low + 1e-9)
    MomentumThreshold = 0.025
    signal = 0
    
    if cooldown_over:
        if position == 0:
            if (
                Momentum > 0
                and MomentumStrength > MomentumThreshold
                and AtrRatio < 0.75
                and DPO_Fast > DPO_Slow
            ):
                signal = 1
            
            elif (
                Momentum < 0
                and MomentumStrength > MomentumThreshold
                and AtrRatio > 0.25
                and DPO_Fast < DPO_Slow
            ):
                signal = -1
        elif position == 1:
            if (
                Momentum < MomentumMA
                or Momentum < MomentumThreshold
                or DPO_Fast < DPO_Slow
            ):
                signal = -1
        else:
            if (
                Momentum > MomentumMA
                or Momentum > -MomentumThreshold
                or DPO_Fast > DPO_Slow
            ):
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
        df['Time_sec'] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
        df = prepare_derived_features(df, strategy_params)

        # --- Initialize State ---
        position = 0
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = [0] * len(df)
        positions = [0] * len(df)

        # --- Process Row by Row ---
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = row['Time_sec']
            is_last_tick = (i == len(df) - 1)

            signal = 0

            # --- EOD Square Off ---
            if is_last_tick and position != 0:
                signal = -position
            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                signal = generate_signal(row, position, cooldown_over, strategy_params)

            # --- Apply Signal ---
            if signal != 0:
                # Prevent redundant same-direction signals
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0
                else:
                    position += signal
                    last_signal_time = current_time

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
