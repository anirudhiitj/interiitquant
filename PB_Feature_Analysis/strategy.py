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

# --- Configuration ---a
TEMP_DIR = "temp_signal_processing"
OUTPUT_FILE = "trading_signals.csv"
COOLDOWN_PERIOD_SECONDS = 15
fast_feature = 'PB6_T4'
long_feature = 'PB6_T11'
REQUIRED_COLUMNS = ['Time', 'Price', fast_feature, long_feature]

def extract_day_num(filepath):
    """Extracts the day number 'n' from a filepath like '.../day{n}.parquet'."""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path) -> str:
    """
    Processes a single day's parquet file to generate signals.
    Saves the result as a CSV in the temp directory.
    """
    try:
        # 1. Read data using dask_cudf, then compute into a cudf DataFrame
        ddf = dask_cudf.read_parquet(file_path, columns=REQUIRED_COLUMNS)
        gdf = ddf.compute()
        
        # 2. Move to Pandas for stateful row-by-row processing
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None

        # 3. Prepare data
        # Convert Time to total seconds from midnight for easy comparison
        df['Time_sec'] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
        
        # 4. Initialize state variables
        position = 0
        # Initialize to allow immediate signal on first tick
        last_signal_time = -COOLDOWN_PERIOD_SECONDS 
        
        signals = [0] * len(df)
        positions = [0] * len(df)
        
        num_rows = len(df)

        # 5. Iterate through each tick (row)
        for i in range(num_rows):
            row = df.iloc[i]
            current_time = row['Time_sec']
            fast_window = row[fast_feature]
            long_window = row[long_feature]
            
            signal = 0
            is_last_tick = (i == num_rows - 1)
            
            # --- EOD Square Off Logic ---
            # This must be checked first, regardless of cooldown
            if is_last_tick and position != 0:
                signal = -position # Force exit
            
            else:
                # --- Standard Signal Logic ---
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                
                # Determine current regime
                current_regime = 0
                if fast_window > long_window:
                    current_regime = 1  # Long regime
                elif fast_window < long_window:
                    current_regime = -1 # Short regime
                
                if cooldown_over and current_regime != 0:
                    # Try to Enter (if flat)
                    if position == 0:
                        signal = current_regime
                    
                    # Try to Exit (if regime flipped)
                    elif (position == 1 and current_regime == -1) or \
                         (position == -1 and current_regime == 1):
                        signal = -position # Exit signal
            
            # --- Apply Signal and Update Position ---
            if signal != 0:
                # Check for position clamping
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    # Trying to go long when already long, or short when already short
                    signal = 0 # Invalidate signal, no change
                else:
                    position += signal
                    last_signal_time = current_time # Reset cooldown timer
            
            signals[i] = signal
            positions[i] = position

        # 6. Store results and save to temp file
        df['Signal'] = signals
        df['Position'] = positions
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = ['Time', 'Price', 'Signal', 'Position', fast_feature, long_feature]
        df[final_columns].to_csv(output_path, index=False)
        
        return str(output_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main(directory: str, max_workers: int):
    """
    Main function to find files, process them in parallel, and concatenate results.
    """
    start_time = time.time()
    
    # 1. Setup Temp Directory
    temp_dir_path = pathlib.Path(TEMP_DIR)
    if temp_dir_path.exists():
        print(f"Removing existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    print(f"Created temp directory: {temp_dir_path}")

    # 2. Find and Sort Input Files
    files_pattern = os.path.join(directory, "day*.parquet")
    all_files = glob.glob(files_pattern)
    
    # Sort files based on the extracted day number (day1, day2, ..., day100)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"Error: No 'day*.parquet' files found in {directory}")
        shutil.rmtree(temp_dir_path)
        return

    print(f"Found {len(sorted_files)} day files to process.")

    processed_files = []
    
    try:
        # 3. Process Files in Parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for f in sorted_files:
                day_num = extract_day_num(f)
                if day_num != -1:
                    future = executor.submit(process_day, f, day_num, temp_dir_path)
                    futures[future] = f
                else:
                    print(f"Warning: Could not extract day number from {f}. Skipping.")
            
            print(f"Submitted {len(futures)} jobs to ProcessPool with {max_workers} workers...")
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                        print(f"Successfully processed: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # 4. Concatenate CSVs
        if not processed_files:
            print("No files were processed successfully. Exiting.")
            return

        print(f"All processing complete. Concatenating {len(processed_files)} CSVs into {OUTPUT_FILE}...")
        
        # Sort the output CSVs in the temp directory by day number
        def get_csv_day_num(p):
            match = re.search(r'day(\d+)\.csv', os.path.basename(p))
            return int(match.group(1)) if match else -1

        sorted_csvs = sorted(processed_files, key=get_csv_day_num)
        
        with open(OUTPUT_FILE, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        # Copy first file completely (including header)
                        shutil.copyfileobj(infile, outfile)
                    else:
                        # Skip header for subsequent files
                        infile.readline() 
                        shutil.copyfileobj(infile, outfile)
        
        print(f"Successfully created: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\nManual interrupt detected! Shutting down all processes...")
        # The 'with' block for ProcessPoolExecutor handles shutdown automatically
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 5. Cleanup
        if temp_dir_path.exists():
            print(f"Cleaning up temp directory: {temp_dir_path}")
            shutil.rmtree(temp_dir_path)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process daily trading parquet files into a single signal CSV."
    )
    parser.add_argument(
        "directory", 
        type=str, 
        help="Directory containing the 'day{n}.parquet' files."
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=os.cpu_count(), 
        help="Maximum number of parallel processes to run."
    )
    
    args = parser.parse_args()
    main(args.directory, args.max_workers)
