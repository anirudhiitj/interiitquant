#!/usr/bin/env python3
"""
High-Performance GPU-Accelerated Trading Signal Generator
Uses dask_cudf for I/O and numba CUDA kernels for strategy execution
Processes multiple day files in parallel with end-of-day square-off

*** MODIFIED LOGIC (Crossover Strategy) ***
*** REFACTORED for parallel file processing ***
"""

import sys
import os
import glob
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

import cudf
import dask_cudf
import numba
from numba import cuda
import numpy as np

#############################################################################
# CONFIGURATION - NEW CROSSOVER STRATEGY
#############################################################################

FEATURE_CONFIG = {
    'fast_ma': 'PB6_T6',            # Used for Fast MA
    'regime_slow_ma': 'PB6_T11',    # Used for Regime Filter AND Slow MA
    'stop_loss_high_band': 'PB10_T4', # Stop-loss for short positions
    'stop_loss_low_band': 'PB11_T4',  # Stop-loss for long positions
}

# Percentage stop-loss (e.g., 0.01 = 1%). Adjust as needed.
#STOP_LOSS_PCT = 0.01

# Cooldown period (in ticks) after taking a position.
# User confirms 15 ticks = 15 seconds for their data.
COOLDOWN_TICKS = 15

#############################################################################

@cuda.jit
def trading_kernel(price, regime_slow_ma, fast_ma,
                   stop_high_band, stop_low_band, cooldown_ticks,
                   signal_out, start_idx, eod_squareoff):
    """
    CUDA kernel for tick-by-tick trading strategy execution.

    Stop-loss uses per-row high/low bands:
      - Long exit if current_price <= stop_low_band[i]
      - Short exit if current_price >= stop_high_band[i]

    On exit we schedule the opposite-entry at the next tick (i+1) if the
    regime at the next tick matches the required opposite-regime and there
    are enough ticks remaining.
    """
    tid = cuda.threadIdx.x

    if tid == 0:
        n = price.shape[0]
        position = 0
        long_cooldown = 0
        short_cooldown = 0
        entry_price = 0.0  # store entry price when a position is opened

        # Scheduler for forced next-tick entry (0 = none, 1/-1 = entry signal)
        scheduled_entry = 0
        scheduled_tick = -1

        # Process each tick sequentially
        for i in range(start_idx, n):
            signal_to_output = 0
            should_exit = False

            # Apply scheduled entry at the start of the tick (if any)
            if i == scheduled_tick:
                if scheduled_entry == 1:
                    position = 1
                    long_cooldown = cooldown_ticks
                    entry_price = price[i]
                    signal_to_output = 1
                    # clear schedule
                    scheduled_entry = 0
                    scheduled_tick = -1
                    signal_out[i] = signal_to_output
                    # skip normal entry/exit logic for this tick
                    continue
                elif scheduled_entry == -1:
                    position = -1
                    short_cooldown = cooldown_ticks
                    entry_price = price[i]
                    signal_to_output = -1
                    scheduled_entry = 0
                    scheduled_tick = -1
                    signal_out[i] = signal_to_output
                    continue

            # Decrement cooldowns
            if long_cooldown > 0:
                long_cooldown -= 1
            if short_cooldown > 0:
                short_cooldown -= 1

            # --- Read current and previous values ---
            curr_price = price[i]
            curr_slow_ma = regime_slow_ma[i]
            curr_fast_ma = fast_ma[i]

            prev_slow_ma = regime_slow_ma[i-1]
            prev_fast_ma = fast_ma[i-1]

            # Per-row stop bands for this tick (may be NaN/0 depending on input)
            curr_stop_high = stop_high_band[i]
            curr_stop_low = stop_low_band[i]

            # --- Determine Regime ---
            regime = 0
            if curr_price > curr_slow_ma:
                regime = 1   # BULL
            else:
                regime = -1  # BEAR

            # --- Calculate Crossover Conditions ---
            bullish_cross = (prev_fast_ma <= prev_slow_ma) and (curr_fast_ma > curr_slow_ma)
            bearish_cross = (prev_fast_ma >= prev_slow_ma) and (curr_fast_ma < curr_slow_ma)

            # --- LONG position logic - check exit conditions first ---
            if position == 1:
                # Only check for exits if cooldown is over
                if long_cooldown == 0:
                    # Exit conditions: regime flip OR bearish cross OR band stoploss hit
                    stop_loss_hit = False
                    # If per-row low band is non-zero use it (assumes 0/NaN indicated by 0)
                    if curr_stop_low != 0.0:
                        if curr_price <= curr_stop_low:
                            stop_loss_hit = True

                    if regime == -1 or bearish_cross or stop_loss_hit:
                        signal_to_output = -1  # Exit long (sell signal)
                        position = 0
                        entry_price = 0.0
                        should_exit = True

                        # Schedule immediate reversal (enter short) at next tick if regime maintained and ticks remain
                        if (i + 1) < n and ((n - 1 - i) >= cooldown_ticks):
                            # determine regime at next tick
                            if price[i+1] > regime_slow_ma[i+1]:
                                next_regime = 1
                            else:
                                next_regime = -1
                            if next_regime == -1:
                                scheduled_entry = -1
                                scheduled_tick = i + 1

            # --- SHORT position logic - check exit conditions first ---
            elif position == -1:
                # Only check for exits if cooldown is over
                if short_cooldown == 0:
                    stop_loss_hit = False
                    if curr_stop_high != 0.0:
                        if curr_price >= curr_stop_high:
                            stop_loss_hit = True

                    if regime == 1 or bullish_cross or stop_loss_hit:
                        signal_to_output = 1   # Exit short (buy signal)
                        position = 0
                        entry_price = 0.0
                        should_exit = True

                        # Schedule immediate reversal (enter long) at next tick if regime maintained and ticks remain
                        if (i + 1) < n and ((n - 1 - i) >= cooldown_ticks):
                            if price[i+1] > regime_slow_ma[i+1]:
                                scheduled_entry = 1
                                scheduled_tick = i + 1

            # --- FLAT position logic - only if we didn't just exit ---
            if position == 0 and not should_exit:
                # Check for LONG entry
                if regime == 1 and bullish_cross and short_cooldown == 0:
                    signal_to_output = 1
                    position = 1
                    long_cooldown = cooldown_ticks
                    entry_price = curr_price  # set entry price for reference

                # Check for SHORT entry
                elif regime == -1 and bearish_cross and long_cooldown == 0:
                    signal_to_output = -1
                    position = -1
                    short_cooldown = cooldown_ticks
                    entry_price = curr_price  # set entry price for reference

            # Store signal
            signal_out[i] = signal_to_output

        # End of day square-off: if we still have a position, close it
        if position != 0 and n > 0:
            # Square off at the last tick
            last_idx = n - 1
            if position == 1:
                signal_out[last_idx] = -1  # Exit long
                eod_squareoff[0] = 1   # Flag that we squared off
            elif position == -1:
                signal_out[last_idx] = 1   # Exit short
                eod_squareoff[0] = 1   # Flag that we squared off


def find_valid_start_idx(df, required_cols):
    """
    Find the first row where all required feature columns are non-NaN
    for BOTH the current row (i) and the previous row (i-1).
    This is necessary for crossover calculations.
    
    Args:
        df: cudf.DataFrame with feature columns
        required_cols: List of required column names
    
    Returns:
        int: Index of first valid row (i)
    """
    # Check for NaN in any required column - use CPU pandas for simplicity
    df_cpu = df[required_cols].to_pandas()
    
    # Mask for valid current row
    valid_mask = ~df_cpu.isna().any(axis=1)
    
    # Mask for valid previous row
    prev_valid_mask = valid_mask.shift(1,fill_value=False)
    
    # Combined mask: both current and previous must be valid
    combined_valid_mask = valid_mask & prev_valid_mask
    
    if combined_valid_mask.any():
        # Return the first index `i` where both `i` and `i-1` are valid
        return combined_valid_mask.idxmax()
    else:
        return len(df)  # No valid data


def process_day_file(filepath, temp_output_path, feature_cols):
    """
    Process a single day's CSV file and generate trading signals.
    Writes results to a temporary file.
    
    Args:
        filepath: Path to input CSV file
        temp_output_path: Path to write the temporary output CSV
        feature_cols: Dictionary of feature column names
    
    Returns:
        (str, str): Tuple of (temp_output_path, status_string)
    
    Raises:
        ValueError: If file is empty, columns are missing, or no valid data exists.
    """
    print(f"Processing {os.path.basename(filepath)}...")
    
    # Load data using dask_cudf
    ddf = dask_cudf.read_parquet(filepath)
    df = ddf.compute()  # Convert to cudf.DataFrame
    
    if len(df) == 0:
        raise ValueError(f"Empty file, skipping.")
    
    # Get required columns: Price + MAs needed for crossover
    required_cols = ['Price', feature_cols['regime_slow_ma'], feature_cols['fast_ma']]

    # Validate columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols}, skipping.")

    # Find first valid row (after warm-up)
    start_idx = find_valid_start_idx(df, required_cols)

    if start_idx >= len(df):
        raise ValueError(f"No valid data for crossover (need 2 ticks), skipping.")

    print(f"  Valid data for crossover in {os.path.basename(filepath)} starts at row {start_idx}")

    # Prepare arrays for kernel
    n = len(df)
    signal_array = cudf.Series([0] * n, dtype='int8')
    eod_squareoff = cudf.Series([0], dtype='int8')  # Flag for EOD square-off

    # --- Extract columns as device arrays (MODIFIED) ---
    price = df['Price'].values
    regime_slow_ma = df[feature_cols['regime_slow_ma']].values
    fast_ma = df[feature_cols['fast_ma']].values

    # Per-row stop bands: fill NaN with 0 -> disabled
    stop_high = df[feature_cols['stop_loss_high_band']].fillna(0).values
    stop_low  = df[feature_cols['stop_loss_low_band']].fillna(0).values

    signal_out = signal_array.values
    eod_flag = eod_squareoff.values

    # --- Launch kernel (single thread for sequential processing) (FIXED) ---
    trading_kernel[1, 1](price, regime_slow_ma, fast_ma,
                         stop_high, stop_low, np.int32(COOLDOWN_TICKS),
                         signal_out, start_idx, eod_flag)
    
    # Wait for kernel completion
    cuda.synchronize()
    
    # Count signals
    total_signals = (signal_array != 0).sum()
    long_signals = (signal_array == 1).sum()
    short_signals = (signal_array == -1).sum()
    
    status_str = f"Generated {total_signals} signals (Long: {long_signals}, Short: {short_signals})"
    
    # Check if EOD square-off occurred
    if eod_squareoff[0] == 1:
        status_str += " (EOD square-off executed)"
    
    # Create output dataframe (removed PB10_T4 / PB11_T4 columns)
    output_df = cudf.DataFrame({
        'Time': df['Time'].reset_index(drop=True),
        'Price': df['Price'].reset_index(drop=True),
        'Signal': signal_array.reset_index(drop=True),
        feature_cols['fast_ma']: df[feature_cols['fast_ma']].reset_index(drop=True),
        feature_cols['regime_slow_ma']: df[feature_cols['regime_slow_ma']].reset_index(drop=True),
    })
    
    # Convert to pandas for CSV writing
    output_pd = output_df.to_pandas()
    
    # Write to its own temporary file with a header
    output_pd.to_csv(temp_output_path, mode='w', header=True, index=False)
    
    return temp_output_path, status_str


def main():
    """Main entry point for the generator."""
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Trading Signal Generator - Edit FEATURE_CONFIG in code to change features'
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help='Directory containing day*.csv files'
    )
    parser.add_argument(
        '--start-day',
        type=int,
        default=None,
        help='Optional start day number (inclusive). e.g., 5'
    )
    parser.add_argument(
        '--end-day',
        type=int,
        default=None,
        help='Optional end day number (inclusive). e.g., 20'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Optional: Maximum number of parallel processes to run. Defaults to all available cores.'
    )
    
    args = parser.parse_args()
    
    # Display configuration
    print("=" * 80)
    print("GPU-ACCELERATED TRADING SIGNAL GENERATOR (CROSSOVER LOGIC)")
    print("=" * 80)
    print("\nFEATURE CONFIGURATION:")
    for key, value in FEATURE_CONFIG.items():
        print(f"  {key:20s}: {value}")
    print(f"  {'cooldown_ticks':20s}: {COOLDOWN_TICKS}")
    print("=" * 80)
    
    # Validate input directory
    if not os.path.isdir(args.input_directory):
        print(f"\nError: Directory '{args.input_directory}' does not exist")
        sys.exit(1)
    
    # Find all day*.csv files
    pattern = os.path.join(args.input_directory, 'day*.parquet')
    all_day_files = sorted(glob.glob(pattern))
    
    # --- Filter files based on start/end day arguments ---
    day_files = []
    if args.start_day is not None or args.end_day is not None:
        print(f"\nFiltering files for range: start={args.start_day or 'N/A'}, end={args.end_day or 'N/A'}")
        for f in all_day_files:
            basename = os.path.basename(f)
            # Extract the first number from the filename
            match = re.search(r'(\d+)', basename)
            
            if not match:
                print(f"  Warning: Skipping file with non-numeric name: {basename}")
                continue
            
            try:
                file_num = int(match.group(1))
            except ValueError:
                print(f"  Warning: Skipping file with unparsable number: {basename}")
                continue
            
            # Apply start filter
            if args.start_day is not None and file_num < args.start_day:
                continue
                
            # Apply end filter
            if args.end_day is not None and file_num > args.end_day:
                continue
                
            # If we get here, the file is in range
            day_files.append(f)
    else:
        day_files = all_day_files
    # --- END OF FILTERING LOGIC ---
    
    if not day_files:
        print(f"\nError: No 'day*.csv' files found in '{args.input_directory}' (after filtering).")
        sys.exit(1)
    
    print(f"\nFound {len(day_files)} day files to process (after filtering):")
    for f in day_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(day_files) > 5:
        print(f"  ... and {len(day_files) - 5} more")
    
    # --- PARALLELISM SETUP ---
    # Output file
    final_output_file = 'trading_signals.csv'
    
    # Temporary directory for partial results
    temp_dir = Path("./temp_output_parts")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Remove existing final output file
    if os.path.exists(final_output_file):
        os.remove(final_output_file)
        print(f"\nRemoved existing {final_output_file}")
    
    # --- UPDATED PRINT STATEMENT ---
    workers_to_use = args.max_workers
    workers_str = f"{workers_to_use}" if workers_to_use is not None else f"Default (up to {os.cpu_count() or 1})"
    print("\n" + "=" * 80)
    print(f"PROCESSING DAY FILES IN PARALLEL (max_workers={workers_str})")
    print("=" * 80)
    
    success_count = 0
    temp_files_map = {} # Map filepath to temp file path for ordered concatenation

    # --- UPDATED EXECUTOR ---
    # Use ProcessPoolExecutor to parallelize file processing
    # Pass max_workers directly from args
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks
        futures_map = {}
        for filepath in day_files:
            temp_output_path = temp_dir / f"part_{os.path.basename(filepath)}"
            future = executor.submit(process_day_file, filepath, temp_output_path, FEATURE_CONFIG)
            futures_map[future] = filepath

        # Process results as they complete
        for future in as_completed(futures_map):
            filepath = futures_map[future]
            try:
                temp_output_path, status_str = future.result()
                print(f"  SUCCESS: {os.path.basename(filepath)} - {status_str}")
                temp_files_map[filepath] = temp_output_path
                success_count += 1
            except Exception as e:
                print(f"  ERROR processing {os.path.basename(filepath)}: {e}")

    print("\n" + "=" * 80)
    print("CONCATENATING RESULTS")
    print("=" * 80)

    # Concatenate results in the original file order
    if success_count > 0:
        with open(final_output_file, 'wb') as outfile:
            # Iterate over the original, sorted, filtered list to ensure order
            for idx, filepath in enumerate(day_files):
                if filepath not in temp_files_map:
                    continue # This file failed, skip it
                
                temp_file = temp_files_map[filepath]
                
                # Check if file has content before processing
                if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                    print(f"  Warning: Skipping empty temp file for {os.path.basename(filepath)}")
                    continue

                with open(temp_file, 'rb') as infile:
                    if outfile.tell() == 0:
                        # This is the very first file, write header
                        outfile.write(infile.read())
                    else:
                        # Skip header for subsequent files
                        infile.readline() 
                        outfile.write(infile.read())
                        
        print(f"Successfully concatenated {success_count} files into {final_output_file}")
    else:
        print("No files were processed successfully. No output file generated.")

    # Clean up temporary directory
    try:
        shutil.rmtree(temp_dir)
        print(f"Removed temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temp directory {temp_dir}: {e}")

    print("\n" + "=" * 80)
    print("SIGNAL GENERATION COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {success_count}/{len(day_files)} files")
    
    # Summary statistics
    if os.path.exists(final_output_file) and success_count > 0:
        try:
            summary_df = dask_cudf.read_csv(final_output_file).compute()
            if len(summary_df) > 0:
                total_rows = len(summary_df)
                total_signals = (summary_df['Signal'] != 0).sum()
                long_signals = (summary_df['Signal'] == 1).sum()
                short_signals = (summary_df['Signal'] == -1).sum()
                
                print(f"\nOVERALL STATISTICS:")
                print(f"  Total rows: {total_rows}")
                print(f"  Total signals: {total_signals}")
                print(f"  Total long signals: {long_signals}")
                print(f"  Total short signals: {short_signals}")

                df=summary_df[summary_df['Signal'] != 0]
                df[['Signal']].to_csv("signal_order.csv", index=False)
            else:
                print("\nOVERALL STATISTICS: Output file is empty.")
        except Exception as e:
            print(f"\nError reading output file for stats: {e}")
    elif not os.path.exists(final_output_file):
        print("\nNo output file generated.")
    print("=" * 80)


if __name__ == '__main__':
    # This check is essential for multiprocessing to work correctly
    main()


