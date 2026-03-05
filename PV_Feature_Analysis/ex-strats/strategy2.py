#!/usr/bin/env python3
import sys
import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial
import traceback

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    # Check if libraries exist, but we won't use them.
    import cudf
    import dask_cudf
    from numba import cuda
    print("GPU libraries (cuDF, Numba) detected, but they will NOT be used.")
    print("The strategy logic is sequential and runs faster on the CPU.")
except ImportError:
    print("GPU libraries not available - using CPU mode")


#############################################################################
# CONFIGURATION - Modify these defaults as needed
#############################################################################

FEATURE_CONFIG = {
    # Feature column names
    'avg_band_col_1': 'PV3_B3_T6',      # First component for avg band offset
    'avg_band_col_2': 'PV3_B5_T6',      # Second component for avg band offset
    'exit_band_col': 'PV3_B4_T6',      # Inner band for exit (mean_band = price - this)
    
    # Strategy parameters
    'min_pressure_seconds': 15,        # Consecutive seconds at band for entry
    'min_hold_seconds': 15,            # Minimum holding period
    'exit_patience_seconds': 1,        # Price must be at mean for this many seconds to exit
    'pressure_tolerance_pct': 0.00001,  
    
    # *** NEW: Strategy mode and Stop Loss ***
    'strategy_mode': 'momentum', # 'mean_reversion' or 'momentum'
    'trailing_stop_loss_offset': 0.5,  # Price offset for trailing stop
    
    # Output configuration
    'output_file': 'trading_signals.csv',
    
    # Execution parameters
    'max_cpu_workers': os.cpu_count() or 4,
}

# This will be populated by main() with command-line overrides
CONFIG = {}

def find_valid_start_idx_cpu(df, required_cols):
    """Find first row where all required columns are non-NaN (CPU version)."""
    valid_mask = ~df[required_cols].isna().any(axis=1)
    
    if valid_mask.any():
        return valid_mask.idxmax()
    else:
        return len(df)


def process_day_file_cpu(filepath, config):
    """
    Process a single day file using CPU.
    Returns a pandas DataFrame on success, None on failure.
    """
    try:
        print(f"Processing {os.path.basename(filepath)} on CPU...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            print(f"  Warning: Empty file {filepath}, skipping.")
            return None
        
        # Get column names from config
        exit_col = config['exit_band_col']
        avg_col_1 = config['avg_band_col_1']
        avg_col_2 = config['avg_band_col_2']
        
        # Validate required columns
        required_cols = ['Time', 'Price', exit_col, avg_col_1, avg_col_2]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Error: Missing columns {missing_cols} in {filepath}, skipping.")
            return None
        
        # Find valid start index
        feature_cols = [exit_col, avg_col_1, avg_col_2]
        start_idx = find_valid_start_idx_cpu(df, feature_cols)
        
        if start_idx >= len(df):
            print(f"  Warning: No valid data after warm-up in {filepath}, skipping.")
            return None
        
        print(f"  Valid data starts at row {start_idx} for {filepath}")
        
        # Initialize arrays
        n = len(df)
        signals = np.zeros(n, dtype=np.int8)
        
        # State variables
        position = 0
        entry_index = -1
        consecutive_upper = 0
        consecutive_lower = 0
        consecutive_at_mean = 0

        # *** NEW: State for trailing stop ***
        high_water_mark = 0.0
        low_water_mark = 0.0
        
        # Extract config values
        min_pressure = config['min_pressure_seconds']
        min_hold = config['min_hold_seconds']
        exit_patience = config['exit_patience_seconds']
        tolerance = config['pressure_tolerance_pct']
        
        # *** NEW: Get strategy mode and stop loss ***
        strategy_mode = config.get('strategy_mode', 'mean_reversion')
        trailing_stop_loss_offset = config.get('trailing_stop_loss_offset', 5.0)

        # Process each tick
        for i in range(start_idx, n):
            price = df.loc[i, 'Price']
            
            # Calculate avg band offset
            avg_offset = (df.loc[i, avg_col_1] + df.loc[i, avg_col_2]) / 2.0
            
            # Calculate bands using avg offset only (not the feature values)
            upper_band = price + avg_offset
            lower_band = price - avg_offset
            mean_band = price - df.loc[i, exit_col]  # Exit band uses feature value
            
            # Calculate pressure tolerance (ONLY for entry)
            upper_tolerance = upper_band * tolerance
            lower_tolerance = lower_band * tolerance
            
            # Check pressure with tolerance for ENTRY
            at_upper = price >= (upper_band - upper_tolerance)
            at_lower = price <= (lower_band + lower_tolerance)
            
            # Check mean band with NO tolerance for EXIT
            at_mean_for_long_exit = price >= mean_band
            at_mean_for_short_exit = price <= mean_band
            
            # Update entry counters
            if at_upper:
                consecutive_upper += 1
                consecutive_lower = 0
            elif at_lower:
                consecutive_lower += 1
                consecutive_upper = 0
            else:
                consecutive_upper = 0
                consecutive_lower = 0
            
            # Update exit counter
            if position == 1 and at_mean_for_long_exit:
                consecutive_at_mean += 1
            elif position == -1 and at_mean_for_short_exit:
                consecutive_at_mean += 1
            else:
                consecutive_at_mean = 0
            
            # Hold duration
            hold_duration = (i - entry_index) if entry_index >= 0 else 0
            
            # Check if last tick
            is_last_tick = (i == n - 1)
            
            # *** NEW: Entry logic based on strategy_mode ***
            if position == 0 and not is_last_tick:
                signal_to_set = 0
                
                if strategy_mode == 'mean_reversion':
                    if consecutive_lower >= min_pressure:
                        signal_to_set = 1
                    elif consecutive_upper >= min_pressure:
                        signal_to_set = -1
                else: # 'momentum'
                    if consecutive_lower >= min_pressure:
                        signal_to_set = -1  # Flipped
                    elif consecutive_upper >= min_pressure:
                        signal_to_set = 1   # Flipped
                
                # Apply signal and reset state if entry triggered
                if signal_to_set != 0:
                    signals[i] = signal_to_set
                    position = signal_to_set
                    entry_index = i
                    consecutive_lower = 0
                    consecutive_upper = 0
                    consecutive_at_mean = 0
                    
                    # *** NEW: Set watermarks on entry ***
                    if position == 1:
                        high_water_mark = price
                    else: # position == -1
                        low_water_mark = price
            
            # *** NEW: Updated Exit logic with Stop Loss ***
            elif position != 0:
                
                # Check Trailing Stop Loss
                stop_loss_triggered = False
                if position == 1:
                    # Update High Water Mark
                    if price > high_water_mark:
                        high_water_mark = price
                    # Check stop loss
                    stop_price = high_water_mark - trailing_stop_loss_offset
                    if price <= stop_price:
                        stop_loss_triggered = True
                
                elif position == -1:
                    # Update Low Water Mark
                    if price < low_water_mark:
                        low_water_mark = price
                    # Check stop loss
                    stop_price = low_water_mark + trailing_stop_loss_offset
                    if price >= stop_price:
                        stop_loss_triggered = True

                # --- Exit Priority ---
                # 1. Force exit on last tick
                if is_last_tick:
                    if position == 1:
                        signals[i] = -1
                    elif position == -1:
                        signals[i] = 1
                    position = 0
                    entry_index = -1
                    print(f"  EOD square-off: Exited position at last tick for {filepath}")
                
                # 2. Trailing Stop Loss Exit
                elif stop_loss_triggered:
                    if position == 1:
                        signals[i] = -1 # Exit long
                    elif position == -1:
                        signals[i] = 1  # Exit short
                    position = 0
                    entry_index = -1
                    consecutive_upper = 0
                    consecutive_lower = 0
                    consecutive_at_mean = 0

                # 3. Normal exit with patience
                elif hold_duration >= min_hold and consecutive_at_mean >= exit_patience:
                    if position == 1:
                        signals[i] = -1
                    elif position == -1:
                        signals[i] = 1
                    position = 0
                    entry_index = -1
                    consecutive_upper = 0
                    consecutive_lower = 0
                    consecutive_at_mean = 0
        
        # Count signals
        total_signals = np.count_nonzero(signals)
        long_signals = np.count_nonzero(signals == 1)
        short_signals = np.count_nonzero(signals == -1)
        
        print(f"  Generated {total_signals} signals (Long: {long_signals}, Short: {short_signals}) for {filepath}")
        
        # Create output
        output_df = pd.DataFrame({
            'Time': df['Time'],
            'Price': df['Price'],
            'Signal': signals
        })
        
        return output_df

    except Exception as e:
        print(f"  CRITICAL Error processing {filepath}: {e}")
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    global CONFIG
    
    parser = argparse.ArgumentParser(
        description='HFT Cascading Band Pressure Strategy (CPU-Optimized)'
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help='Directory containing day*.csv files'
    )
    # *** NEW ARGUMENTS ***
    parser.add_argument(
        '--strategy-mode',
        type=str,
        choices=['mean_reversion', 'momentum'],
        help=f'Strategy logic to apply (default: {FEATURE_CONFIG["strategy_mode"]})'
    )
    parser.add_argument(
        '--trailing-stop-offset',
        type=float,
        help=f'Price offset for trailing stop loss (default: {FEATURE_CONFIG["trailing_stop_loss_offset"]})'
    )
    # ---
    parser.add_argument(
        '--min-pressure',
        type=int,
        help=f'Minimum consecutive seconds at band for entry (default: {FEATURE_CONFIG["min_pressure_seconds"]})'
    )
    parser.add_argument(
        '--min-hold',
        type=int,
        help=f'Minimum holding period in seconds (default: {FEATURE_CONFIG["min_hold_seconds"]})'
    )
    parser.add_argument(
        '--exit-patience',
        type=int,
        help=f'Seconds price must be at mean band to exit (default: {FEATURE_CONFIG["exit_patience_seconds"]})'
    )
    parser.add_argument(
        '--pressure-tolerance',
        type=float,
        help=f'Pressure tolerance as percentage (default: {FEATURE_CONFIG["pressure_tolerance_pct"]})'
    )
    
    args = parser.parse_args()
    
    # Start with feature config defaults
    CONFIG = FEATURE_CONFIG.copy()
    CONFIG['input_directory'] = args.input_directory
    
    # Override with command-line arguments if provided
    if args.strategy_mode is not None:
        CONFIG['strategy_mode'] = args.strategy_mode
    if args.trailing_stop_offset is not None:
        CONFIG['trailing_stop_loss_offset'] = args.trailing_stop_offset
    if args.min_pressure is not None:
        CONFIG['min_pressure_seconds'] = args.min_pressure
    if args.min_hold is not None:
        CONFIG['min_hold_seconds'] = args.min_hold
    if args.exit_patience is not None:
        CONFIG['exit_patience_seconds'] = args.exit_patience
    if args.pressure_tolerance is not None:
        CONFIG['pressure_tolerance_pct'] = args.pressure_tolerance
    
    # Display configuration
    print("=" * 80)
    print("HFT CASCADING BAND PRESSURE STRATEGY (CPU-OPTIMIZED)")
    print("=" * 80)
    print(f"\nCONFIGURATION:")
    print(f"  Acceleration mode: {'CPU (Optimized)'}") # Always CPU
    print(f"  Input directory: {CONFIG['input_directory']}")
    print(f"  Output file: {CONFIG['output_file']}")
    print(f"\nFEATURE COLUMNS:")
    print(f"  Avg band component 1: {CONFIG['avg_band_col_1']}")
    print(f"  Avg band component 2: {CONFIG['avg_band_col_2']}")
    print(f"  Exit band: {CONFIG['exit_band_col']}")
    print(f"\nBAND CALCULATION:")
    print(f"  Avg offset = ({CONFIG['avg_band_col_1']} + {CONFIG['avg_band_col_2']}) / 2")
    print(f"  Upper band = Price + Avg offset")
    print(f"  Lower band = Price - Avg offset")
    print(f"  Mean band = Price - {CONFIG['exit_band_col']}")
    print(f"\nSTRATEGY PARAMETERS:")
    # *** NEW: Print new params ***
    print(f"  Strategy mode: {CONFIG['strategy_mode']}")
    print(f"  Trailing stop offset: {CONFIG['trailing_stop_loss_offset']}")
    # ---
    print(f"  Min pressure seconds: {CONFIG['min_pressure_seconds']}")
    print(f"  Min hold seconds: {CONFIG['min_hold_seconds']}")
    print(f"  Exit patience seconds: {CONFIG['exit_patience_seconds']}")
    print(f"  Pressure tolerance: {CONFIG['pressure_tolerance_pct']:.4%}")
    print(f"\nEXECUTION:")
    print(f"  Max CPU workers: {CONFIG['max_cpu_workers']}")
    print("=" * 80)
    
    # Validate input directory
    if not os.path.isdir(CONFIG['input_directory']):
        print(f"\nError: Directory '{CONFIG['input_directory']}' does not exist")
        sys.exit(1)
    
    # Find day files and sort chronologically
    pattern = os.path.join(CONFIG['input_directory'], 'day*.csv')
    day_files = glob.glob(pattern)
    
    def extract_day_number(filepath):
        filename = os.path.basename(filepath)
        num_str = filename.replace('day', '').replace('.csv', '')
        try:
            return int(num_str)
        except ValueError:
            return 0
    
    day_files = sorted(day_files, key=extract_day_number)
    
    if not day_files:
        print(f"\nError: No 'day*.csv' files found in '{CONFIG['input_directory']}'")
        sys.exit(1)
    
    print(f"\nFound {len(day_files)} day files to process:")
    for f in day_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(day_files) > 5:
        print(f"  ... and {len(day_files) - 5} more")
    
    # Output file setup
    output_file = CONFIG['output_file']
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"\nRemoved existing {output_file}")
    
    print("\n" + "=" * 80)
    print("PROCESSING DAY FILES CONCURRENTLY")
    print("=" * 80)
    
    # Choose processing function and worker count
    # We ALWAYS use the CPU function now
    process_func = process_day_file_cpu
    max_workers = CONFIG['max_cpu_workers']
    print(f"Using ProcessPoolExecutor with {max_workers} CPU worker(s)...")

    # Use partial to pre-load the 'config' argument for executor.map
    partial_process_func = partial(process_func, config=CONFIG)
    
    results = []
    failed_files = []
    
    # The 'with' statement ensures the pool is properly shut down
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            future_to_file = {executor.submit(partial_process_func, f): f for f in day_files}
            
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"\n--- Error processing {os.path.basename(filepath)} ---")
                    print(f"Exception: {e}")
                    traceback.print_exc()
                    failed_files.append(filepath)
                    results.append(None)
        except Exception as e:
            print(f"\n--- A CRITICAL ERROR occurred during parallel execution ---")
            print(e)
            traceback.print_exc()
            print("--- Shutting down pool ---")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE - AGGREGATING RESULTS")
    print("=" * 80)
    
    # Filter out None results from failed/skipped files
    successful_results = [df for df in results if df is not None]
    success_count = len(successful_results)
    
    if failed_files:
        print(f"\nWARNING: {len(failed_files)} file(s) failed to process:")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")
    
    if success_count == 0:
        print("\nNo files were processed successfully.")
        print("=" * 80)
        sys.exit(1)
        
    print(f"\nSuccessfully processed: {success_count}/{len(day_files)} files")
    
    # Concatenate all results into final DataFrame
    print("Concatenating results into final DataFrame...")
    final_df = pd.concat(successful_results, ignore_index=True)
    
    print(f"Saving results to: {output_file}")
    final_df.to_csv(output_file, index=False)
    
    # Summary statistics
    total_rows = len(final_df)
    total_signals = (final_df['Signal'] != 0).sum()
    long_signals = (final_df['Signal'] == 1).sum()
    short_signals = (final_df['Signal'] == -1).sum()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total signals: {total_signals:,}")
    print(f"  Long signals: {long_signals:,}")
    print(f"  Short signals: {short_signals:,}")
    print("=" * 80)


if __name__ == '__main__':
    main()