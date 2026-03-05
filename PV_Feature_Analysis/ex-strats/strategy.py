#!/usr/bin/env python3
"""
OPTIMIZED MEAN REVERSION STRATEGY - PARALLEL PROCESSING
Parallel CSV loading with ProcessPoolExecutor
Each day processed efficiently on CPU with pandas/numpy
"""

import sys
import os
import glob
import argparse
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial
import traceback
from pathlib import Path

#############################################################################
# FEATURE CONFIGURATION
#############################################################################

FEATURE_CONFIG = {
    # Feature column names
    'B1_feature': 'PV3_B1_T6',
    'B2_feature': 'PV3_B2_T6',
    'B3_feature': 'PV3_B3_T6',
    'B4_feature': 'PV3_B4_T6',
    'B5_feature': 'PV3_B5_T6',
    'B6_feature': 'PV3_B6_T6',
    'B7_feature': 'PV3_B7_T6',
    'B4_short_tf': 'PV3_B4_T6',
    'B4_long_tf': 'PV3_B4_T12',
    
    # Strategy parameters - OPTIMIZED VALUES
    'divergence_entry_threshold': 2.5,
    'b4_momentum_threshold': -0.001,
    'volatility_regime_factor': 1.2,
    'divergence_exit_threshold': 0.3,
    'b4_profit_target_zscore': 0.5,
    'stop_loss_zscore': 2.0,
    'max_hold_seconds': 120,
    'min_hold_seconds': 20,
    'cooldown_after_exit': 10,
    'lookback_window': 400,
    
    # Output configuration
    'output_file': 'trading_signals.csv',
    
    # Execution parameters
    'max_cpu_workers': os.cpu_count() or 4,
}

CONFIG = {}


def calculate_zscore(value, rolling_mean, rolling_std):
    """Calculate z-score with division-by-zero protection."""
    if rolling_std < 1e-9:
        return 0.0
    return (value - rolling_mean) / rolling_std


def classify_volatility_regime(b1_values, current_idx, window):
    """
    Classify current volatility regime based on B1 z-score.
    Returns: 'HIGH_VOL', 'LOW_VOL', or 'NORMAL'
    """
    start_idx = max(0, current_idx - window)
    if start_idx >= current_idx:
        return 'NORMAL'
    
    b1_slice = b1_values[start_idx:current_idx]
    if len(b1_slice) < 10:
        return 'NORMAL'
    
    mean = np.mean(b1_slice)
    std = np.std(b1_slice)
    current = b1_values[current_idx]
    
    if std < 1e-9:
        return 'NORMAL'
    
    z = (current - mean) / std
    
    if z > 1.0:
        return 'HIGH_VOL'
    elif z < -1.0:
        return 'LOW_VOL'
    else:
        return 'NORMAL'


def get_adaptive_lookback(vol_regime, base_lookback):
    """
    Adapt lookback window based on volatility regime.
    High vol = shorter window, Low vol = longer window
    """
    if vol_regime == 'HIGH_VOL':
        return int(base_lookback * 0.7)  # 210 seconds
    elif vol_regime == 'LOW_VOL':
        return int(base_lookback * 1.3)  # 390 seconds
    return base_lookback


def find_valid_start_idx(df, required_cols):
    """Find first row where all required columns are non-NaN."""
    valid_mask = ~df[required_cols].isna().any(axis=1)
    
    if valid_mask.any():
        return valid_mask.idxmax()
    else:
        return len(df)


def process_day_file(filepath, config):
    """
    Process single day file using optimized CPU operations.
    Returns a pandas DataFrame on success, None on failure.
    """
    try:
        filename = os.path.basename(filepath)
        
        # Load CSV with pandas (fast and efficient)
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            print(f"  Warning: Empty file {filename}, skipping.")
            return None
        
        # Extract feature names from config
        B1 = config['B1_feature']
        B2 = config['B2_feature']
        B3 = config['B3_feature']
        B4 = config['B4_feature']
        B5 = config['B5_feature']
        B6 = config['B6_feature']
        B7 = config['B7_feature']
        B4_short = config['B4_short_tf']
        B4_long = config['B4_long_tf']
        
        # Validate columns
        required_cols = ['Time', 'Price', B1, B2, B3, B4, B5, B6, B7, B4_short, B4_long]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Error: Missing columns {missing_cols} in {filename}, skipping.")
            return None
        
        # Find first valid row (all features available)
        feature_cols = [B1, B2, B3, B4, B5, B6, B7, B4_short, B4_long]
        start_idx = find_valid_start_idx(df, feature_cols)
        
        if start_idx >= len(df):
            print(f"  Warning: No valid data in {filename}, skipping.")
            return None
        
        # Extract numpy arrays for faster processing
        n = len(df)
        price = df['Price'].values
        b1_vals = df[B1].values
        b2_vals = df[B2].values
        b3_vals = df[B3].values
        b4_vals = df[B4].values
        b5_vals = df[B5].values
        b6_vals = df[B6].values
        b7_vals = df[B7].values
        b4_short_vals = df[B4_short].values
        b4_long_vals = df[B4_long].values
        
        # Initialize signals array
        signals = np.zeros(n, dtype=np.int8)
        
        # Extract parameters
        base_lookback = config['lookback_window']
        div_entry_thresh = config['divergence_entry_threshold']
        div_exit_thresh = config['divergence_exit_threshold']
        b4_momentum_thresh = config['b4_momentum_threshold']
        stop_loss_z = config['stop_loss_zscore']
        b4_profit_z = config['b4_profit_target_zscore']
        min_hold = config['min_hold_seconds']
        max_hold = config['max_hold_seconds']
        cooldown = config['cooldown_after_exit']
        vol_regime_factor = config['volatility_regime_factor']
        
        # State tracking variables
        position = 0
        entry_idx = -1
        entry_divergence = 0.0
        entry_price = 0.0
        cooldown_until = -1
        
        # Process each timestamp
        for i in range(start_idx, n):
            # Skip if in cooldown period
            if i < cooldown_until:
                continue
            
            # Determine volatility regime for adaptive lookback
            vol_regime = classify_volatility_regime(b1_vals, i, base_lookback)
            lookback = get_adaptive_lookback(vol_regime, base_lookback)
            
            # Calculate rolling window
            start = max(start_idx, i - lookback)
            if i - start < 30:  # Need minimum data points
                continue
            
            # Calculate statistics for B1
            b1_mean = np.mean(b1_vals[start:i])
            b1_std = np.std(b1_vals[start:i])
            b1_z = calculate_zscore(b1_vals[i], b1_mean, b1_std)
            
            # Calculate statistics for B7
            b7_mean = np.mean(b7_vals[start:i])
            b7_std = np.std(b7_vals[start:i])
            b7_z = calculate_zscore(b7_vals[i], b7_mean, b7_std)
            
            # Calculate statistics for B4
            b4_mean = np.mean(b4_vals[start:i])
            b4_std = np.std(b4_vals[start:i])
            b4_z = calculate_zscore(b4_vals[i], b4_mean, b4_std)
            
            # Core signal: divergence between B1 and B7
            divergence = b1_z - b7_z
            
            # Momentum confirmation from cross-timeframe B4
            b4_momentum = b4_short_vals[i] - b4_long_vals[i]
            
            # Adjust entry threshold based on volatility regime
            effective_entry_thresh = div_entry_thresh * vol_regime_factor if vol_regime == 'HIGH_VOL' else div_entry_thresh
            
            # Calculate hold duration
            hold_duration = (i - entry_idx) if entry_idx >= 0 else 0
            
            # Check if last tick of the day - STRICT EOD CHECK
            is_last_tick = (i == n - 1)
            
            # === POSITION MANAGEMENT ===
            
            if position == 0:
                # ENTRY LOGIC - NO ENTRIES ON LAST TICK
                if not is_last_tick and abs(divergence) > effective_entry_thresh:
                    # Determine potential trade direction (mean reversion)
                    potential_direction = -np.sign(divergence)
                    
                    # Check if B4 momentum supports the trade
                    b4_supports = (potential_direction * b4_momentum) < b4_momentum_thresh
                    
                    # Enter if momentum confirms
                    if b4_supports:
                        signals[i] = potential_direction
                        position = potential_direction
                        entry_idx = i
                        entry_divergence = divergence
                        entry_price = price[i]
            
            else:
                # EXIT LOGIC
                exit_signal = 0
                
                # PRIORITY 1: FORCE EXIT ON LAST TICK (STRICT EOD SQUARE-OFF)
                if is_last_tick:
                    exit_signal = -position
                    print(f"  [EOD SQUARE-OFF] {filename}, idx {i}: Closing position {position}")
                
                # Check exit conditions after minimum hold period
                elif hold_duration >= min_hold:
                    # Stop loss: divergence increases beyond stop loss threshold
                    if abs(divergence) > stop_loss_z and np.sign(divergence) == np.sign(entry_divergence):
                        exit_signal = -position
                    
                    # Mean reversion complete: divergence converged
                    elif abs(divergence) < div_exit_thresh:
                        exit_signal = -position
                    
                    # Profit target: B4 reached target zone
                    elif (position == 1 and b4_z > b4_profit_z) or (position == -1 and b4_z < -b4_profit_z):
                        exit_signal = -position
                    
                    # Maximum hold time exceeded
                    elif hold_duration >= max_hold:
                        exit_signal = -position
                
                # Execute exit if triggered
                if exit_signal != 0:
                    signals[i] = exit_signal
                    position = 0
                    entry_idx = -1
                    entry_divergence = 0.0
                    entry_price = 0.0
                    cooldown_until = i + cooldown
        
        # Count signals for logging
        total_signals = np.count_nonzero(signals)
        long_signals = np.count_nonzero(signals == 1)
        short_signals = np.count_nonzero(signals == -1)
        
        # VERIFY EOD SQUARE-OFF: Check if positions are balanced
        if abs(long_signals - short_signals) > 1:
            print(f"  WARNING {filename}: Position imbalance detected! Long: {long_signals}, Short: {short_signals}")
        
        print(f"  Processed {filename}: {total_signals} signals (Long: {long_signals}, Short: {short_signals})")
        
        # Return result DataFrame
        return pd.DataFrame({
            'Time': df['Time'],
            'Price': df['Price'],
            'Signal': signals
        })
    
    except Exception as e:
        print(f"  ERROR processing {os.path.basename(filepath)}: {e}")
        traceback.print_exc()
        return None


def main():
    """Main entry point with command-line argument parsing."""
    global CONFIG
    
    parser = argparse.ArgumentParser(
        description='Optimized Mean Reversion Strategy - Parallel Processing'
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help='Directory containing day*.csv files'
    )
    parser.add_argument(
        '--divergence-entry',
        type=float,
        help=f'Divergence entry threshold (default: {FEATURE_CONFIG["divergence_entry_threshold"]})'
    )
    parser.add_argument(
        '--divergence-exit',
        type=float,
        help=f'Divergence exit threshold (default: {FEATURE_CONFIG["divergence_exit_threshold"]})'
    )
    parser.add_argument(
        '--min-hold',
        type=int,
        help=f'Minimum hold seconds (default: {FEATURE_CONFIG["min_hold_seconds"]})'
    )
    parser.add_argument(
        '--max-hold',
        type=int,
        help=f'Maximum hold seconds (default: {FEATURE_CONFIG["max_hold_seconds"]})'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help=f'Number of CPU workers (default: {FEATURE_CONFIG["max_cpu_workers"]})'
    )
    
    args = parser.parse_args()
    
    # Initialize config with defaults
    CONFIG = FEATURE_CONFIG.copy()
    CONFIG['input_directory'] = args.input_directory
    
    # Override with command-line arguments
    if args.divergence_entry is not None:
        CONFIG['divergence_entry_threshold'] = args.divergence_entry
    if args.divergence_exit is not None:
        CONFIG['divergence_exit_threshold'] = args.divergence_exit
    if args.min_hold is not None:
        CONFIG['min_hold_seconds'] = args.min_hold
    if args.max_hold is not None:
        CONFIG['max_hold_seconds'] = args.max_hold
    if args.workers is not None:
        CONFIG['max_cpu_workers'] = args.workers
    
    # Display configuration
    print("\n" + "="*80)
    print("OPTIMIZED MEAN REVERSION STRATEGY - PARALLEL PROCESSING")
    print("="*80)
    print(f"\nCONFIGURATION:")
    print(f"  Input directory: {CONFIG['input_directory']}")
    print(f"  Output file: {CONFIG['output_file']}")
    print(f"  CPU workers: {CONFIG['max_cpu_workers']}")
    print(f"\nFEATURE COLUMNS:")
    print(f"  B1: {CONFIG['B1_feature']}")
    print(f"  B7: {CONFIG['B7_feature']}")
    print(f"  B4 (short): {CONFIG['B4_short_tf']}")
    print(f"  B4 (long): {CONFIG['B4_long_tf']}")
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"  Divergence entry threshold: {CONFIG['divergence_entry_threshold']}")
    print(f"  Divergence exit threshold: {CONFIG['divergence_exit_threshold']}")
    print(f"  B4 momentum threshold: {CONFIG['b4_momentum_threshold']}")
    print(f"  Stop loss z-score: {CONFIG['stop_loss_zscore']}")
    print(f"  Profit target z-score: {CONFIG['b4_profit_target_zscore']}")
    print(f"  Min hold seconds: {CONFIG['min_hold_seconds']}")
    print(f"  Max hold seconds: {CONFIG['max_hold_seconds']}")
    print(f"  Cooldown seconds: {CONFIG['cooldown_after_exit']}")
    print(f"  Lookback window: {CONFIG['lookback_window']}")
    print("="*80)
    
    # Validate input directory
    if not os.path.isdir(CONFIG['input_directory']):
        print(f"\nError: Directory '{CONFIG['input_directory']}' does not exist")
        sys.exit(1)
    
    # Find and sort day files
    pattern = os.path.join(CONFIG['input_directory'], 'day*.csv')
    day_files = glob.glob(pattern)
    
    def extract_day_number(filepath):
        """Extract day number from filename for sorting."""
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
    
    print(f"\nFound {len(day_files)} day files to process")
    for f in day_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(day_files) > 5:
        print(f"  ... and {len(day_files) - 5} more")
    
    # Remove existing output file
    output_file = CONFIG['output_file']
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"\nRemoved existing {output_file}")
    
    print("\n" + "="*80)
    print("PROCESSING FILES IN PARALLEL")
    print("="*80)
    
    # Process files in parallel using ProcessPoolExecutor
    partial_process = partial(process_day_file, config=CONFIG)
    results = []
    failed_files = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['max_cpu_workers']) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(partial_process, f): f for f in day_files}
        
        # Collect results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            filepath = future_to_file[future]
            completed += 1
            
            # Progress update every 10 files
            if completed % 10 == 0:
                print(f"\n>>> Progress: {completed}/{len(day_files)} files completed")
            
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\n  CRITICAL: Exception for {os.path.basename(filepath)}: {e}")
                traceback.print_exc()
                failed_files.append(filepath)
                results.append(None)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - AGGREGATING RESULTS")
    print("="*80)
    
    # Filter successful results
    successful_results = [df for df in results if df is not None]
    success_count = len(successful_results)
    
    if failed_files:
        print(f"\nWARNING: {len(failed_files)} file(s) failed:")
        for f in failed_files[:10]:
            print(f"  - {os.path.basename(f)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    if success_count == 0:
        print("\nERROR: No files processed successfully")
        sys.exit(1)
    
    print(f"\nSuccessfully processed: {success_count}/{len(day_files)} files")
    
    # Concatenate all results
    print("Concatenating results...")
    final_df = pd.concat(successful_results, ignore_index=True)
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    
    # Calculate and display statistics
    total_rows = len(final_df)
    total_signals = (final_df['Signal'] != 0).sum()
    long_signals = (final_df['Signal'] == 1).sum()
    short_signals = (final_df['Signal'] == -1).sum()
    
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total signals: {total_signals:,}")
    print(f"  Long signals (+1): {long_signals:,}")
    print(f"  Short signals (-1): {short_signals:,}")
    print(f"  Balance: {abs(long_signals - short_signals)}")
    print(f"  Signal rate: {(total_signals/total_rows)*100:.2f}%")
    print("="*80)
    print(f"\nOutput saved to: {output_file}")
    print("Done!\n")


if __name__ == '__main__':
    main()