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
import warnings
import gc
warnings.filterwarnings('ignore')

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    import xgboost as xgb
    GPU_AVAILABLE = xgb.XGBClassifier().get_params()['device'] != 'cpu'
    print("GPU acceleration enabled for XGBoost")
except Exception:
    print("GPU not available for XGBoost - using CPU mode")

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

#############################################################################
# CONFIGURATION - Modify these defaults as needed
#############################################################################

FEATURE_CONFIG = {
    # Strategy parameters
    'lookback_period': 500,              # Seconds for feature warm-up
    'lookahead_period': 1,               # Seconds to look ahead for target
    'price_threshold': 0.001,            # Price change threshold for signal
    'min_hold_seconds': 15,              # Minimum holding period
    
    # XGBoost parameters
    'n_estimators': 120,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    
    # Output configuration
    'output_file': 'trading_signals.csv',
    
    # Execution parameters - NEW: Separate CPU and GPU workers
    'max_cpu_workers': 1,                # Workers for CPU mode
    'max_gpu_workers': 32,                # Workers for GPU mode (lower to save VRAM)
    'gpu_memory_fraction': 0.4,          # Fraction of GPU memory per worker
}

# This will be populated by main() with command-line overrides
CONFIG = {}

#############################################################################


def time_to_seconds(time_str):
    """Convert HH:MM:SS to total seconds."""
    parts = str(time_str).split(':')
    if len(parts) == 3:
        try:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return 0
    return 0


def enforce_min_hold_strict(predictions, min_hold, max_trades):
    """
    Strictly enforce minimum holding period with proper entry/exit pairing.
    
    STRICT RULES:
    1. Maximum trades per day = max_trades (calculated from day length / min_hold)
    2. Each entry MUST have a corresponding exit (equal long and short signals)
    3. ALL positions MUST be closed by end of day
    4. Minimum hold period of min_hold seconds enforced
    5. No new position until previous is exited
    
    Args:
        predictions: Model predictions (0=hold, 1=long, -1=short)
        min_hold: Minimum seconds to hold position
        max_trades: Maximum number of complete trades allowed
        
    Returns:
        signals: Entry/exit signals where:
                 +1 = enter long
                 -1 = exit long (can also be interpreted as short, but we only do longs)
                 0 = no action
    """
    n = len(predictions)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0  # 0: flat, 1: long
    entry_idx = -1
    trade_count = 0
    
    for i in range(n):
        pred = predictions[i]
        is_last = (i == n - 1)
        
        # Calculate hold duration
        hold_duration = (i - entry_idx) if entry_idx >= 0 else 0
        
        # If we're flat, look for entry signal
        if position == 0:
            # Check if we can still enter new trades
            if trade_count >= max_trades:
                continue  # No more trades allowed
            
            # Don't enter if we can't meet min_hold before end of day
            if i + min_hold >= n:
                continue
            
            if pred == 1:  # Want to go long
                signals[i] = 1  # Enter long
                position = 1
                entry_idx = i
        
        # If we have a position, look for exit signal
        elif position == 1:
            # Force exit on last tick if still in position
            if is_last:
                signals[i] = -1  # Exit long
                position = 0
                entry_idx = -1
                trade_count += 1
            # Check for exit signal with min hold satisfied
            elif hold_duration >= min_hold:
                # Exit long: model says hold/short OR we need to exit
                if pred != 1:
                    signals[i] = -1  # Exit long
                    position = 0
                    entry_idx = -1
                    trade_count += 1
            # Force exit if approaching end of day and can't hold longer
            elif i + (min_hold - hold_duration) >= n:
                # We can't hold until min_hold and stay within day
                # Exit now at min_hold
                if hold_duration >= min_hold:
                    signals[i] = -1  # Exit long
                    position = 0
                    entry_idx = -1
                    trade_count += 1
    
    # CRITICAL: Ensure all positions are closed
    # If we still have an open position, force close on last available tick
    if position != 0:
        # Find last non-zero signal (should be entry)
        last_entry = np.where(signals != 0)[0]
        if len(last_entry) > 0:
            entry_pos = last_entry[-1]
            # Force exit at the end
            signals[n-1] = -1
            trade_count += 1
    
    # CRITICAL: Ensure equal long and short signals
    long_count = np.sum(signals == 1)
    short_count = np.sum(signals == -1)
    
    if long_count != short_count:
        # Remove excess entries if we have more longs than shorts
        if long_count > short_count:
            # Find and remove the last unpaired entry
            long_indices = np.where(signals == 1)[0]
            for idx in reversed(long_indices):
                if long_count <= short_count:
                    break
                signals[idx] = 0
                long_count -= 1
    
    return signals

def validate_signals(signals, min_hold):
    """
    Validate that signals follow proper entry/exit rules.
    Returns (is_valid, error_message, stats)
    """
    long_entries = np.sum(signals == 1)
    short_entries = np.sum(signals == -1)
    
    # Track position to validate pairing
    position = 0
    entry_idx = -1
    violations = []
    
    for i in range(len(signals)):
        if signals[i] == 0:
            continue
            
        if position == 0:
            # Should be entry
            if signals[i] == 1:
                position = 1
                entry_idx = i
            elif signals[i] == -1:
                position = -1
                entry_idx = i
        else:
            # Should be exit
            hold_duration = i - entry_idx
            
            if position == 1 and signals[i] == -1:
                if hold_duration < min_hold:
                    violations.append(f"Long exit at {i} held only {hold_duration}s")
                position = 0
                entry_idx = -1
            elif position == -1 and signals[i] == 1:
                if hold_duration < min_hold:
                    violations.append(f"Short exit at {i} held only {hold_duration}s")
                position = 0
                entry_idx = -1
            else:
                violations.append(f"Invalid signal at {i}: position={position}, signal={signals[i]}")
    
    # Check if position is closed at end (last signal should close position)
    if position != 0:
        violations.append(f"Position not closed at end of day: {position}")
    
    stats = {
        'long_entries': long_entries,
        'short_entries': short_entries,
        'total_signals': long_entries + short_entries,
        'violations': len(violations)
    }
    
    is_valid = len(violations) == 0
    error_msg = "\n    ".join(violations) if violations else "All validations passed"
    
    return is_valid, error_msg, stats


def process_day_file(filepath, config):
    """
    Process a single day file with XGBoost.
    Each day trains its own model independently.
    Returns a pandas DataFrame on success, None on failure.
    
    Memory optimizations:
    - Explicit garbage collection
    - Delete intermediate variables
    - Use float32 instead of float64
    """
    try:
        print(f"\nProcessing {os.path.basename(filepath)}...")
        
        # Load data
        df = pd.read_csv(filepath)
        
        if len(df) == 0:
            print(f"  Warning: Empty file {filepath}, skipping.")
            return None
        
        # Validate required columns
        if 'Time' not in df.columns or 'Price' not in df.columns:
            print(f"  Error: Missing Time or Price column in {filepath}, skipping.")
            return None
        
        # Convert time to seconds
        df['time_numeric'] = df['Time'].apply(time_to_seconds)
        
        # Feature columns (exclude Time and Price)
        feature_cols = [col for col in df.columns 
                       if col not in ['Time', 'Price', 'time_numeric']]
        
        if len(feature_cols) == 0:
            print(f"  Error: No feature columns found in {filepath}, skipping.")
            return None
        
        print(f"  Found {len(feature_cols)} features, {len(df):,} rows")
        print(f"  Max possible trades (with {config['min_hold_seconds']}s hold): {len(df) // config['min_hold_seconds']}")
        
        # Create target variable
        df['price_ahead'] = df['Price'].shift(-config['lookahead_period'])
        df['price_change'] = df['price_ahead'] - df['Price']
        df['target'] = 0
        
        threshold = config['price_threshold']
        df.loc[df['price_change'] > threshold, 'target'] = 1   # Long
        df.loc[df['price_change'] < -threshold, 'target'] = 2  # Short
        
        # Training data (exclude NaN targets)
        df_train = df.dropna(subset=['target']).copy()
        
        if len(df_train) == 0:
            print(f"  Warning: No valid training data in {filepath}, skipping.")
            return None
        
        # Check class distribution
        target_counts = df_train['target'].value_counts()
        if len(target_counts) < 2:
            print(f"  Warning: Insufficient class diversity in {filepath}, skipping.")
            return None
        
        print(f"  Training samples: {len(df_train):,}")
        print(f"    Hold: {(df_train['target']==0).sum():,}")
        print(f"    Long: {(df_train['target']==1).sum():,}")
        print(f"    Short: {(df_train['target']==2).sum():,}")
        
        # Prepare features - USE FLOAT32 to save memory
        X_train = df_train[feature_cols].fillna(0).astype(np.float32).values
        y_train = df_train['target'].values.astype(np.int32)
        
        # Delete df_train to free memory
        del df_train
        gc.collect()
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        
        # Delete original X_train
        del X_train
        gc.collect()
        
        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([class_weights[int(y)] for y in y_train], dtype=np.float32)
        
        # Train XGBoost model for this day
        print(f"  Training XGBoost model...")
        
        device = 'cuda' if GPU_AVAILABLE else 'cpu'
        
        # GPU memory management parameters
        xgb_params = {
            'n_estimators': config['n_estimators'],
            'max_depth': config['max_depth'],
            'learning_rate': config['learning_rate'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'device': device,
            'tree_method': 'hist' if device == 'cpu' else 'hist',
        }
        
        # Add GPU-specific memory optimization
        if device == 'cuda':
            xgb_params['max_bin'] = 256  # Reduce from default to save memory
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Delete training data
        del X_train_scaled, y_train, sample_weights
        gc.collect()
        
        # Generate signals for all data
        print(f"  Generating predictions...")
        X_all = df[feature_cols].fillna(0).astype(np.float32).values
        X_all_scaled = scaler.transform(X_all).astype(np.float32)
        
        # Delete X_all
        del X_all
        gc.collect()
        
        # Find lookback end index
        start_time = df['time_numeric'].min()
        lookback_end_idx = (df['time_numeric'] - start_time >= config['lookback_period']).idxmax()
        
        # Initialize predictions
        predictions = np.zeros(len(df), dtype=np.int8)
        
        # Predict only after lookback period
        if lookback_end_idx < len(df):
            raw_predictions = model.predict(X_all_scaled[lookback_end_idx:])
            predictions[lookback_end_idx:] = raw_predictions
        
        # Delete model and scaled data
        del model, X_all_scaled
        gc.collect()
        
        # Map predictions: 0->0 (hold), 1->1 (long), 2->-1 (short)
        predictions = np.where(predictions == 2, -1, predictions)
        
        # Enforce minimum holding period with strict entry/exit pairing
        print(f"  Enforcing strict {config['min_hold_seconds']}s minimum hold with proper entry/exit...")
        final_signals = enforce_min_hold_strict(predictions, config['min_hold_seconds'])
        
        # Validate signals
        is_valid, error_msg, stats = validate_signals(final_signals, config['min_hold_seconds'])
        
        print(f"  Signal validation: {'PASSED' if is_valid else 'FAILED'}")
        if not is_valid:
            print(f"    Violations:\n    {error_msg}")
        
        print(f"  Generated signals:")
        print(f"    Long entries (+1): {stats['long_entries']}")
        print(f"    Short entries/Long exits (-1): {stats['short_entries']}")
        print(f"    Total signals: {stats['total_signals']}")
        print(f"    Entry/Exit balance: {'+1 and -1 counts EQUAL ✓' if stats['long_entries'] == stats['short_entries'] else 'UNBALANCED ✗'}")
        
        # Create output
        output_df = pd.DataFrame({
            'Time': df['Time'],
            'Price': df['Price'],
            'Signal': final_signals
        })
        
        # Clean up
        del df, predictions, final_signals
        gc.collect()
        
        return output_df
        
    except Exception as e:
        print(f"  CRITICAL Error processing {filepath}: {e}")
        traceback.print_exc()
        # Force garbage collection on error
        gc.collect()
        return None


def main():
    """Main entry point."""
    global CONFIG
    
    parser = argparse.ArgumentParser(
        description='XGBoost Day-wise Trading Signal Generator with Strict Entry/Exit Pairing'
    )
    parser.add_argument(
        'input_directory',
        type=str,
        help='Directory containing day*.csv files'
    )
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of parallel workers (default: 2 for GPU, 4 for CPU)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        help=f'Lookback period in seconds (default: {FEATURE_CONFIG["lookback_period"]})'
    )
    parser.add_argument(
        '--lookahead',
        type=int,
        help=f'Lookahead period in seconds (default: {FEATURE_CONFIG["lookahead_period"]})'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help=f'Price threshold for signals (default: {FEATURE_CONFIG["price_threshold"]})'
    )
    parser.add_argument(
        '--min-hold',
        type=int,
        help=f'Minimum holding period in seconds (default: {FEATURE_CONFIG["min_hold_seconds"]})'
    )
    
    args = parser.parse_args()
    
    # Start with feature config defaults
    CONFIG = FEATURE_CONFIG.copy()
    CONFIG['input_directory'] = args.input_directory
    
    # Override with command-line arguments if provided
    if args.lookback is not None:
        CONFIG['lookback_period'] = args.lookback
    if args.lookahead is not None:
        CONFIG['lookahead_period'] = args.lookahead
    if args.threshold is not None:
        CONFIG['price_threshold'] = args.threshold
    if args.min_hold is not None:
        CONFIG['min_hold_seconds'] = args.min_hold
    
    # Set worker count based on mode and user input
    if args.workers is not None:
        max_workers = args.workers
    else:
        max_workers = CONFIG['max_gpu_workers'] if GPU_AVAILABLE else CONFIG['max_cpu_workers']
    
    # Display configuration
    print("=" * 80)
    print("XGBOOST DAY-WISE TRADING SIGNAL GENERATOR")
    print("=" * 80)
    print(f"\nCONFIGURATION:")
    print(f"  Acceleration mode: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")
    print(f"  Parallel workers: {max_workers}")
    if GPU_AVAILABLE:
        print(f"  GPU memory optimization: ENABLED")
        print(f"    - Using float32 precision")
        print(f"    - Aggressive garbage collection")
        print(f"    - Reduced max_bin (256)")
    print(f"  Input directory: {CONFIG['input_directory']}")
    print(f"  Output file: {CONFIG['output_file']}")
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"  Lookback period: {CONFIG['lookback_period']} seconds")
    print(f"  Lookahead period: {CONFIG['lookahead_period']} seconds")
    print(f"  Price threshold: {CONFIG['price_threshold']}")
    print(f"  Min hold seconds: {CONFIG['min_hold_seconds']} (STRICTLY ENFORCED)")
    print(f"\nSIGNAL RULES:")
    print(f"  +1 = Enter long position")
    print(f"  -1 = Enter short OR exit long position")
    print(f"  Each entry MUST have corresponding exit")
    print(f"  All positions closed at end of day")
    print(f"  Long entries and exits must be EQUAL (or differ by 1 at EOD)")
    print(f"\nXGBOOST PARAMETERS:")
    print(f"  n_estimators: {CONFIG['n_estimators']}")
    print(f"  max_depth: {CONFIG['max_depth']}")
    print(f"  learning_rate: {CONFIG['learning_rate']}")
    print(f"\nMEMORY MANAGEMENT:")
    print(f"  Each worker processes 1 file at a time")
    print(f"  With {max_workers} workers and 20GB VRAM, ~{20//max_workers}GB per worker")
    print(f"  Tip: If OOM errors persist, reduce --workers value")
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
    
    # Use partial to pre-load the 'config' argument
    partial_process_func = partial(process_day_file, config=CONFIG)
    
    results = []
    failed_files = []
    
    # Process files concurrently with controlled worker count
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
    
    # Concatenate all results
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
    print(f"  Long entries (+1): {long_signals:,}")
    print(f"  Short entries/Long exits (-1): {short_signals:,}")
    print(f"  Entry/Exit Balance: {'BALANCED ✓' if long_signals == short_signals else f'UNBALANCED ✗ (diff: {abs(long_signals - short_signals)})'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
