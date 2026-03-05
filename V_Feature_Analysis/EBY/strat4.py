import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# CONFIGURATION
Strategy_Config = {
    # Data
    'DATA_DIR': '/data/quant14/EBX/',
    'NUM_DAYS': 510,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'mean_sigma_threshold_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_mean_sigma_threshold',
    
    # Processing
    'MAX_WORKERS': 30,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 15 * 60,  # 30 minutes = 1800 seconds
    'MEAN_THRESHOLD': 100.0,     # μ threshold
    'SIGMA_MIN': 0.1,           # σ minimum threshold
    'SIGMA_MAX': 0.2,           # σ maximum threshold
    
    # Take Profit (absolute dollar amounts)
    'TP_LONG': 0.95,   # $0.80 for long
    'TP_SHORT': 0.85,  # $0.75 for short
    
    # Trailing Stop Loss Interval (post-TP only)
    'TRAILING_SL_INTERVAL': 1.5 * 60,  # 2 minutes
}


def calculate_direction(df: pd.DataFrame, config: dict) -> int:
    """
    Calculate trading direction based on first 30 minutes data.
    
    Entry Rules:
    - Long:  μ > 100 AND 0.08 < σ < 0.12
    - Short: μ < 100 AND 0.08 < σ < 0.12
    
    Returns:
        1 = Long, -1 = Short, 0 = No trade
    """
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    
    if len(first_30_data) < 10:
        return 0
    
    prices = first_30_data[config['PRICE_COLUMN']]
    mean_price = prices.mean()
    sigma_price = prices.std()
    
    mean_threshold = config['MEAN_THRESHOLD']
    sigma_min = config['SIGMA_MIN']
    sigma_max = config['SIGMA_MAX']
    
    # Entry conditions - sigma must be between min and max
    if sigma_min < sigma_price < sigma_max:
        if mean_price > mean_threshold:
            return 1  # Long
        elif mean_price < mean_threshold:
            return -1  # Short
    
    return 0  # No trade


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate fractional trading signals:
    1) Entry: +1 (long) or -1 (short) for 100% position
    2) TP: Exit 50% at $0.80 (long) or $0.75 (short) → signal = ±0.5
    3) After TP: 2-minute trailing SL on remaining 50%
    4) EOD: Square off any remaining position
    """
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals
    
    direction = calculate_direction(df, config)
    if direction == 0:
        return signals
    
    # Entry
    decision_window = config['DECISION_WINDOW']
    entry_idx_candidates = df[df['Time_sec'] >= decision_window]
    if entry_idx_candidates.empty:
        return signals
    
    entry_idx = entry_idx_candidates.index[0]
    signals[entry_idx] = direction  # +1 or -1
    entry_price = df.iloc[entry_idx][config['PRICE_COLUMN']]
    entry_time = df.iloc[entry_idx]['Time_sec']
    
    # TP thresholds (absolute dollar amounts)
    if direction == 1:  # LONG
        tp_price = entry_price + config['TP_LONG']
    else:  # SHORT
        tp_price = entry_price - config['TP_SHORT']
    
    # Tracking variables
    position_size = 1.0  # 100% initially
    tp_hit = False
    
    # Trailing SL checkpoint (only used after TP)
    checkpoint_time = None
    checkpoint_price = None
    
    # Iterate through remaining data
    remaining = df[df['Time_sec'] >= entry_time].iloc[1:]
    
    for idx in remaining.index:
        row = df.iloc[idx]
        cur_price = row[config['PRICE_COLUMN']]
        cur_time = row['Time_sec']
        
        # === BEFORE TP ===
        if not tp_hit:
            # Check if TP is hit
            tp_hit_now = False
            if direction == 1:  # LONG
                if cur_price >= tp_price:
                    tp_hit_now = True
            else:  # SHORT
                if cur_price <= tp_price:
                    tp_hit_now = True
            
            if tp_hit_now:
                # Exit 50% of position, keep 50%
                signals[idx] = -0.05 * direction
                position_size = 0.95
                tp_hit = True
                
                # Initialize trailing SL checkpoint
                checkpoint_time = cur_time + config['TRAILING_SL_INTERVAL']
                checkpoint_price = None
                continue
        
        # === AFTER TP ===
        else:
            # 2-minute trailing SL check
            if cur_time >= checkpoint_time:
                if checkpoint_price is not None:
                    should_exit = False
                    if direction == 1:  # LONG
                        if cur_price <= checkpoint_price:
                            should_exit = True
                    else:  # SHORT
                        if cur_price >= checkpoint_price:
                            should_exit = True
                    
                    if should_exit:
                        # Exit remaining 50%
                        signals[idx] = -0.95 * direction
                        position_size = 0.0
                        break
                
                # Update checkpoint
                checkpoint_price = cur_price
                checkpoint_time = cur_time + config['TRAILING_SL_INTERVAL']
    
    # === EOD SQUARE-OFF ===
    if position_size > 0:
        signals[-1] = -position_size * direction
    
    return signals


def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, 
                      config: dict) -> dict:
    """Process a single day file and generate signals"""
    try:
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate signals
        signals = generate_signals(df.copy(), config)
        
        # Determine direction
        entry_signal = signals[signals != 0][0] if len(signals[signals != 0]) > 0 else 0
        direction = 'LONG' if entry_signal > 0 else ('SHORT' if entry_signal < 0 else 'FLAT')
        
        # Save result with Day column
        output_df = pd.DataFrame({
            'Day': day_num,
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'direction': direction
        }
    
    except Exception as e:
        print(f"Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Mean-Sigma Threshold Strategy (Sigma Range Filter)")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=Strategy_Config['DATA_DIR'],
        help=f"Directory containing the 'day*.parquet' files. Default: {Strategy_Config['DATA_DIR']}"
    )
    parser.add_argument(
        '--mean-threshold',
        type=float,
        default=Strategy_Config['MEAN_THRESHOLD'],
        help=f"Mean threshold for entry. Default: {Strategy_Config['MEAN_THRESHOLD']}"
    )
    parser.add_argument(
        '--sigma-min',
        type=float,
        default=Strategy_Config['SIGMA_MIN'],
        help=f"Minimum sigma threshold for entry. Default: {Strategy_Config['SIGMA_MIN']}"
    )
    parser.add_argument(
        '--sigma-max',
        type=float,
        default=Strategy_Config['SIGMA_MAX'],
        help=f"Maximum sigma threshold for entry. Default: {Strategy_Config['SIGMA_MAX']}"
    )
    args = parser.parse_args()
    
    Strategy_Config['DATA_DIR'] = args.data_dir
    Strategy_Config['MEAN_THRESHOLD'] = args.mean_threshold
    Strategy_Config['SIGMA_MIN'] = args.sigma_min
    Strategy_Config['SIGMA_MAX'] = args.sigma_max
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("MEAN-SIGMA THRESHOLD STRATEGY (SIGMA RANGE FILTER)")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"\nStrategy Parameters:")
    print(f"  Decision Window: {config['DECISION_WINDOW']}s ({config['DECISION_WINDOW']/60:.0f} min)")
    print(f"  Mean Threshold (μ): {config['MEAN_THRESHOLD']}")
    print(f"  Sigma Range (σ): {config['SIGMA_MIN']} < σ < {config['SIGMA_MAX']}")
    print(f"  Long TP: ${config['TP_LONG']} (exit 50%, keep 50%)")
    print(f"  Short TP: ${config['TP_SHORT']} (exit 50%, keep 50%)")
    print(f"  Post-TP Trailing SL: {config['TRAILING_SL_INTERVAL']/60:.0f} min intervals")
    
    print(f"\nEntry Logic (First 30 minutes):")
    print(f"  Long:  μ > {config['MEAN_THRESHOLD']} AND {config['SIGMA_MIN']} < σ < {config['SIGMA_MAX']} → Signal = +1.0")
    print(f"  Short: μ < {config['MEAN_THRESHOLD']} AND {config['SIGMA_MIN']} < σ < {config['SIGMA_MAX']} → Signal = -1.0")
    
    print(f"\nExit Logic:")
    print(f"  Phase 1 (100% position):")
    print(f"    - Long: If price hits entry + ${config['TP_LONG']} → Exit 50% (Signal = -0.5), keep 50%")
    print(f"    - Short: If price hits entry - ${config['TP_SHORT']} → Exit 50% (Signal = +0.5), keep 50%")
    print(f"  Phase 2 (50% position after TP):")
    print(f"    - 2-minute trailing SL → Exit remaining 50% (Signal = ±0.5)")
    print(f"  EOD: Square off any remaining position")
    print("="*80)
    
    # Setup temp directory
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    print(f"Created temp directory: {temp_dir_path}")
    
    # Find input files
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    filtered_files = [f for f in sorted_files]
    
    if not filtered_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(filtered_files)} day files to process")
    
    # Process all days in parallel
    print(f"\nProcessing with {config['MAX_WORKERS']} workers...")
    processed_files = []
    direction_counts = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    
    try:
        with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, config): f
                for f in filtered_files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result['path'])
                        direction_counts[result['direction']] += 1
                        print(f"✓ Processed day{result['day_num']:3d} | Direction: {result['direction']:5s}")
                except Exception as e:
                    print(f"✗ Error in {file_path}: {e}")
        
        # Combine results
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
        
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()  # Skip header
                        shutil.copyfileobj(infile, outfile)
        
        print(f"\n✓ Created combined output: {output_path}")
        
        # Statistics
        final_df = pd.read_csv(output_path)
        entry_signals = final_df[final_df['Signal'].abs() == 1.0]
        partial_exits_50 = final_df[final_df['Signal'].abs() == 0.5]
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:         {len(final_df):,}")
        print(f"Entry signals:      {len(entry_signals):,} (±1.0)")
        print(f"Exits (50%):        {len(partial_exits_50):,} (±0.5)")
        print(f"\nDaily Direction Distribution:")
        print(f"  Long days:        {direction_counts['LONG']:,}")
        print(f"  Short days:       {direction_counts['SHORT']:,}")
        print(f"  Flat days:        {direction_counts['FLAT']:,}")
        print(f"  Total days:       {len(processed_files):,}")
        print(f"  Trade rate:       {((direction_counts['LONG'] + direction_counts['SHORT']) / len(processed_files) * 100):.1f}%")
        print(f"{'='*80}")
        
    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print(f"\nCleaned up temp directory")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"✓ Output saved to: {output_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()