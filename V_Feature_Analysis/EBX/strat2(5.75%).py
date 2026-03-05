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
    'OUTPUT_FILE': 'z_mean_sigma_30min_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_mean_sigma',
    
    # Processing
    'MAX_WORKERS': 24,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,  # 30 minutes = 1800 seconds
    'SIGMA_MULTIPLIER_L': 1.6,     # k value for mean ± k*sigma
    'SIGMA_MULTIPLIER_S': 1.2,
    
    # Take Profit
    'FIRST_TP_PERCENT': 0.0065,  # 0.75% - exit 70% of position
    
    # Trailing Stop Loss Intervals
    'CHECK_INTERVAL_LONG': 45* 60,   # 45 minutes for main trailing SL
    'CHECK_INTERVAL_SHORT': 1.25 * 60,   # 1.5 minutes for aggressive trailing SL (post-TP only)
    
    # Optional: Volatility filters (set to None to disable)
    'MIN_SIGMA_THRESHOLD': None,
    'MAX_SIGMA_THRESHOLD': None,
}


def calculate_direction(df: pd.DataFrame, config: dict) -> int:
    """
    Calculate trading direction based on first 30 minutes data.
    
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
    
    # Volatility filters
    if config['MIN_SIGMA_THRESHOLD'] is not None:
        if sigma_price < config['MIN_SIGMA_THRESHOLD']:
            return 0
    
    if config['MAX_SIGMA_THRESHOLD'] is not None:
        if sigma_price > config['MAX_SIGMA_THRESHOLD']:
            return 0
    
    price_at_30min = first_30_data.iloc[-1][config['PRICE_COLUMN']]
    
    # Calculate bands
    k = config['SIGMA_MULTIPLIER_L']
    kk = config['SIGMA_MULTIPLIER_S']
    upper_band = mean_price + k * sigma_price
    lower_band = mean_price - kk * sigma_price
    
    # Determine direction
    if price_at_30min > upper_band:
        return 1  # Long
    elif price_at_30min < lower_band:
        return -1  # Short
    else:
        return 0  # No trade


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate fractional trading signals:
    1) Entry: +1 (long) or -1 (short) for 100% position
    2) First TP at 0.75%: Exit 70% → signal = -0.7 or +0.7 (remaining 30%)
    3) After TP: Dual trailing SL (45-min and 1.5-min intervals) on remaining 30%
    4) Before TP: Single trailing SL (45-min interval only) on full 100%
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
    
    # TP thresholds
    tp_percent = config['FIRST_TP_PERCENT']
    if direction == 1:  # LONG
        first_tp_price = entry_price * (1 + tp_percent)
    else:  # SHORT
        first_tp_price = entry_price * (1 - tp_percent)
    
    # Tracking variables
    position_size = 1.0  # 100% initially
    first_tp_hit = False
    first_tp_price_actual = None
    
    # Trailing checkpoints BEFORE TP (45-min only)
    checkpoint_45_time = entry_time + config['CHECK_INTERVAL_LONG']
    checkpoint_45_price = None
    
    # Trailing checkpoints AFTER TP (both 45-min and 1.5-min)
    checkpoint_45_time_post = None
    checkpoint_45_price_post = None
    checkpoint_5_time = None
    checkpoint_5_price = None
    
    # Iterate through remaining data
    remaining = df[df['Time_sec'] >= entry_time].iloc[1:]
    
    for idx in remaining.index:
        row = df.iloc[idx]
        cur_price = row[config['PRICE_COLUMN']]
        cur_time = row['Time_sec']
        
        # === PHASE 1: BEFORE FIRST TP (100% position) ===
        if not first_tp_hit:
            # Check if first TP is hit
            tp_hit_now = False
            if direction == 1:  # LONG
                if cur_price >= first_tp_price:
                    tp_hit_now = True
            else:  # SHORT
                if cur_price <= first_tp_price:
                    tp_hit_now = True
            
            if tp_hit_now:
                # Exit 70% of position, keep 30%
                signals[idx] = -0.01 * direction
                position_size = 0.99
                first_tp_hit = True
                first_tp_price_actual = cur_price
                
                # Initialize post-TP trailing checkpoints
                checkpoint_45_time_post = cur_time + config['CHECK_INTERVAL_LONG']
                checkpoint_5_time = cur_time + config['CHECK_INTERVAL_SHORT']
                checkpoint_45_price_post = None
                checkpoint_5_price = None
                continue
            
            # 45-min trailing check (before TP)
            if cur_time >= checkpoint_45_time:
                if checkpoint_45_price is not None:
                    should_exit = False
                    if direction == 1:  # LONG
                        if cur_price <= checkpoint_45_price:
                            should_exit = True
                    else:  # SHORT
                        if cur_price >= checkpoint_45_price:
                            should_exit = True
                    
                    if should_exit:
                        # Exit 100% position
                        signals[idx] = -1.0 * direction
                        position_size = 0.0
                        break
                
                # Update checkpoint
                checkpoint_45_price = cur_price
                checkpoint_45_time = cur_time + config['CHECK_INTERVAL_LONG']
        
        # === PHASE 2: AFTER FIRST TP (30% position remaining) ===
        else:
            # 45-min trailing check (post-TP)
            if cur_time >= checkpoint_45_time_post:
                if checkpoint_45_price_post is not None:
                    should_exit = False
                    if direction == 1:  # LONG
                        if cur_price <= checkpoint_45_price_post:
                            should_exit = True
                    else:  # SHORT
                        if cur_price >= checkpoint_45_price_post:
                            should_exit = True
                    
                    if should_exit:
                        # Exit remaining 30%
                        signals[idx] = -0.99 * direction
                        position_size = 0.0
                        break
                
                # Update 45-min checkpoint
                checkpoint_45_price_post = cur_price
                checkpoint_45_time_post = cur_time + config['CHECK_INTERVAL_LONG']
            
            # 1.5-min trailing check (post-TP only)
            if cur_time >= checkpoint_5_time:
                if checkpoint_5_price is not None:
                    should_exit = False
                    if direction == 1:  # LONG
                        if cur_price <= checkpoint_5_price:
                            should_exit = True
                    else:  # SHORT
                        if cur_price >= checkpoint_5_price:
                            should_exit = True
                    
                    if should_exit:
                        # Exit remaining 30%
                        signals[idx] = -0.99 * direction
                        position_size = 0.0
                        break
                
                # Update 1.5-min checkpoint
                checkpoint_5_price = cur_price
                checkpoint_5_time = cur_time + config['CHECK_INTERVAL_SHORT']
    
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
        
        # Count entries and fractional exits
        entry_signal = signals[signals != 0][0] if len(signals[signals != 0]) > 0 else 0
        direction = 'LONG' if entry_signal > 0 else ('SHORT' if entry_signal < 0 else 'FLAT')
        
        # Save result with Day column
        output_df = pd.DataFrame({
            'Day': day_num,  # Add Day column
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
    parser = argparse.ArgumentParser(description="Fractional Signal Strategy with Dual Trailing SL (70/30 Split)")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=Strategy_Config['DATA_DIR'],
        help=f"Directory containing the 'day*.parquet' files. Default: {Strategy_Config['DATA_DIR']}"
    )
    parser.add_argument(
        '--sigma-multiplier',
        type=float,
        default=Strategy_Config['SIGMA_MULTIPLIER_L'],
        help=f"Sigma multiplier (k) for bands. Default: {Strategy_Config['SIGMA_MULTIPLIER_L']}"
    )
    args = parser.parse_args()
    
    Strategy_Config['DATA_DIR'] = args.data_dir
    Strategy_Config['SIGMA_MULTIPLIER_L'] = args.sigma_multiplier
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("FRACTIONAL SIGNAL STRATEGY - DUAL TRAILING STOP LOSS (70/30 SPLIT)")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"\nStrategy Parameters:")
    print(f"  Decision Window: {config['DECISION_WINDOW']}s ({config['DECISION_WINDOW']/60:.0f} min)")
    print(f"  Sigma Multiplier (k): {config['SIGMA_MULTIPLIER_L']}")
    print(f"  First TP: {config['FIRST_TP_PERCENT']*100:.2f}% (exit 70% of position, keep 30%)")
    print(f"  Pre-TP Trailing: {config['CHECK_INTERVAL_LONG']/60:.0f} min intervals")
    print(f"  Post-TP Trailing: {config['CHECK_INTERVAL_LONG']/60:.0f} min + {config['CHECK_INTERVAL_SHORT']/60:.1f} min intervals")
    
    print(f"\nEntry Logic:")
    print(f"  Long:  Price at 30-min > mean + {config['SIGMA_MULTIPLIER_L']}*sigma → Signal = +1.0")
    print(f"  Short: Price at 30-min < mean - {config['SIGMA_MULTIPLIER_S']}*sigma → Signal = -1.0")
    
    print(f"\nExit Logic:")
    print(f"  Phase 1 (100% position):")
    print(f"    - If price hits {config['FIRST_TP_PERCENT']*100:.2f}% TP → Exit 70% (Signal = ±0.7), keep 30%")
    print(f"    - Otherwise: 45-min trailing SL checks")
    print(f"  Phase 2 (30% position after TP):")
    print(f"    - Dual trailing SL: 45-min AND 1.5-min intervals (independent)")
    print(f"    - Whichever hits first → Exit remaining 30% (Signal = ±0.3)")
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
    filtered_files = [f for f in sorted_files if not (600 <= extract_day_num(f) <= 790)]
    
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
        partial_exits_70 = final_df[final_df['Signal'].abs() == 0.7]
        partial_exits_30 = final_df[final_df['Signal'].abs() == 0.3]
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:         {len(final_df):,}")
        print(f"Entry signals:      {len(entry_signals):,} (±1.0)")
        print(f"First exits (70%):  {len(partial_exits_70):,} (±0.7)")
        print(f"Second exits (30%): {len(partial_exits_30):,} (±0.3)")
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