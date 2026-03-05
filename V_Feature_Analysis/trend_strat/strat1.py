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
import math

# CONFIGURATION
Strategy_Config = {
    # Data
    'DATA_DIR': '/data/quant14/EBX/',
    'NUM_DAYS': 510,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'kama_eama_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_kama_eama',
    
    # Processing
    'MAX_WORKERS': 30,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # KAMA Parameters
    'KAMA_PERIOD': 15,
    'KAMA_FAST': 25,
    'KAMA_SLOW': 100,
    
    # EAMA Parameters
    'EAMA_PERIOD': 15,
    'EAMA_FAST': 30,
    'EAMA_SLOW': 120,
    
    # Strategy Parameters
    'SLOPE_WINDOW': 5,
    'SLOPE_THRESHOLD': 0.0005,
    'SLOPE_LOOKBACK': 5,
    'EXIT_COOLDOWN': 5,
    
    # Day filter parameters
    'MIN_SIGMA_THRESHOLD': 0.15,
    'WARMUP_PERIOD': 1800,  # 30 minutes in seconds
    
    # Blacklisted days
    'BLACKLISTED_DAYS': {},
}


def get_super_smoother_array(prices, period):
    """Ehlers Super Smoother for EAMA calculation"""
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1

    ss = np.zeros(n)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]

    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]

    return ss


def apply_recursive_filter(series, sc):
    """Apply adaptive recursive filter for KAMA/EAMA"""
    n = len(series)
    out = np.full(n, np.nan)

    start_idx = 0
    if np.isnan(series[0]):
        valid = np.where(~np.isnan(series))[0]
        if len(valid) > 0:
            start_idx = valid[0]
        else:
            return out

    out[start_idx] = series[start_idx]

    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        val = series[i]
        if not np.isnan(val):
            out[i] = out[i-1] + c * (val - out[i-1])
        else:
            out[i] = out[i-1]

    return out


def calculate_kama(df, price_col="Price", config=None):
    """Calculate KAMA"""
    if config is None:
        config = Strategy_Config
    
    prices = df[price_col].values
    period = config['KAMA_PERIOD']
    fast = config['KAMA_FAST']
    slow = config['KAMA_SLOW']
    
    direction = df[price_col].diff(period).abs()
    volatility = df[price_col].diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0)

    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = (((er * (fast_sc - slow_sc)) + slow_sc)**2).values

    df['KAMA'] = apply_recursive_filter(prices, sc)
    return df


def calculate_eama(df, price_col="Price", config=None):
    """Calculate EAMA using Super Smoother + adaptive MA"""
    if config is None:
        config = Strategy_Config
    
    prices = df[price_col].values
    
    ss = get_super_smoother_array(prices, 30)
    df["SuperSmoother"] = ss
    
    period = config['EAMA_PERIOD']
    fast = config['EAMA_FAST']
    slow = config['EAMA_SLOW']
    
    ss_series = pd.Series(ss)
    direction = ss_series.diff(period).abs()
    volatility = ss_series.diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    
    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = ((er*(fast_sc-slow_sc))+slow_sc)**2

    df["EAMA"] = apply_recursive_filter(ss, sc)
    return df


def calculate_slope(series, window):
    """Calculate slope of a series using linear regression over a rolling window (no lookahead)"""
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def check_day_eligibility(df, config):
    """
    Check if a day is eligible for trading based on:
    1. Sigma (std dev) of first 30 minutes must be > 0.09
    
    Returns: (is_eligible, sigma_value, warmup_cutoff_idx)
    """
    warmup_cutoff = config['WARMUP_PERIOD']
    warmup_mask = df['Time_sec'] <= warmup_cutoff
    
    if warmup_mask.sum() == 0:
        return False, 0.0, 0
    
    warmup_prices = df.loc[warmup_mask, config['PRICE_COLUMN']]
    sigma = warmup_prices.std()
    
    warmup_cutoff_idx = warmup_mask.sum()
    
    is_eligible = sigma > config['MIN_SIGMA_THRESHOLD']
    
    return is_eligible, sigma, warmup_cutoff_idx


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate trading signals:
    - Long entry (1.0): EAMA > KAMA crossover + |slope_5s_ago| > 0.0001
    - Short entry (-1.0): KAMA > EAMA crossover + |slope_5s_ago| > 0.0001
    - Long exit (-1.0): KAMA > EAMA crossover (no slope condition)
    - Short exit (1.0): EAMA > KAMA crossover (no slope condition)
    - 5 second cooldown after exit before next entry
    - Only trade after 30 minute warmup period
    """
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals
    
    # Check eligibility
    is_eligible, sigma, warmup_cutoff_idx = check_day_eligibility(df, config)
    if not is_eligible:
        return signals
    
    # Calculate indicators
    df = calculate_kama(df, config=config)
    df = calculate_eama(df, config=config)
    
    # Calculate difference and slope
    df['Diff'] = df['KAMA'] - df['EAMA']
    df['Diff_Slope'] = calculate_slope(df['Diff'], config['SLOPE_WINDOW'])
    
    # Detect crossovers
    df['KAMA_Prev'] = df['KAMA'].shift(1)
    df['EAMA_Prev'] = df['EAMA'].shift(1)
    
    # Long signal: EAMA crosses above KAMA
    cross_long = (df['KAMA'] < df['EAMA']) & (df['KAMA_Prev'] >= df['EAMA_Prev'])
    # Short signal: KAMA crosses above EAMA
    cross_short = (df['KAMA'] > df['EAMA']) & (df['KAMA_Prev'] <= df['EAMA_Prev'])
    
    # Generate signals
    position = 0
    last_exit_time = -999999
    
    for i in range(warmup_cutoff_idx, len(df)):
        current_time = df['Time_sec'].iloc[i]
        
        # Get slope from SLOPE_LOOKBACK seconds ago
        lookback_idx = i - config['SLOPE_LOOKBACK']
        if lookback_idx >= 0 and not pd.isna(df['Diff_Slope'].iloc[lookback_idx]):
            slope_lookback = abs(df['Diff_Slope'].iloc[lookback_idx])
        else:
            slope_lookback = 0
        
        # Check if cooldown period has passed
        cooldown_passed = (current_time - last_exit_time) >= config['EXIT_COOLDOWN']
        
        # Entry logic
        if position == 0 and cooldown_passed:
            # Long entry: EAMA > KAMA crossover with slope condition
            if cross_long.iloc[i] and slope_lookback < config['SLOPE_THRESHOLD']:
                signals[i] = 1.0
                position = 1
            
            # Short entry: KAMA > EAMA crossover with slope condition
            elif cross_short.iloc[i] and slope_lookback > config['SLOPE_THRESHOLD']:
                signals[i] = -1.0
                position = -1
        
        # Exit logic (no slope condition required)
        elif position == 1:
            # Exit long: KAMA crosses above EAMA
            if cross_short.iloc[i]:
                signals[i] = -1.0
                position = 0
                last_exit_time = current_time
        
        elif position == -1:
            # Exit short: EAMA crosses above KAMA
            if cross_long.iloc[i]:
                signals[i] = 1.0
                position = 0
                last_exit_time = current_time
    
    # EOD square-off
    if position != 0:
        signals[-1] = -position
    
    return signals


def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, 
                      config: dict) -> dict:
    """Process a single day file and generate signals"""
    try:
        # Check if blacklisted
        if day_num in config['BLACKLISTED_DAYS']:
            return None
        
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate signals
        signals = generate_signals(df.copy(), config)
        
        # Determine direction
        entry_signals = signals[signals != 0]
        if len(entry_signals) > 0:
            first_entry = entry_signals[0]
            direction = 'LONG' if first_entry > 0 else 'SHORT'
        else:
            direction = 'FLAT'
        
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
    parser = argparse.ArgumentParser(description="KAMA-EAMA Crossover Strategy Signal Generation")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=Strategy_Config['DATA_DIR'],
        help=f"Directory containing the 'day*.parquet' files. Default: {Strategy_Config['DATA_DIR']}"
    )
    parser.add_argument(
        '--slope-threshold',
        type=float,
        default=Strategy_Config['SLOPE_THRESHOLD'],
        help=f"Absolute slope threshold for entry. Default: {Strategy_Config['SLOPE_THRESHOLD']}"
    )
    parser.add_argument(
        '--sigma-min',
        type=float,
        default=Strategy_Config['MIN_SIGMA_THRESHOLD'],
        help=f"Minimum sigma threshold for day eligibility. Default: {Strategy_Config['MIN_SIGMA_THRESHOLD']}"
    )
    args = parser.parse_args()
    
    Strategy_Config['DATA_DIR'] = args.data_dir
    Strategy_Config['SLOPE_THRESHOLD'] = args.slope_threshold
    Strategy_Config['MIN_SIGMA_THRESHOLD'] = args.sigma_min
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("KAMA-EAMA CROSSOVER STRATEGY - SIGNAL GENERATION")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"\nBlacklisted Days: {sorted(config['BLACKLISTED_DAYS'])}")
    print(f"\nDay Eligibility Filter:")
    print(f"  Min Sigma (30min): > {config['MIN_SIGMA_THRESHOLD']}")
    print(f"  Trading starts after: {config['WARMUP_PERIOD']}s (30 minutes)")
    print(f"\nIndicator Parameters:")
    print(f"  KAMA: Period={config['KAMA_PERIOD']}, Fast={config['KAMA_FAST']}, Slow={config['KAMA_SLOW']}")
    print(f"  EAMA: Period={config['EAMA_PERIOD']}, Fast={config['EAMA_FAST']}, Slow={config['EAMA_SLOW']}")
    print(f"  Slope Window: {config['SLOPE_WINDOW']} bars")
    print(f"\nEntry Logic:")
    print(f"  Long Entry:  EAMA > KAMA crossover + |slope_{config['SLOPE_LOOKBACK']}s_ago| > {config['SLOPE_THRESHOLD']} → Signal = +1.0")
    print(f"  Short Entry: KAMA > EAMA crossover + |slope_{config['SLOPE_LOOKBACK']}s_ago| > {config['SLOPE_THRESHOLD']} → Signal = -1.0")
    print(f"\nExit Logic:")
    print(f"  Long Exit:  KAMA > EAMA crossover (no slope condition) → Signal = -1.0")
    print(f"  Short Exit: EAMA > KAMA crossover (no slope condition) → Signal = +1.0")
    print(f"  Cooldown: {config['EXIT_COOLDOWN']} seconds after exit")
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
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    
    # Process all days in parallel
    print(f"\nProcessing with {config['MAX_WORKERS']} workers...")
    processed_files = []
    direction_counts = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    blacklisted_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, config): f
                for f in sorted_files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result['path'])
                        direction_counts[result['direction']] += 1
                        print(f"✓ Processed day{result['day_num']:3d} | Direction: {result['direction']:5s}")
                    else:
                        day_num = extract_day_num(file_path)
                        if day_num in config['BLACKLISTED_DAYS']:
                            blacklisted_count += 1
                            print(f"⊗ Skipped day{day_num:3d} | Blacklisted")
                        else:
                            print(f"✗ Filtered day{day_num:3d} | Low sigma or error")
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
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:             {len(final_df):,}")
        print(f"Entry/Exit signals:     {len(entry_signals):,} (±1.0)")
        print(f"\nDaily Direction Distribution:")
        print(f"  Long days:            {direction_counts['LONG']:,}")
        print(f"  Short days:           {direction_counts['SHORT']:,}")
        print(f"  Flat days:            {direction_counts['FLAT']:,}")
        print(f"  Blacklisted days:     {blacklisted_count:,}")
        print(f"  Total processed:      {len(processed_files):,}")
        print(f"  Trade rate:           {((direction_counts['LONG'] + direction_counts['SHORT']) / len(processed_files) * 100):.1f}%")
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