import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import time
import math
import gc
from multiprocessing import Pool
from typing import Tuple

# ==========================================
# CONFIGURATION
# ==========================================
Strategy_Config = {
    # Data & I/O
    'DATA_DIR': '/data/quant14/EBY/',
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'eama_kama_sma_crossover_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_eama_kama_sma_crossover',
    
    # Processing
    'MAX_WORKERS': 32,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,         # 30 minutes = 1800 seconds
    'SIGMA_THRESHOLD': 0.12,            # 0.12 σ threshold
    'RESAMPLE_PERIOD': 120,             # 120 seconds for HA candles
    'REENTRY_WAIT_CANDLES': 1,          # Wait 1 candle before re-entry
    'STOP_LOSS': 0.5,                   # Static stop loss in price points
    
    # Indicator Parameters
    'EAMA_SS_PERIOD': 25,
    'EAMA_ER_PERIOD': 15,
    'EAMA_FAST_PERIOD': 30,
    'EAMA_SLOW_PERIOD': 120,
    
    'KAMA_ER_PERIOD': 15,
    'KAMA_FAST_PERIOD': 25,
    'KAMA_SLOW_PERIOD': 100,
    
    'SMA_PERIOD': 30,
}


# ==========================================
# RESAMPLING & HEIKIN-ASHI
# ==========================================

def resample_to_candles(data_df, period_seconds=90):
    """Resample tick data to OHLC candles"""
    data_df = data_df.copy()
    
    if 'Time' in data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(data_df['Time']):
            data_df['Time'] = pd.to_timedelta(data_df['Time'])
            data_df['DateTime'] = pd.Timestamp('2024-01-01') + data_df['Time']
        else:
            data_df['DateTime'] = data_df['Time']
    
    data_df = data_df.set_index('DateTime')
    ohlc = data_df['Price'].resample(f'{period_seconds}s').ohlc()
    ohlc = ohlc.dropna()
    
    return ohlc


def calculate_heikin_ashi(ohlc_df):
    """Calculate Heikin-Ashi candles from OHLC data"""
    ha_df = pd.DataFrame(index=ohlc_df.index)
    
    ha_df['HA_Close'] = (ohlc_df['open'] + ohlc_df['high'] + 
                         ohlc_df['low'] + ohlc_df['close']) / 4
    
    ha_open = np.zeros(len(ohlc_df), dtype=np.float64)
    ha_open[0] = (ohlc_df['open'].iloc[0] + ohlc_df['close'].iloc[0]) / 2
    
    ha_close_values = ha_df['HA_Close'].values
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close_values[i-1]) / 2
    
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = pd.concat([ohlc_df['high'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).max(axis=1)
    ha_df['HA_Low'] = pd.concat([ohlc_df['low'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).min(axis=1)
    
    return ha_df


# ==========================================
# CORE MATH & INDICATOR HELPERS
# ==========================================

def apply_recursive_filter(series: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Generic recursive filter: out[i] = out[i-1] + sc[i] * (series[i] - out[i-1])"""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float32)
    
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
        else:
            return out 
            
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        prev_out = out[i-1] if not np.isnan(out[i-1]) else target
        
        if not np.isnan(target):
            out[i] = prev_out + c * (target - prev_out)
        else:
            out[i] = prev_out
            
    return out


def get_super_smoother_array(prices: np.ndarray, period: int) -> np.ndarray:
    """Returns the numpy array for SuperSmoother (Ehlers Filter)."""
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    
    ss = np.zeros(n, dtype=np.float32)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss


def calculate_eama(prices: np.ndarray, ss_period: int, er_period: int, fast: int, slow: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates EAMA and returns both EAMA and the SuperSmoother source."""
    ss = get_super_smoother_array(prices, ss_period)
    
    ss_series = pd.Series(ss)
    direction = ss_series.diff(er_period).abs()
    volatility = ss_series.diff(1).abs().rolling(er_period).sum()
    er = (direction / volatility).fillna(0).values

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2

    eama = apply_recursive_filter(ss, sc)
    return eama, ss


def calculate_kama(prices: np.ndarray, er_period: int, fast: int, slow: int) -> np.ndarray:
    """Calculates KAMA (ER on Price)."""
    price_series = pd.Series(prices)
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    sc = (((direction/volatility).fillna(0) * (2/(fast+1) - 2/(slow+1))) + (2/(slow+1)))**2
    
    kama_vals = apply_recursive_filter(prices, sc.values)
    return kama_vals


# ==========================================
# STRATEGY LOGIC
# ==========================================

def calculate_mu_sigma_condition(df: pd.DataFrame, config: dict) -> bool:
    """
    Check if volatility condition is met in the first 30 minutes.
    Returns: True if tradable (avg sigma > threshold), False otherwise.
    """
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    
    if len(first_30_data) < 10:
        return False
    
    prices = first_30_data[config['PRICE_COLUMN']]
    avg_sigma = prices.std()
    
    return avg_sigma > config['SIGMA_THRESHOLD']


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate signals based on EAMA/KAMA crossover with SMA filter and static stop loss.
    
    Strategy Rules:
    1. First 30 min: No trading, check volatility (σ > 0.12)
    2. At 30min + 1 candle: Initial entry based on EAMA vs KAMA + SMA confirmation
    3. Exit: On crossover (NO SMA confirmation) OR stop loss hit
    4. Stop Loss: 0.5 points from entry (Long: entry - 0.5, Short: entry + 0.5)
    5. Re-entry: Wait 1 candle, then require crossover + SMA confirmation
    
    Lookahead Bias Prevention:
    - All candle data (Price, SMA) shifted by 2 periods
    - EAMA/KAMA shifted by 1 period
    - Crossover = compare current vs previous indicator values
    """
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals
    
    # 1. Volatility Filter (on tick data in first 30 min)
    if not calculate_mu_sigma_condition(df.copy(), config):
        return signals
    
    # 2. Prepare data for resampling
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[config['TIME_COLUMN']]):
        df_copy[config['TIME_COLUMN']] = pd.to_timedelta(df_copy[config['TIME_COLUMN']])
        df_copy['DateTime'] = pd.Timestamp('2024-01-01') + df_copy[config['TIME_COLUMN']]
    else:
        df_copy['DateTime'] = df_copy[config['TIME_COLUMN']]
    
    # 3. Calculate indicators on TICK data
    prices = df_copy[config['PRICE_COLUMN']].values
    
    eama_tick, _ = calculate_eama(
        prices, 
        config['EAMA_SS_PERIOD'],
        config['EAMA_ER_PERIOD'],
        config['EAMA_FAST_PERIOD'],
        config['EAMA_SLOW_PERIOD']
    )
    kama_tick = calculate_kama(
        prices, 
        config['KAMA_ER_PERIOD'],
        config['KAMA_FAST_PERIOD'],
        config['KAMA_SLOW_PERIOD']
    )
    
    df_copy['EAMA'] = eama_tick
    df_copy['KAMA'] = kama_tick
    
    # 4. Resample to candles
    period_seconds = config['RESAMPLE_PERIOD']
    
    data_for_candles = df_copy[['DateTime', config['PRICE_COLUMN']]].copy()
    data_for_candles = data_for_candles.rename(columns={'DateTime': 'Time'})
    ohlc = resample_to_candles(data_for_candles, period_seconds)
    
    if len(ohlc) < 10:
        return signals
    
    # Calculate Heikin-Ashi
    ha_df = calculate_heikin_ashi(ohlc)
    
    # 5. Resample indicators to candle times (use .last() - value at candle close)
    df_indexed = df_copy.set_index('DateTime')
    
    for col in ['EAMA', 'KAMA', config['PRICE_COLUMN']]:
        resampled = df_indexed[col].resample(f'{period_seconds}s').last()
        ha_df[col] = resampled.reindex(ha_df.index).ffill()
    
    # 6. Calculate SMA on resampled price
    ha_df['SMA'] = ha_df[config['PRICE_COLUMN']].rolling(
        window=config['SMA_PERIOD'], 
        min_periods=1
    ).mean()
    
    # 7. SHIFT candle-level data by 2 periods (avoid lookahead bias)
    ha_df['Price_shifted'] = ha_df[config['PRICE_COLUMN']].shift(2)
    ha_df['SMA_shifted'] = ha_df['SMA'].shift(2)
    
    # Shift indicators by 1 period for crossover detection
    ha_df['EAMA_prev'] = ha_df['EAMA'].shift(2)
    ha_df['KAMA_prev'] = ha_df['KAMA'].shift(2)
    ha_df['EAMA'] = ha_df['EAMA'].shift(1)
    ha_df['KAMA'] = ha_df['KAMA'].shift(1)
    
    # 8. Mark 30-minute cutoff
    start_time = ha_df.index[0]
    cutoff_time = start_time + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    
    # 9. Generate signals on resampled data
    resampled_signals = np.zeros(len(ha_df), dtype=np.float32)
    position = 0.0  # 0 = flat, 1 = long, -1 = short
    exit_candle_idx = -999  # Track when we exited (for re-entry wait)
    entry_price = 0.0  # Track entry price for stop loss
    stop_loss_level = 0.0  # Track stop loss level
    
    eama_vals = ha_df['EAMA'].values
    kama_vals = ha_df['KAMA'].values
    eama_prev_vals = ha_df['EAMA_prev'].values
    kama_prev_vals = ha_df['KAMA_prev'].values
    price_vals = ha_df['Price_shifted'].values
    sma_vals = ha_df['SMA_shifted'].values
    
    cutoff_idx = ha_df.index.get_indexer([cutoff_time], method='nearest')[0]
    first_tradable_idx = cutoff_idx + 1
    
    for i in range(first_tradable_idx, len(ha_df)):
        # Skip if any data is NaN
        if (np.isnan(eama_vals[i]) or np.isnan(kama_vals[i]) or 
            np.isnan(eama_prev_vals[i]) or np.isnan(kama_prev_vals[i]) or 
            np.isnan(sma_vals[i]) or np.isnan(price_vals[i])):
            continue
        
        eama_cur = eama_vals[i]
        kama_cur = kama_vals[i]
        eama_prev = eama_prev_vals[i]
        kama_prev = kama_prev_vals[i]
        price_cur = price_vals[i]
        sma_cur = sma_vals[i]
        
        # Detect crossovers
        bullish_crossover = (eama_prev <= kama_prev) and (eama_cur > kama_cur)
        bearish_crossover = (eama_prev >= kama_prev) and (eama_cur < kama_cur)
        
        # ============================================
        # STOP LOSS CHECK (Priority: Check first)
        # ============================================
        if position == 1.0:  # Long position
            if price_cur <= stop_loss_level:
                resampled_signals[i] = -1.0  # Exit long on stop loss
                position = 0.0
                exit_candle_idx = i
                entry_price = 0.0
                stop_loss_level = 0.0
                continue
        
        elif position == -1.0:  # Short position
            if price_cur >= stop_loss_level:
                resampled_signals[i] = +1.0  # Exit short on stop loss
                position = 0.0
                exit_candle_idx = i
                entry_price = 0.0
                stop_loss_level = 0.0
                continue
        
        # ============================================
        # EXIT LOGIC (Crossover - No SMA confirmation needed)
        # ============================================
        if position == 1.0 and bearish_crossover:
            resampled_signals[i] = -1.0  # Exit long
            position = 0.0
            exit_candle_idx = i
            entry_price = 0.0
            stop_loss_level = 0.0
            continue
            
        elif position == -1.0 and bullish_crossover:
            resampled_signals[i] = +1.0  # Exit short
            position = 0.0
            exit_candle_idx = i
            entry_price = 0.0
            stop_loss_level = 0.0
            continue
        
        # ============================================
        # ENTRY LOGIC
        # ============================================
        if position == 0.0:
            # Case 1: Initial entry at first candle after 30 min
            if i == first_tradable_idx:
                # Long: EAMA > KAMA AND Price > SMA
                if eama_cur > kama_cur and price_cur > sma_cur:
                    resampled_signals[i] = +1.0
                    position = 1.0
                    entry_price = price_cur
                    stop_loss_level = entry_price - config['STOP_LOSS']
                
                # Short: EAMA < KAMA AND Price < SMA
                elif eama_cur < kama_cur and price_cur < sma_cur:
                    resampled_signals[i] = -1.0
                    position = -1.0
                    entry_price = price_cur
                    stop_loss_level = entry_price + config['STOP_LOSS']
            
            # Case 2: Re-entry after exit (must wait 1 candle)
            else:
                # Check if we've waited long enough after last exit
                candles_since_exit = i - exit_candle_idx
                
                if candles_since_exit >= config['REENTRY_WAIT_CANDLES']:
                    # Long entry: Bullish crossover + Price > SMA
                    if bullish_crossover and price_cur > sma_cur:
                        resampled_signals[i] = +1.0
                        position = 1.0
                        entry_price = price_cur
                        stop_loss_level = entry_price - config['STOP_LOSS']
                    
                    # Short entry: Bearish crossover + Price < SMA
                    elif bearish_crossover and price_cur < sma_cur:
                        resampled_signals[i] = -1.0
                        position = -1.0
                        entry_price = price_cur
                        stop_loss_level = entry_price + config['STOP_LOSS']
    
    # EOD SQUARE-OFF
    if position != 0.0:
        resampled_signals[-1] = -position
    
    # 10. Map signals back to original tick data
    ha_df['Signal'] = resampled_signals
    signal_series = ha_df['Signal']
    
    df_indexed['Signal'] = signal_series.reindex(df_indexed.index).ffill().fillna(0)
    
    return df_indexed['Signal'].values


# ==========================================
# FILE/EXECUTION UTILITIES
# ==========================================

def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, 
                        config: dict) -> dict:
    """Process a single day file and generate signals"""
    
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    
    try:
        # GPU loading
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()  # Convert to CPU
        del ddf, gdf
        gc.collect()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate signals
        signals = generate_signals(df.copy(), config)
        
        # Determine direction
        entry_signal = signals[signals.astype(bool)][0] if len(signals[signals.astype(bool)]) > 0 else 0
        direction = 'LONG' if entry_signal > 0 else ('SHORT' if entry_signal < 0 else 'FLAT')
        
        # Save result
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
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("EAMA/KAMA CROSSOVER STRATEGY - WITH 0.5 STATIC STOP LOSS")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"\nStrategy Parameters:")
    print(f"  Vol Filter Window: {config['DECISION_WINDOW']}s ({config['DECISION_WINDOW']/60:.0f} min)")
    print(f"  Avg Sigma Threshold: {config['SIGMA_THRESHOLD']}")
    print(f"  Resample Period: {config['RESAMPLE_PERIOD']}s (HA candles)")
    print(f"  SMA Period: {config['SMA_PERIOD']} (on resampled data)")
    print(f"  Re-entry Wait: {config['REENTRY_WAIT_CANDLES']} candle(s) = {config['REENTRY_WAIT_CANDLES'] * config['RESAMPLE_PERIOD']}s")
    print(f"  Stop Loss: {config['STOP_LOSS']} points")
    print(f"\nStrategy Rules:")
    print(f"  1. First 30 min: Observe sigma, no trades")
    print(f"  2. At 30min + 1 candle: Initial entry (EAMA vs KAMA + SMA filter)")
    print(f"  3. Exit: On crossover OR stop loss hit")
    print(f"  4. Stop Loss: Long = Entry - {config['STOP_LOSS']}, Short = Entry + {config['STOP_LOSS']}")
    print(f"  5. Re-entry: Wait {config['REENTRY_WAIT_CANDLES']} candle(s), then check crossover + SMA")
    print(f"\nLookahead Bias Prevention:")
    print(f"  - Candle data (Price, SMA) SHIFTED by 2 periods")
    print(f"  - EAMA/KAMA SHIFTED by 1 period")
    print(f"\nIndicator Parameters:")
    print(f"  EAMA: SS={config['EAMA_SS_PERIOD']}, ER={config['EAMA_ER_PERIOD']}, Fast={config['EAMA_FAST_PERIOD']}, Slow={config['EAMA_SLOW_PERIOD']}")
    print(f"  KAMA: ER={config['KAMA_ER_PERIOD']}, Fast={config['KAMA_FAST_PERIOD']}, Slow={config['KAMA_SLOW_PERIOD']}")
    print("="*80)
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created temp directory: {temp_dir_path}")
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    filtered_files = sorted_files
    
    if not filtered_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(filtered_files)} day files to process")
    
    pool_args = [
        (f, extract_day_num(f), temp_dir_path, config) 
        for f in filtered_files
    ]
    
    print(f"\nProcessing with {config['MAX_WORKERS']} workers using multiprocessing.Pool...")
    processed_files = []
    direction_counts = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    
    try:
        with Pool(processes=config['MAX_WORKERS']) as pool:
            results = pool.starmap(process_single_day, pool_args)
        
        for result in results:
            if result:
                processed_files.append(result['path'])
                direction_counts[result['direction']] += 1
                print(f"✓ Processed day {result['day_num']}: {result['direction']}")

        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
        
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)
        
        print(f"\n✓ Created combined output: {output_path}")
        
        final_df = pd.read_csv(output_path)
        entry_signals = final_df[final_df['Signal'].abs() == 1.0]
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:        {len(final_df):,}")
        print(f"Entry signals:     {len(entry_signals):,} (±1.0)")
        print(f"\nDaily Direction Distribution:")
        print(f"  Long days:       {direction_counts['LONG']:,}")
        print(f"  Short days:      {direction_counts['SHORT']:,}")
        print(f"  Flat days:       {direction_counts['FLAT']:,}")
        print(f"  Total trading days: {direction_counts['LONG'] + direction_counts['SHORT']:,}")
        print(f"  Total days:      {len(processed_files):,}")
        if len(processed_files) > 0:
            print(f"  Trade rate:      {((direction_counts['LONG'] + direction_counts['SHORT']) / len(processed_files) * 100):.1f}%")
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