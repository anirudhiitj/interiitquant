import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import time
import math
import gc
from multiprocessing import Pool
from typing import Tuple
from numba import njit, float64, int64

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
    'MAX_WORKERS': 32, # Adjust based on CPU cores available
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,         # 30 minutes
    'SIGMA_THRESHOLD': 0.25,            # 0.12 σ threshold
    'RESAMPLE_PERIOD': 300,             # 300 seconds (5 min) or user defined
    'REENTRY_WAIT_CANDLES': 1,          # Wait 1 candle before re-entry
    
    # Indicator Parameters
    'EAMA_SS_PERIOD': 25,
    'EAMA_ER_PERIOD': 15,
    'EAMA_FAST_PERIOD': 30,
    'EAMA_SLOW_PERIOD': 120,
    
    'KAMA_ER_PERIOD': 15,
    'KAMA_FAST_PERIOD': 25,
    'KAMA_SLOW_PERIOD': 100,
    
    'SMA_PERIOD': 100,
}

# ==========================================
# NUMBA OPTIMIZED MATH FUNCTIONS
# ==========================================

@njit(fastmath=True)
def apply_recursive_filter_numba(series: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Numba optimized recursive filter"""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid index
    start_idx = -1
    for i in range(n):
        if not np.isnan(series[i]):
            start_idx = i
            break
            
    if start_idx == -1:
        return out 
            
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0.0
        target = series[i]
        
        # Use previous output or current target if previous is nan (restart)
        prev_out = out[i-1]
        if np.isnan(prev_out):
            prev_out = target
        
        if not np.isnan(target):
            out[i] = prev_out + c * (target - prev_out)
        else:
            out[i] = prev_out
            
    return out

@njit(fastmath=True)
def get_super_smoother_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """Numba optimized SuperSmoother"""
    n = len(prices)
    pi = np.pi
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2 * a1 * np.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    
    ss = np.zeros(n, dtype=np.float64)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss

@njit(fastmath=True)
def calculate_ha_open_numba(open_vals: np.ndarray, close_vals: np.ndarray, ha_close: np.ndarray) -> np.ndarray:
    """Numba optimized iterative HA Open calculation"""
    n = len(open_vals)
    ha_open = np.zeros(n, dtype=np.float64)
    
    # Initialize first candle
    if n > 0:
        ha_open[0] = (open_vals[0] + close_vals[0]) / 2.0
    
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        
    return ha_open

@njit(fastmath=True)
def _core_signal_logic(
    eama_vals: np.ndarray, 
    kama_vals: np.ndarray, 
    eama_prev_vals: np.ndarray, 
    kama_prev_vals: np.ndarray, 
    price_vals: np.ndarray, 
    sma_vals: np.ndarray, 
    first_tradable_idx: int,
    reentry_wait_candles: int
) -> np.ndarray:
    """
    Numba optimized core state machine for signal generation.
    Returns the signal array.
    """
    n = len(eama_vals)
    signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    exit_candle_idx = -999
    
    for i in range(first_tradable_idx, n):
        # Check for NaNs
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
        
        # Detect crossovers (Transition from i-1 to i)
        bullish_crossover = (eama_prev <= kama_prev) and (eama_cur > kama_cur)
        bearish_crossover = (eama_prev >= kama_prev) and (eama_cur < kama_cur)
        
        # ============================================
        # EXIT LOGIC
        # ============================================
        if position == 1.0 and bearish_crossover:
            signals[i] = -1.0
            position = 0.0
            exit_candle_idx = i
            continue
            
        elif position == -1.0 and bullish_crossover:
            signals[i] = +1.0
            position = 0.0
            exit_candle_idx = i
            continue
        
        # ============================================
        # ENTRY LOGIC
        # ============================================
        if position == 0.0:
            if i == first_tradable_idx:
                if eama_cur > kama_cur and price_cur > sma_cur:
                    signals[i] = +1.0
                    position = 1.0
                elif eama_cur < kama_cur and price_cur < sma_cur:
                    signals[i] = -1.0
                    position = -1.0
            else:
                candles_since_exit = i - exit_candle_idx
                
                if candles_since_exit >= reentry_wait_candles:
                    if bullish_crossover and price_cur > sma_cur:
                        signals[i] = +1.0
                        position = 1.0
                    elif bearish_crossover and price_cur < sma_cur:
                        signals[i] = -1.0
                        position = -1.0
                        
    # Close open position at end logic handled in main mapper now to ensure tick precision
        
    return signals

# ==========================================
# RESAMPLING & INDICATOR WRAPPERS
# ==========================================

def resample_to_candles(data_df, period_seconds=45):
    """Resample tick data to OHLC candles with strict right-edge labeling"""
    # Use pandas native resampling (C-optimized)
    data_df = data_df.copy()
    
    if 'Time' in data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(data_df['Time']):
            data_df['Time'] = pd.to_timedelta(data_df['Time'])
            data_df['DateTime'] = pd.Timestamp('2024-01-01') + data_df['Time']
        else:
            data_df['DateTime'] = data_df['Time']
    
    data_df = data_df.set_index('DateTime')
    
    ohlc = data_df['Price'].resample(f'{period_seconds}s', label='right', closed='right').ohlc()
    ohlc = ohlc.dropna()
    
    return ohlc


def calculate_heikin_ashi_optimized(ohlc_df):
    """Calculate Heikin-Ashi candles using Numba for the loop"""
    ha_df = pd.DataFrame(index=ohlc_df.index)
    
    # Vectorized calculation for Close
    ha_close_vals = (ohlc_df['open'].values + ohlc_df['high'].values + 
                     ohlc_df['low'].values + ohlc_df['close'].values) / 4
    
    # Numba optimized calculation for Open (iterative dependency)
    ha_open_vals = calculate_ha_open_numba(
        ohlc_df['open'].values, 
        ohlc_df['close'].values, 
        ha_close_vals
    )
    
    ha_df['HA_Close'] = ha_close_vals
    ha_df['HA_Open'] = ha_open_vals
    
    # Vectorized Min/Max for High/Low
    # Stack arrays: [High, HA_Open, HA_Close]
    high_stack = np.vstack((ohlc_df['high'].values, ha_open_vals, ha_close_vals))
    low_stack = np.vstack((ohlc_df['low'].values, ha_open_vals, ha_close_vals))
    
    ha_df['HA_High'] = np.max(high_stack, axis=0)
    ha_df['HA_Low'] = np.min(low_stack, axis=0)
    
    return ha_df


def calculate_eama(prices: np.ndarray, ss_period: int, er_period: int, fast: int, slow: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates EAMA using Numba optimized math."""
    ss = get_super_smoother_numba(prices, ss_period)
    
    ss_series = pd.Series(ss)
    direction = ss_series.diff(er_period).abs()
    volatility = ss_series.diff(1).abs().rolling(er_period).sum()
    er = (direction / volatility).fillna(0).values

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2

    eama = apply_recursive_filter_numba(ss, sc)
    return eama, ss


def calculate_kama(prices: np.ndarray, er_period: int, fast: int, slow: int) -> np.ndarray:
    """Calculates KAMA using Numba optimized math."""
    price_series = pd.Series(prices)
    # price_series[0:60] = np.nan
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    sc = (((direction/volatility).fillna(0) * (2/(fast+1) - 2/(slow+1))) + (2/(slow+1)))**2
    
    kama_vals = apply_recursive_filter_numba(prices, sc.values)
    return kama_vals


# ==========================================
# STRATEGY LOGIC
# ==========================================

def calculate_mu_sigma_condition(df: pd.DataFrame, config: dict) -> bool:
    """Check if volatility condition is met in the first 30 minutes."""
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    
    if len(first_30_data) < 10:
        return False
    
    prices = first_30_data[config['PRICE_COLUMN']]
    avg_sigma = prices.std()
    
    return avg_sigma > config['SIGMA_THRESHOLD']


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate signals based on EAMA/KAMA crossover with SMA filter.
    Uses Numba for the core signal generation loop.
    """
    signals = np.zeros(len(df), dtype=np.float64)
    
    if df.empty or len(df) < 10:
        return signals
    
    # 1. Volatility Filter
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
    # Convert to float64 numpy array for Numba compatibility
    prices = df_copy[config['PRICE_COLUMN']].values.astype(np.float64)
    
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
    
    # Calculate Heikin-Ashi (Optimized)
    ha_df = calculate_heikin_ashi_optimized(ohlc)
    
    # 5. Resample indicators to candle times (value at candle close)
    df_indexed = df_copy.set_index('DateTime')
    
    for col in ['EAMA', 'KAMA', config['PRICE_COLUMN']]:
        resampled = df_indexed[col].resample(f'{period_seconds}s', label='right', closed='right').last()
        ha_df[col] = resampled.reindex(ha_df.index).ffill()
    
    # 6. Calculate SMA on resampled price
    ha_df['SMA'] = ha_df[config['PRICE_COLUMN']].rolling(
        window=config['SMA_PERIOD'], 
        min_periods=1
    ).mean()
    
    # 7. Setup Previous Values for Crossover Detection
    ha_df['EAMA_prev'] = ha_df['EAMA'].shift(1)
    ha_df['KAMA_prev'] = ha_df['KAMA'].shift(1)
    
    # 8. Mark 30-minute cutoff
    start_time = ha_df.index[0]
    cutoff_time = start_time + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    
    # 9. Prepare Arrays for Numba Signal Generation
    cutoff_idx = ha_df.index.get_indexer([cutoff_time], method='nearest')[0]
    first_tradable_idx = int(cutoff_idx + 1)
    
    # Extract arrays (ensure float64)
    eama_vals = ha_df['EAMA'].values.astype(np.float64)
    kama_vals = ha_df['KAMA'].values.astype(np.float64)
    eama_prev_vals = ha_df['EAMA_prev'].values.astype(np.float64)
    kama_prev_vals = ha_df['KAMA_prev'].values.astype(np.float64)
    price_vals = ha_df[config['PRICE_COLUMN']].values.astype(np.float64)
    sma_vals = ha_df['SMA'].values.astype(np.float64)
    
    # Call Numba Function
    resampled_signals = _core_signal_logic(
        eama_vals, kama_vals, eama_prev_vals, kama_prev_vals, 
        price_vals, sma_vals, first_tradable_idx, 
        config['REENTRY_WAIT_CANDLES']
    )
    
    # 10. Map signals back to original tick data (Sparse mapping)
    ha_df['Signal'] = resampled_signals
    
    # Filter for only non-zero signals
    sparse_signals = ha_df[ha_df['Signal'] != 0]
    
    # Initialize full tick signal array with zeros
    tick_signals = np.zeros(len(df_indexed), dtype=np.float64)
    
    if not sparse_signals.empty:
        # Get timestamps from ticks and signals
        tick_times = df_indexed.index
        signal_times = sparse_signals.index
        
        # searchsorted finds the index in tick_times where signal_times would be inserted
        # This effectively finds the first tick >= signal timestamp
        indices = tick_times.searchsorted(signal_times)
        
        # Ensure we don't go out of bounds (if signal is after last tick)
        valid_mask = indices < len(tick_times)
        valid_indices = indices[valid_mask]
        
        # Assign signals to those specific tick indices only
        tick_signals[valid_indices] = sparse_signals['Signal'].values[valid_mask]

    # --- 11. End Of Day (EOD) Square Off Logic ---
    # Calculate current position to see if we are net long or short
    current_position = np.sum(tick_signals)
    
    # If there is an open position at the very last tick, close it
    if current_position != 0:
        # If position is 1 (Long), we need -1 to close. 
        # If position is -1 (Short), we need 1 to close.
        tick_signals[-1] = -current_position

    return tick_signals


# ==========================================
# FILE/EXECUTION UTILITIES
# ==========================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, 
                       config: dict) -> dict:
    
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    
    try:
        # CPU loading with pandas
        df = pd.read_parquet(file_path, columns=required_cols)
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        # Ensure Time_sec matches config
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate signals
        signals = generate_signals(df.copy(), config)
        
        entry_signal = 0
        # Check for non-zero signals (since array is now sparse)
        valid_signals = signals[~np.isnan(signals) & (signals != 0)]
        
        if len(valid_signals) > 0:
            entry_signal = valid_signals[0]
            
        direction = 'LONG' if entry_signal > 0 else ('SHORT' if entry_signal < 0 else 'FLAT')
        
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

def generate_trade_reports_csv(output_file):
    import pandas as pd
    import os

    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)

    signals_file = output_file

    try:
        df = pd.read_csv(signals_file)
    except FileNotFoundError:
        print("Error: Combined signals file not found:", signals_file)
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found in final signal file.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        
        if pd.isna(signal):
            continue

        if position is None:
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            if signal == -position:
                trades.append({
                    "day": entry["Day"],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": (float(row["Price"]) - float(entry["Price"])) * position
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades found.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)

    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")


def main():
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("EAMA/KAMA CROSSOVER (CPU/NUMBA OPTIMIZED)")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"Workers: {config['MAX_WORKERS']}")
    print("="*80)
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    
    pool_args = [
        (f, extract_day_num(f), temp_dir_path, config) 
        for f in sorted_files
    ]
    
    processed_files = []
    direction_counts = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    
    try:
        # Use Multiprocessing to utilize CPU cores
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
        # Check non-NaN absolute signals
        entry_signals = final_df[(final_df['Signal'].notna()) & (final_df['Signal'].abs() == 1.0)]

        print("\nGenerating trade_reports.csv ...")
        generate_trade_reports_csv(output_path)
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:        {len(final_df):,}")
        print(f"Entry signals:     {len(entry_signals):,} (±1.0)")
        print(f"Total trading days: {direction_counts['LONG'] + direction_counts['SHORT']:,}")
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