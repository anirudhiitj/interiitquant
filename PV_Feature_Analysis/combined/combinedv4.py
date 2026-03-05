import os
import glob
import re
import shutil
import sys
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import cudf
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try to import Numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'managed_signals_v4.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 15.0,  # seconds between ANY signals
    'MAX_WORKERS': 24,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy 1 (Momentum/Volatility) - Priority 1
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_ATR_PERIOD': 25,
    'P1_STDDEV_PERIOD': 20,
    'P1_ATR_THRESHOLD': 0.01,
    'P1_STDDEV_THRESHOLD': 0.06,
    'P1_MOMENTUM_PERIOD': 10,
    'P1_VOLATILITY_PERIOD': 20,
    'P1_MIN_HOLD_SECONDS': 15.0,
    'P1_TAKE_PROFIT': 0.155,
    'P1_BLOCKED_HOURS': [5],  # No entries in hour 5 (05:00-06:00)
    
    # Strategy 2 (KAMA Slope-Based) - Priority 2
    'P2_VOLATILITY_FEATURE': 'PB9_T1',
    'P2_KAMA_PERIOD': 30,
    'P2_KAMA_FAST': 60,
    'P2_KAMA_SLOW': 300,
    'P2_KAMA_SLOPE_LOOKBACK': 2,
    'P2_KAMA_SLOPE_ENTRY_THRESHOLD': 0.007,
    'P2_ATR_PERIOD': 30,
    'P2_STDDEV_PERIOD': 20,
    'P2_ATR_THRESHOLD': 0.01,
    'P2_STDDEV_THRESHOLD': 0.06,
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_PRICE_EXIT_TARGET': 0.35,
}

# ============================================================================
# STRATEGY 1 (MOMENTUM/VOLATILITY) - PRIORITY 1 FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period=20):
    n = len(prices)
    atr = np.zeros(n)
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = abs(prices[i] - prices[i-1])
    
    for i in range(period, n):
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    return atr


@jit(nopython=True, fastmath=True)
def compute_stddev_fast(prices, period=20):
    n = len(prices)
    stddev = np.zeros(n)
    if n < period:
        return stddev
    
    for i in range(period-1, n):
        window = prices[i-period+1:i+1]
        stddev[i] = np.std(window)
    
    return stddev


@jit(nopython=True, fastmath=True)
def compute_momentum_fast(prices, period=10):
    n = len(prices)
    momentum = np.zeros(n)
    period_safe = max(period, 1)
    for i in range(period_safe, n):
        momentum[i] = prices[i] - prices[i - period_safe]
    return momentum


@jit(nopython=True, fastmath=True)
def compute_volatility_ratio(prices, period=20):
    n = len(prices)
    vol_ratio = np.zeros(n)
    if n < period:
        return vol_ratio
    
    for i in range(period-1, n):
        window = prices[i-period+1:i+1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        if abs(mean_val) > 1e-10:
            vol_ratio[i] = std_val / abs(mean_val)
        else:
            vol_ratio[i] = 0.0
    
    return vol_ratio


@jit(nopython=True, fastmath=True)
def generate_p1_signal_single(atr, stddev, momentum, volatility, atr_thresh, stddev_thresh):
    if atr < atr_thresh or stddev < stddev_thresh:
        return 0
    
    if abs(volatility) < 1e-10:
        return 0
    
    momentum_vol_ratio = momentum / max(abs(volatility), 1e-10)
    
    if momentum_vol_ratio > 0.1:
        return 1
    elif momentum_vol_ratio < -0.1:
        return -1
    else:
        return 0


@jit(nopython=True, fastmath=True)
def generate_p1_signals_internal(
    timestamps,
    prices,
    atr,
    stddev,
    momentum,
    volatility,
    atr_thresh,
    stddev_thresh,
    min_hold_seconds,
    take_profit,
    cooldown_seconds,
    blocked_hours
):
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    entry_idx = -1
    entry_price = 0.0
    last_exit_time = 0.0
    last_direction = 0
    
    for i in range(1, n):
        t = timestamps[i]
        price = prices[i]
        if price < 1e-10:
            continue
        
        current_hour = int((t % 86400) // 3600)
        
        hour_blocked = False
        for blocked_hour in blocked_hours:
            if current_hour == blocked_hour:
                hour_blocked = True
                break
        
        if hour_blocked:
            if position != 0:
                pnl = (price - entry_price) * position
                holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
                
                tp_triggered = pnl >= take_profit
                eod_triggered = i == n - 1
                min_hold_ok = holding_time >= min_hold_seconds
                
                if min_hold_ok and (tp_triggered or eod_triggered):
                    signals[i] = -position
                    last_direction = position
                    position = 0
                    last_exit_time = t
                    entry_idx = -1
            continue
        
        sig = generate_p1_signal_single(
            atr[i], stddev[i], momentum[i], volatility[i],
            atr_thresh, stddev_thresh
        )
        
        if position == 0 and sig != 0:
            if last_direction != 0 and sig == last_direction:
                continue
            
            if last_exit_time == 0.0 or (t - last_exit_time) >= cooldown_seconds:
                position = sig
                entry_price = max(price, 1e-10)
                entry_idx = i
                signals[i] = position
        
        elif position != 0:
            pnl = (price - entry_price) * position
            holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
            
            tp_triggered = pnl >= take_profit
            eod_triggered = i == n - 1
            min_hold_ok = holding_time >= min_hold_seconds
            
            if min_hold_ok and (tp_triggered or eod_triggered):
                signals[i] = -position
                last_direction = position
                position = 0
                last_exit_time = t
                entry_idx = -1
    
    return signals


def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    pb9_values = df[config['P1_FEATURE_COLUMN']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    
    atr = compute_atr_fast(pb9_values, config['P1_ATR_PERIOD'])
    stddev = compute_stddev_fast(pb9_values, config['P1_STDDEV_PERIOD'])
    momentum = compute_momentum_fast(pb9_values, config['P1_MOMENTUM_PERIOD'])
    volatility = compute_volatility_ratio(pb9_values, config['P1_VOLATILITY_PERIOD'])
    
    blocked_hours = np.array(config['P1_BLOCKED_HOURS'], dtype=np.int64)
    
    signals = generate_p1_signals_internal(
        timestamps=timestamps,
        prices=prices,
        atr=atr,
        stddev=stddev,
        momentum=momentum,
        volatility=volatility,
        atr_thresh=config['P1_ATR_THRESHOLD'],
        stddev_thresh=config['P1_STDDEV_THRESHOLD'],
        min_hold_seconds=config['P1_MIN_HOLD_SECONDS'],
        take_profit=config['P1_TAKE_PROFIT'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        blocked_hours=blocked_hours
    )
    
    return signals


# ============================================================================
# STRATEGY 2 (KAMA SLOPE-BASED) - PRIORITY 2 FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_kama_numba(prices, period, fast_period, slow_period):
    n = len(prices)
    kama = np.zeros(n, dtype=np.float64)
    signal = np.zeros(n, dtype=np.float64)
    noise = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        signal[i] = abs(prices[i] - prices[i - period])
        noise_sum = 0.0
        for j in range(period):
            noise_sum += abs(prices[i - j] - prices[i - j - 1])
        noise[i] = noise_sum
    
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)
    
    first_valid = period
    kama[first_valid] = prices[first_valid]
    
    for i in range(first_valid + 1, n):
        er = signal[i] / noise[i] if noise[i] > 1e-10 else 0.0
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    
    return kama


@jit(nopython=True, fastmath=True)
def calculate_kama_slope_numba(kama, lookback):
    n = len(kama)
    slope = np.zeros(n, dtype=np.float64)
    for i in range(lookback, n):
        slope[i] = kama[i] - kama[i - lookback]
    return slope


@jit(nopython=True, fastmath=True)
def compute_expanding_max(arr):
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    max_val = arr[0] if n > 0 else 0.0
    for i in range(n):
        if arr[i] > max_val:
            max_val = arr[i]
        result[i] = max_val
    return result


@jit(nopython=True, fastmath=True)
def generate_p2_signals_internal(
    prices,
    timestamps,
    kama_slope,
    atr_high,
    std_high,
    atr_threshold,
    std_threshold,
    slope_entry_threshold,
    min_hold_seconds,
    price_exit_target,
    cooldown_seconds
):
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    entry_price = 0.0
    entry_time = 0.0
    last_exit_time = 0.0
    
    for i in range(1, n):
        current_time = timestamps[i]
        price = prices[i]
        is_last_bar = (i == n - 1)
        
        signal = 0
        
        if is_last_bar and position != 0:
            signal = -position
        
        cooldown_ok = (current_time - last_exit_time) >= cooldown_seconds
        
        if position == 0 and cooldown_ok and not is_last_bar:
            volatility_confirmed = (atr_high[i] > atr_threshold and 
                                   std_high[i] > std_threshold)
            
            if volatility_confirmed:
                if kama_slope[i] > slope_entry_threshold:
                    signal = 1
                elif kama_slope[i] < -slope_entry_threshold:
                    signal = -1
        
        elif position != 0 and cooldown_ok:
            holding_time = current_time - entry_time
            
            if holding_time >= min_hold_seconds:
                if position == 1:
                    price_diff = price - entry_price
                    if price_diff >= price_exit_target:
                        signal = -1
                elif position == -1:
                    price_diff = entry_price - price
                    if price_diff >= price_exit_target:
                        signal = 1
        
        if signal != 0:
            position += signal
            if position == 0:
                last_exit_time = current_time
                entry_price = 0.0
                entry_time = 0.0
            else:
                entry_price = price
                entry_time = current_time
            signals[i] = signal
    
    return signals


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    pb9_values = df[config['P2_VOLATILITY_FEATURE']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    
    kama = calculate_kama_numba(
        pb9_values,
        config['P2_KAMA_PERIOD'],
        config['P2_KAMA_FAST'],
        config['P2_KAMA_SLOW']
    )
    kama_slope = calculate_kama_slope_numba(kama, config['P2_KAMA_SLOPE_LOOKBACK'])
    atr = compute_atr_fast(pb9_values, config['P2_ATR_PERIOD'])
    atr_high = compute_expanding_max(atr)
    stddev = compute_stddev_fast(pb9_values, config['P2_STDDEV_PERIOD'])
    std_high = compute_expanding_max(stddev)
    
    signals = generate_p2_signals_internal(
        prices=prices,
        timestamps=timestamps,
        kama_slope=kama_slope,
        atr_high=atr_high,
        std_high=std_high,
        atr_threshold=config['P2_ATR_THRESHOLD'],
        std_threshold=config['P2_STDDEV_THRESHOLD'],
        slope_entry_threshold=config['P2_KAMA_SLOPE_ENTRY_THRESHOLD'],
        min_hold_seconds=config['P2_MIN_HOLD_SECONDS'],
        price_exit_target=config['P2_PRICE_EXIT_TARGET'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN']
    )
    
    return signals


# ============================================================================
# MANAGER LOGIC (FIXED priority_2_locked daily reset)
# ============================================================================

def apply_manager_logic(df: pd.DataFrame, p1_signals: np.ndarray, p2_signals: np.ndarray, 
                       config: dict) -> np.ndarray:
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    position_owner = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    priority_2_locked = False
    last_day = -1

    for i in range(n):
        current_time = df.iloc[i]['Time_sec']
        current_day = int(current_time // 86400)
        
        # --- FIX: Reset priority_2_locked each new day ---
        if current_day != last_day:
            priority_2_locked = False
            last_day = current_day
        
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        is_last_bar = (i == n - 1)
        
        cooldown_ok = (current_time - last_signal_time) >= config['UNIVERSAL_COOLDOWN']
        
        final_sig = 0
        
        if is_last_bar and position != 0:
            final_sig = -position
            position = 0
            position_owner = None
            last_signal_time = current_time
        
        elif position == 0:
            if cooldown_ok:
                if p1_sig != 0:
                    final_sig = p1_sig
                    position = p1_sig
                    position_owner = "P1"
                    priority_2_locked = True
                    last_signal_time = current_time
                elif p2_sig != 0 and not priority_2_locked:
                    final_sig = p2_sig
                    position = p2_sig
                    position_owner = "P2"
                    last_signal_time = current_time
        
        elif position != 0:
            if position_owner == "P1":
                if cooldown_ok and p1_sig == -position:
                    final_sig = p1_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
            elif position_owner == "P2":
                if p1_sig != 0 and p1_sig != position:
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_2_locked = True
                        last_signal_time = current_time
                elif cooldown_ok and p2_sig == -position:
                    final_sig = p2_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
        
        final_signals[i] = final_sig
    
    return final_signals


# ============================================================================
# DAY PROCESSING AND MAIN (unchanged)
# ============================================================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> str:
    try:
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P1_FEATURE_COLUMN'],
            config['P2_VOLATILITY_FEATURE'],
        ]
        required_cols = list(dict.fromkeys(required_cols))
        
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        p1_signals = generate_p1_signals(df, config)
        p2_signals = generate_p2_signals(df, config)
        final_signals = apply_manager_logic(df, p1_signals, p2_signals, config)
        
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': final_signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return None


def main(config):
    start_time = time.time()
    
    print("="*80)
    print("STRATEGY MANAGER - PRIORITY-BASED SIGNAL ROUTING")
    print("="*80)
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    print(f"Created temp directory: {temp_dir_path}")
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    print(f"\nProcessing with {config['MAX_WORKERS']} workers...")
    processed_files = []
    
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
                        processed_files.append(result)
                        print(f"✓ Processed: day{extract_day_num(file_path)}")
                except Exception as e:
                    print(f"✗ Error in {file_path}: {e}")
        
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
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)
        
        print(f"\n✓ Created combined output: {output_path}")
        
        final_df = pd.read_csv(output_path)
        num_long = (final_df['Signal'] == 1).sum()
        num_short = (final_df['Signal'] == -1).sum()
        num_hold = (final_df['Signal'] == 0).sum()
        total_bars = len(final_df)
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:        {total_bars:,}")
        print(f"Long entries:      {num_long:,}")
        print(f"Short entries:     {num_short:,}")
        print(f"Hold/Wait:         {num_hold:,}")
        print(f"Total entries:     {(num_long + num_short):,}")
        print(f"Avg per day:       {(num_long + num_short)/len(processed_files):.1f}")
        print(f"Signal density:    {((num_long + num_short)/total_bars)*100:.3f}%")
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
    main(CONFIG)
