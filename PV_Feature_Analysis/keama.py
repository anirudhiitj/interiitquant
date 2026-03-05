import sys
import os
import glob
import re
import shutil
import pathlib
import time
import math
import pandas as pd
import numpy as np
import multiprocessing
import gc
import traceback

# Try to import Numba
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python (slower)")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

#############################################################################
# STRATEGY CONFIGURATION
#############################################################################

CONFIG = {
    # Data paths
    'DATA_DIR': '/data/quant14/EBX/',
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'keama_signals.csv',
    'TEMP_DIR': '/tmp/keama_ekf_temp',

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # --- INDICATOR PARAMETERS ---
    
    # KEAMA (Fast Line)
    'keama_er_period': 15,
    'keama_fast': 30,
    'keama_slow': 120,
    
    # EKFTrend (Slow Line/Signal)
    'ekf_q': 0.07,
    'ekf_r': 0.2,
    
    # KAMA Slope Confirmation
    'kama_slope_er_period': 30,
    'kama_slope_fast': 60,
    'kama_slope_slow': 300,
    'kama_slope_lookback': 2,
    'kama_slope_threshold': 0.0002,

    # --- ACTIVATION PARAMETERS ---
    'use_dual_activation': False,
    
    # ATR Activation
    'atr_window': 20,
    'atr_threshold': 0.05,
    
    # Std Dev Activation
    'stddev_window': 20,
    'stddev_threshold': 0.06,

    # --- TRADING LOGIC ---
    'take_profit_points': 1.0,
    'min_signal_gap_seconds': 5,

    # Execution constraints
    'strategy_start_tick': 300,
    'MAX_WORKERS': 32,
}

EPSILON = 1e-9

#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def apply_recursive_filter_numba(series, sc):
    """
    Apply recursive KAMA-style filter
    """
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid index
    start_idx = 0
    for i in range(n):
        if not np.isnan(series[i]):
            start_idx = i
            break
    
    if start_idx >= n:
        return out
    
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0.0
        val = series[i]
        if not np.isnan(val):
            out[i] = out[i-1] + c * (val - out[i-1])
        else:
            out[i] = out[i-1]
    
    return out

@jit(nopython=True, fastmath=True)
def calculate_keama_numba(eama_values, er_period, fast, slow):
    """
    Calculate KEAMA: KAMA applied on EAMA
    """
    n = len(eama_values)
    
    # Calculate Efficiency Ratio
    direction = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        direction[i] = abs(eama_values[i] - eama_values[i - er_period])
    
    volatility = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        vol_sum = 0.0
        for j in range(er_period):
            vol_sum += abs(eama_values[i - j] - eama_values[i - j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if volatility[i] > 1e-10:
            er[i] = direction[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sc[i] = ((er[i] * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # Apply recursive filter
    keama = apply_recursive_filter_numba(eama_values, sc)
    
    return keama

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(prices, er_period, fast, slow):
    """
    Calculate EAMA (Efficiency-adjusted MA)
    """
    n = len(prices)
    
    # Calculate Efficiency Ratio
    direction = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        direction[i] = abs(prices[i] - prices[i - er_period])
    
    volatility = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        vol_sum = 0.0
        for j in range(er_period):
            vol_sum += abs(prices[i - j] - prices[i - j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if volatility[i] > 1e-10:
            er[i] = direction[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sc[i] = ((er[i] * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # Apply recursive filter
    eama = apply_recursive_filter_numba(prices, sc)
    
    return eama

@jit(nopython=True, fastmath=True)
def calculate_kama_slope_numba(prices, er_period, fast, slow):
    """
    Calculate KAMA for slope confirmation
    """
    n = len(prices)
    
    # Calculate Efficiency Ratio
    direction = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        direction[i] = abs(prices[i] - prices[i - er_period])
    
    volatility = np.zeros(n, dtype=np.float64)
    for i in range(er_period, n):
        vol_sum = 0.0
        for j in range(er_period):
            vol_sum += abs(prices[i - j] - prices[i - j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if volatility[i] > 1e-10:
            er[i] = direction[i] / volatility[i]
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sc[i] = ((er[i] * (fast_sc - slow_sc)) + slow_sc) ** 2
    
    # Apply recursive filter
    kama = apply_recursive_filter_numba(prices, sc)
    
    return kama

@jit(nopython=True, fastmath=True)
def calculate_atr_simple_numba(prices, window):
    """
    Calculate simple ATR on raw tick data
    """
    n = len(prices)
    atr_values = np.full(n, np.nan, dtype=np.float64)
    
    if n < 2:
        return atr_values
    
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        changes[i] = abs(prices[i] - prices[i-1])
    
    for i in range(window, n):
        sum_changes = 0.0
        for j in range(window):
            sum_changes += changes[i - j]
        atr_values[i] = sum_changes / window
    
    return atr_values

@jit(nopython=True, fastmath=True)
def calculate_rolling_stddev_numba(prices, window):
    """
    Calculate rolling standard deviation
    """
    n = len(prices)
    stddev_values = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return stddev_values
    
    for i in range(window, n):
        sum_prices = 0.0
        for j in range(window):
            sum_prices += prices[i - j]
        mean = sum_prices / window
        
        sum_sq_diff = 0.0
        for j in range(window):
            diff = prices[i - j] - mean
            sum_sq_diff += diff * diff
        variance = sum_sq_diff / window
        stddev_values[i] = math.sqrt(variance)
    
    return stddev_values

@jit(nopython=True, fastmath=True)
def check_dual_activation(atr_values, stddev_values, atr_threshold, stddev_threshold):
    """
    Check if BOTH ATR and StdDev thresholds are met
    """
    n = len(atr_values)
    atr_activated = False
    stddev_activated = False
    max_atr = 0.0
    max_stddev = 0.0
    
    for i in range(n):
        if not np.isnan(atr_values[i]):
            if atr_values[i] > max_atr:
                max_atr = atr_values[i]
            if atr_values[i] >= atr_threshold:
                atr_activated = True
        
        if not np.isnan(stddev_values[i]):
            if stddev_values[i] > max_stddev:
                max_stddev = stddev_values[i]
            if stddev_values[i] >= stddev_threshold:
                stddev_activated = True
    
    is_active = atr_activated and stddev_activated
    return is_active, max_atr, max_stddev

@jit(nopython=True, fastmath=True)
def detect_crossover(series1, series2, i):
    """
    Detect crossover between two series
    Returns: 1 for bullish cross, -1 for bearish cross, 0 for no cross
    """
    epsilon = 1e-10
    if i < 1:
        return 0
    
    curr_diff = series1[i] - series2[i]
    prev_diff = series1[i-1] - series2[i-1]
    
    if prev_diff <= epsilon and curr_diff > epsilon:
        return 1  # Bullish crossover
    if prev_diff >= -epsilon and curr_diff < -epsilon:
        return -1  # Bearish crossover
    
    return 0

@jit(nopython=True, fastmath=True)
def check_kama_slope_confirmation(kama_slope, i, lookback, threshold):
    """
    Check if KAMA slope confirms the trend
    Returns True if abs(slope) > threshold for lookback period
    """
    if i < lookback:
        return False
    
    for j in range(lookback + 1):
        if abs(kama_slope[i - j]) <= threshold:
            return False
    
    return True

@jit(nopython=True, fastmath=True)
def generate_signals(
    timestamps, prices, keama, ekf_trend, kama_slope,
    atr_values, stddev_values,
    tp_points, min_gap_seconds,
    strategy_start_tick, 
    use_dual_activation,
    atr_threshold,
    stddev_threshold,
    kama_slope_lookback,
    kama_slope_threshold
):
    """
    Generate trading signals based on KEAMA-EKFTrend crossover with KAMA slope confirmation
    Entry: EKFTrend > KEAMA for long (with slope confirmation)
    Exit: Opposite crossover or take profit
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    eps = 1e-9
    
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = 2
    
    state = STATE_NEUTRAL
    tp_price = 0.0
    last_signal_time = -999999.0
    
    # Dual Activation logic
    day_activated = False
    pending_entry_type = 0  # 0=none, 1=long, -1=short
    
    for i in range(n):
        if i < strategy_start_tick:
            continue
        
        current_price = prices[i]
        current_time = timestamps[i]
        time_since_last = current_time - last_signal_time
        is_last_tick = (i == n - 1)
        
        # Check Dual Activation
        if use_dual_activation and not day_activated:
            atr_ok = not np.isnan(atr_values[i]) and atr_values[i] >= atr_threshold
            stddev_ok = not np.isnan(stddev_values[i]) and stddev_values[i] >= stddev_threshold
            
            if atr_ok and stddev_ok:
                day_activated = True
                
                # Execute pending entry
                if pending_entry_type != 0:
                    if pending_entry_type == 1:  # Long
                        signals[i] = 1.0
                        state = STATE_LONG
                        tp_price = current_price + tp_points
                        last_signal_time = current_time
                        pending_entry_type = 0
                        continue
                    elif pending_entry_type == -1:  # Short
                        signals[i] = -1.0
                        state = STATE_SHORT
                        tp_price = current_price - tp_points
                        last_signal_time = current_time
                        pending_entry_type = 0
                        continue
        
        # Detect crossover
        crossover = detect_crossover(ekf_trend, keama, i)
        
        # Check KAMA slope confirmation
        slope_confirmed = check_kama_slope_confirmation(
            kama_slope, i, kama_slope_lookback, kama_slope_threshold
        )
        
        # ===================================================================
        # ENTRY LOGIC
        # ===================================================================
        
        if state == STATE_NEUTRAL:
            # Long Entry: EKFTrend crosses above KEAMA with slope confirmation
            if crossover == 1 and slope_confirmed and time_since_last >= min_gap_seconds and not is_last_tick:
                if use_dual_activation:
                    if day_activated:
                        signals[i] = 1.0
                        state = STATE_LONG
                        tp_price = current_price + tp_points
                        last_signal_time = current_time
                    else:
                        pending_entry_type = 1
                else:
                    signals[i] = 1.0
                    state = STATE_LONG
                    tp_price = current_price + tp_points
                    last_signal_time = current_time
                continue
            
            # Short Entry: EKFTrend crosses below KEAMA with slope confirmation
            elif crossover == -1 and slope_confirmed and time_since_last >= min_gap_seconds and not is_last_tick:
                if use_dual_activation:
                    if day_activated:
                        signals[i] = -1.0
                        state = STATE_SHORT
                        tp_price = current_price - tp_points
                        last_signal_time = current_time
                    else:
                        pending_entry_type = -1
                else:
                    signals[i] = -1.0
                    state = STATE_SHORT
                    tp_price = current_price - tp_points
                    last_signal_time = current_time
                continue
        
        # ===================================================================
        # EXIT LOGIC - LONG
        # ===================================================================
        
        elif state == STATE_LONG:
            # Force close on last tick
            if is_last_tick:
                signals[i] = -1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # Take Profit
            if current_price >= (tp_price - eps) and time_since_last >= min_gap_seconds:
                signals[i] = -1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # Opposite Crossover Exit (EKFTrend crosses below KEAMA)
            if crossover == -1 and time_since_last >= min_gap_seconds:
                signals[i] = -1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
        
        # ===================================================================
        # EXIT LOGIC - SHORT
        # ===================================================================
        
        elif state == STATE_SHORT:
            # Force close on last tick
            if is_last_tick:
                signals[i] = 1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # Take Profit
            if current_price <= (tp_price + eps) and time_since_last >= min_gap_seconds:
                signals[i] = 1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # Opposite Crossover Exit (EKFTrend crosses above KEAMA)
            if crossover == 1 and time_since_last >= min_gap_seconds:
                signals[i] = 1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
    
    return signals

#############################################################################
# EKF CALCULATION (Non-Numba due to matrix operations)
#############################################################################

def calculate_ekf_trend(prices):
    """
    Calculate Extended Kalman Filter trend
    Using simplified implementation for performance
    """
    n = len(prices)
    ekf_trend = np.zeros(n)
    
    # Initialize
    x = prices[0]  # State estimate
    P = 1.0        # Error covariance
    Q = 0.07       # Process noise
    R = 0.2        # Measurement noise
    
    ekf_trend[0] = x
    
    for i in range(1, n):
        # Prediction
        x_pred = x
        P_pred = P + Q
        
        # Update
        z = np.log(prices[i]) if prices[i] > 0 else np.log(0.01)
        h = np.log(x_pred) if x_pred > 0 else np.log(0.01)
        
        # Jacobian of measurement function
        H = 1.0 / x_pred if x_pred > 0 else 1.0
        
        # Innovation
        y = z - h
        S = H * P_pred * H + R
        
        # Kalman gain
        K = (P_pred * H) / S
        
        # Update state
        x = x_pred + K * y
        P = (1 - K * H) * P_pred
        
        ekf_trend[i] = x
    
    return ekf_trend

#############################################################################
# FILE PROCESSING
#############################################################################

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: str, config: dict) -> dict:
    try:
        temp_dir_path = pathlib.Path(temp_dir)
        
        # 1. READ DATA
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        df = pd.read_parquet(file_path, columns=required_cols)
        if df.empty:
            return None
        df = df.reset_index(drop=True)
        
        # Convert time to seconds
        if pd.api.types.is_numeric_dtype(df[config['TIME_COLUMN']]):
            df['Time_sec'] = df[config['TIME_COLUMN']].astype(float)
        else:
            df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        df[config['PRICE_COLUMN']] = df[config['PRICE_COLUMN']].replace(0, np.nan).ffill().bfill()
        
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        
        if np.isnan(prices).any():
            prices = np.nan_to_num(prices, nan=1.0)
        
        # Remove first 60 seconds
        cutoff_time = timestamps[0] + 60.0
        cutoff_idx = np.searchsorted(timestamps, cutoff_time, side='right')
        timestamps = timestamps[cutoff_idx:]
        prices = prices[cutoff_idx:]
        
        if len(prices) < config['strategy_start_tick'] + 100:
            print(f"  [Day {day_num}] Not enough data for indicators.")
            return None
        
        # 2. CALCULATE ACTIVATION METRICS
        atr_values = calculate_atr_simple_numba(prices, config['atr_window'])
        stddev_values = calculate_rolling_stddev_numba(prices, config['stddev_window'])
        
        is_active = True
        actual_atr = 0.0
        actual_stddev = 0.0
        
        if config['use_dual_activation']:
            is_active, actual_atr, actual_stddev = check_dual_activation(
                atr_values, 
                stddev_values, 
                config['atr_threshold'],
                config['stddev_threshold']
            )
        
        # 3. CALCULATE INDICATORS
        
        # Calculate EAMA first (needed for KEAMA)
        eama = calculate_eama_numba(
            prices,
            config['keama_er_period'],
            config['keama_fast'],
            config['keama_slow']
        )
        
        # Calculate KEAMA (KAMA on EAMA)
        keama = calculate_keama_numba(
            eama,
            config['keama_er_period'],
            config['keama_fast'],
            config['keama_slow']
        )
        
        # Calculate EKF Trend
        ekf_trend = calculate_ekf_trend(prices)
        
        # Calculate KAMA for slope confirmation
        kama_slope_raw = calculate_kama_slope_numba(
            prices,
            config['kama_slope_er_period'],
            config['kama_slope_fast'],
            config['kama_slope_slow']
        )
        
        # Calculate absolute slope
        kama_slope = np.zeros(len(kama_slope_raw))
        kama_slope[1:] = np.abs(np.diff(kama_slope_raw))
        
        # 4. GENERATE SIGNALS
        signals = generate_signals(
            timestamps=timestamps,
            prices=prices,
            keama=keama,
            ekf_trend=ekf_trend,
            kama_slope=kama_slope,
            atr_values=atr_values,
            stddev_values=stddev_values,
            tp_points=config['take_profit_points'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            strategy_start_tick=config['strategy_start_tick'],
            use_dual_activation=config['use_dual_activation'],
            atr_threshold=config['atr_threshold'],
            stddev_threshold=config['stddev_threshold'],
            kama_slope_lookback=config['kama_slope_lookback'],
            kama_slope_threshold=config['kama_slope_threshold']
        )
        
        long_entries = int(np.sum(signals == 1.0))
        short_entries = int(np.sum(signals == -1.0))
        
        activation_status = "ACTIVE" if is_active else "INACTIVE"
        stats_info = f"(ATR:{actual_atr:.4f}, StdDev:{actual_stddev:.4f})" if config['use_dual_activation'] else ""
        
        print(f"  [Day {day_num}] {activation_status:10s} {stats_info} | Long: {long_entries} | Short: {short_entries}")
        
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: timestamps,
            config['PRICE_COLUMN']: prices,
            'Signal': signals,
            'KEAMA': keama,
            'EKFTrend': ekf_trend,
            'KAMA_Slope': kama_slope,
            'EAMA': eama,
            'ATR': atr_values,
            'StdDev': stddev_values
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'total_signals': long_entries + short_entries,
            'activated': is_active
        }
    
    except Exception as e:
        print(f"  ERROR processing day {day_num}: {e}")
        traceback.print_exc()
        return None

def process_wrapper(args):
    return process_single_day(*args)

#############################################################################
# MAIN EXECUTION
#############################################################################

def main(config):
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STRATEGY: KEAMA-EKFTREND CROSSOVER WITH KAMA SLOPE CONFIRMATION")
    print("="*80)
    print(f"Data Dir:        {config['DATA_DIR']}")
    print(f"Output:          {config['OUTPUT_FILE']}")
    
    print(f"\n--- Strategy Components ---")
    print(f"Fast Line:       KEAMA (ER:{config['keama_er_period']}, {config['keama_fast']}-{config['keama_slow']})")
    print(f"Signal Line:     EKFTrend (Q:{config['ekf_q']}, R:{config['ekf_r']})")
    print(f"Confirmation:    KAMA Slope (ER:{config['kama_slope_er_period']}, {config['kama_slope_fast']}-{config['kama_slope_slow']})")
    print(f"Slope Lookback:  {config['kama_slope_lookback']} ticks")
    print(f"Slope Threshold: {config['kama_slope_threshold']}")
    print(f"Entry Logic:     EKFTrend > KEAMA = Long (with slope confirmation)")
    print(f"Exit Logic:      Opposite crossover OR Take Profit")
    
    if config['use_dual_activation']:
        print(f"\n--- Dual Activation (ATR AND StdDev) ---")
        print(f"ATR Window:      {config['atr_window']} ticks")
        print(f"ATR Threshold:   {config['atr_threshold']}")
        print(f"StdDev Window:   {config['stddev_window']} ticks")
        print(f"StdDev Thresh:   {config['stddev_threshold']}")
    
    print(f"\nWorkers:         {config['MAX_WORKERS']}")
    print("="*80 + "\n")
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    files_pattern = os.path.join(config['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    filtered_files = [f for f in all_files if (extract_day_num(f) not in range(60,80) and extract_day_num(f) not in [87])]
    sorted_files = sorted(filtered_files, key=extract_day_num)
    
    if not sorted_files:
        print("❌ No parquet files found.")
        sys.exit(1)
    
    print(f"📁 Found {len(sorted_files)} files")
    
    tasks = [
        (f, extract_day_num(f), str(temp_dir_path), config) 
        for f in sorted_files
    ]
    
    processed_files = []
    total_signals = 0
    active_days = 0
    
    with multiprocessing.Pool(processes=config['MAX_WORKERS']) as pool:
        for res in pool.imap_unordered(process_wrapper, tasks):
            if res:
                processed_files.append(res['path'])
                total_signals += res['total_signals']
                if res['activated']:
                    active_days += 1
    
    if processed_files:
        print(f"\n{'='*80}")
        print("📊 COMBINING RESULTS...")
        
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i != 0:
                        infile.readline()  # Skip header for all but first file
                    shutil.copyfileobj(infile, outfile)
        
        elapsed = time.time() - start_time
        
        print(f"{'='*80}")
        print("✅ SIGNALS FILE COMPLETE")
        print(f"{'='*80}")
        print(f"Output File:     {output_path}")
        print(f"Total Signals:   {total_signals:,}")
        if config['use_dual_activation']:
            print(f"Active Days:     {active_days}/{len(processed_files)} ({100*active_days/len(processed_files):.1f}%)")
        print(f"Execution Time:  {elapsed:.2f}s")
        print(f"{'='*80}\n")
        
        print(f"\n✅ FULL PIPELINE COMPLETE!")
        print(f"Total Time: {time.time() - start_time:.2f}s\n")
    else:
        print("❌ No data processed successfully.")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main(CONFIG)