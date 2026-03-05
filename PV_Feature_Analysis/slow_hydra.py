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
    'OUTPUT_FILE': 'jma_eama_signals.csv',
    'TEMP_DIR': '/tmp/jma_eama_temp',

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # --- INDICATOR PARAMETERS ---
    'super_smoother_period': 30,
    
    # EAMA_Slow (Signal Line)
    'eama_slow_er_period': 15,
    'eama_slow_fast': 60,
    'eama_slow_slow': 180,

    # JMA (Fast Line)
    'jma_length': 1800,

    # --- ACTIVATION PARAMETERS ---
    'use_dual_activation': True,
    
    # ATR Activation
    'atr_window': 15,           # Window for ATR calculation
    'atr_threshold': 0.05,      # ATR activation threshold
    
    # Std Dev Activation
    'stddev_window': 20,        # 20 ticks for std dev
    'stddev_threshold': 0.15,    # Std dev activation threshold

    # --- TRADING LOGIC ---
    'take_profit_points': 1.0,
    'trailing_stop_points': 0.15,
    'min_signal_gap_seconds': 240,
    'flip_delay_seconds': 240,     # Delay before flipping position

    # Execution constraints
    'strategy_start_tick': 1800,
    'MAX_WORKERS': 32,
}

EPSILON = 1e-9

#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    if n == 0: return ss
    pi = 3.14159265358979323846
    arg = 1.414 * pi / period
    a1 = np.exp(-arg)
    b1 = 2.0 * a1 * np.cos(arg)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, er_period, fast, slow):
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    if n == 0: return eama
    eama[0] = ss_values[0]
    fast_sc_limit = 2.0 / (fast + 1.0)
    slow_sc_limit = 2.0 / (slow + 1.0)
    for i in range(1, n):
        er = 0.0
        if i >= er_period:
            direction = abs(ss_values[i] - ss_values[i - er_period])
            volatility = 0.0
            for j in range(er_period):
                volatility += abs(ss_values[i - j] - ss_values[i - j - 1])
            if volatility > 1e-10:
                er = direction / volatility
        sc = ((er * (fast_sc_limit - slow_sc_limit)) + slow_sc_limit) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
    return eama

@jit(nopython=True, fastmath=True)
def calculate_jma_numba(prices, length):
    """
    Jurik Moving Average with adaptive smoothing based on volatility
    """
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    if n == 0: return out
    
    # Initialize volatility calculation
    vol = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        vol[i] = abs(prices[i] - prices[i-1])
    
    # Calculate rolling volatility smoothing
    vol_smooth = np.zeros(n, dtype=np.float64)
    vol_mean = np.zeros(n, dtype=np.float64)
    
    # Initial values
    vol_smooth[0] = vol[0]
    vol_mean[0] = vol[0] if vol[0] > 0 else 1.0
    
    # Smooth volatility
    for i in range(1, min(length, n)):
        sum_vol = 0.0
        for j in range(i + 1):
            sum_vol += vol[j]
        vol_smooth[i] = sum_vol / (i + 1)
        vol_mean[i] = vol_smooth[i] if vol_smooth[i] > 0 else 1.0
    
    for i in range(length, n):
        sum_vol = 0.0
        for j in range(length):
            sum_vol += vol[i - j]
        vol_smooth[i] = sum_vol / length
        
        sum_mean = 0.0
        for j in range(length):
            sum_mean += vol_smooth[i - j]
        vol_mean[i] = sum_mean / length
        if vol_mean[i] == 0:
            vol_mean[i] = 1.0
    
    # Calculate normalized volatility and adaptive alpha
    vol_norm = np.zeros(n, dtype=np.float64)
    alpha_adapt = np.zeros(n, dtype=np.float64)
    alpha = 2.0 / (length + 1.0)
    
    for i in range(n):
        vol_norm[i] = vol_smooth[i] / vol_mean[i]
        if vol_norm[i] < 0.1:
            vol_norm[i] = 0.1
        elif vol_norm[i] > 10.0:
            vol_norm[i] = 10.0
        
        alpha_adapt[i] = alpha * (1.0 + 0.5 * (vol_norm[i] - 1.0))
        if alpha_adapt[i] < 0.001:
            alpha_adapt[i] = 0.001
        elif alpha_adapt[i] > 0.999:
            alpha_adapt[i] = 0.999
    
    # Apply adaptive EMA
    out[0] = prices[0]
    for i in range(1, n):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    
    return out

@jit(nopython=True, fastmath=True)
def calculate_atr_simple_numba(prices, window):
    """
    Calculate simple ATR on raw tick data (no resampling)
    ATR = Simple moving average of absolute price changes
    """
    n = len(prices)
    atr_values = np.full(n, np.nan, dtype=np.float64)
    
    if n < 2:
        return atr_values
    
    # Calculate absolute price changes
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        changes[i] = abs(prices[i] - prices[i-1])
    
    # Simple moving average of changes as ATR
    for i in range(window, n):
        sum_changes = 0.0
        for j in range(window):
            sum_changes += changes[i - j]
        atr_values[i] = sum_changes / window
    
    return atr_values

@jit(nopython=True, fastmath=True)
def calculate_rolling_stddev_numba(prices, window):
    """
    Calculate rolling standard deviation on raw tick data
    """
    n = len(prices)
    stddev_values = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return stddev_values
    
    # Calculate rolling std dev
    for i in range(window, n):
        # Calculate mean
        sum_prices = 0.0
        for j in range(window):
            sum_prices += prices[i - j]
        mean = sum_prices / window
        
        # Calculate variance
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
    Check if BOTH ATR and StdDev thresholds are met somewhere in the day
    Returns: (is_active, max_atr, max_stddev)
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
    epsilon = 1e-10 
    if i < 1: return 0
    curr_diff = series1[i] - series2[i]
    prev_diff = series1[i-1] - series2[i-1]
    if prev_diff <= epsilon and curr_diff > epsilon: return 1   # JMA crosses above EAMA_Slow
    if prev_diff >= -epsilon and curr_diff < -epsilon: return -1  # JMA crosses below EAMA_Slow
    return 0

@jit(nopython=True, fastmath=True)
def generate_crossover_signals(
    timestamps, prices, jma, eama_slow, atr_values, stddev_values,
    tp_points, trail_points, min_gap_seconds,
    flip_delay_seconds,
    strategy_start_tick, 
    use_dual_activation,
    atr_threshold,
    stddev_threshold
):
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    eps = 1e-9 
    
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_LONG_TRAILING = 2
    STATE_SHORT = 3
    STATE_SHORT_TRAILING = 4
    STATE_WAIT_TO_SHORT = 5  # Waiting to flip to short
    STATE_WAIT_TO_LONG = 6   # Waiting to flip to long
    
    state = STATE_NEUTRAL
    tp_price = 0.0
    trail_stop = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    last_signal_time = -999999.0
    flip_wait_until = -999999.0  # Time when we can flip position
    
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
        
        # Check Dual Activation (ATR AND StdDev)
        if use_dual_activation and not day_activated:
            atr_ok = not np.isnan(atr_values[i]) and atr_values[i] >= atr_threshold
            stddev_ok = not np.isnan(stddev_values[i]) and stddev_values[i] >= stddev_threshold
            
            if atr_ok and stddev_ok:
                day_activated = True
                
                # If we have a pending entry, execute it now
                if pending_entry_type != 0:
                    if pending_entry_type == 1:  # Long entry
                        signals[i] = 1.0
                        state = STATE_LONG
                        tp_price = current_price + tp_points
                        highest_price = current_price
                        last_signal_time = current_time
                        pending_entry_type = 0
                        continue
                    elif pending_entry_type == -1:  # Short entry
                        signals[i] = -1.0
                        state = STATE_SHORT
                        tp_price = current_price - tp_points
                        lowest_price = current_price
                        last_signal_time = current_time
                        pending_entry_type = 0
                        continue
        
        crossover = detect_crossover(jma, eama_slow, i)
        
        # ===================================================================
        # WAITING STATES - Check if enough time has passed to flip
        # ===================================================================
        
        if state == STATE_WAIT_TO_SHORT:
            if current_time >= flip_wait_until and time_since_last >= min_gap_seconds and not is_last_tick:
                signals[i] = -1.0
                state = STATE_SHORT
                tp_price = current_price - tp_points
                lowest_price = current_price
                last_signal_time = current_time
            continue
        
        elif state == STATE_WAIT_TO_LONG:
            if current_time >= flip_wait_until and time_since_last >= min_gap_seconds and not is_last_tick:
                signals[i] = 1.0
                state = STATE_LONG
                tp_price = current_price + tp_points
                highest_price = current_price
                last_signal_time = current_time
            continue
        
        # ===================================================================
        # ENTRY LOGIC
        # ===================================================================
        
        if state == STATE_NEUTRAL:
            # Check for crossovers
            if crossover == 1 and time_since_last >= min_gap_seconds and not is_last_tick:
                # JMA crosses above EAMA_Slow -> Long signal
                if use_dual_activation:
                    if day_activated:
                        signals[i] = 1.0
                        state = STATE_LONG
                        tp_price = current_price + tp_points
                        highest_price = current_price
                        last_signal_time = current_time
                    else:
                        # Store as pending entry
                        pending_entry_type = 1
                else:
                    signals[i] = 1.0
                    state = STATE_LONG
                    tp_price = current_price + tp_points
                    highest_price = current_price
                    last_signal_time = current_time
                continue
            
            elif crossover == -1 and time_since_last >= min_gap_seconds and not is_last_tick:
                # JMA crosses below EAMA_Slow -> Short signal
                if use_dual_activation:
                    if day_activated:
                        signals[i] = -1.0
                        state = STATE_SHORT
                        tp_price = current_price - tp_points
                        lowest_price = current_price
                        last_signal_time = current_time
                    else:
                        # Store as pending entry
                        pending_entry_type = -1
                else:
                    signals[i] = -1.0
                    state = STATE_SHORT
                    tp_price = current_price - tp_points
                    lowest_price = current_price
                    last_signal_time = current_time
                continue
        
        # ===================================================================
        # EXIT LOGIC - LONG POSITION
        # ===================================================================
        
        elif state == STATE_LONG:
            # Force close on last tick
            if is_last_tick:
                signals[i] = -1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # REVERSE CROSSOVER EXIT (JMA crosses below EAMA_Slow)
            if crossover == -1 and time_since_last >= min_gap_seconds:
                signals[i] = -1.0
                # Wait before flipping to short
                state = STATE_WAIT_TO_SHORT
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time
                continue
            
            # Track Highest Price
            if current_price > highest_price:
                highest_price = current_price
            
            # TAKE PROFIT CHECK
            if current_price >= (tp_price - eps) and time_since_last >= min_gap_seconds:
                if jma[i] > eama_slow[i]:
                    state = STATE_LONG_TRAILING 
                    trail_stop = current_price - trail_points
                else:
                    signals[i] = -1.0
                    # Wait before flipping to short
                    state = STATE_WAIT_TO_SHORT
                    flip_wait_until = current_time + flip_delay_seconds
                    last_signal_time = current_time
        
        elif state == STATE_LONG_TRAILING:
            # Force close on last tick
            if is_last_tick:
                signals[i] = -1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # REVERSE CROSSOVER EXIT
            if crossover == -1 and time_since_last >= min_gap_seconds:
                signals[i] = -1.0
                # Wait before flipping to short
                state = STATE_WAIT_TO_SHORT
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time
                continue
            
            # Update Trail
            if current_price > highest_price:
                highest_price = current_price
                trail_stop = current_price - trail_points
            
            # HIT TRAILING STOP
            if current_price <= (trail_stop + eps) and time_since_last >= min_gap_seconds:
                signals[i] = -1.0
                # Wait before flipping to short
                state = STATE_WAIT_TO_SHORT
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time
        
        # ===================================================================
        # EXIT LOGIC - SHORT POSITION
        # ===================================================================
        
        elif state == STATE_SHORT:
            # Force close on last tick
            if is_last_tick:
                signals[i] = 1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # REVERSE CROSSOVER EXIT (JMA crosses above EAMA_Slow)
            if crossover == 1 and time_since_last >= min_gap_seconds:
                signals[i] = 1.0
                # Wait before flipping to long
                state = STATE_WAIT_TO_LONG
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time
                continue
            
            # Track Lowest Price
            if current_price < lowest_price:
                lowest_price = current_price
            
            # TAKE PROFIT CHECK
            if current_price <= (tp_price + eps) and time_since_last >= min_gap_seconds:
                if jma[i] < eama_slow[i]:
                    state = STATE_SHORT_TRAILING 
                    trail_stop = current_price + trail_points
                else:
                    signals[i] = 1.0
                    # Wait before flipping to long
                    state = STATE_WAIT_TO_LONG
                    flip_wait_until = current_time + flip_delay_seconds
                    last_signal_time = current_time
        
        elif state == STATE_SHORT_TRAILING:
            # Force close on last tick
            if is_last_tick:
                signals[i] = 1.0
                state = STATE_NEUTRAL
                last_signal_time = current_time
                continue
            
            # REVERSE CROSSOVER EXIT
            if crossover == 1 and time_since_last >= min_gap_seconds:
                signals[i] = 1.0
                # Wait before flipping to long
                state = STATE_WAIT_TO_LONG
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time
                continue
            
            # Update Trail
            if current_price < lowest_price:
                lowest_price = current_price
                trail_stop = current_price + trail_points
            
            # HIT TRAILING STOP
            if current_price >= (trail_stop - eps) and time_since_last >= min_gap_seconds:
                signals[i] = 1.0
                # Wait before flipping to long
                state = STATE_WAIT_TO_LONG
                flip_wait_until = current_time + flip_delay_seconds
                last_signal_time = current_time

    return signals  # Flip signals as requested


#############################################################################
# FILE PROCESSING FUNCTIONS
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
        if df.empty: return None
        df = df.reset_index(drop=True)
        
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

        if len(prices) < config['jma_length'] + 100:
            print(f"  [Day {day_num}] Not enough data for indicators.")
            return None
        
        # 2. CALCULATE ACTIVATION METRICS
        atr_values = calculate_atr_simple_numba(prices, config['atr_window'])
        stddev_values = calculate_rolling_stddev_numba(prices, config['stddev_window'])
        
        # Check if day gets activated (both conditions must be met)
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
        
        # 3. STRATEGY INDICATORS
        super_smoother = calculate_super_smoother_numba(prices, config['super_smoother_period'])
        
        eama_slow = calculate_eama_numba(
            super_smoother, 
            config['eama_slow_er_period'], 
            config['eama_slow_fast'], 
            config['eama_slow_slow']
        )
        
        jma = calculate_jma_numba(prices, config['jma_length'])
        
        # 4. GENERATE SIGNALS
        signals = generate_crossover_signals(
            timestamps=timestamps,
            prices=prices,
            jma=jma,
            eama_slow=eama_slow,
            atr_values=atr_values,
            stddev_values=stddev_values,
            tp_points=config['take_profit_points'],
            trail_points=config['trailing_stop_points'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            flip_delay_seconds=config['flip_delay_seconds'],
            strategy_start_tick=config['strategy_start_tick'],
            use_dual_activation=config['use_dual_activation'],
            atr_threshold=config['atr_threshold'],
            stddev_threshold=config['stddev_threshold']
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
            'JMA': jma,
            'EAMA_Slow': eama_slow,
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
    print("STRATEGY: JMA vs EAMA_SLOW CROSSOVER (BIDIRECTIONAL)")
    print("="*80)
    print(f"Data Dir:        {config['DATA_DIR']}")
    print(f"Output:          {config['OUTPUT_FILE']}")
    print(f"Execution:       Long Entry = 1.0, Short Entry = -1.0")
    
    print(f"\n--- Strategy Components ---")
    print(f"Fast Line:       JMA (Length {config['jma_length']})")
    print(f"Signal Line:     EAMA_Slow (Period {config['eama_slow_fast']}-{config['eama_slow_slow']})")
    print(f"Entry:           JMA > EAMA_Slow = Long, JMA < EAMA_Slow = Short")
    print(f"Flip Delay:      {config['flip_delay_seconds']}s after exit")
    
    if config['use_dual_activation']:
        print(f"\n--- Dual Activation (ATR AND StdDev) ---")
        print(f"ATR Window:      {config['atr_window']} ticks")
        print(f"ATR Threshold:   {config['atr_threshold']}")
        print(f"StdDev Window:   {config['stddev_window']} ticks")
        print(f"StdDev Thresh:   {config['stddev_threshold']}")
        print(f"Logic:           BOTH must activate somewhere in day")
    
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
                        infile.readline()  
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
    
    shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main(CONFIG)