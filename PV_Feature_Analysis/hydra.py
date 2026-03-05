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
    'DATA_DIR': '/data/quant14/EBY/',
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'hydra_signals.csv',
    'TEMP_DIR': '/tmp/eama_hydra_temp',

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # --- INDICATOR PARAMETERS ---
    'super_smoother_period': 30,
    'eama_er_period': 15,
    'eama_fast': 30,
    'eama_slow': 120,

    # EAMA_Slow for HYDRA MA
    'eama_slow_er_period': 15,
    'eama_slow_fast': 60,
    'eama_slow_slow': 180,

    # HYDRA MA regime detection
    'hydra_window': 300,
    'hydra_std_period': 120,

    # --- KAMA FILTER PARAMETERS ---
    'kama_period': 15,
    'kama_fast': 30,
    'kama_slow': 180,
    'use_kama_slope_filter': True, 
    'kama_slope_threshold': 0.0005,
    'kama_slope_lookback': 2,

    # --- KAMA REGIME FILTER (High Volatility Lockout) ---
    'use_kama_regime_filter': False,  # Set True to ONLY trade when Regime is 0
    'kama_regime_window': 120,       # Rolling window for slope count
    'kama_regime_count_thresh': 20,  # Threshold of high slope ticks
    'kama_regime_lockout': 60,       # Lockout duration
    'kama_regime_warmup_min': 30,    # Minutes to calc threshold (P99/2)

    # --- ACTIVATION PARAMETERS (NEW) ---
    'use_daily_activation': False,
    'activation_window_seconds': 1800, # First 30 minutes
    'activation_sigma_threshold': 0.15, # Standard Deviation threshold

    # --- TRADING LOGIC ---
    'take_profit_points': 0.35,
    'trailing_stop_points': 0.15,
    'min_signal_gap_seconds': 150,
    
    # [NEW] Post Exit Cooldown
    'post_exit_wait': 1800, # 30 Minutes wait after an exit before entering again

    # Execution constraints
    'strategy_start_tick': 1800,
    'std_warmup_seconds': 3600,
    'MAX_WORKERS': 24,
}

# Global Epsilon for floating point comparisons
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
def calculate_kama_numba(prices, period, fast, slow):
    n = len(prices)
    kama = np.zeros(n, dtype=np.float64)
    if n == 0: return kama
    kama[0] = prices[0]
    fast_sc_limit = 2.0 / (fast + 1.0)
    slow_sc_limit = 2.0 / (slow + 1.0)
    for i in range(1, n):
        er = 0.0
        if i >= period:
            direction = abs(prices[i] - prices[i - period])
            volatility = 0.0
            for j in range(period):
                volatility += abs(prices[i - j] - prices[i - j - 1])
            if volatility > 1e-10:
                er = direction / volatility
        sc = ((er * (fast_sc_limit - slow_sc_limit)) + slow_sc_limit) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    return kama

@jit(nopython=True, fastmath=True)
def calculate_kama_regime_numba(timestamps, kama, window, count_thresh, lockout, warmup_minutes):
    """
    Calculates KAMA Regime using Numba.
    0 = Normal (Trade Allowed)
    1 = High Volatility/Choppy (Trade Locked)
    """
    n = len(kama)
    regime = np.zeros(n, dtype=np.int8)
    if n < window: return regime, 0.0

    # 1. Calculate Slope (Abs Diff)
    slope = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        slope[i] = abs(kama[i] - kama[i-1])

    # 2. Dynamic Threshold Calculation
    warmup_seconds = warmup_minutes * 60.0
    start_time = timestamps[0]
    warmup_indices = 0
    for i in range(n):
        if timestamps[i] - start_time > warmup_seconds:
            warmup_indices = i
            break
            
    threshold = 0.0003
    if warmup_indices > 10:
        warmup_slopes = slope[:warmup_indices]
        sorted_slopes = np.sort(warmup_slopes)
        p99_idx = int(len(sorted_slopes) * 0.99)
        if p99_idx >= len(sorted_slopes): p99_idx = len(sorted_slopes) - 1
        threshold = sorted_slopes[p99_idx] / 2.0

    # 3. Rolling Window & Lockout Logic
    rolling_sum = 0
    lockout_timer = 0
    
    is_high_slope = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if slope[i] > threshold: is_high_slope[i] = 1

    for i in range(n):
        rolling_sum += is_high_slope[i]
        if i >= window:
            rolling_sum -= is_high_slope[i - window]
            
        raw_regime = 1 if rolling_sum > count_thresh else 0
        
        if raw_regime == 1:
            lockout_timer = lockout
            
        if lockout_timer > 0:
            regime[i] = 1
            lockout_timer -= 1
        else:
            regime[i] = 0
            
    return regime, threshold

@jit(nopython=True, fastmath=True)
def calculate_rolling_std(series, window):
    n = len(series)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_data = series[start_idx:i+1]
        if len(window_data) > 1:
            result[i] = np.std(window_data)
        else:
            result[i] = 0.0
    return result

@jit(nopython=True, fastmath=True)
def calculate_diff_std_std(prices, std_period):
    n = len(prices)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = prices[i] - prices[i-1]
    diff_std = calculate_rolling_std(diff, std_period)
    diff_std_std = calculate_rolling_std(diff_std, std_period)
    return diff_std_std

@jit(nopython=True, fastmath=True)
def calculate_hydra_threshold(diff_std_std, std_warmup_ticks):
    if std_warmup_ticks < 10: return 0.0
    first_chunk = diff_std_std[:std_warmup_ticks]
    valid_data = first_chunk[first_chunk > 0]
    if len(valid_data) == 0: return 0.0
    sorted_data = np.sort(valid_data)
    idx = int(len(sorted_data) * 0.99)
    if idx >= len(sorted_data): idx = len(sorted_data) - 1
    return sorted_data[idx] / 2.5

@jit(nopython=True, fastmath=True)
def calculate_hydra_ma(eama, eama_slow, diff_std_std, threshold, window, std_warmup_ticks):
    n = len(eama)
    hydra = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < std_warmup_ticks:
            hydra[i] = eama_slow[i]
            continue
        start_idx = max(0, i - window + 1)
        high_vol_count = 0.0
        for j in range(start_idx, i + 1):
            if diff_std_std[j] > threshold:
                high_vol_count += 1.0
        weight = high_vol_count / window
        hydra[i] = weight * eama[i] + (1.0 - weight) * eama_slow[i]
    return hydra

@jit(nopython=True, fastmath=True)
def check_early_volatility_numba(timestamps, prices, window_seconds, sigma_threshold):
    """
    Calculates the standard deviation of PRICE in the first 'window_seconds'.
    Returns True if std_dev > sigma_threshold.
    """
    n = len(timestamps)
    if n == 0: return False, 0.0
    
    start_time = timestamps[0]
    cutoff_time = start_time + window_seconds
    
    # Calculate Mean first
    sum_price = 0.0
    count = 0
    for i in range(n):
        if timestamps[i] > cutoff_time:
            break
        sum_price += prices[i]
        count += 1
        
    if count < 10: return False, 0.0
    
    mean_price = sum_price / count
    
    # Calculate Variance
    sum_sq_diff = 0.0
    for i in range(count):
        diff = prices[i] - mean_price
        sum_sq_diff += diff * diff
        
    variance = sum_sq_diff / count
    std_dev = math.sqrt(variance)
    
    return std_dev > sigma_threshold, std_dev

@jit(nopython=True, fastmath=True)
def detect_crossover(series1, series2, i):
    # CHANGED: Added EPSILON to prevent micro-noise triggering signals
    # 1e-10 ensures we aren't triggered by floating point drift
    epsilon = 1e-10 
    
    if i < 1: return 0
    curr_diff = series1[i] - series2[i]
    prev_diff = series1[i-1] - series2[i-1]
    
    # Strict inequality > epsilon (not just > 0)
    if prev_diff <= epsilon and curr_diff > epsilon: return 1
    if prev_diff >= -epsilon and curr_diff < -epsilon: return -1
    return 0

@jit(nopython=True, fastmath=True)
def generate_crossover_signals(
    timestamps, prices, eama, hydra, kama,
    kama_regime,            
    use_kama_regime_filter, 
    tp_points, trail_points, min_gap_seconds,
    post_exit_wait, # NEW: 30 min cooldown parameter
    use_kama_filter, slope_threshold, slope_lookback,
    strategy_start_tick, 
    is_active_day 
):
    n = len(prices)
    # CHANGED: Use float64 for signals instead of int8
    signals = np.zeros(n, dtype=np.float64)
    
    # Global Epsilon for safe price comparisons
    eps = 1e-9 
    
    if not is_active_day:
        return signals
    
    # State constants
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = -1
    STATE_LONG_TRAILING = 2
    STATE_SHORT_TRAILING = -2

    state = STATE_NEUTRAL
    tp_price = 0.0
    trail_stop = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Init last signal time
    last_signal_time = -999999.0
    
    # NEW: Init last exit time (Start far back so first trade is valid)
    last_exit_time = -999999.0

    for i in range(n):

        if i < strategy_start_tick:
            continue
            
        is_regime_safe = True
        if use_kama_regime_filter and kama_regime[i] != 0:
            is_regime_safe = False

        current_price = prices[i]
        current_time = timestamps[i]
        time_since_last = current_time - last_signal_time
        time_since_exit = current_time - last_exit_time # NEW
        is_last_tick = (i == n - 1)

        crossover = detect_crossover(eama, hydra, i)

        # ENTRY LOGIC: NEUTRAL -> LONG/SHORT
        if state == STATE_NEUTRAL and crossover != 0 and time_since_last >= min_gap_seconds and not is_last_tick and is_regime_safe:
            
            # NEW: CHECK POST EXIT WAIT (30 MIN)
            if time_since_exit < post_exit_wait:
                continue

            kama_confirmed = True
            if use_kama_filter and i >= slope_lookback:
                slope_val = kama[i] - kama[i - slope_lookback]
                kama_confirmed = abs(slope_val) > slope_threshold
            if not kama_confirmed:
                continue

            # CHANGED: Assign 1.0 or -1.0 (Floats)
            signals[i] = 1.0 if crossover == 1 else -1.0
            last_signal_time = current_time
            
            if signals[i] == 1.0:
                state = STATE_LONG
                tp_price = current_price + tp_points
                highest_price = current_price
            else:
                state = STATE_SHORT
                tp_price = current_price - tp_points
                lowest_price = current_price

            continue

        # ===================================================================
        # POSITION MANAGEMENT (EXIT LOGIC)
        # ===================================================================

        # LONG
        elif state == STATE_LONG:
            if is_last_tick:
                signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if crossover == -1 and time_since_last >= min_gap_seconds:
                signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if current_price > highest_price:
                highest_price = current_price
            
            # Check TP (Price >= TP - epsilon)
            if current_price >= (tp_price - eps) and time_since_last >= min_gap_seconds:
                if eama[i] > hydra[i]:
                    state = STATE_LONG_TRAILING 
                    trail_stop = current_price - trail_points
                else:
                    signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time

        # LONG TRAILING
        elif state == STATE_LONG_TRAILING:
            if is_last_tick:
                signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if crossover == -1 and time_since_last >= min_gap_seconds:
                signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if current_price > highest_price:
                highest_price = current_price
                trail_stop = current_price - trail_points
            
            # Check Trail (Price <= Trail + epsilon)
            if current_price <= (trail_stop + eps) and time_since_last >= min_gap_seconds:
                signals[i] = -1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time

        # SHORT
        elif state == STATE_SHORT:
            if is_last_tick:
                signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if crossover == 1 and time_since_last >= min_gap_seconds:
                signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if current_price < lowest_price:
                lowest_price = current_price
            
            # Check TP (Price <= TP + epsilon)
            if current_price <= (tp_price + eps) and time_since_last >= min_gap_seconds:
                if eama[i] < hydra[i]:
                    state = STATE_SHORT_TRAILING
                    trail_stop = current_price + trail_points
                else:
                    signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time

        # SHORT TRAILING
        elif state == STATE_SHORT_TRAILING:
            if is_last_tick:
                signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if crossover == 1 and time_since_last >= min_gap_seconds:
                signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time; continue
            
            if current_price < lowest_price:
                lowest_price = current_price
                trail_stop = current_price + trail_points
            
            # Check Trail (Price >= Trail - epsilon)
            if current_price >= (trail_stop - eps) and time_since_last >= min_gap_seconds:
                signals[i] = 1.0; state = STATE_NEUTRAL; last_signal_time = current_time; last_exit_time = current_time

    return signals


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
        
        # Convert Time to seconds
        if pd.api.types.is_numeric_dtype(df[config['TIME_COLUMN']]):
            df['Time_sec'] = df[config['TIME_COLUMN']].astype(float)
        else:
            df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        # ================================================================
        # CHANGED: CLEANER PRICE HANDLING
        # Replaced the unsafe '1e-10' fill with FFILL/BFILL.
        # This prevents EAMA/KAMA from crashing or distorting on gap ticks.
        # ================================================================
        df[config['PRICE_COLUMN']] = df[config['PRICE_COLUMN']].replace(0, np.nan).ffill().bfill()
        
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        
        # Fallback if entire file is NaN (rare)
        if np.isnan(prices).any():
             prices = np.nan_to_num(prices, nan=1.0) # Safe fallback

        # ================================================================
        # REMOVE FIRST 60 SECONDS
        # ================================================================
        cutoff_time = timestamps[0] + 60.0
        cutoff_idx = np.searchsorted(timestamps, cutoff_time, side='right')

        timestamps = timestamps[cutoff_idx:]
        prices = prices[cutoff_idx:]

        if len(prices) < 200:
            print(f"  [Day {day_num}] Not enough data after 60s cutoff.")
            return None
        # ================================================================

        # 2. DAILY ACTIVATION CHECK
        is_active = True
        actual_sigma = 0.0
        if config['use_daily_activation']:
            is_active, actual_sigma = check_early_volatility_numba(
                timestamps, prices, 
                config['activation_window_seconds'], 
                config['activation_sigma_threshold']
            )
        
        # 3. STRATEGY INDICATORS
        super_smoother = calculate_super_smoother_numba(prices, config['super_smoother_period'])
        eama = calculate_eama_numba(super_smoother, config['eama_er_period'], config['eama_fast'], config['eama_slow'])
        eama_slow = calculate_eama_numba(super_smoother, config['eama_slow_er_period'], config['eama_slow_fast'], config['eama_slow_slow'])
        diff_std_std = calculate_diff_std_std(prices, config['hydra_std_period'])
        std_warmup_ticks = int(config['std_warmup_seconds']) 
        threshold = calculate_hydra_threshold(diff_std_std, std_warmup_ticks)
        hydra_ma = calculate_hydra_ma(eama, eama_slow, diff_std_std, threshold, config['hydra_window'], std_warmup_ticks)
        kama = calculate_kama_numba(prices, config['kama_period'], config['kama_fast'], config['kama_slow'])

        kama_regime = np.zeros(len(prices), dtype=np.int8)
        regime_threshold = 0.0
        if config['use_kama_regime_filter']:
            kama_regime, regime_threshold = calculate_kama_regime_numba(
                timestamps, kama, config['kama_regime_window'],
                config['kama_regime_count_thresh'], 
                config['kama_regime_lockout'], 
                config['kama_regime_warmup_min']
            )
        
        # 4. GENERATE SIGNALS
        signals = generate_crossover_signals(
            timestamps=timestamps,
            prices=prices,
            eama=eama,
            hydra=hydra_ma,
            kama=kama,
            kama_regime=kama_regime,
            use_kama_regime_filter=config['use_kama_regime_filter'],
            tp_points=config['take_profit_points'],
            trail_points=config['trailing_stop_points'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            post_exit_wait=config['post_exit_wait'], # NEW PARAMETER
            use_kama_filter=config['use_kama_slope_filter'],
            slope_threshold=config['kama_slope_threshold'],
            slope_lookback=config['kama_slope_lookback'],
            strategy_start_tick=config['strategy_start_tick'],
            is_active_day=is_active
        )

        long_entries = int(np.sum(signals == 1.0))
        short_entries = int(np.sum(signals == -1.0))
        total_signals = int(long_entries + short_entries)

        activation_status = "ACTIVE" if is_active else "INACTIVE"
        sigma_info = f"Sig:{actual_sigma:.4f}" if config['use_daily_activation'] else ""
        
        regime_info = ""
        if config['use_kama_regime_filter']:
            regime_info = f"| RegTh:{regime_threshold:.5f}"

        print(f"  [Day {day_num}] {activation_status:10s} ({sigma_info}) {regime_info} | Sig: {total_signals} (L:{long_entries} S:{short_entries})")
        
        # Save output CSV
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: timestamps,
            config['PRICE_COLUMN']: prices,
            'Signal': signals, # Signals are now float
            'KAMA_Regime': kama_regime
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'total_signals': total_signals,
            'long_entries': long_entries,
            'short_entries': short_entries,
            'activated': is_active
        }
    
    except Exception as e:
        print(f"  ERROR processing day {day_num}: {e}")
        traceback.print_exc()
        return None


def process_wrapper(args):
    return process_single_day(*args)

# ------------------- ACCELERATED TRADE REPORTING -------------------
@jit(nopython=True, fastmath=True)
def _parse_trades_fast(prices, signals, day_indices):
    """Numba-accelerated core trade parsing loop (Strict Entry/Exit)."""
    N = len(prices)
    max_trades = N // 2
    
    # Pre-allocated NumPy arrays for speed
    trade_number_arr = np.zeros(max_trades, dtype=np.int32)
    day_arr = np.zeros(max_trades, dtype=np.int32)
    # CHANGED: direction_arr to float64 to support float signals
    direction_arr = np.zeros(max_trades, dtype=np.float64) 
    entry_price_arr = np.zeros(max_trades, dtype=np.float64)
    exit_price_arr = np.zeros(max_trades, dtype=np.float64)
    raw_pnl_arr = np.zeros(max_trades, dtype=np.float64)
    entry_idx_arr = np.zeros(max_trades, dtype=np.int32)
    exit_idx_arr = np.zeros(max_trades, dtype=np.int32)

    position = 0.0 # Float
    entry_price = 0.0
    entry_idx = 0
    trade_count = 0

    for i in range(N):
        sig = signals[i]
        
        if sig == 0.0: continue

        if position == 0.0:
            # Check if sig is Long (1.0) or Short (-1.0)
            if abs(sig) > 0.5:
                position = sig
                entry_price = prices[i]
                entry_idx = i
        
        else:
            # Check for reversal or close
            # if sig == -position works for floats 1.0 vs -1.0
            if sig == -position: 
                if trade_count < max_trades:
                    xp = prices[i]
                    raw = (xp - entry_price) * position
                    
                    trade_number_arr[trade_count] = trade_count + 1
                    day_arr[trade_count] = day_indices[entry_idx]
                    direction_arr[trade_count] = position
                    entry_price_arr[trade_count] = entry_price
                    exit_price_arr[trade_count] = xp
                    raw_pnl_arr[trade_count] = raw
                    entry_idx_arr[trade_count] = entry_idx
                    exit_idx_arr[trade_count] = i
                    trade_count += 1
                
                position = 0.0
                
    return (
        trade_number_arr[:trade_count], day_arr[:trade_count], direction_arr[:trade_count],
        entry_price_arr[:trade_count], exit_price_arr[:trade_count], raw_pnl_arr[:trade_count],
        entry_idx_arr[:trade_count], exit_idx_arr[:trade_count]
    )

def generate_trade_report(signals_file, out_file):
    try:
        # Load necessary data
        df = pd.read_csv(signals_file, usecols=['Time', 'Price', 'Signal'])
    except Exception as e:
        print(f"[ERR] Failed to read {signals_file}: {e}")
        return False
        
    if df.empty:
        pd.DataFrame(columns=["trade_number", "day", "direction", "entry_time", "entry_price", "exit_time", "exit_price", "raw_pnl", "pnl_pct"]).to_csv(out_file, index=False)
        print("[OK] Trade report: 0 trades")
        return True

    # 1. Detect day boundaries and prepare arrays
    time_series = df['Time'].str.split(':').str[0].astype(int)
    day_changes = (time_series.diff() < 0).fillna(False)
    
    day_idx = np.zeros(len(df), dtype=int)
    day_idx[0] = 1
    day_num = 1
    for i in range(1, len(df)):
        if day_changes[i]:
            day_num += 1
        day_idx[i] = day_num
    
    prices = df["Price"].values
    signals = df["Signal"].values
    
    # 2. Run Numba core trade parsing
    (
        trade_number_arr, day_arr, direction_arr,
        entry_price_arr, exit_price_arr, raw_pnl_arr,
        entry_idx_arr, exit_idx_arr
    ) = _parse_trades_fast(prices, signals, day_idx)

    trade_count = len(trade_number_arr)
    
    if trade_count == 0:
        pd.DataFrame(columns=["trade_number", "day", "direction", "entry_time", "entry_price", "exit_time", "exit_price", "raw_pnl", "pnl_pct"]).to_csv(out_file, index=False)
        print("[OK] Trade report: 0 trades")
        return True

    # 3. Reconstruct DataFrame
    valid_indices = entry_price_arr != 0
    pnl_pct_arr = np.zeros_like(raw_pnl_arr)
    pnl_pct_arr[valid_indices] = (raw_pnl_arr[valid_indices] / entry_price_arr[valid_indices]) * 100
    
    entry_time_series = df["Time"].iloc[entry_idx_arr].reset_index(drop=True)
    exit_time_series = df["Time"].iloc[exit_idx_arr].reset_index(drop=True)
    
    direction_str = np.where(direction_arr == 1.0, "LONG", "SHORT")

    trade_df = pd.DataFrame({
        "trade_number": trade_number_arr, "day": day_arr, "direction": direction_str,
        "entry_time": entry_time_series, "entry_price": entry_price_arr,
        "exit_time": exit_time_series, "exit_price": exit_price_arr,
        "raw_pnl": raw_pnl_arr, "pnl_pct": pnl_pct_arr
    })
    
    # 4. Save report and print summary
    trade_df.to_csv(out_file, index=False)
    
    total_pnl = trade_df['raw_pnl'].sum()
    winning_trades = len(trade_df[trade_df['raw_pnl'] > 0])
    
    print(f"\n{'='*60}")
    print(f"TRADE REPORT SUMMARY (ACCELERATED)")
    print(f"{'='*60}")
    print(f"Total Trades: {trade_count}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {(winning_trades/trade_count*100):.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    if trade_count > 0:
        print(f"Avg PnL per Trade: {total_pnl/trade_count:.4f}")
    print(f"{'='*60}\n")

    return True


#############################################################################
# MAIN EXECUTION
#############################################################################

def main(config):
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STRATEGY: HYDRA CROSSOVER + VOLATILITY ACTIVATION (FLOAT-FIXED)")
    print("="*80)
    print(f"Data Dir:        {config['DATA_DIR']}")
    print(f"Output:          {config['OUTPUT_FILE']}")
    print(f"\n--- Filters ---")
    print(f"KAMA Regime:     {config['use_kama_regime_filter']} (Block entry if High Volatility)")

    print(f"\n--- Daily Activation (Volatility Check) ---")
    print(f"Activation:      {config['use_daily_activation']}")
    if config['use_daily_activation']:
        print(f"Window:          {config['activation_window_seconds']}s (First {config['activation_window_seconds']/60:.0f} mins)")
        print(f"Sigma Thresh:    {config['activation_sigma_threshold']} (StdDev of Price)")
    
    print(f"\n--- Cooldown ---")
    print(f"Post Exit Wait:  {config['post_exit_wait']}s ({config['post_exit_wait']/60:.0f} mins)")

    print(f"\nWorkers:         {config['MAX_WORKERS']}")
    print("="*80 + "\n")
    
    # Setup Temp Directory
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    # Find Files
    files_pattern = os.path.join(config['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    filtered_files = [f for f in all_files if extract_day_num(f) not in [6, 25, 104, 110, 115, 135, 165, 209]]
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
    total_long = 0
    total_short = 0
    active_days = 0

    # CPU Multiprocessing
    with multiprocessing.Pool(processes=config['MAX_WORKERS']) as pool:
        for res in pool.imap_unordered(process_wrapper, tasks):
            if res:
                processed_files.append(res['path'])
                total_signals += res['total_signals']
                total_long += res['long_entries']
                total_short += res['short_entries']
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
                        infile.readline()  # Skip header for subsequent files
                    shutil.copyfileobj(infile, outfile)
        
        elapsed = time.time() - start_time
        
        print(f"{'='*80}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Output File:     {output_path}")
        print(f"Total Signals:   {total_signals:,} (L:{total_long:,} S:{total_short:,})")
        if config['use_daily_activation']:
            print(f"Active Days:     {active_days}/{len(processed_files)} ({100*active_days/len(processed_files):.1f}%)")
        print(f"Execution Time:  {elapsed:.2f}s")
        print(f"{'='*80}\n")
    else:
        print("❌ No data processed successfully.")
    
    shutil.rmtree(temp_dir_path)
    #generate_trade_report(str(output_path), "./trade_report.csv")

if __name__ == "__main__":
    main(CONFIG)