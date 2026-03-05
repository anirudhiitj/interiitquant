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
    'OUTPUT_FILE': 'managed_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 15.0,  # seconds between ANY signals
    'MAX_WORKERS': 20,
    
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
    'P1_TAKE_PROFIT': 0.16,
    'P1_TRAILING_STOP_FROM_PEAK': 0.01,  # Exit if PnL drops this much from peak after TP hit
    'P1_BLOCKED_HOURS': [],  # No entries in hour 5 (05:00-06:00)
    
    # Strategy 2 (BB Crossover with KAMA Filter) - Priority 2
    'P2_BB4_COLUMN': 'BB4_T9',
    'P2_BB6_COLUMN': 'BB6_T11',
    'P2_VOLATILITY_FEATURE': 'PB9_T1',
    'P2_V_PATTERN_FEATURE': 'PB6_T4',  # For V-pattern detection
    'P2_V_PATTERN_ATR_PERIOD': 60,  # ATR period for V-pattern momentum check
    'P2_V_PATTERN_ATR_THRESHOLD': 0.007,  # ATR threshold to enable V-pattern detection
    'P2_ATR_PERIOD': 30,
    'P2_ACTIVATION_ATR_THRESHOLD': 0.0,
    'P2_ACTIVATION_STDDEV_THRESHOLD': 0.0,
    
    # KAMA Filter Parameters (matching document 3)
    'P2_KAMA_PERIOD': 15,
    'P2_KAMA_FAST': 30,
    'P2_KAMA_SLOW': 180,
    'P2_KAMA_SLOPE_LOOKBACK': 2,
    'P2_KAMA_SLOPE_THRESHOLD': 0.0007,
    
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_MAX_HOLD_SECONDS': 3600*2,
    'P2_USE_TAKE_PROFIT': True,
    'P2_TAKE_PROFIT_OFFSET': 0.8,
    'P2_STRATEGY_START_BAR': 1800,
}

# ============================================================================
# STRATEGY 1 (MOMENTUM/VOLATILITY) - PRIORITY 1 FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period=20):
    """Compute Average True Range on price series"""
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
    """Compute rolling standard deviation"""
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
    """Compute momentum (rate of change)"""
    n = len(prices)
    momentum = np.zeros(n)
    period_safe = max(period, 1)
    for i in range(period_safe, n):
        momentum[i] = prices[i] - prices[i - period_safe]
    return momentum


@jit(nopython=True, fastmath=True)
def compute_volatility_ratio(prices, period=20):
    """Compute volatility ratio as stddev/mean to normalize volatility"""
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
    """
    Generate single bar trading signal based on:
    1. Volatility filter: ATR > threshold AND StdDev > threshold
    2. Direction: momentum/volatility ratio determines long/short
    """
    # First check volatility filter
    if atr < atr_thresh or stddev < stddev_thresh:
        return 0  # Not eligible to trade
    
    # If volatility is too low, don't trade
    if abs(volatility) < 1e-10:
        return 0
    
    # Use momentum/volatility to determine direction
    momentum_vol_ratio = momentum / max(abs(volatility), 1e-10)
    
    # Positive momentum/vol ratio = bullish (long)
    # Negative momentum/vol ratio = bearish (short)
    if momentum_vol_ratio > 0.1:  # Small threshold to avoid noise
        return 1  # Long
    elif momentum_vol_ratio < -0.1:
        return -1  # Short
    else:
        return 0  # No trade


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
    trailing_stop,
    cooldown_seconds,
    blocked_hours
):
    """
    Generate Priority 1 (Momentum/Volatility) signals with PURE trailing stop loss
    
    Exit Strategy:
    - After hitting TP threshold, use ONLY trailing stop loss
    - Track peak PnL after TP hit
    - Exit when: current_pnl < (peak_pnl - trailing_stop)
    
    NO ENTRIES IN BLOCKED HOURS (default: hour 5 = 05:00-06:00)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    entry_idx = -1
    entry_price = 0.0
    last_exit_time = 0.0
    last_direction = 0
    
    # Trailing stop tracking
    tp_hit = False
    peak_pnl = 0.0
    
    for i in range(1, n):
        t = timestamps[i]
        price = prices[i]
        
        if price < 1e-10:
            continue
        
        # Calculate current hour (0-5 for 6-hour trading session)
        current_hour = int(t // 3600)
        
        # Check if current hour is blocked
        hour_blocked = False
        for blocked_hour in blocked_hours:
            if current_hour == blocked_hour:
                hour_blocked = True
                break
        
        # BLOCK ENTRIES IN BLOCKED HOURS
        if hour_blocked:
            # Still allow exits during blocked hours, just no new entries
            if position != 0:
                pnl = (price - entry_price) * position
                holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
                
                # Check if we've hit TP threshold
                if not tp_hit and pnl >= take_profit:
                    tp_hit = True
                    peak_pnl = pnl
                
                # Update peak PnL if TP hit
                if tp_hit and pnl > peak_pnl:
                    peak_pnl = pnl
                
                # Exit conditions
                eod_triggered = i == n - 1
                min_hold_ok = holding_time >= min_hold_seconds
                
                # PURE TRAILING STOP LOGIC
                should_exit = False
                if tp_hit:
                    # Exit ONLY when PnL drops from peak
                    if pnl < (peak_pnl - trailing_stop):
                        should_exit = True
                
                if min_hold_ok and (should_exit or eod_triggered):
                    signals[i] = -position
                    last_direction = position
                    position = 0
                    last_exit_time = t
                    entry_idx = -1
                    tp_hit = False
                    peak_pnl = 0.0
            continue  # Skip to next iteration - NO NEW ENTRIES IN BLOCKED HOURS
        
        # Generate signal for current bar
        sig = generate_p1_signal_single(
            atr[i], stddev[i], momentum[i], volatility[i],
            atr_thresh, stddev_thresh
        )
        
        # Entry logic
        if position == 0 and sig != 0:
            # No consecutive same-direction trades
            if last_direction != 0 and sig == last_direction:
                continue
            
            # Check cooldown
            if last_exit_time == 0.0 or (t - last_exit_time) >= cooldown_seconds:
                position = sig
                entry_price = max(price, 1e-10)
                entry_idx = i
                signals[i] = position
                tp_hit = False  # Reset TP flag for new position
                peak_pnl = 0.0  # Reset peak PnL
        
        # Exit logic - PURE trailing stop loss after TP hit
        elif position != 0:
            pnl = (price - entry_price) * position
            holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
            
            # Check if we've hit TP threshold
            if not tp_hit and pnl >= take_profit:
                tp_hit = True
                peak_pnl = pnl  # Initialize peak at TP level
            
            # Update peak PnL if TP hit
            if tp_hit and pnl > peak_pnl:
                peak_pnl = pnl
            
            # Exit conditions
            eod_triggered = i == n - 1
            
            # Enforce minimum hold period
            min_hold_ok = holding_time >= min_hold_seconds
            
            # PURE TRAILING STOP LOGIC
            should_exit = False
            if tp_hit:
                # Exit ONLY when PnL drops from peak
                if pnl < (peak_pnl - trailing_stop):
                    should_exit = True
            
            exit_condition = min_hold_ok and (should_exit or eod_triggered)
            
            if exit_condition:
                signals[i] = -position
                last_direction = position
                position = 0
                last_exit_time = t
                entry_idx = -1
                tp_hit = False
                peak_pnl = 0.0
    
    return signals


def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 1 (Momentum/Volatility) signals with trailing stop loss"""
    pb9_values = df[config['P1_FEATURE_COLUMN']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    # Ensure no invalid values
    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    
    # Calculate all indicators
    atr = compute_atr_fast(pb9_values, config['P1_ATR_PERIOD'])
    stddev = compute_stddev_fast(pb9_values, config['P1_STDDEV_PERIOD'])
    momentum = compute_momentum_fast(pb9_values, config['P1_MOMENTUM_PERIOD'])
    volatility = compute_volatility_ratio(pb9_values, config['P1_VOLATILITY_PERIOD'])
    
    # Convert blocked hours list to numpy array
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
        trailing_stop=config['P1_TRAILING_STOP_FROM_PEAK'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        blocked_hours=blocked_hours
    )
    
    return signals


# ============================================================================
# STRATEGY 2 (BB CROSSOVER WITH KAMA FILTER) - PRIORITY 2 FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_true_range_from_pb9(pb9_t1_values):
    """Calculate True Range using PB9_T1"""
    n = len(pb9_t1_values)
    true_range = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        true_range[i] = abs(pb9_t1_values[i] - pb9_t1_values[i-1])
    return true_range


@jit(nopython=True, fastmath=True)
def rolling_mean_numba(arr, window):
    """Calculate rolling mean with Numba acceleration"""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        result[i] = np.mean(arr[start_idx:i+1])
    return result


@jit(nopython=True, fastmath=True)
def rolling_std_numba(arr, window):
    """Calculate rolling standard deviation with Numba acceleration"""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        result[i] = np.std(arr[start_idx:i+1])
    return result


@jit(nopython=True, fastmath=True)
def calculate_kama(prices, period, fast_period, slow_period):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    Matches implementation from document 3
    
    Parameters:
    - prices: array of prices (PB9_T1)
    - period: lookback period for efficiency ratio (30)
    - fast_period: fast EMA period (60)
    - slow_period: slow EMA period (300)
    """
    n = len(prices)
    kama = np.zeros(n, dtype=np.float64)
    
    # Calculate efficiency ratio components
    signal = np.zeros(n, dtype=np.float64)
    noise = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        # Signal: absolute change over period
        signal[i] = abs(prices[i] - prices[i - period])
        
        # Noise: sum of absolute changes
        noise_sum = 0.0
        for j in range(period):
            noise_sum += abs(prices[i - j] - prices[i - j - 1])
        noise[i] = noise_sum
    
    # Calculate smoothing constants
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    first_valid = period
    kama[first_valid] = prices[first_valid]
    
    for i in range(first_valid + 1, n):
        # Efficiency ratio
        if noise[i] > 1e-10:
            er = signal[i] / noise[i]
        else:
            er = 0.0
        
        # Smoothing constant
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        
        # KAMA calculation
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
    
    return kama


@jit(nopython=True, fastmath=True)
def calculate_kama_slope(kama, lookback):
    """Calculate KAMA slope with specified lookback period"""
    n = len(kama)
    slope = np.zeros(n, dtype=np.float64)
    
    for i in range(lookback, n):
        slope[i] = kama[i] - kama[i - lookback]
    
    return slope


@jit(nopython=True, fastmath=True)
def detect_v_pattern(pb6_t4_values, v_pattern_atr_values, index, signal_direction, atr_threshold):
    """
    Detect V-pattern using PB6_T4 feature, but ONLY when momentum is high (ATR > threshold)
    
    Momentum Check (REQUIRED):
    - ATR (60-period on PB9_T1) must cross above threshold (0.07)
    - ATR[i] > threshold AND ATR[i-1] <= threshold (crossing condition)
    
    V-Pattern Check (if momentum condition met):
    For SHORT signal: check if PB6_T4 is ASCENDING (V-bottom pattern)
        PB6_T4[i] > PB6_T4[i-1] > PB6_T4[i-2] > PB6_T4[i-3]
    
    For LONG signal: check if PB6_T4 is DESCENDING (V-top pattern)
        PB6_T4[i] < PB6_T4[i-1] < PB6_T4[i-2] < PB6_T4[i-3]
    
    Returns True if BOTH momentum condition AND V-pattern detected
    """
    # Need at least 4 bars for 3-second window
    if index < 3:
        return False
    
    # CRITICAL: Check momentum condition first (ATR crossing above threshold)
    current_atr = v_pattern_atr_values[index]
    prev_atr = v_pattern_atr_values[index - 1]
    
    # ATR must cross above threshold (current > threshold AND previous <= threshold)
    atr_crossing = (current_atr > atr_threshold) and (prev_atr <= atr_threshold)
    
    if not atr_crossing:
        return False  # No momentum - skip V-pattern check
    
    # Momentum condition met - now check V-pattern
    v0 = pb6_t4_values[index]
    v1 = pb6_t4_values[index - 1]
    v2 = pb6_t4_values[index - 2]
    v3 = pb6_t4_values[index - 3]
    
    if signal_direction == -1:  # Short signal
        # Check for ascending pattern (V-bottom)
        is_ascending = (v0 > v1) and (v1 > v2) and (v2 > v3)
        return is_ascending
    
    elif signal_direction == 1:  # Long signal
        # Check for descending pattern (V-top)
        is_descending = (v0 < v1) and (v1 < v2) and (v2 < v3)
        return is_descending
    
    return False


@jit(nopython=True, fastmath=True)
def generate_p2_signals_internal(
    prices,
    bb4_values,
    bb6_values,
    timestamps,
    atr_values,
    pb9_stddev_values,
    kama_slope,
    pb6_t4_values,
    v_pattern_atr_values,
    activation_atr_threshold,
    activation_stddev_threshold,
    kama_slope_threshold,
    v_pattern_atr_threshold,
    min_hold_seconds,
    max_hold_seconds,
    cooldown_seconds,
    use_take_profit,
    take_profit_offset,
    strategy_start_bar
):
    """
    Generate Priority 2 (BB Crossover with KAMA Filter) signals
    
    KAMA Directional Confirmation:
    - Long entry: KAMA slope > threshold (uptrend)
    - Short entry: KAMA slope < -threshold (downtrend)
    
    V-Pattern Detection (at entry time, ONLY when momentum is high):
    - REQUIRES: ATR (60-period on PB9_T1) crossing above 0.07 threshold
    - IF momentum condition met:
      - Short signal + ascending PB6_T4 → reverse to long
      - Long signal + descending PB6_T4 → reverse to short
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    strategy_activated = False
    current_position = 0
    entry_time = 0.0
    entry_price = 0.0
    last_exit_time = 0.0
    
    # V-pattern reversal tracking
    pending_reversal = 0  # 0=none, 1=long, -1=short
    reversal_available_time = 0.0
    
    prev_bb4 = bb4_values[0]
    prev_bb6 = bb6_values[0]
    
    for i in range(1, n):
        current_time = timestamps[i]
        current_price = prices[i]
        current_bb4 = bb4_values[i]
        current_bb6 = bb6_values[i]
        current_atr = atr_values[i]
        current_pb9_stddev = pb9_stddev_values[i]
        current_kama_slope = kama_slope[i]
        
        # Check activation
        if not strategy_activated:
            if (current_atr > activation_atr_threshold and 
                current_pb9_stddev > activation_stddev_threshold):
                strategy_activated = True
        
        can_trade = strategy_activated and (i >= strategy_start_bar)
        
        # Detect crossovers
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        # KAMA directional filter
        kama_bullish = current_kama_slope > kama_slope_threshold
        kama_bearish = current_kama_slope < -kama_slope_threshold
        
        is_last_bar = (i == n - 1)
        
        # Position management
        if current_position == 0:
            # Check if we have a pending reversal from V-pattern
            if pending_reversal != 0 and current_time >= reversal_available_time:
                # Enter reverse position (no KAMA filter for reversal)
                current_position = pending_reversal
                entry_time = current_time
                entry_price = current_price
                signals[i] = pending_reversal
                pending_reversal = 0
                reversal_available_time = 0.0
            
            # Normal entry logic
            else:
                cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)
                
                if (can_trade and cooldown_elapsed and not is_last_bar):
                    initial_signal = 0
                    
                    # Long entry: BB crossover + KAMA uptrend
                    if bullish_bb_cross and kama_bullish:
                        initial_signal = 1
                    # Short entry: BB crossover + KAMA downtrend
                    elif bearish_bb_cross and kama_bearish:
                        initial_signal = -1
                    
                    # Check for V-pattern if we have a signal
                    if initial_signal != 0:
                        v_pattern_detected = detect_v_pattern(
                            pb6_t4_values, v_pattern_atr_values, i, initial_signal, v_pattern_atr_threshold
                        )
                        
                        if v_pattern_detected:
                            # V-pattern detected - schedule reversal
                            # Exit immediately (position is 0, so just mark exit time)
                            # and schedule reverse entry after cooldown
                            pending_reversal = -initial_signal  # Reverse direction
                            reversal_available_time = current_time + cooldown_seconds
                            last_exit_time = current_time
                            # No signal this bar - reversal happens next eligible bar
                        else:
                            # No V-pattern - normal entry
                            current_position = initial_signal
                            entry_time = current_time
                            entry_price = current_price
                            signals[i] = initial_signal
        
        elif current_position == 1:
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                if is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1
            else:
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price >= (entry_price + take_profit_offset))
                
                # Exit on bearish BB crossover
                should_exit = (bearish_bb_cross or max_hold_exceeded or is_last_bar or take_profit_hit)
                
                if should_exit:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1
        
        elif current_position == -1:
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                if is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1
            else:
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price <= (entry_price - take_profit_offset))
                
                # Exit on bullish BB crossover
                should_exit = (bullish_bb_cross or max_hold_exceeded or is_last_bar or take_profit_hit)
                
                if should_exit:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1
        
        prev_bb4 = current_bb4
        prev_bb6 = current_bb6
    
    return signals


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 2 (BB Crossover with KAMA Filter) signals"""
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    bb4_values = df[config['P2_BB4_COLUMN']].values.astype(np.float64)
    bb6_values = df[config['P2_BB6_COLUMN']].values.astype(np.float64)
    pb9_t1_values = df[config['P2_VOLATILITY_FEATURE']].values.astype(np.float64)
    pb6_t4_values = df[config['P2_V_PATTERN_FEATURE']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    # Calculate ATR for strategy activation (30-period)
    true_range = calculate_true_range_from_pb9(pb9_t1_values)
    atr_values = rolling_mean_numba(true_range, config['P2_ATR_PERIOD'])
    
    # Calculate ATR for V-pattern momentum check (60-period)
    v_pattern_atr_values = rolling_mean_numba(true_range, config['P2_V_PATTERN_ATR_PERIOD'])
    
    # Calculate StdDev
    pb9_stddev_values = rolling_std_numba(pb9_t1_values, config['P2_ATR_PERIOD'])
    
    # Calculate KAMA and slope (matching document 3 implementation)
    kama = calculate_kama(
        pb9_t1_values,
        config['P2_KAMA_PERIOD'],
        config['P2_KAMA_FAST'],
        config['P2_KAMA_SLOW']
    )
    kama_slope = calculate_kama_slope(kama, config['P2_KAMA_SLOPE_LOOKBACK'])
    
    signals = generate_p2_signals_internal(
        prices=prices,
        bb4_values=bb4_values,
        bb6_values=bb6_values,
        timestamps=timestamps,
        atr_values=atr_values,
        pb9_stddev_values=pb9_stddev_values,
        kama_slope=kama_slope,
        pb6_t4_values=pb6_t4_values,
        v_pattern_atr_values=v_pattern_atr_values,
        activation_atr_threshold=config['P2_ACTIVATION_ATR_THRESHOLD'],
        activation_stddev_threshold=config['P2_ACTIVATION_STDDEV_THRESHOLD'],
        kama_slope_threshold=config['P2_KAMA_SLOPE_THRESHOLD'],
        v_pattern_atr_threshold=config['P2_V_PATTERN_ATR_THRESHOLD'],
        min_hold_seconds=config['P2_MIN_HOLD_SECONDS'],
        max_hold_seconds=config['P2_MAX_HOLD_SECONDS'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        use_take_profit=config['P2_USE_TAKE_PROFIT'],
        take_profit_offset=config['P2_TAKE_PROFIT_OFFSET'],
        strategy_start_bar=config['P2_STRATEGY_START_BAR']
    )
    
    return signals


# ============================================================================
# MANAGER LOGIC
# ============================================================================

def apply_manager_logic(df: pd.DataFrame, p1_signals: np.ndarray, p2_signals: np.ndarray, 
                       config: dict) -> np.ndarray:
    """
    Apply manager logic to combine Priority 1 and Priority 2 signals
    
    Rules:
    1. Daily reset
    2. Once P1 trades -> P2 locked for entire day
    3. 15s universal cooldown between ANY signals
    4. If in P2 position and P1 wants to reverse -> exit P2, wait 15s, then P1 can enter
    5. Unified EOD square-off
    """
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    
    # Manager state
    position = 0  # 0=flat, 1=long, -1=short
    position_owner = None  # None, "P1", "P2"
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    priority_2_locked = False
    
    for i in range(n):
        current_time = df.iloc[i]['Time_sec']
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        is_last_bar = (i == n - 1)
        
        # Check cooldown
        cooldown_ok = (current_time - last_signal_time) >= config['UNIVERSAL_COOLDOWN']
        
        final_sig = 0
        
        # ====================================================================
        # EOD SQUARE-OFF (Unified - overrides everything)
        # ====================================================================
        if is_last_bar and position != 0:
            final_sig = -position
            position = 0
            position_owner = None
            last_signal_time = current_time
        
        # ====================================================================
        # FLAT POSITION
        # ====================================================================
        elif position == 0:
            if cooldown_ok and not is_last_bar:
                # Priority 1 entry
                if p1_sig != 0:
                    final_sig = p1_sig
                    position = p1_sig
                    position_owner = "P1"
                    priority_2_locked = True
                    last_signal_time = current_time
                
                # Priority 2 entry (only if P1 hasn't locked it)
                elif p2_sig != 0 and not priority_2_locked:
                    final_sig = p2_sig
                    position = p2_sig
                    position_owner = "P2"
                    last_signal_time = current_time
        
        # ====================================================================
        # IN POSITION
        # ====================================================================
        elif position != 0:
            if position_owner == "P1":
                # Priority 1 manages its own exit
                if cooldown_ok and p1_sig == -position:
                    final_sig = p1_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
            
            elif position_owner == "P2":
                # Check if Priority 1 wants to override/reverse
                if p1_sig != 0 and p1_sig != position:
                    # Priority 1 wants to reverse - exit P2 position
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_2_locked = True
                        last_signal_time = current_time
                        # Note: P1 will enter on next bar after 15s cooldown
                
                # Otherwise check P2 exit
                elif cooldown_ok and p2_sig == -position:
                    final_sig = p2_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
        
        final_signals[i] = final_sig
    
    return final_signals


# ============================================================================
# DAY PROCESSING
# ============================================================================

def extract_day_num(filepath):
    """Extract day number from filepath"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    """Process a single day with both strategies and manager logic"""
    try:
        # ====================================================================
        # 1. READ DATA
        # ====================================================================
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P1_FEATURE_COLUMN'],  # PB9_T1 for P1
            config['P2_BB4_COLUMN'],       # BB4_T9 for P2
            config['P2_BB6_COLUMN'],       # BB6_T11 for P2
            config['P2_VOLATILITY_FEATURE'],  # PB9_T1 for P2
            config['P2_V_PATTERN_FEATURE'],   # PB6_T4 for V-pattern detection
        ]
        
        # Remove duplicates (P1_FEATURE_COLUMN and P2_VOLATILITY_FEATURE are both PB9_T1)
        required_cols = list(dict.fromkeys(required_cols))
        
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # ====================================================================
        # 2. GENERATE PRIORITY 1 SIGNALS (Momentum/Volatility)
        # ====================================================================
        p1_signals = generate_p1_signals(df, config)
        
        # ====================================================================
        # 3. GENERATE PRIORITY 2 SIGNALS (BB Crossover with KAMA)
        # ====================================================================
        p2_signals = generate_p2_signals(df, config)
        
        # ====================================================================
        # 4. APPLY MANAGER LOGIC
        # ====================================================================
        final_signals = apply_manager_logic(df, p1_signals, p2_signals, config)
        
        # ====================================================================
        # 5. COUNT SIGNALS PER STRATEGY
        # ====================================================================
        p1_entries = np.sum(np.abs(p1_signals) == 1)
        p2_entries = np.sum(np.abs(p2_signals) == 1)
        final_entries = np.sum(np.abs(final_signals) == 1)
        
        # ====================================================================
        # 6. SAVE RESULT
        # ====================================================================
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': final_signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'p1_entries': p1_entries,
            'p2_entries': p2_entries,
            'final_entries': final_entries
        }
    
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main(config):
    start_time = time.time()
    
    print("="*80)
    print("STRATEGY MANAGER - PRIORITY-BASED SIGNAL ROUTING")
    print("="*80)
    print(f"\nPriority 1: Momentum/Volatility Strategy (PB9_T1 based)")
    print(f"  - Volatility Filter: ATR>{config['P1_ATR_THRESHOLD']}, StdDev>{config['P1_STDDEV_THRESHOLD']}")
    print(f"  - Take Profit: {config['P1_TAKE_PROFIT']} absolute")
    print(f"  - 🎯 PURE TRAILING STOP: Exit when PnL drops {config['P1_TRAILING_STOP_FROM_PEAK']} from peak (after TP hit)")
    print(f"  - Blocked Hours: {config['P1_BLOCKED_HOURS']} (no entries)")
    print(f"\nPriority 2: BB Crossover Strategy with KAMA Filter (BB4_T9/BB6_T11 based)")
    print(f"  - Activation: ATR>{config['P2_ACTIVATION_ATR_THRESHOLD']}, StdDev>{config['P2_ACTIVATION_STDDEV_THRESHOLD']}")
    print(f"  - KAMA Filter: Period={config['P2_KAMA_PERIOD']}, Fast={config['P2_KAMA_FAST']}, Slow={config['P2_KAMA_SLOW']}")
    print(f"  - KAMA Slope: Lookback={config['P2_KAMA_SLOPE_LOOKBACK']}, Threshold={config['P2_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Directional Confirmation: Long if slope>threshold, Short if slope<-threshold")
    print(f"  - V-Pattern Detection: Using {config['P2_V_PATTERN_FEATURE']} (3-second window)")
    print(f"    * Momentum Gate: ATR({config['P2_V_PATTERN_ATR_PERIOD']}) crossing >{config['P2_V_PATTERN_ATR_THRESHOLD']}")
    print(f"    * Only checks V-pattern when momentum condition met")
    print(f"  - Take Profit: {config['P2_TAKE_PROFIT_OFFSET']} absolute")
    print(f"\nUniversal Cooldown: {config['UNIVERSAL_COOLDOWN']}s between ANY signals")
    print(f"Daily Reset: Yes (priority lock resets each day)")
    print(f"EOD Handling: Unified manager-level square-off")
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
    total_p1_entries = 0
    total_p2_entries = 0
    total_final_entries = 0
    
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
                        total_p1_entries += result['p1_entries']
                        total_p2_entries += result['p2_entries']
                        total_final_entries += result['final_entries']
                        print(f"✓ Processed day{result['day_num']:3d} | P1: {result['p1_entries']:3d} | P2: {result['p2_entries']:3d} | Final: {result['final_entries']:3d}")
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
        print(f"\nPer-Strategy Breakdown (before manager):")
        print(f"P1 entries:        {total_p1_entries:,}")
        print(f"P2 entries:        {total_p2_entries:,}")
        print(f"Final entries:     {total_final_entries:,}")
        print(f"\nAvg per day:       {(num_long + num_short)/len(processed_files):.1f}")
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