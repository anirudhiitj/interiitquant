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
from pykalman import KalmanFilter

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
    'OUTPUT_FILE': 'managed_signals_3.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_3',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 5.0,  # seconds between ANY signals
    'MAX_WORKERS': 24,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Feature Selection (set to True to use Price instead of PB9_T1 for indicators)
    'USE_PRICE_FOR_INDICATORS': False,  # If True, uses Price; if False, uses PB9_T1
    
    # Strategy 1 (Kalman Filter) - Priority 1
    'P1_FEATURE_COLUMN': 'PB9_T1',  # Will be overridden if USE_PRICE_FOR_INDICATORS=True
    'P1_KF_FAST_TRANSITION_COV': 0.24,
    'P1_KF_FAST_OBSERVATION_COV': 0.7,
    'P1_KF_FAST_INITIAL_STATE_COV': 100,
    'P1_TAKE_PROFIT': 0.4,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_UCM_ALPHA': 0.1,
    'P1_UCM_BETA': 0.02,
    'P1_ENTRY_KAMA_SLOPE_THRESHOLD': 0.001,
    'P1_EXIT_KAMA_SLOPE_THRESHOLD': 0.0005,
    'P1_MIN_HOLD_SECONDS': 15.0,
    
    # Strategy 3 (BB Crossover with KAMA Filter and V-pattern) - Priority 2
    'P3_BB4_COLUMN': 'BB4_T9',
    'P3_BB6_COLUMN': 'BB6_T11',
    'P3_VOLATILITY_FEATURE': 'PB9_T1',  # Will be overridden if USE_PRICE_FOR_INDICATORS=True
    'P3_V_PATTERN_FEATURE': 'PB6_T4',
    'P3_V_PATTERN_ATR_PERIOD': 20,
    'P3_V_PATTERN_ATR_THRESHOLD': 0.006,
    'P3_KAMA_PERIOD': 15,
    'P3_KAMA_FAST': 30,
    'P3_KAMA_SLOW': 180,
    'P3_KAMA_SLOPE_LOOKBACK': 2,
    'P3_KAMA_SLOPE_THRESHOLD': 0.0008,
    'P3_MIN_HOLD_SECONDS': 15.0,
    'P3_MAX_HOLD_SECONDS': 3600*2,
    'P3_USE_TAKE_PROFIT': True,
    'P3_TAKE_PROFIT_OFFSET': 0.5,
    'P3_STRATEGY_START_BAR': 1800,
    
    # Strategy 4 (KAMA-based Simple) - Priority 3
    'P4_FEATURE_COLUMN': 'PB9_T1',  # Will be overridden if USE_PRICE_FOR_INDICATORS=True
    'P4_KAMA_WINDOW': 30,
    'P4_KAMA_FAST_PERIOD': 60,
    'P4_KAMA_SLOW_PERIOD': 300,
    'P4_KAMA_SLOPE_DIFF': 2,
    'P4_ATR_SPAN': 30,
    'P4_ATR_THRESHOLD': 0.01,
    'P4_STD_THRESHOLD': 0.06,
    'P4_KAMA_SLOPE_ENTRY': 0.0008,
    'P4_TAKE_PROFIT': 0.35,
    'P4_MIN_HOLD_SECONDS': 15.0,
}

def apply_feature_selection(config):
    """Override feature columns based on USE_PRICE_FOR_INDICATORS setting"""
    if config['USE_PRICE_FOR_INDICATORS']:
        config['P1_FEATURE_COLUMN'] = config['PRICE_COLUMN']
        config['P3_VOLATILITY_FEATURE'] = config['PRICE_COLUMN']
        config['P4_FEATURE_COLUMN'] = config['PRICE_COLUMN']
        print("\n🔧 Using PRICE for all indicator calculations")
    else:
        print("\n🔧 Using PB9_T1 for indicator calculations")
    return config

# Apply feature selection
CONFIG = apply_feature_selection(CONFIG)

# ============================================================================
# SHARED HELPER FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_true_range_from_pb9(pb9_t1_values):
    """Calculate True Range using PB9_T1 or Price (depending on config)"""
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
def detect_v_pattern(pb6_t4_values, v_pattern_atr_values, index, signal_direction, atr_threshold):
    """Detect V-pattern using PB6_T4 feature"""
    if index < 2:
        return False
    
    current_atr = v_pattern_atr_values[index]
    prev_atr = v_pattern_atr_values[index - 1]
    
    atr_crossing = (current_atr > atr_threshold) and (prev_atr <= atr_threshold)
    
    if not atr_crossing:
        return False
    
    v0 = pb6_t4_values[index]
    v1 = pb6_t4_values[index - 1]
    v2 = pb6_t4_values[index - 2]
    
    if signal_direction == -1:
        is_ascending = (v0 > v1) and (v1 > v2)
        return is_ascending
    
    elif signal_direction == 1:
        is_descending = (v0 < v1) and (v1 < v2)
        return is_descending
    
    return False


# ============================================================================
# STRATEGY 1 (KALMAN FILTER) - PRIORITY 1 FUNCTIONS
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare all derived features for Kalman Filter strategy (Priority 1)"""
    
    # KFTrendFast
    kf_fast = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_FAST_TRANSITION_COV'],
        observation_covariance=config['P1_KF_FAST_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_FAST_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf_fast.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)
    
    df["Take_Profit"] = config['P1_TAKE_PROFIT']
    
    # KAMA (primary)
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]
    
    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
    
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # UCM (Unobserved Components Model)
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError(f"{config['P1_FEATURE_COLUMN']} has no valid values")
    
    alpha = config['P1_UCM_ALPHA']
    beta = config['P1_UCM_BETA']
    n = len(series)
    
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
    
    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue
        
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]
        
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]
    
    df["UCM"] = filtered
    df["UCM_Lagged"] = df["UCM"].shift(1)
    
    return df


def generate_p1_signal_single(row, position, cooldown_over, config):
    """Generate signal for Kalman Filter strategy (Priority 1)"""
    signal = 0
    
    if cooldown_over:
        if position == 0:
            if row["KFTrendFast"] >= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = 1
            elif row["KFTrendFast"] <= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = -1
        elif position == 1:
            if (row["KFTrendFast_Lagged"] > row["UCM_Lagged"] and 
                row["KFTrendFast"] < row["UCM"] and 
                row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                signal = -1
        elif position == -1:
            if (row["KFTrendFast_Lagged"] < row["UCM_Lagged"] and 
                row["KFTrendFast"] > row["UCM"] and 
                row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                signal = 1
    
    return -1 * signal


def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 1 (Kalman Filter) signals"""
    # Prepare features
    df = prepare_p1_features(df, config)
    
    # Initialize state
    position = 0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = [0] * len(df)
    
    # Process row by row
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)
        
        signal = 0
        
        # EOD Square Off
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P1_MIN_HOLD_SECONDS']
            signal = generate_p1_signal_single(row, position, cooldown_over, config)
            
            # Entry Logic
            if position == 0 and signal != 0:
                entry_price = price
                take_profit = row["Take_Profit"]
            
            # Exit Logic (Price-based confirmation)
            elif cooldown_over and position != 0:
                if position == 1:
                    price_diff = price - entry_price
                    if price_diff >= take_profit:
                        signal = -1
                elif position == -1:
                    price_diff = entry_price - price
                    if price_diff >= take_profit:
                        signal = 1
        
        # Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0  # redundant signal
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:  # position closed
                    entry_price = None
        
        signals[i] = signal
    
    return np.array(signals, dtype=np.int8), df


# ============================================================================
# STRATEGY 3 (BB CROSSOVER WITH KAMA FILTER AND V-PATTERN) - PRIORITY 2 FUNCTIONS
# ============================================================================

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
    """Calculate Kaufman's Adaptive Moving Average (KAMA)"""
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
        if noise[i] > 1e-10:
            er = signal[i] / noise[i]
        else:
            er = 0.0
        
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
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
def generate_p3_signals_internal(
    prices,
    bb4_values,
    bb6_values,
    timestamps,
    kama_slope,
    pb6_t4_values,
    v_pattern_atr_values,
    kama_slope_threshold,
    v_pattern_atr_threshold,
    min_hold_seconds,
    max_hold_seconds,
    cooldown_seconds,
    use_take_profit,
    take_profit_offset,
    strategy_start_bar
):
    """Generate Priority 2 (BB Crossover with KAMA Filter) signals with immediate V-pattern reversal"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    current_position = 0
    entry_time = 0.0
    entry_price = 0.0
    last_exit_time = 0.0
    
    prev_bb4 = bb4_values[0]
    prev_bb6 = bb6_values[0]
    
    for i in range(1, n):
        current_time = timestamps[i]
        current_price = prices[i]
        current_bb4 = bb4_values[i]
        current_bb6 = bb6_values[i]
        current_kama_slope = kama_slope[i]
        
        can_trade = (i >= strategy_start_bar)
        
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        kama_bullish = current_kama_slope > kama_slope_threshold
        kama_bearish = current_kama_slope < -kama_slope_threshold
        
        is_last_bar = (i == n - 1)
        
        if current_position == 0:
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)
            
            if (can_trade and cooldown_elapsed and not is_last_bar):
                initial_signal = 0
                
                if bullish_bb_cross and kama_bullish:
                    initial_signal = 1
                elif bearish_bb_cross and kama_bearish:
                    initial_signal = -1
                
                # Apply V-pattern reversal if initial signal exists
                if initial_signal != 0:
                    v_pattern_detected = detect_v_pattern(
                        pb6_t4_values, v_pattern_atr_values, i, initial_signal, v_pattern_atr_threshold
                    )
                    
                    if v_pattern_detected:
                        # IMMEDIATE REVERSAL - flip the signal on same bar
                        final_signal = -initial_signal
                    else:
                        final_signal = initial_signal
                    
                    current_position = final_signal
                    entry_time = current_time
                    entry_price = current_price
                    signals[i] = final_signal
        
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


def generate_p3_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 2 (BB Crossover with KAMA Filter) signals with V-pattern reversal"""
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    bb4_values = df[config['P3_BB4_COLUMN']].values.astype(np.float64)
    bb6_values = df[config['P3_BB6_COLUMN']].values.astype(np.float64)
    pb9_t1_values = df[config['P3_VOLATILITY_FEATURE']].values.astype(np.float64)
    pb6_t4_values = df[config['P3_V_PATTERN_FEATURE']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    true_range = calculate_true_range_from_pb9(pb9_t1_values)
    v_pattern_atr_values = rolling_mean_numba(true_range, config['P3_V_PATTERN_ATR_PERIOD'])
    
    kama = calculate_kama(
        pb9_t1_values,
        config['P3_KAMA_PERIOD'],
        config['P3_KAMA_FAST'],
        config['P3_KAMA_SLOW']
    )
    kama_slope = calculate_kama_slope(kama, config['P3_KAMA_SLOPE_LOOKBACK'])
    
    signals = generate_p3_signals_internal(
        prices=prices,
        bb4_values=bb4_values,
        bb6_values=bb6_values,
        timestamps=timestamps,
        kama_slope=kama_slope,
        pb6_t4_values=pb6_t4_values,
        v_pattern_atr_values=v_pattern_atr_values,
        kama_slope_threshold=config['P3_KAMA_SLOPE_THRESHOLD'],
        v_pattern_atr_threshold=config['P3_V_PATTERN_ATR_THRESHOLD'],
        min_hold_seconds=config['P3_MIN_HOLD_SECONDS'],
        max_hold_seconds=config['P3_MAX_HOLD_SECONDS'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        use_take_profit=config['P3_USE_TAKE_PROFIT'],
        take_profit_offset=config['P3_TAKE_PROFIT_OFFSET'],
        strategy_start_bar=config['P3_STRATEGY_START_BAR']
    )
    
    return signals


# ============================================================================
# STRATEGY 4 (KAMA-BASED SIMPLE) - PRIORITY 3 FUNCTIONS
# ============================================================================

def prepare_p4_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare features for KAMA-based strategy (Priority 3)"""
    
    # KAMA calculation
    price = df[config['P4_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P4_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (config['P4_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P4_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["P4_KAMA"] = kama
    df["P4_KAMA_Slope"] = df["P4_KAMA"].diff(config['P4_KAMA_SLOPE_DIFF']).shift(1)
    df["P4_KAMA_Slope_abs"] = df["P4_KAMA_Slope"].abs()

    # ATR calculation
    tr = df[config['P4_FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P4_ATR_SPAN'], adjust=False).mean()
    df["P4_ATR"] = atr.shift(1)
    df["P4_ATR_High"] = df["P4_ATR"].expanding(min_periods=1).max()

    # Standard Deviation
    std = df[config['P4_FEATURE_COLUMN']].rolling(window=20, min_periods=1).std()
    df["P4_STD"] = std.shift(1)
    df["P4_STD_High"] = df["P4_STD"].expanding(min_periods=1).max()
    
    return df


def generate_p4_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 3 (KAMA-based Simple) signals"""
    # Prepare features
    df = prepare_p4_features(df, config)
    
    # Initialize state
    position = 0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = [0] * len(df)
    
    # Process row by row
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)
        
        signal = 0
        
        # EOD Square Off
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P4_MIN_HOLD_SECONDS']
            
            if cooldown_over:
                # Volatility filter
                volatility_confirmed = (row["P4_ATR_High"] > config['P4_ATR_THRESHOLD'] and 
                                       row["P4_STD_High"] > config['P4_STD_THRESHOLD'])
                
                # Entry conditions
                if position == 0 and volatility_confirmed:
                    if row["P4_KAMA_Slope"] > config['P4_KAMA_SLOPE_ENTRY']:
                        signal = 1
                    elif row["P4_KAMA_Slope"] < -config['P4_KAMA_SLOPE_ENTRY']:
                        signal = -1
                
                # Exit conditions (price-based)
                elif position != 0:
                    if position == 1:
                        price_diff = price - entry_price
                        if price_diff >= config['P4_TAKE_PROFIT']:
                            signal = -1
                    elif position == -1:
                        price_diff = entry_price - price
                        if price_diff >= config['P4_TAKE_PROFIT']:
                            signal = 1
        
        # Apply Signal
        if signal != 0:
            if position == 0 and signal != 0:
                entry_price = price
            
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0  # redundant signal
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:  # position closed
                    entry_price = None
        
        signals[i] = signal
    
    return np.array(signals, dtype=np.int8), df


# ============================================================================
# MANAGER LOGIC (3 PRIORITIES)
# ============================================================================

def apply_manager_logic(df: pd.DataFrame, p1_signals: np.ndarray, 
                       p3_signals: np.ndarray, p4_signals: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply manager logic to combine Priority 1, 2 (P3), and 3 (P4) signals
    
    Rules:
    1. Daily reset
    2. Once P1 trades -> P3, P4 locked for entire day
    3. Once P3 trades (if P1 hasn't) -> P4 locked for entire day
    4. P4 can only trade if P1, P3 haven't traded
    5. 15s universal cooldown between ANY signals
    6. Priority override: Higher priority can force exit of lower priority
    7. Unified EOD square-off
    """
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    
    # Manager state
    position = 0
    position_owner = None  # None, "P1", "P3", "P4"
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    priority_3_locked = False  # P3 locked
    priority_4_locked = False  # P4 locked
    
    for i in range(n):
        current_time = df.iloc[i]['Time_sec']
        p1_sig = p1_signals[i]
        p3_sig = p3_signals[i]
        p4_sig = p4_signals[i]
        is_last_bar = (i == n - 1)
        
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
                # Priority 1 entry (highest priority)
                if p1_sig != 0:
                    final_sig = p1_sig
                    position = p1_sig
                    position_owner = "P1"
                    priority_3_locked = True
                    priority_4_locked = True
                    last_signal_time = current_time
                
                # Priority 2 (P3) entry (only if P1 hasn't locked it)
                elif p3_sig != 0 and not priority_3_locked:
                    final_sig = p3_sig
                    position = p3_sig
                    position_owner = "P3"
                    priority_4_locked = True
                    last_signal_time = current_time
                
                # Priority 3 (P4) entry (only if P1, P3 haven't locked it)
                elif p4_sig != 0 and not priority_4_locked:
                    final_sig = p4_sig
                    position = p4_sig
                    position_owner = "P4"
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
            
            elif position_owner == "P3":
                # Check if Priority 1 wants to override/reverse
                if p1_sig != 0 and p1_sig != position:
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_3_locked = True
                        priority_4_locked = True
                        last_signal_time = current_time
                
                # Otherwise check P3 exit
                elif cooldown_ok and p3_sig == -position:
                    final_sig = p3_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
            
            elif position_owner == "P4":
                # Check if Priority 1 wants to override/reverse
                if p1_sig != 0 and p1_sig != position:
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_3_locked = True
                        priority_4_locked = True
                        last_signal_time = current_time
                
                # Check if Priority 2 (P3) wants to override/reverse
                elif p3_sig != 0 and p3_sig != position:
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_4_locked = True
                        last_signal_time = current_time
                
                # Otherwise check P4 exit
                elif cooldown_ok and p4_sig == -position:
                    final_sig = p4_sig
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
    """Process a single day with all three strategies and manager logic"""
    try:
        # ====================================================================
        # 1. READ DATA
        # ====================================================================
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P3_BB4_COLUMN'],
            config['P3_BB6_COLUMN'],
            config['P3_V_PATTERN_FEATURE'],
        ]
        
        # Add feature columns only if not using Price
        if not config['USE_PRICE_FOR_INDICATORS']:
            required_cols.extend([
                'PB9_T1',  # Used by P1, P3, P4 if USE_PRICE_FOR_INDICATORS=False
            ])
        
        # Remove duplicates
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
        # 2. GENERATE PRIORITY 1 SIGNALS (Kalman Filter)
        # ====================================================================
        p1_signals, df_with_features = generate_p1_signals(df, config)
        
        # ====================================================================
        # 3. GENERATE PRIORITY 2 SIGNALS (BB Crossover - formerly P3)
        # ====================================================================
        p3_signals = generate_p3_signals(df, config)
        
        # ====================================================================
        # 4. GENERATE PRIORITY 3 SIGNALS (KAMA-based - formerly P4)
        # ====================================================================
        p4_signals, df_with_p4_features = generate_p4_signals(df, config)
        
        # ====================================================================
        # 5. APPLY MANAGER LOGIC
        # ====================================================================
        final_signals = apply_manager_logic(df, p1_signals, p3_signals, p4_signals, config)
        
        # ====================================================================
        # 6. COUNT SIGNALS PER STRATEGY
        # ====================================================================
        p1_entries = np.sum(np.abs(p1_signals) == 1)
        p3_entries = np.sum(np.abs(p3_signals) == 1)
        p4_entries = np.sum(np.abs(p4_signals) == 1)
        final_entries = np.sum(np.abs(final_signals) == 1)
        
        # ====================================================================
        # 7. SAVE RESULT
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
            'p3_entries': p3_entries,
            'p4_entries': p4_entries,
            'final_entries': final_entries
        }
    
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN
# ============================================================================

def main(config):
    start_time = time.time()
    
    feature_type = "PRICE" if config['USE_PRICE_FOR_INDICATORS'] else "PB9_T1"
    
    print("="*80)
    print("STRATEGY MANAGER - 3-PRIORITY SIGNAL ROUTING")
    print("="*80)
    print(f"\n📊 INDICATOR FEATURE: {feature_type}")
    print(f"\nPriority 1: Kalman Filter Strategy ({config['P1_FEATURE_COLUMN']} based)")
    print(f"  - KFTrendFast vs UCM crossovers with KAMA_Slope_abs filter")
    print(f"  - Entry threshold: KAMA_Slope_abs > {config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Exit threshold: KAMA_Slope_abs > {config['P1_EXIT_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Take Profit: {config['P1_TAKE_PROFIT']} absolute")
    print(f"  - Min Hold: {config['P1_MIN_HOLD_SECONDS']}s")
    print(f"\nPriority 2: BB Crossover Strategy with KAMA Filter and V-Pattern ({config['P3_BB4_COLUMN']}/{config['P3_BB6_COLUMN']} based)")
    print(f"  - Volatility Feature: {config['P3_VOLATILITY_FEATURE']}")
    print(f"  - Starts trading from bar: {config['P3_STRATEGY_START_BAR']}")
    print(f"  - KAMA Filter: Period={config['P3_KAMA_PERIOD']}, Fast={config['P3_KAMA_FAST']}, Slow={config['P3_KAMA_SLOW']}")
    print(f"  - KAMA Slope: Lookback={config['P3_KAMA_SLOPE_LOOKBACK']}, Threshold={config['P3_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - 🔄 V-PATTERN REVERSAL: Using {config['P3_V_PATTERN_FEATURE']}, ATR threshold={config['P3_V_PATTERN_ATR_THRESHOLD']}")
    print(f"  - Take Profit: {config['P3_TAKE_PROFIT_OFFSET']} absolute")
    print(f"\nPriority 3: KAMA-based Simple Strategy ({config['P4_FEATURE_COLUMN']} based)")
    print(f"  - KAMA Slope Entry Threshold: {config['P4_KAMA_SLOPE_ENTRY']}")
    print(f"  - Volatility Filter: ATR>{config['P4_ATR_THRESHOLD']}, StdDev>{config['P4_STD_THRESHOLD']}")
    print(f"  - Take Profit: {config['P4_TAKE_PROFIT']} absolute")
    print(f"  - Min Hold: {config['P4_MIN_HOLD_SECONDS']}s")
    print(f"\nUniversal Cooldown: {config['UNIVERSAL_COOLDOWN']}s between ANY signals")
    print(f"Priority Locking: P1→P3+P4 | P3→P4")
    print(f"Daily Reset: Yes (priority locks reset each day)")
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
    total_p3_entries = 0
    total_p4_entries = 0
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
                        total_p3_entries += result['p3_entries']
                        total_p4_entries += result['p4_entries']
                        total_final_entries += result['final_entries']
                        print(f"✓ Processed day{result['day_num']:3d} | P1: {result['p1_entries']:3d} | P3: {result['p3_entries']:3d} | P4: {result['p4_entries']:3d} | Final: {result['final_entries']:3d}")
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
        print(f"Feature Used:      {feature_type}")
        print(f"Total bars:        {total_bars:,}")
        print(f"Long entries:      {num_long:,}")
        print(f"Short entries:     {num_short:,}")
        print(f"Hold/Wait:         {num_hold:,}")
        print(f"Total entries:     {(num_long + num_short):,}")
        print(f"\nPer-Strategy Breakdown (before manager):")
        print(f"P1 entries:        {total_p1_entries:,}")
        print(f"P3 entries:        {total_p3_entries:,}")
        print(f"P4 entries:        {total_p4_entries:,}")
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