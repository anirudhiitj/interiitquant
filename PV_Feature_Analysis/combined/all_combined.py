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
# MASTER CONFIGURATION - ALL STRATEGIES
# ============================================================================

CONFIG = {
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'combined_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_4priority',
    
    # ========================================================================
    # MANAGER SETTINGS
    # ========================================================================
    'UNIVERSAL_COOLDOWN': 15.0,  # seconds between ANY signals
    'MAX_WORKERS': 24,
    
    # ========================================================================
    # REQUIRED COLUMNS
    # ========================================================================
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # ========================================================================
    # PRIORITY 1: KALMAN/UCM STRATEGY
    # ========================================================================
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_MIN_HOLD_SECONDS': 15.0,
    'P1_TAKE_PROFIT': 0.35,
    'P1_BLOCKED_HOURS': [],  # No blocked hours
    
    # Kalman Filter - Slow Trend
    'P1_KF_TRANSITION_COV': 0.01,
    'P1_KF_OBSERVATION_COV': 0.7,
    'P1_KF_INITIAL_STATE_COV': 100,
    'P1_KF_INITIAL_STATE_MEAN': 100,
    
    # UCM (Unobserved Components Model)
    'P1_UCM_ALPHA': 0.1,
    'P1_UCM_BETA': 0.02,
    
    # KAMA for slope confirmation
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST': 60,
    'P1_KAMA_SLOW': 300,
    'P1_KAMA_SLOPE_LOOKBACK': 2,
    'P1_KAMA_SLOPE_THRESHOLD': 0.001,
    
    # Volatility filters
    'P1_ATR_PERIOD': 30,
    'P1_ATR_THRESHOLD': 0.01,
    'P1_STDDEV_PERIOD': 20,
    'P1_STDDEV_THRESHOLD': 0.05,
    
    # ========================================================================
    # PRIORITY 2: MOMENTUM/VOLATILITY STRATEGY
    # ========================================================================
    'P2_FEATURE_COLUMN': 'PB9_T1',
    'P2_ATR_PERIOD': 25,
    'P2_STDDEV_PERIOD': 20,
    'P2_ATR_THRESHOLD': 0.01,
    'P2_STDDEV_THRESHOLD': 0.06,
    'P2_MOMENTUM_PERIOD': 10,
    'P2_VOLATILITY_PERIOD': 20,
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_TAKE_PROFIT': 0.155,
    'P2_BLOCKED_HOURS': [5],  # No entries in hour 5 (05:00-06:00)
    'P2_MOMENTUM_THRESHOLD': 0.1,  # Threshold for momentum/volatility ratio
    
    # ========================================================================
    # PRIORITY 3: BB CROSSOVER WITH KAMA FILTER
    # ========================================================================
    'P3_BB4_COLUMN': 'BB4_T9',
    'P3_BB6_COLUMN': 'BB6_T11',
    'P3_VOLATILITY_FEATURE': 'PB9_T1',
    'P3_ATR_PERIOD': 30,
    'P3_ACTIVATION_ATR_THRESHOLD': 0.01,
    'P3_ACTIVATION_STDDEV_THRESHOLD': 0.05,
    
    # KAMA Filter Parameters
    'P3_KAMA_PERIOD': 15,
    'P3_KAMA_FAST': 30,
    'P3_KAMA_SLOW': 180,
    'P3_KAMA_SLOPE_LOOKBACK': 2,
    'P3_KAMA_SLOPE_THRESHOLD': 0.0005,
    
    'P3_MIN_HOLD_SECONDS': 15.0,
    'P3_MAX_HOLD_SECONDS': 3600*2.5,
    'P3_USE_TAKE_PROFIT': True,
    'P3_TAKE_PROFIT_OFFSET': 0.8,
    'P3_STRATEGY_START_BAR': 2400,
    'P3_BLOCKED_HOURS': [],  # No blocked hours
    
    # ========================================================================
    # PRIORITY 4: KAMA SLOPE STRATEGY
    # ========================================================================
    'P4_FEATURE_COLUMN': 'PB9_T1',
    'P4_MIN_HOLD_SECONDS': 15.0,
    'P4_TAKE_PROFIT': 0.35,
    'P4_BLOCKED_HOURS': [],  # No blocked hours
    
    # KAMA Parameters
    'P4_KAMA_WINDOW': 30,
    'P4_KAMA_FAST': 60,
    'P4_KAMA_SLOW': 300,
    'P4_KAMA_SLOPE_LOOKBACK': 2,
    'P4_KAMA_SLOPE_ENTRY': 0.0007,
    
    # Volatility filters
    'P4_ATR_PERIOD': 30,
    'P4_ATR_THRESHOLD': 0.01,
    'P4_STDDEV_PERIOD': 20,
    'P4_STDDEV_THRESHOLD': 0.06,
}

# ============================================================================
# PRIORITY 1 (KALMAN/UCM) - FEATURE PREPARATION
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare features for Priority 1 (Kalman/UCM) strategy.
    All features use proper lagging to avoid lookahead bias.
    """
    # Kalman Filter - Slow Trend
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_TRANSITION_COV'],
        observation_covariance=config['P1_KF_OBSERVATION_COV'],
        initial_state_mean=config['P1_KF_INITIAL_STATE_MEAN'],
        initial_state_covariance=config['P1_KF_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), 
                               index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    df["KFTrendSlow"] = smoothed_prices
    df["KFTrendSlow_Lagged"] = df["KFTrendSlow"].shift(1)
    
    # UCM (Unobserved Components Model)
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("P1_FEATURE_COLUMN has no valid values")
    
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
    
    # KAMA for slope confirmation
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW'] + 1)
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
    
    df["P1_KAMA"] = kama
    df["P1_KAMA_Slope"] = df["P1_KAMA"].diff(config['P1_KAMA_SLOPE_LOOKBACK'])
    df["P1_KAMA_Slope_abs"] = df["P1_KAMA_Slope"].abs()
    
    # ATR and StdDev for volatility filters
    tr = df[config['P1_FEATURE_COLUMN']].diff().abs()
    df["P1_ATR"] = tr.ewm(span=config['P1_ATR_PERIOD']).mean()
    df["P1_ATR_High"] = df["P1_ATR"].expanding(min_periods=1).max()
    
    pb9_ma = df[config['P1_FEATURE_COLUMN']].rolling(window=config['P1_STDDEV_PERIOD'], min_periods=1).mean()
    df["P1_STD"] = pb9_ma.rolling(window=config['P1_STDDEV_PERIOD'], min_periods=1).std()
    df["P1_STD_High"] = df["P1_STD"].expanding(min_periods=1).max()
    
    return df


# ============================================================================
# PRIORITY 2 (MOMENTUM/VOLATILITY) - NUMBA FUNCTIONS
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


# ============================================================================
# PRIORITY 3 (BB CROSSOVER) - NUMBA FUNCTIONS
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
def calculate_kama_p3(prices, period, fast_period, slow_period):
    """Calculate KAMA for Priority 3"""
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


# ============================================================================
# PRIORITY 4 (KAMA SLOPE) - FEATURE PREPARATION
# ============================================================================

def prepare_p4_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare features for Priority 4 (KAMA Slope) strategy.
    All features use proper shift() to avoid lookahead bias.
    """
    # KAMA Calculation
    price = df[config['P4_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P4_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (config['P4_KAMA_FAST'] + 1)
    sc_slow = 2 / (config['P4_KAMA_SLOW'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found for P4.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["P4_KAMA"] = kama
    df["P4_KAMA_Slope"] = df["P4_KAMA"].diff(config['P4_KAMA_SLOPE_LOOKBACK']).shift(1)
    df["P4_KAMA_Slope_abs"] = df["P4_KAMA_Slope"].abs()

    # ATR Calculation
    tr = df[config['P4_FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P4_ATR_PERIOD'], adjust=False).mean()
    df["P4_ATR"] = atr.shift(1)
    df["P4_ATR_High"] = df["P4_ATR"].expanding(min_periods=1).max()

    # Standard Deviation
    std = df[config['P4_FEATURE_COLUMN']].rolling(window=config['P4_STDDEV_PERIOD'], min_periods=1).std()
    df["P4_STD"] = std.shift(1)
    df["P4_STD_High"] = df["P4_STD"].expanding(min_periods=1).max()
    
    return df


# ============================================================================
# STRATEGY SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_p1_signal(row, position, cooldown_over, config):
    """
    Priority 1 (Kalman/UCM) signal generation.
    Entry: UCM vs KFTrendSlow crossover + volatility filter
    Exit: Opposite crossover + KAMA slope filter
    """
    signal = 0
    
    if cooldown_over:
        volatility_confirmed = (row["P1_ATR_High"] > config['P1_ATR_THRESHOLD'] and 
                               row["P1_STD_High"] > config['P1_STDDEV_THRESHOLD'])
        
        if position == 0 and volatility_confirmed:
            # Entry signals
            if (row["UCM"] >= row["KFTrendSlow"] and 
                row["P1_KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_THRESHOLD']):
                signal = 1
            elif (row["UCM"] <= row["KFTrendSlow"] and 
                  row["P1_KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_THRESHOLD']):
                signal = -1
        
        elif position == 1:
            # Exit long on bearish crossover
            if ((row["UCM_Lagged"] > row["KFTrendSlow_Lagged"] and 
                 row["UCM"] < row["KFTrendSlow"]) and 
                row["P1_KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_THRESHOLD'] * 0.7):
                signal = -1
        
        elif position == -1:
            # Exit short on bullish crossover
            if ((row["UCM_Lagged"] < row["KFTrendSlow_Lagged"] and 
                 row["UCM"] > row["KFTrendSlow"]) and 
                row["P1_KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_THRESHOLD'] * 0.7):
                signal = 1
    
    return signal


@jit(nopython=True, fastmath=True)
def generate_p2_signal_single(atr, stddev, momentum, volatility, atr_thresh, 
                               stddev_thresh, momentum_thresh):
    """
    Priority 2 (Momentum/Volatility) signal generation - single bar.
    Entry based on momentum/volatility ratio with volatility filter.
    """
    if atr < atr_thresh or stddev < stddev_thresh:
        return 0
    
    if abs(volatility) < 1e-10:
        return 0
    
    momentum_vol_ratio = momentum / max(abs(volatility), 1e-10)
    
    if momentum_vol_ratio > momentum_thresh:
        return 1
    elif momentum_vol_ratio < -momentum_thresh:
        return -1
    else:
        return 0


def generate_p3_signal(row, prev_row, position, strategy_activated, 
                       current_bar_idx, config):
    """
    Priority 3 (BB Crossover with KAMA) signal generation.
    Entry: BB crossover + KAMA directional filter
    Exit: Opposite crossover
    """
    signal = 0
    
    # Check if strategy is activated
    if not strategy_activated:
        if (row["P3_ATR"] > config['P3_ACTIVATION_ATR_THRESHOLD'] and 
            row["P3_PB9_STDDEV"] > config['P3_ACTIVATION_STDDEV_THRESHOLD']):
            strategy_activated = True
    
    can_trade = strategy_activated and (current_bar_idx >= config['P3_STRATEGY_START_BAR'])
    
    if not can_trade:
        return signal, strategy_activated
    
    # Detect crossovers
    bullish_bb_cross = (prev_row["P3_BB4"] <= prev_row["P3_BB6"] and 
                        row["P3_BB4"] > row["P3_BB6"])
    bearish_bb_cross = (prev_row["P3_BB4"] >= prev_row["P3_BB6"] and 
                        row["P3_BB4"] < row["P3_BB6"])
    
    # KAMA directional filter
    kama_bullish = row["P3_KAMA_Slope"] > config['P3_KAMA_SLOPE_THRESHOLD']
    kama_bearish = row["P3_KAMA_Slope"] < -config['P3_KAMA_SLOPE_THRESHOLD']
    
    if position == 0:
        if bullish_bb_cross and kama_bullish:
            signal = 1
        elif bearish_bb_cross and kama_bearish:
            signal = -1
    
    elif position == 1:
        if bearish_bb_cross:
            signal = -1
    
    elif position == -1:
        if bullish_bb_cross:
            signal = 1
    
    return signal, strategy_activated


def generate_p4_signal(row, position, cooldown_over, config):
    """
    Priority 4 (KAMA Slope) signal generation.
    Entry: KAMA slope direction + ATR/StdDev thresholds
    """
    signal = 0
    
    if cooldown_over:
        volatility_confirmed = (row["P4_ATR_High"] > config['P4_ATR_THRESHOLD'] and 
                               row["P4_STD_High"] > config['P4_STDDEV_THRESHOLD'])
        
        if position == 0 and volatility_confirmed:
            if row["P4_KAMA_Slope"] > config['P4_KAMA_SLOPE_ENTRY']:
                signal = 1
            elif row["P4_KAMA_Slope"] < -config['P4_KAMA_SLOPE_ENTRY']:
                signal = -1
    
    return signal


# ============================================================================
# MANAGER LOGIC
# ============================================================================

def apply_manager_logic(df: pd.DataFrame, config: dict) -> tuple:
    """
    Apply 4-priority manager logic.
    
    Priority hierarchy:
    P1 trades → locks P2, P3, P4 for day
    P2 trades → locks P3, P4 for day
    P3 trades → locks P4 for day
    P4 trades → no locks
    
    Override rule: Higher priority can override lower priority position
    - Exit lower priority position
    - Wait 15s cooldown
    - Enter higher priority position
    """
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    
    # Manager state
    position = 0
    position_owner = None  # "P1", "P2", "P3", "P4"
    entry_price = 0.0
    entry_time = 0.0
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    
    # Priority locks (reset daily)
    p2_locked = False
    p3_locked = False
    p4_locked = False
    
    # P2 state (for blocked hours check)
    p2_atr = None
    p2_stddev = None
    p2_momentum = None
    p2_volatility = None
    
    # P3 state (for activation and crossover detection)
    p3_strategy_activated = False
    p3_prev_bb4 = df.iloc[0]["P3_BB4"] if "P3_BB4" in df.columns else 0
    p3_prev_bb6 = df.iloc[0]["P3_BB6"] if "P3_BB6" in df.columns else 0
    
    for i in range(n):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_bar = (i == n - 1)
        
        # Calculate current hour for blocked hours check
        current_hour = int(current_time // 3600)
        
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
        # GENERATE INDIVIDUAL STRATEGY SIGNALS
        # ====================================================================
        else:
            # Priority 1 signal
            p1_sig = generate_p1_signal(row, position if position_owner == "P1" else 0, 
                                       cooldown_ok, config)
            
            # Priority 2 signal (check blocked hours)
            p2_sig = 0
            p2_hour_blocked = current_hour in config['P2_BLOCKED_HOURS']
            if not p2_hour_blocked and not p2_locked:
                # Prepare P2 indicators if not already done
                if p2_atr is None:
                    pb9_values = df[config['P2_FEATURE_COLUMN']].values.astype(np.float64)
                    p2_atr = compute_atr_fast(pb9_values, config['P2_ATR_PERIOD'])
                    p2_stddev = compute_stddev_fast(pb9_values, config['P2_STDDEV_PERIOD'])
                    p2_momentum = compute_momentum_fast(pb9_values, config['P2_MOMENTUM_PERIOD'])
                    p2_volatility = compute_volatility_ratio(pb9_values, config['P2_VOLATILITY_PERIOD'])
                
                p2_sig = generate_p2_signal_single(
                    p2_atr[i], p2_stddev[i], p2_momentum[i], p2_volatility[i],
                    config['P2_ATR_THRESHOLD'], config['P2_STDDEV_THRESHOLD'],
                    config['P2_MOMENTUM_THRESHOLD']
                )
                
                # P2 only generates entry signals when flat
                if position_owner == "P2":
                    # P2 exits only via take profit (handled later)
                    p2_sig = 0
                elif position != 0:
                    p2_sig = 0
            
            # Priority 3 signal
            p3_sig = 0
            if not p3_locked:
                prev_row_data = {
                    "P3_BB4": p3_prev_bb4,
                    "P3_BB6": p3_prev_bb6
                }
                p3_sig, p3_strategy_activated = generate_p3_signal(
                    row, prev_row_data, position if position_owner == "P3" else 0,
                    p3_strategy_activated, i, config
                )
                p3_prev_bb4 = row["P3_BB4"]
                p3_prev_bb6 = row["P3_BB6"]
            
            # Priority 4 signal
            p4_sig = 0
            if not p4_locked:
                p4_sig = generate_p4_signal(row, position if position_owner == "P4" else 0, 
                                           cooldown_ok, config)
            
            # ================================================================
            # PRIORITY ROUTING
            # ================================================================
            
            if position == 0:
                # FLAT - check priorities in order
                if cooldown_ok:
                    if p1_sig != 0:
                        final_sig = p1_sig
                        position = p1_sig
                        position_owner = "P1"
                        entry_price = price
                        entry_time = current_time
                        last_signal_time = current_time
                        # Lock all lower priorities
                        p2_locked = p3_locked = p4_locked = True
                    
                    elif p2_sig != 0 and not p2_locked:
                        final_sig = p2_sig
                        position = p2_sig
                        position_owner = "P2"
                        entry_price = price
                        entry_time = current_time
                        last_signal_time = current_time
                        # Lock lower priorities
                        p3_locked = p4_locked = True
                    
                    elif p3_sig != 0 and not p3_locked:
                        final_sig = p3_sig
                        position = p3_sig
                        position_owner = "P3"
                        entry_price = price
                        entry_time = current_time
                        last_signal_time = current_time
                        # Lock lower priority
                        p4_locked = True
                    
                    elif p4_sig != 0 and not p4_locked:
                        final_sig = p4_sig
                        position = p4_sig
                        position_owner = "P4"
                        entry_price = price
                        entry_time = current_time
                        last_signal_time = current_time
            
            else:
                # IN POSITION - check for exits or overrides
                holding_time = current_time - entry_time
                
                # Check for higher priority override
                override_happened = False
                
                if position_owner == "P4" and p3_sig != 0 and p3_sig != position:
                    # P3 wants to override P4
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        last_signal_time = current_time
                        override_happened = True
                        p4_locked = True
                
                elif position_owner in ["P3", "P4"] and p2_sig != 0 and p2_sig != position:
                    # P2 wants to override P3/P4
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        last_signal_time = current_time
                        override_happened = True
                        p3_locked = p4_locked = True
                
                elif position_owner in ["P2", "P3", "P4"] and p1_sig != 0 and p1_sig != position:
                    # P1 wants to override P2/P3/P4
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        last_signal_time = current_time
                        override_happened = True
                        p2_locked = p3_locked = p4_locked = True
                
                # If no override, check strategy-specific exits
                if not override_happened and position != 0:
                    # Check take profit conditions
                    pnl = (price - entry_price) * position
                    
                    # Check minimum hold time
                    min_hold_map = {
                        "P1": config['P1_MIN_HOLD_SECONDS'],
                        "P2": config['P2_MIN_HOLD_SECONDS'],
                        "P3": config['P3_MIN_HOLD_SECONDS'],
                        "P4": config['P4_MIN_HOLD_SECONDS']
                    }
                    min_hold_ok = holding_time >= min_hold_map.get(position_owner, 15.0)
                    
                    if min_hold_ok and cooldown_ok:
                        # P1 exits
                        if position_owner == "P1":
                            if p1_sig == -position:
                                final_sig = p1_sig
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                            elif pnl >= config['P1_TAKE_PROFIT']:
                                final_sig = -position
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                        
                        # P2 exits (take profit only)
                        elif position_owner == "P2":
                            if pnl >= config['P2_TAKE_PROFIT']:
                                final_sig = -position
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                        
                        # P3 exits
                        elif position_owner == "P3":
                            if p3_sig == -position:
                                final_sig = p3_sig
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                            elif config['P3_USE_TAKE_PROFIT'] and pnl >= config['P3_TAKE_PROFIT_OFFSET']:
                                final_sig = -position
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                            elif config['P3_MAX_HOLD_SECONDS'] > 0 and holding_time >= config['P3_MAX_HOLD_SECONDS']:
                                final_sig = -position
                                position = 0
                                position_owner = None
                                last_signal_time = current_time
                        
                        # P4 exits (take profit only)
                        elif position_owner == "P4":
                            if pnl >= config['P4_TAKE_PROFIT']:
                                final_sig = -position
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


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> str:
    """Process a single day with all 4 strategies and manager logic"""
    try:
        # ====================================================================
        # 1. READ DATA
        # ====================================================================
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P1_FEATURE_COLUMN'],
            config['P2_FEATURE_COLUMN'],
            config['P3_BB4_COLUMN'],
            config['P3_BB6_COLUMN'],
            config['P3_VOLATILITY_FEATURE'],
            config['P4_FEATURE_COLUMN'],
        ]
        
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
        # 2. PREPARE FEATURES FOR ALL STRATEGIES
        # ====================================================================
        df = prepare_p1_features(df, config)
        df = prepare_p4_features(df, config)
        
        # P3 features
        pb9_t1_values = df[config['P3_VOLATILITY_FEATURE']].values.astype(np.float64)
        bb4_values = df[config['P3_BB4_COLUMN']].values.astype(np.float64)
        bb6_values = df[config['P3_BB6_COLUMN']].values.astype(np.float64)
        
        true_range = calculate_true_range_from_pb9(pb9_t1_values)
        df["P3_ATR"] = rolling_mean_numba(true_range, config['P3_ATR_PERIOD'])
        df["P3_PB9_STDDEV"] = rolling_std_numba(pb9_t1_values, config['P3_ATR_PERIOD'])
        df["P3_BB4"] = bb4_values
        df["P3_BB6"] = bb6_values
        
        kama_p3 = calculate_kama_p3(
            pb9_t1_values,
            config['P3_KAMA_PERIOD'],
            config['P3_KAMA_FAST'],
            config['P3_KAMA_SLOW']
        )
        df["P3_KAMA_Slope"] = calculate_kama_slope(kama_p3, config['P3_KAMA_SLOPE_LOOKBACK'])
        
        # ====================================================================
        # 3. APPLY MANAGER LOGIC
        # ====================================================================
        final_signals = apply_manager_logic(df, config)
        
        # ====================================================================
        # 4. SAVE RESULT
        # ====================================================================
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
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN
# ============================================================================

def main(config):
    start_time = time.time()
    
    print("="*80)
    print("4-PRIORITY STRATEGY MANAGER")
    print("="*80)
    print(f"\nPriority 1: Kalman/UCM Strategy")
    print(f"  - Entry: UCM/KFTrendSlow crossover + volatility")
    print(f"  - Exit: Opposite crossover + KAMA slope OR TP ({config['P1_TAKE_PROFIT']})")
    print(f"  - Locks: P2, P3, P4 for rest of day")
    
    print(f"\nPriority 2: Momentum/Volatility Strategy")
    print(f"  - Entry: Momentum/Vol ratio + ATR/StdDev filter")
    print(f"  - Exit: TP only ({config['P2_TAKE_PROFIT']})")
    print(f"  - Blocked Hours: {config['P2_BLOCKED_HOURS']}")
    print(f"  - Locks: P3, P4 for rest of day")
    
    print(f"\nPriority 3: BB Crossover + KAMA Filter")
    print(f"  - Entry: BB4/BB6 crossover + KAMA directional")
    print(f"  - Exit: Opposite crossover OR TP ({config['P3_TAKE_PROFIT_OFFSET']})")
    print(f"  - Locks: P4 for rest of day")
    
    print(f"\nPriority 4: KAMA Slope Strategy")
    print(f"  - Entry: KAMA slope + ATR/StdDev filter")
    print(f"  - Exit: TP only ({config['P4_TAKE_PROFIT']})")
    print(f"  - Locks: None")
    
    print(f"\nUniversal Cooldown: {config['UNIVERSAL_COOLDOWN']}s")
    print(f"Daily Reset: Yes (all locks reset each day)")
    print(f"Override: Higher priority can override lower (exit → 15s → entry)")
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