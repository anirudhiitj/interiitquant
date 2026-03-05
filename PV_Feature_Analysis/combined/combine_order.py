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
from statsmodels.tsa.seasonal import STL
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
    'OUTPUT_FILE': 'managed_signals_order.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_window',
    
    # Manager Settings
    'WINDOW_SIZE_SECONDS': 3600,  # 30 minutes
    'UNIVERSAL_COOLDOWN': 15.0,  # seconds between ANY signals
    'MAX_WORKERS': 12,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy 1 (Kalman Filter) - Priority 1
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_KF_TREND_TRANSITION_COV': 0.01,
    'P1_KF_TREND_OBSERVATION_COV': 1.0,
    'P1_KF_TREND_INITIAL_STATE_COV': 1.0,
    'P1_KF_TREND_INITIAL_STATE_MEAN': 100,
    'P1_KF_FAST_TRANSITION_COV': 0.24,
    'P1_KF_FAST_OBSERVATION_COV': 0.7,
    'P1_KF_FAST_INITIAL_STATE_COV': 100,
    'P1_KF_SLOW_TRANSITION_COV': 0.005,
    'P1_KF_SLOW_OBSERVATION_COV': 0.7,
    'P1_KF_SLOW_INITIAL_STATE_COV': 100,
    'P1_SMA_WINDOW': 20,
    'P1_STD_WINDOW': 20,
    'P1_TAKE_PROFIT': 0.4,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_KAMA_TREND_WINDOW': 10,
    'P1_KAMA_TREND_FAST_PERIOD': 10,
    'P1_KAMA_TREND_SLOW_PERIOD': 15,
    'P1_UCM_ALPHA': 0.1,
    'P1_UCM_BETA': 0.02,
    'P1_ATR_SPAN': 30,
    'P1_ENTRY_KAMA_SLOPE_THRESHOLD': 0.001,
    'P1_EXIT_KAMA_SLOPE_THRESHOLD': 0.0005,
    'P1_MIN_HOLD_SECONDS': 15.0,
    
    # Strategy 2 (Momentum/Volatility) - Priority 2
    'P2_FEATURE_COLUMN': 'PB9_T1',
    'P2_ATR_PERIOD': 25,
    'P2_STDDEV_PERIOD': 20,
    'P2_ATR_THRESHOLD': 0.01,
    'P2_STDDEV_THRESHOLD': 0.06,
    'P2_MOMENTUM_PERIOD': 10,
    'P2_VOLATILITY_PERIOD': 20,
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_TAKE_PROFIT': 0.16,
    'P2_TRAILING_STOP_FROM_PEAK': 0.01,
    'P2_BLOCKED_HOURS': [],
    
    # Strategy 3 (BB Crossover with KAMA Filter) - Priority 3
    'P3_BB4_COLUMN': 'BB4_T9',
    'P3_BB6_COLUMN': 'BB6_T11',
    'P3_VOLATILITY_FEATURE': 'PB9_T1',
    'P3_V_PATTERN_FEATURE': 'PB6_T4',
    'P3_V_PATTERN_ATR_PERIOD': 20,
    'P3_V_PATTERN_ATR_THRESHOLD': 0.005,
    'P3_ATR_PERIOD': 30,
    'P3_ACTIVATION_ATR_THRESHOLD': 0.0,
    'P3_ACTIVATION_STDDEV_THRESHOLD': 0.0,
    'P3_KAMA_PERIOD': 15,
    'P3_KAMA_FAST': 30,
    'P3_KAMA_SLOW': 180,
    'P3_KAMA_SLOPE_LOOKBACK': 2,
    'P3_KAMA_SLOPE_THRESHOLD': 0.0008,
    'P3_MIN_HOLD_SECONDS': 15.0,
    'P3_MAX_HOLD_SECONDS': 3600*2,
    'P3_USE_TAKE_PROFIT': True,
    'P3_TAKE_PROFIT_OFFSET': 0.3,
    'P3_STRATEGY_START_BAR': 1800,
}

# ============================================================================
# STRATEGY 1 (KALMAN FILTER) - PRIORITY 1 FUNCTIONS
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare all derived features for Kalman Filter strategy (Priority 1)"""
    
    # KFTrend (main Kalman filter)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_TREND_TRANSITION_COV'],
        observation_covariance=config['P1_KF_TREND_OBSERVATION_COV'],
        initial_state_mean=config['P1_KF_TREND_INITIAL_STATE_MEAN'],
        initial_state_covariance=config['P1_KF_TREND_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    df["KFTrend"] = smoothed_prices
    
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
    
    df["KFTrendFast"] = smoothed_prices
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)
    
    # KFTrendSlow
    kf_slow = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_SLOW_TRANSITION_COV'],
        observation_covariance=config['P1_KF_SLOW_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_SLOW_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf_slow.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    
    df["KFTrendSlow"] = smoothed_prices
    df["KFTrendSlow_Lagged"] = df["KFTrendSlow"].shift(1)
    
    # SMA and STD
    df["SMA_20"] = df[config['P1_FEATURE_COLUMN']].rolling(window=config['P1_SMA_WINDOW'], min_periods=1).mean()
    df["STD_SMA"] = df["SMA_20"].rolling(window=config['P1_STD_WINDOW'], min_periods=1).std()
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
        if np.isnan(price[i]) or np.isnan(sc[i]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i] * (price[i] - kama[i - 1])
    
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # KAMA_Trend (secondary)
    window_trend = config['P1_KAMA_TREND_WINDOW']
    kama_signal_trend = np.abs(pd.Series(price).diff(window_trend))
    kama_noise_trend = np.abs(pd.Series(price).diff()).rolling(window=window_trend, min_periods=1).sum()
    er_trend = (kama_signal_trend / kama_noise_trend.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast_trend = 2 / (config['P1_KAMA_TREND_FAST_PERIOD'] + 1)
    sc_slow_trend = 2 / (config['P1_KAMA_TREND_SLOW_PERIOD'] + 1)
    sc_trend = ((er_trend * (sc_fast_trend - sc_slow_trend)) + sc_slow_trend) ** 2
    
    kama_trend = np.full(len(price), np.nan)
    kama_trend[start] = price[start]
    
    for i in range(start + 1, len(price)):
        if np.isnan(price[i]) or np.isnan(sc_trend[i]) or np.isnan(kama_trend[i - 1]):
            kama_trend[i] = kama_trend[i - 1]
        else:
            kama_trend[i] = kama_trend[i - 1] + sc_trend[i] * (price[i] - kama_trend[i - 1])
    
    df["KAMA_Trend"] = kama_trend
    df["KAMA_Trend_Lagged"] = df["KAMA_Trend"].shift(1)
    df["KAMA_Trend_Slope"] = df["KAMA_Trend"].diff()
    
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
    
    # ATR
    tr = df[config['P1_FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P1_ATR_SPAN']).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()
    
    # Additional features
    df["PB9_T1_MA"] = df[config['P1_FEATURE_COLUMN']].rolling(window=20, min_periods=1).mean()
    df["STD"] = df["PB9_T1_MA"].rolling(window=20, min_periods=1).std()
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()
    df["Price_Dev"] = df[config['P1_FEATURE_COLUMN']] - 100
    
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
# STRATEGY 2 (MOMENTUM/VOLATILITY) - PRIORITY 2 FUNCTIONS
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
def generate_p2_signal_single(atr, stddev, momentum, volatility, atr_thresh, stddev_thresh):
    """Generate single bar trading signal for Priority 2"""
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
def generate_p2_signals_internal(
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
    """Generate Priority 2 (Momentum/Volatility) signals with pure trailing stop loss"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    entry_idx = -1
    entry_price = 0.0
    last_exit_time = 0.0
    last_direction = 0
    
    tp_hit = False
    peak_pnl = 0.0
    
    for i in range(1, n):
        t = timestamps[i]
        price = prices[i]
        
        if price < 1e-10:
            continue
        
        current_hour = int(t // 3600)
        
        hour_blocked = False
        for blocked_hour in blocked_hours:
            if current_hour == blocked_hour:
                hour_blocked = True
                break
        
        if hour_blocked:
            if position != 0:
                pnl = (price - entry_price) * position
                holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
                
                if not tp_hit and pnl >= take_profit:
                    tp_hit = True
                    peak_pnl = pnl
                
                if tp_hit and pnl > peak_pnl:
                    peak_pnl = pnl
                
                eod_triggered = i == n - 1
                min_hold_ok = holding_time >= min_hold_seconds
                
                should_exit = False
                if tp_hit:
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
            continue
        
        sig = generate_p2_signal_single(
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
                tp_hit = False
                peak_pnl = 0.0
        
        elif position != 0:
            pnl = (price - entry_price) * position
            holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
            
            if not tp_hit and pnl >= take_profit:
                tp_hit = True
                peak_pnl = pnl
            
            if tp_hit and pnl > peak_pnl:
                peak_pnl = pnl
            
            eod_triggered = i == n - 1
            min_hold_ok = holding_time >= min_hold_seconds
            
            should_exit = False
            if tp_hit:
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


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 2 (Momentum/Volatility) signals"""
    pb9_values = df[config['P2_FEATURE_COLUMN']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    
    atr = compute_atr_fast(pb9_values, config['P2_ATR_PERIOD'])
    stddev = compute_stddev_fast(pb9_values, config['P2_STDDEV_PERIOD'])
    momentum = compute_momentum_fast(pb9_values, config['P2_MOMENTUM_PERIOD'])
    volatility = compute_volatility_ratio(pb9_values, config['P2_VOLATILITY_PERIOD'])
    
    blocked_hours = np.array(config['P2_BLOCKED_HOURS'], dtype=np.int64)
    
    signals = generate_p2_signals_internal(
        timestamps=timestamps,
        prices=prices,
        atr=atr,
        stddev=stddev,
        momentum=momentum,
        volatility=volatility,
        atr_thresh=config['P2_ATR_THRESHOLD'],
        stddev_thresh=config['P2_STDDEV_THRESHOLD'],
        min_hold_seconds=config['P2_MIN_HOLD_SECONDS'],
        take_profit=config['P2_TAKE_PROFIT'],
        trailing_stop=config['P2_TRAILING_STOP_FROM_PEAK'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        blocked_hours=blocked_hours
    )
    
    return signals


# ============================================================================
# STRATEGY 3 (BB CROSSOVER WITH KAMA FILTER) - PRIORITY 3 FUNCTIONS
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
def detect_v_pattern(pb6_t4_values, v_pattern_atr_values, index, signal_direction, atr_threshold):
    """Detect V-pattern using PB6_T4 feature"""
    if index < 3:
        return False
    
    current_atr = v_pattern_atr_values[index]
    prev_atr = v_pattern_atr_values[index - 1]
    
    atr_crossing = (current_atr > atr_threshold) and (prev_atr <= atr_threshold)
    
    if not atr_crossing:
        return False
    
    v0 = pb6_t4_values[index]
    v1 = pb6_t4_values[index - 1]
    v2 = pb6_t4_values[index - 2]
    v3 = pb6_t4_values[index - 3]
    
    if signal_direction == -1:
        is_ascending = (v0 > v1) and (v1 > v2) and (v2 > v3)
        return is_ascending
    
    elif signal_direction == 1:
        is_descending = (v0 < v1) and (v1 < v2) and (v2 < v3)
        return is_descending
    
    return False


@jit(nopython=True, fastmath=True)
def generate_p3_signals_internal(
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
    """Generate Priority 3 (BB Crossover with KAMA Filter) signals"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    strategy_activated = False
    current_position = 0
    entry_time = 0.0
    entry_price = 0.0
    last_exit_time = 0.0
    
    pending_reversal = 0
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
        
        if not strategy_activated:
            if (current_atr > activation_atr_threshold and 
                current_pb9_stddev > activation_stddev_threshold):
                strategy_activated = True
        
        can_trade = strategy_activated and (i >= strategy_start_bar)
        
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        kama_bullish = current_kama_slope > kama_slope_threshold
        kama_bearish = current_kama_slope < -kama_slope_threshold
        
        is_last_bar = (i == n - 1)
        
        if current_position == 0:
            if pending_reversal != 0 and current_time >= reversal_available_time:
                current_position = pending_reversal
                entry_time = current_time
                entry_price = current_price
                signals[i] = pending_reversal
                pending_reversal = 0
                reversal_available_time = 0.0
            
            else:
                cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)
                
                if (can_trade and cooldown_elapsed and not is_last_bar):
                    initial_signal = 0
                    
                    if bullish_bb_cross and kama_bullish:
                        initial_signal = 1
                    elif bearish_bb_cross and kama_bearish:
                        initial_signal = -1
                    
                    if initial_signal != 0:
                        v_pattern_detected = detect_v_pattern(
                            pb6_t4_values, v_pattern_atr_values, i, initial_signal, v_pattern_atr_threshold
                        )
                        
                        if v_pattern_detected:
                            pending_reversal = -initial_signal
                            reversal_available_time = current_time + cooldown_seconds
                            last_exit_time = current_time
                        else:
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
    """Generate Priority 3 (BB Crossover with KAMA Filter) signals"""
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    bb4_values = df[config['P3_BB4_COLUMN']].values.astype(np.float64)
    bb6_values = df[config['P3_BB6_COLUMN']].values.astype(np.float64)
    pb9_t1_values = df[config['P3_VOLATILITY_FEATURE']].values.astype(np.float64)
    pb6_t4_values = df[config['P3_V_PATTERN_FEATURE']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    true_range = calculate_true_range_from_pb9(pb9_t1_values)
    atr_values = rolling_mean_numba(true_range, config['P3_ATR_PERIOD'])
    v_pattern_atr_values = rolling_mean_numba(true_range, config['P3_V_PATTERN_ATR_PERIOD'])
    pb9_stddev_values = rolling_std_numba(pb9_t1_values, config['P3_ATR_PERIOD'])
    
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
        atr_values=atr_values,
        pb9_stddev_values=pb9_stddev_values,
        kama_slope=kama_slope,
        pb6_t4_values=pb6_t4_values,
        v_pattern_atr_values=v_pattern_atr_values,
        activation_atr_threshold=config['P3_ACTIVATION_ATR_THRESHOLD'],
        activation_stddev_threshold=config['P3_ACTIVATION_STDDEV_THRESHOLD'],
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
# WINDOW-BASED MANAGER LOGIC WITH ORDER BOOK
# ============================================================================

def calculate_strategy_pnl(signals: np.ndarray, prices: np.ndarray, start_idx: int, end_idx: int) -> dict:
    """
    Calculate total PnL (realized + unrealized) for a strategy in a window
    
    Returns:
        dict with 'total_pnl', 'realized_pnl', 'unrealized_pnl', 'num_trades', 'final_position'
    """
    position = 0
    entry_price = 0.0
    realized_pnl = 0.0
    num_trades = 0
    
    for i in range(start_idx, end_idx + 1):
        signal = signals[i]
        price = prices[i]
        
        if signal != 0:
            if position == 0:
                # Entry
                position = signal
                entry_price = price
                num_trades += 1
            elif (position == 1 and signal == -1) or (position == -1 and signal == 1):
                # Exit
                pnl = (price - entry_price) * position
                realized_pnl += pnl
                position = 0
                entry_price = 0.0
    
    # Calculate unrealized PnL
    unrealized_pnl = 0.0
    if position != 0:
        final_price = prices[end_idx]
        unrealized_pnl = (final_price - entry_price) * position
    
    total_pnl = realized_pnl + unrealized_pnl
    
    return {
        'total_pnl': total_pnl,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'num_trades': num_trades,
        'final_position': position
    }


def determine_window_winner(p1_perf: dict, p2_perf: dict, p3_perf: dict) -> str:
    """
    Determine winning strategy based on total PnL
    If all zero or tie, use priority order: P1 > P2 > P3
    """
    p1_pnl = p1_perf['total_pnl']
    p2_pnl = p2_perf['total_pnl']
    p3_pnl = p3_perf['total_pnl']
    
    # If all zero, use priority order
    if p1_pnl == 0.0 and p2_pnl == 0.0 and p3_pnl == 0.0:
        return 'P1'
    
    # Find max PnL
    max_pnl = max(p1_pnl, p2_pnl, p3_pnl)
    
    # Priority-based tie breaking
    if p1_pnl == max_pnl:
        return 'P1'
    elif p2_pnl == max_pnl:
        return 'P2'
    else:
        return 'P3'


def apply_window_manager_logic(
    df: pd.DataFrame, 
    p1_signals: np.ndarray, 
    p2_signals: np.ndarray,
    p3_signals: np.ndarray, 
    config: dict
) -> tuple:
    """
    Apply window-based manager logic with order book approach
    
    Returns:
        (final_signals, strategy_sources, window_stats)
    """
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    strategy_sources = [''] * n  # Track which strategy generated each signal
    
    prices = df[config['PRICE_COLUMN']].values
    timestamps = df['Time_sec'].values
    
    window_size = config['WINDOW_SIZE_SECONDS']
    cooldown = config['UNIVERSAL_COOLDOWN']
    
    # Find window boundaries
    windows = []
    start_time = timestamps[0]
    start_idx = 0
    
    for i in range(n):
        if timestamps[i] >= start_time + window_size or i == n - 1:
            end_idx = i if i == n - 1 else i - 1
            windows.append((start_idx, end_idx, start_time, timestamps[end_idx]))
            start_idx = i
            start_time = timestamps[i]
    
    # Manager state
    current_position = 0
    current_owner = None
    entry_price = 0.0
    last_signal_time = -cooldown
    previous_winner = None
    window_stats = []
    
    for window_num, (win_start, win_end, t_start, t_end) in enumerate(windows):
        # ====================================================================
        # 1. SIMULATE ALL STRATEGIES IN PREVIOUS WINDOW (for order book)
        # ====================================================================
        if window_num > 0:
            prev_start, prev_end, _, _ = windows[window_num - 1]
            
            p1_perf = calculate_strategy_pnl(p1_signals, prices, prev_start, prev_end)
            p2_perf = calculate_strategy_pnl(p2_signals, prices, prev_start, prev_end)
            p3_perf = calculate_strategy_pnl(p3_signals, prices, prev_start, prev_end)
            
            winner = determine_window_winner(p1_perf, p2_perf, p3_perf)
            
            window_stats.append({
                'window': window_num - 1,
                'start_time': windows[window_num - 1][2],
                'end_time': windows[window_num - 1][3],
                'p1_pnl': p1_perf['total_pnl'],
                'p2_pnl': p2_perf['total_pnl'],
                'p3_pnl': p3_perf['total_pnl'],
                'p1_trades': p1_perf['num_trades'],
                'p2_trades': p2_perf['num_trades'],
                'p3_trades': p3_perf['num_trades'],
                'winner': winner
            })
        else:
            # First window: use priority order
            winner = 'P1'
        
        # ====================================================================
        # 2. HANDLE POSITION HANDOFF AT WINDOW BOUNDARY
        # ====================================================================
        if window_num > 0 and current_position != 0:
            # Get new winner's signal at window start
            winner_signals = {
                'P1': p1_signals,
                'P2': p2_signals,
                'P3': p3_signals
            }[winner]
            
            new_signal = winner_signals[win_start]
            
            # Handoff logic
            if new_signal == current_position:
                # Match: continue holding
                current_owner = winner
                pass
            elif new_signal == 0:
                # No signal: square off
                final_signals[win_start] = -current_position
                strategy_sources[win_start] = f'{current_owner}_exit'
                current_position = 0
                current_owner = None
                last_signal_time = timestamps[win_start]
            else:
                # Opposite: square off, wait 15s, then enter opposite (Option A)
                final_signals[win_start] = -current_position
                strategy_sources[win_start] = f'{current_owner}_exit'
                last_signal_time = timestamps[win_start]
                
                # Find bar after 15s cooldown
                entry_after_cooldown_idx = None
                for j in range(win_start + 1, win_end + 1):
                    if timestamps[j] >= last_signal_time + cooldown:
                        entry_after_cooldown_idx = j
                        break
                
                if entry_after_cooldown_idx is not None:
                    final_signals[entry_after_cooldown_idx] = new_signal
                    strategy_sources[entry_after_cooldown_idx] = winner
                    current_position = new_signal
                    current_owner = winner
                    entry_price = prices[entry_after_cooldown_idx]
                    last_signal_time = timestamps[entry_after_cooldown_idx]
                else:
                    current_position = 0
                    current_owner = None
        
        # ====================================================================
        # 3. PROCESS WINDOW WITH WINNING STRATEGY
        # ====================================================================
        active_signals = {
            'P1': p1_signals,
            'P2': p2_signals,
            'P3': p3_signals
        }[winner]
        
        for i in range(win_start, win_end + 1):
            current_time = timestamps[i]
            price = prices[i]
            signal = active_signals[i]
            is_last_bar = (i == n - 1)
            
            # Skip if already processed (handoff)
            if final_signals[i] != 0:
                continue
            
            # Check cooldown
            cooldown_ok = (current_time - last_signal_time) >= cooldown
            
            # EOD square off
            if is_last_bar and current_position != 0:
                final_signals[i] = -current_position
                strategy_sources[i] = f'{winner}_eod'
                current_position = 0
                current_owner = None
                last_signal_time = current_time
                continue
            
            # Process signal
            if signal != 0 and cooldown_ok:
                if current_position == 0:
                    # Entry
                    final_signals[i] = signal
                    strategy_sources[i] = winner
                    current_position = signal
                    current_owner = winner
                    entry_price = price
                    last_signal_time = current_time
                
                elif signal == -current_position:
                    # Exit
                    final_signals[i] = signal
                    strategy_sources[i] = winner
                    current_position = 0
                    current_owner = None
                    last_signal_time = current_time
    
    return final_signals, strategy_sources, window_stats


# ============================================================================
# DAY PROCESSING
# ============================================================================

def extract_day_num(filepath):
    """Extract day number from filepath"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    """Process a single day with all three strategies and window manager logic"""
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
            config['P3_V_PATTERN_FEATURE'],
        ]
        
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
        # 2. GENERATE ALL STRATEGY SIGNALS INDEPENDENTLY
        # ====================================================================
        p1_signals, df_with_features = generate_p1_signals(df, config)
        p2_signals = generate_p2_signals(df, config)
        p3_signals = generate_p3_signals(df, config)
        
        # ====================================================================
        # 3. APPLY WINDOW MANAGER LOGIC
        # ====================================================================
        final_signals, strategy_sources, window_stats = apply_window_manager_logic(
            df, p1_signals, p2_signals, p3_signals, config
        )
        
        # ====================================================================
        # 4. COUNT SIGNALS
        # ====================================================================
        final_entries = np.sum(np.abs(final_signals) == 1)
        
        # ====================================================================
        # 5. SAVE RESULT
        # ====================================================================
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': final_signals,
            'Strategy': strategy_sources
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'final_entries': final_entries,
            'window_stats': window_stats
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
    
    print("="*80)
    print("STRATEGY MANAGER - WINDOW-BASED ORDER BOOK APPROACH")
    print("="*80)
    print(f"\n🪟 Window Size: {config['WINDOW_SIZE_SECONDS']}s (30 minutes)")
    print(f"📊 Performance Metric: Total PnL (realized + unrealized)")
    print(f"🎯 Selection: Best performing strategy from previous window")
    print(f"⏱️  Universal Cooldown: {config['UNIVERSAL_COOLDOWN']}s between ANY signals")
    print(f"🔄 Handoff Logic (Option A):")
    print(f"   - Match position → Continue")
    print(f"   - No signal → Square off")
    print(f"   - Opposite → Square off, wait 15s, enter opposite")
    print(f"🎲 Tie Breaking: Priority order (P1 > P2 > P3)")
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
    total_final_entries = 0
    all_window_stats = []
    
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
                        total_final_entries += result['final_entries']
                        all_window_stats.extend(result['window_stats'])
                        print(f"✓ Processed day{result['day_num']:3d} | Final entries: {result['final_entries']:3d} | Windows: {len(result['window_stats'])}")
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
        
        # Strategy usage statistics
        p1_signals = final_df['Strategy'].str.startswith('P1').sum()
        p2_signals = final_df['Strategy'].str.startswith('P2').sum()
        p3_signals = final_df['Strategy'].str.startswith('P3').sum()
        
        # Window statistics
        total_windows = len(all_window_stats)
        if total_windows > 0:
            p1_wins = sum(1 for w in all_window_stats if w['winner'] == 'P1')
            p2_wins = sum(1 for w in all_window_stats if w['winner'] == 'P2')
            p3_wins = sum(1 for w in all_window_stats if w['winner'] == 'P3')
            
            avg_p1_pnl = np.mean([w['p1_pnl'] for w in all_window_stats])
            avg_p2_pnl = np.mean([w['p2_pnl'] for w in all_window_stats])
            avg_p3_pnl = np.mean([w['p3_pnl'] for w in all_window_stats])
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:        {total_bars:,}")
        print(f"Long entries:      {num_long:,}")
        print(f"Short entries:     {num_short:,}")
        print(f"Hold/Wait:         {num_hold:,}")
        print(f"Total entries:     {(num_long + num_short):,}")
        print(f"\nStrategy Usage:")
        print(f"P1 signals:        {p1_signals:,}")
        print(f"P2 signals:        {p2_signals:,}")
        print(f"P3 signals:        {p3_signals:,}")
        
        if total_windows > 0:
            print(f"\nWindow Performance:")
            print(f"Total windows:     {total_windows}")
            print(f"P1 wins:           {p1_wins} ({p1_wins/total_windows*100:.1f}%)")
            print(f"P2 wins:           {p2_wins} ({p2_wins/total_windows*100:.1f}%)")
            print(f"P3 wins:           {p3_wins} ({p3_wins/total_windows*100:.1f}%)")
            print(f"\nAvg Window PnL:")
            print(f"P1:                {avg_p1_pnl:.4f}")
            print(f"P2:                {avg_p2_pnl:.4f}")
            print(f"P3:                {avg_p3_pnl:.4f}")
        
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