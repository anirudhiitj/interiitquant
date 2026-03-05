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
    'OUTPUT_FILE': 'managed_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_swapped',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 15.0,  # seconds between ANY signals
    'MAX_WORKERS': 24,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    'FEATURE_COLUMN': 'PB9_T1',
    
    # Strategy 1 (KF/UCM-Based) - Priority 1
    'P1_PRICE_EXIT_TARGET': 0.3,
    'P1_ATR_THRESHOLD': 0.01,
    'P1_STD_THRESHOLD': 0.06,
    'P1_KAMA_SLOPE_THRESHOLD_ENTRY': 0.001,
    'P1_KAMA_SLOPE_THRESHOLD_EXIT': 0.0005,
    'P1_MIN_HOLD_SECONDS': 15.0,
    
    # Strategy 2 (Momentum/Volatility) - Priority 2
    'P2_ATR_PERIOD': 25,
    'P2_STDDEV_PERIOD': 20,
    'P2_ATR_THRESHOLD': 0.01,
    'P2_STDDEV_THRESHOLD': 0.06,
    'P2_MOMENTUM_PERIOD': 10,
    'P2_VOLATILITY_PERIOD': 20,
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_TAKE_PROFIT': 0.15,
    'P2_BLOCKED_HOURS': [5],  # No entries in hour 5 (05:00-06:00)
}

# ============================================================================
# STRATEGY 1 (KF/UCM-BASED) - PRIORITY 1 FUNCTIONS
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare all derived features for Priority 1 (KF/UCM) strategy"""
    
    # Kalman Filter - Fast Trend
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.49
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = df["Price"].iloc[0]
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(df[config['FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['FEATURE_COLUMN']].dropna().index)
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)

    # Kalman Filter - Slow Trend
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.01
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = df["Price"].iloc[0]
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(df[config['FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['FEATURE_COLUMN']].dropna().index)
    df["KFTrendSlow"] = smoothed_prices.shift(1)
    df["KFTrendSlow_Lagged"] = df["KFTrendSlow"].shift(1)

    # KAMA for slope calculation
    price = df[config['FEATURE_COLUMN']].to_numpy(dtype=float)
    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
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
    df["KAMA_Slope"] = df["KAMA"].diff(2)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    # UCM (Unobserved Components Model)
    series = df[config['FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("PB9_T1 has no valid values")

    alpha = 0.1
    beta = 0.02
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

    # ATR and STD for volatility filter
    tr = df[config['FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=30).mean().shift(1)
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max().shift(1)
    
    df["PB9_T1_MA"] = df[config['FEATURE_COLUMN']].rolling(window=20, min_periods=1).mean().shift(1)
    df["STD"] = df["PB9_T1_MA"].rolling(window=20, min_periods=1).std().shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max().shift(1)

    return df


def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 1 (KF/UCM-Based) signals"""
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)
    
    position = 0
    entry_price = 0.0
    entry_time = 0.0
    last_exit_time = -config['UNIVERSAL_COOLDOWN']
    
    for i in range(1, n):
        current_time = df.iloc[i]['Time_sec']
        price = df.iloc[i][config['PRICE_COLUMN']]
        row = df.iloc[i]
        is_last_bar = (i == n - 1)
        
        signal = 0
        
        # EOD square-off
        if is_last_bar and position != 0:
            signal = -position
        
        # Check cooldown
        cooldown_ok = (current_time - last_exit_time) >= config['UNIVERSAL_COOLDOWN']
        
        # Entry logic
        if position == 0 and cooldown_ok and not is_last_bar:
            atr_high = row["ATR_High"]
            std_high = row["STD_High"]
            kama_slope_abs = row["KAMA_Slope_abs"]
            ucm = row["UCM"]
            kf_slow = row["KFTrendSlow"]
            
            # Check volatility filter
            if atr_high > config['P1_ATR_THRESHOLD'] and std_high > config['P1_STD_THRESHOLD'] and kama_slope_abs > config['P1_KAMA_SLOPE_THRESHOLD_ENTRY']:
                if ucm >= kf_slow:
                    signal = 1
                elif ucm <= kf_slow:
                    signal = -1
        
        # Exit logic
        elif position != 0 and cooldown_ok:
            holding_time = current_time - entry_time
            
            # Enforce minimum hold
            if holding_time >= config['P1_MIN_HOLD_SECONDS']:
                # Check crossover exit
                ucm_lagged = row["UCM_Lagged"]
                kf_slow_lagged = row["KFTrendSlow_Lagged"]
                ucm = row["UCM"]
                kf_slow = row["KFTrendSlow"]
                kama_slope_abs = row["KAMA_Slope_abs"]
                
                if position == 1:
                    if (ucm_lagged > kf_slow_lagged and ucm < kf_slow) and kama_slope_abs > config['P1_KAMA_SLOPE_THRESHOLD_EXIT']:
                        signal = -1
                elif position == -1:
                    if (ucm_lagged < kf_slow_lagged and ucm > kf_slow) and kama_slope_abs > config['P1_KAMA_SLOPE_THRESHOLD_EXIT']:
                        signal = 1
                
                # Check price target exit
                if signal == 0:
                    if position == 1:
                        price_diff = price - entry_price
                        if price_diff >= config['P1_PRICE_EXIT_TARGET']:
                            signal = -1
                    elif position == -1:
                        price_diff = entry_price - price
                        if price_diff >= config['P1_PRICE_EXIT_TARGET']:
                            signal = 1
        
        # Apply signal
        if signal != 0:
            position += signal
            
            if position == 0:  # Exit
                last_exit_time = current_time
                entry_price = 0.0
                entry_time = 0.0
            else:  # Entry
                entry_price = price
                entry_time = current_time
            
            signals[i] = signal
    
    return signals


# ============================================================================
# STRATEGY 2 (MOMENTUM/VOLATILITY) - PRIORITY 2 FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period=20):
    """
    Compute Average True Range on price series
    """
    n = len(prices)
    atr = np.zeros(n)
    if n < period + 1:
        return atr
    
    # Calculate true range
    tr = np.zeros(n)
    for i in range(1, n):
        high_low = abs(prices[i] - prices[i-1])
        tr[i] = high_low
    
    # Calculate ATR as moving average of TR
    for i in range(period, n):
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    return atr


@jit(nopython=True, fastmath=True)
def compute_stddev_fast(prices, period=20):
    """
    Compute rolling standard deviation
    """
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
    """
    Compute momentum (rate of change)
    """
    n = len(prices)
    momentum = np.zeros(n)
    period_safe = max(period, 1)
    for i in range(period_safe, n):
        momentum[i] = prices[i] - prices[i - period_safe]
    return momentum


@jit(nopython=True, fastmath=True)
def compute_volatility_ratio(prices, period=20):
    """
    Compute volatility ratio as stddev/mean to normalize volatility
    """
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
    """
    Generate trading signal based on:
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
    if momentum_vol_ratio > 8:  # Small threshold to avoid noise
        return 1  # Long
    elif momentum_vol_ratio < 0:
        return -1  # Short
    else:
        return 0  # No trade


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
    cooldown_seconds,
    blocked_hours
):
    """
    Generate Priority 2 (Momentum/Volatility) signals
    NO ENTRIES IN BLOCKED HOURS (default: hour 5 = 05:00-06:00)
    ONLY CHANGE: Exit negative positions at 4:00:00 (14400 seconds)
    """
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
                
                tp_triggered = pnl >= take_profit
                eod_triggered = i == n - 1
                min_hold_ok = holding_time >= min_hold_seconds
                
                # ONLY NEW LOGIC: Exit negative positions at 4:00:00
                at_4hr = t >= 21000.0  # At or after 4:00:00 (14400 seconds)
                negative_position = pnl < 0
                exit_at_4hr = at_4hr and negative_position
                
                if min_hold_ok and (tp_triggered or eod_triggered or exit_at_4hr):
                    signals[i] = -position
                    last_direction = position
                    position = 0
                    last_exit_time = t
                    entry_idx = -1
            continue  # Skip to next iteration - NO NEW ENTRIES IN BLOCKED HOURS
        
        # Generate signal for current bar
        sig = generate_p2_signal_single(
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
        
        # Exit logic - only take profit (no stop loss)
        elif position != 0:
            pnl = (price - entry_price) * position
            holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0
            
            # Check for take profit or end of day
            tp_triggered = pnl >= take_profit
            eod_triggered = i == n - 1
            
            # ONLY NEW LOGIC: Exit negative positions at 4:00:00
            at_4hr = t >= 14400.0  # At or after 4:00:00 (14400 seconds)
            negative_position = pnl < 0
            exit_at_4hr = at_4hr and negative_position
            
            # Enforce minimum hold period (15 seconds)
            min_hold_ok = holding_time >= min_hold_seconds
            
            exit_condition = min_hold_ok and (tp_triggered or eod_triggered or exit_at_4hr)
            
            if exit_condition:
                signals[i] = -position
                last_direction = position
                position = 0
                last_exit_time = t
                entry_idx = -1
    
    return signals


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 2 (Momentum/Volatility) signals"""
    pb9_values = df[config['FEATURE_COLUMN']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    # Ensure no invalid values
    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
    
    # Calculate all indicators
    atr = compute_atr_fast(pb9_values, config['P2_ATR_PERIOD'])
    stddev = compute_stddev_fast(pb9_values, config['P2_STDDEV_PERIOD'])
    momentum = compute_momentum_fast(pb9_values, config['P2_MOMENTUM_PERIOD'])
    volatility = compute_volatility_ratio(pb9_values, config['P2_VOLATILITY_PERIOD'])
    
    # Convert blocked hours list to numpy array
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
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        blocked_hours=blocked_hours
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
            if cooldown_ok:
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


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> str:
    """Process a single day with both strategies and manager logic"""
    try:
        # ====================================================================
        # 1. READ DATA
        # ====================================================================
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['FEATURE_COLUMN'],
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
        # 2. PREPARE FEATURES FOR PRIORITY 1 (KF/UCM)
        # ====================================================================
        df = prepare_p1_features(df, config)
        
        # ====================================================================
        # 3. GENERATE PRIORITY 1 SIGNALS (KF/UCM-Based)
        # ====================================================================
        p1_signals = generate_p1_signals(df, config)
        
        # ====================================================================
        # 4. GENERATE PRIORITY 2 SIGNALS (Momentum/Volatility)
        # ====================================================================
        p2_signals = generate_p2_signals(df, config)
        
        # ====================================================================
        # 5. APPLY MANAGER LOGIC
        # ====================================================================
        final_signals = apply_manager_logic(df, p1_signals, p2_signals, config)
        
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
        
        return str(output_path)
    
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main(config):
    start_time = time.time()
    
    print("="*80)
    print("STRATEGY MANAGER - PRIORITY-BASED SIGNAL ROUTING (SWAPPED)")
    print("="*80)
    print(f"\nPriority 1: KF/UCM-Based Strategy (PB9_T1 based)")
    print(f"  - Entry: UCM vs KFTrendSlow crossover with volatility filters")
    print(f"  - Volatility Filter: ATR_High>{config['P1_ATR_THRESHOLD']}, STD_High>{config['P1_STD_THRESHOLD']}, KAMA_Slope_abs>{config['P1_KAMA_SLOPE_THRESHOLD_ENTRY']}")
    print(f"  - Exit: Opposite crossover (KAMA_Slope_abs>{config['P1_KAMA_SLOPE_THRESHOLD_EXIT']}) OR price target {config['P1_PRICE_EXIT_TARGET']} absolute")
    print(f"\nPriority 2: Momentum/Volatility Strategy (PB9_T1 based)")
    print(f"  - Volatility Filter: ATR>{config['P2_ATR_THRESHOLD']}, StdDev>{config['P2_STDDEV_THRESHOLD']}")
    print(f"  - Take Profit: {config['P2_TAKE_PROFIT']} absolute | No Stop Loss")
    print(f"  - Blocked Hours: {config['P2_BLOCKED_HOURS']} (no entries)")
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