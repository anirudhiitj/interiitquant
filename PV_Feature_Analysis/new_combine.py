import os
import glob
import re
import shutil
import sys
import pathlib
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import gc  # Essential for memory management
from pykalman import KalmanFilter

# Try to import Numba for acceleration
try:
    from numba import jit, int64, float64
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
# CONFIGURATION (cleaned for new P1)
# ============================================================================

CONFIG = {
    # Data
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'combined_EBY.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_4',

    # Manager Settings
    'UNIVERSAL_COOLDOWN': 5.0,  # seconds between ANY signals (Exit -> Entry)
    'MAX_WORKERS': 32,

    # Required Columns (names used in parquet)
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',

    # P1 minimal params
    'P1_FEATURE_COLUMN': 'PB9_T1',  # the feature P1 uses
    'P1_MIN_TRADE_SECONDS': 300,    # earliest time to allow trading (like time>=300 in P1)
    'P1_COOLDOWN_SECONDS': 5.0,     # used internally by P1 logic (kept aligned with UNIVERSAL_COOLDOWN)

    'P2_BLACKLIST_DAYS': {311, 421, 446, 87, 273},
    'P2_DECISION_WINDOW': 30 * 60,    # 30 minutes
    'P2_SIGMA_THRESH': 0.15,
    'P2_MU_THRESH': 100.0,
    'P2_ENTRY_DROP': 0.8,
    'P2_LONG_SL': 0.7,
    'P2_LONG_TP_ACT': 1.0,
    'P2_LONG_TRAIL': 0.1,
    'P2_REV_RISE': 0.2,               # Rise required to trigger reversal short
    'P2_SHORT_SL': 4.4,
    'P2_SHORT_TP_ACT': 6.2,
    'P2_SHORT_TRAIL': 0.04,
    'P2_MIN_HOLD_SECONDS': 5.0,

    # Strategy 3 (BB Crossover with KAMA Filter and V-pattern) - Priority 3
    'P3_BB4_COLUMN': 'BB4_T9',
    'P3_BB6_COLUMN': 'BB6_T11',
    'P3_VOLATILITY_FEATURE': 'PB9_T1',
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

    # Strategy 4 (KAMA-based Simple) - Priority 4
    'P4_FEATURE_COLUMN': 'PB9_T1',
    'P4_KAMA_WINDOW': 30,
    'P4_KAMA_FAST_PERIOD': 60,
    'P4_KAMA_SLOW_PERIOD': 300,
    'P4_KAMA_SLOPE_DIFF': 2,
    'P4_ATR_SPAN': 30,
    'P4_ATR_THRESHOLD': 0.01,
    'P4_STD_THRESHOLD': 0.06,
    'P4_KAMA_SLOPE_ENTRY': 0.0007,
    'P4_TAKE_PROFIT': 0.35,
    'P4_MIN_HOLD_SECONDS': 15.0,
}

# ============================================================================
# SHARED HELPER FUNCTIONS
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
# STRATEGY 1 (REPLACED WITH CUSTOM KALMAN + KAMA + UCM STRATEGY) - PRIORITY 1
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare derived features EXACTLY as per user's replacement strategy."""

    # --- Kalman Filter (Same parameters as user logic) ---
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=0.24,
        observation_covariance=0.7,
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=100
    )
    # ensure PB9_T1 exists
    if config['P1_FEATURE_COLUMN'] not in df.columns:
        raise KeyError(f"Required column {config['P1_FEATURE_COLUMN']} not found for P1")

    filtered_state_means, _ = kf.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)

    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)
    df["Take_Profit"] = 0.4

    # --- KAMA (EXACT Implementation from user) ---
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    price[0:60] = np.nan
    window = 30

    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)

    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid PB9_T1 values found.")
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
    df["KAMA_Slope_abs_Lagged_2"] = df["KAMA_Slope_abs"].shift(2)
    df["KAMA_Sum"] = df["KAMA_Slope_abs"].rolling(window=5, min_periods=1).sum()

    # --- UCM Trend Filter (EXACT from user) ---
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError(f"{config['P1_FEATURE_COLUMN']} has no valid values")

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

    filtered[0:60] = np.nan

    df["UCM"] = filtered
    df["UCM_Lagged"] = df["UCM"].shift(1)

    # --- ATR High (from Price) ---
    tr = df[config['PRICE_COLUMN']].diff().abs()
    df["ATR"] = tr.ewm(span=30).mean()
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    return df


def p1_generate_single_full(row, position, cooldown_over, state):
    """
    EXACT user's signal generation logic including partial TP exits.
    Returns (signal, state)
    """
    position = round(position, 2)
    signal = 0.0
    time = int(row['Time_sec'])

    if cooldown_over and time >= CONFIG['P1_MIN_TRADE_SECONDS']:
        if position == 0:
            if (row["KFTrendFast"] >= row["UCM"]
                    and row["KAMA_Slope_abs_Lagged_2"] > 0.0007
                    and row["KAMA_Sum"] > 0.005
                    and row["ATR_High"] > 0.03):
                state["applyTP"] = True
                # Note: user script used signal = -1 for this branch (short)
                signal = -1.0

            elif (row["KFTrendFast"] <= row["UCM"]
                    and row["KAMA_Slope_abs_Lagged_2"] > 0.0007
                    and row["KAMA_Sum"] > 0.005
                    and row["ATR_High"] > 0.02):
                state["applyTP"] = True
                signal = 1.0

    return float(signal), state


def generate_p1_signals(df: pd.DataFrame, config: dict):
    """
    Implements the full P1 day-wise logic (entry, TP ladder with partial exit signals,
    EOD square-off), returning float signals and the enriched dataframe.
    This follows the processing block from the user's attached script.
    """
    df = prepare_p1_features(df, config)

    state = {"applyTP": True}

    position = 0.0
    entry_price = None
    last_signal_time = -config['P1_COOLDOWN_SECONDS']
    signals = [0.0] * len(df)
    positions = [0.0] * len(df)

    # Variables to persist between entries (per day)
    tp_stage = 0
    applySL = False
    sl = 0.5  # as per user's script

    for i in range(len(df)):
        position = round(position, 2)
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)

        signal = 0.0

        # --- EOD Square Off ---
        if is_last_tick and position != 0.0:
            signal = -position

        else:
            cooldown_over = (current_time - last_signal_time) >= config['P1_COOLDOWN_SECONDS']
            local_signal, state = p1_generate_single_full(row, position, cooldown_over, state)

            # Copy local_signal to signal variable
            signal = float(local_signal)

            # --- Entry Logic ---
            if cooldown_over and position == 0.0 and signal != 0.0:
                entry_price = price
                tp1 = row["Take_Profit"] * 1.0
                tp2 = row["Take_Profit"] * 1.25
                tp3 = row["Take_Profit"] * 1.75

                tp_stage = 0
                applySL = False
                sl = 0.5

            # --- Exit / TP Ladder Logic (partial signals) ---
            elif cooldown_over and position != 0.0 and signal == 0.0:
                if state.get("applyTP", False):
                    if position > 0.0:
                        price_diff = price - entry_price
                        tp1pos = 0.7
                        tp2pos = 0.5
                    elif position < 0.0:
                        price_diff = entry_price - price
                        tp1pos = -0.7
                        tp2pos = -0.5
                    else:
                        price_diff = 0.0
                        tp1pos = 0.0
                        tp2pos = 0.0

                    # Note: The user's original script had a complex order of checks.
                    # We'll apply the same order and produce fractional signals like:
                    # -position + tp1pos  etc.
                    if price_diff >= tp3 and tp_stage < 4:
                        applySL = False
                        signal = -position
                    elif price_diff >= tp3 and tp_stage < 3:
                        tp_stage = 3
                        applySL = True
                        if row["KAMA_Slope_abs"] < 0.001:
                            signal = -position
                        else:
                            signal = 0.0
                    elif price_diff >= tp2 and tp_stage < 2:
                        tp_stage = 2
                        applySL = True
                        if row["KAMA_Slope_abs"] < 0.001:
                            # partial exit as per user's logic
                            signal = -position + tp2pos
                        else:
                            signal = 0.0
                    elif price_diff >= tp1 and tp_stage < 1:
                        tp_stage = 1
                        applySL = True
                        if row["KAMA_Slope_abs"] < 0.001:
                            signal = -position + tp1pos
                        else:
                            signal = 0.0

                    # If TP-stage applied, check SL pullback
                    if applySL:
                        if tp_stage == 1:
                            if price_diff - tp1 < -sl:
                                signal = -position
                        elif tp_stage == 2:
                            if price_diff - tp2 < -sl:
                                signal = -position
                        elif tp_stage == 3:
                            if price_diff - tp3 < -sl:
                                signal = -position

        # --- Apply Signal (clamping to maximum position size of +/-1) ---
        if signal != 0.0:
            # When adapting partial signals, respect that position is continuous.
            # However enforce overall position limits in absolute terms [-1, +1].
            proposed_position = position + signal

            if proposed_position > 1.0:
                # convert to incremental signal to reach +1
                signal = 1.0 - position
                position = 1.0
            elif proposed_position < -1.0:
                signal = -1.0 - position
                position = -1.0
            else:
                position = proposed_position

            last_signal_time = current_time
            if abs(round(position, 2)) < 1e-9:
                entry_price = None

        signals[i] = float(signal)
        positions[i] = float(position)

    # Return float signals (allow partials) and the enriched df (df_p1)
    return np.array(signals, dtype=float), df

# ============================================================================
# STRATEGY 2 (PURE MOMENTUM/VOLATILITY WITH DYNAMIC TP) - PRIORITY 2 FUNCTIONS
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
def calculate_dynamic_tp(stddev, tp_low, tp_medium, tp_high, thresh_low, thresh_high):
    """Calculate dynamic TP based on stddev at entry"""
    if stddev >= thresh_high:
        return tp_high  # High volatility
    elif stddev >= thresh_low:
        return tp_medium  # Medium volatility
    else:
        return tp_low  # Low volatility


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
    kama_slope,
    atr_thresh,
    stddev_thresh,
    min_hold_seconds,
    tp_low,
    tp_medium,
    tp_high,
    stddev_tp_low,
    stddev_tp_high,
    trailing_stop,
    cooldown_seconds,
    conditional_sl,
    kama_slope_threshold
):
    """Generate Priority 2 (Pure Momentum/Volatility with Dynamic TP and Conditional SL) signals"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)

    position = 0.0
    entry_idx = -1
    entry_price = 0.0
    last_exit_time = 0.0
    last_direction = 0

    tp_hit = False
    peak_pnl = 0.0
    dynamic_tp = 0.0

    for i in range(1, n):
        t = timestamps[i]
        price = prices[i]

        if price < 1e-10:
            continue

        sig = generate_p2_signal_single(
            atr[i], stddev[i], momentum[i], volatility[i],
            atr_thresh, stddev_thresh
        )

        if position == 0.0 and sig != 0:
            if last_direction != 0 and sig == last_direction:
                continue

            if last_exit_time == 0.0 or (t - last_exit_time) >= cooldown_seconds:
                position = float(sig)
                entry_price = max(price, 1e-10)
                entry_idx = i
                signals[i] = position

                # Calculate dynamic TP based on stddev at entry
                dynamic_tp = calculate_dynamic_tp(
                    stddev[i], tp_low, tp_medium, tp_high,
                    stddev_tp_low, stddev_tp_high
                )

                tp_hit = False
                peak_pnl = 0.0

        elif position != 0.0:
            pnl = (price - entry_price) * position
            holding_time = t - timestamps[entry_idx] if entry_idx >= 0 else 0.0

            if not tp_hit and pnl >= dynamic_tp:
                tp_hit = True
                peak_pnl = pnl

            if tp_hit and pnl > peak_pnl:
                peak_pnl = pnl

            eod_triggered = i == n - 1
            min_hold_ok = holding_time >= min_hold_seconds

            should_exit = False
            should_reverse = False

            # Check conditional stop-loss with reversal (after min hold)
            if min_hold_ok:
                # Check if price moved against position
                adverse_move = 0.0
                if position == 1.0:  # Long position
                    adverse_move = entry_price - price  # Positive if price dropped
                elif position == -1.0:  # Short position
                    adverse_move = price - entry_price  # Positive if price rose

                # If adverse move exceeds conditional SL threshold
                if adverse_move >= conditional_sl:
                    # Check KAMA slope in opposite direction
                    if position == 1.0:
                        # Long position losing - check if KAMA slope is bearish
                        if kama_slope[i] < -kama_slope_threshold:
                            should_reverse = True
                    elif position == -1.0:
                        # Short position losing - check if KAMA slope is bullish
                        if kama_slope[i] > kama_slope_threshold:
                            should_reverse = True

            # Check trailing stop from peak (if TP was hit)
            if tp_hit:
                if pnl < (peak_pnl - trailing_stop):
                    should_exit = True

            exit_condition = min_hold_ok and (should_exit or eod_triggered)

            if should_reverse:
                signals[i] = -position  # Exit signal
                last_direction = int(np.sign(position))

                # Immediate reversal (same bar) - reverse position sign
                position = -position
                entry_price = max(price, 1e-10)
                entry_idx = i
                last_exit_time = t

                # Reset TP tracking for new reversed position
                dynamic_tp = calculate_dynamic_tp(
                    stddev[i], tp_low, tp_medium, tp_high,
                    stddev_tp_low, stddev_tp_high
                )
                tp_hit = False
                peak_pnl = 0.0

            elif exit_condition:
                signals[i] = -position
                last_direction = int(np.sign(position))
                position = 0.0
                last_exit_time = t
                entry_idx = -1
                tp_hit = False
                peak_pnl = 0.0

    return signals


def generate_p2_signals(df: pd.DataFrame, config: dict, p4_kama_slope: np.ndarray) -> np.ndarray:
    """Generate Priority 2 signals (float/int-like)"""
    pb9_values = df[config['P2_FEATURE_COLUMN']].values.astype(np.float64)
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)

    pb9_values = np.where(np.isnan(pb9_values) | (np.abs(pb9_values) < 1e-10), 1e-10, pb9_values)
    prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)

    atr = compute_atr_fast(pb9_values, config['P2_ATR_PERIOD'])
    stddev = compute_stddev_fast(pb9_values, config['P2_STDDEV_PERIOD'])
    momentum = compute_momentum_fast(pb9_values, config['P2_MOMENTUM_PERIOD'])
    volatility = compute_volatility_ratio(pb9_values, config['P2_VOLATILITY_PERIOD'])

    signals = generate_p2_signals_internal(
        timestamps=timestamps,
        prices=prices,
        atr=atr,
        stddev=stddev,
        momentum=momentum,
        volatility=volatility,
        kama_slope=p4_kama_slope,
        atr_thresh=config['P2_ATR_THRESHOLD'],
        stddev_thresh=config['P2_STDDEV_THRESHOLD'],
        min_hold_seconds=config['P2_MIN_HOLD_SECONDS'],
        tp_low=config['P2_TP_LOW'],
        tp_medium=config['P2_TP_MEDIUM'],
        tp_high=config['P2_TP_HIGH'],
        stddev_tp_low=config['P2_STDDEV_TP_LOW'],
        stddev_tp_high=config['P2_STDDEV_TP_HIGH'],
        trailing_stop=config['P2_TRAILING_STOP_FROM_PEAK'],
        cooldown_seconds=config['UNIVERSAL_COOLDOWN'],
        conditional_sl=config['P2_CONDITIONAL_SL'],
        kama_slope_threshold=config['P2_KAMA_SLOPE_THRESHOLD']
    )

    return signals


# ============================================================================
# STRATEGY 3 (BB CROSSOVER WITH KAMA FILTER AND V-PATTERN) - PRIORITY 3 FUNCTIONS
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
    """Generate Priority 3 (BB Crossover with KAMA Filter) signals with immediate V-pattern reversal"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)

    current_position = 0.0
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

        if current_position == 0.0:
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)

            if (can_trade and cooldown_elapsed and not is_last_bar):
                initial_signal = 0.0

                if bullish_bb_cross and kama_bullish:
                    initial_signal = 1.0
                elif bearish_bb_cross and kama_bearish:
                    initial_signal = -1.0

                # Apply V-pattern reversal if initial signal exists
                if initial_signal != 0.0:
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

        elif current_position == 1.0:
            hold_time = current_time - entry_time

            if hold_time < min_hold_seconds:
                if is_last_bar:
                    current_position = 0.0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1.0
            else:
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price >= (entry_price + take_profit_offset))

                should_exit = (bearish_bb_cross or max_hold_exceeded or is_last_bar or take_profit_hit)

                if should_exit:
                    current_position = 0.0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1.0

        elif current_position == -1.0:
            hold_time = current_time - entry_time

            if hold_time < min_hold_seconds:
                if is_last_bar:
                    current_position = 0.0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1.0
            else:
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price <= (entry_price - take_profit_offset))

                should_exit = (bullish_bb_cross or max_hold_exceeded or is_last_bar or take_profit_hit)

                if should_exit:
                    current_position = 0.0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1.0

        prev_bb4 = current_bb4
        prev_bb6 = current_bb6

    return signals


def generate_p3_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 3 (BB Crossover with KAMA Filter) signals with V-pattern reversal"""
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
# STRATEGY 4 (KAMA-BASED SIMPLE) - PRIORITY 4 FUNCTIONS
# ============================================================================

def prepare_p4_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare features for KAMA-based strategy (Priority 4)"""

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


def generate_p4_signals(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate Priority 4 (KAMA-based Simple) signals"""
    # Prepare features
    df = prepare_p4_features(df, config)

    # Initialize state
    position = 0.0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = [0.0] * len(df)

    # Process row by row
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)

        signal = 0.0

        # EOD Square Off
        if is_last_tick and position != 0.0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P4_MIN_HOLD_SECONDS']

            if cooldown_over:
                # Volatility filter
                volatility_confirmed = (row["P4_ATR_High"] > config['P4_ATR_THRESHOLD'] and
                                        row["P4_STD_High"] > config['P4_STD_THRESHOLD'])

                # Entry conditions
                if position == 0.0 and volatility_confirmed:
                    if row["P4_KAMA_Slope"] > config['P4_KAMA_SLOPE_ENTRY']:
                        signal = 1.0
                    elif row["P4_KAMA_Slope"] < -config['P4_KAMA_SLOPE_ENTRY']:
                        signal = -1.0

                # Exit conditions (price-based)
                elif position != 0.0:
                    if position == 1.0:
                        price_diff = price - entry_price
                        if price_diff >= config['P4_TAKE_PROFIT']:
                            signal = -1.0
                    elif position == -1.0:
                        price_diff = entry_price - price
                        if price_diff >= config['P4_TAKE_PROFIT']:
                            signal = 1.0

        # Apply Signal
        if signal != 0.0:
            if position == 0.0 and signal != 0.0:
                entry_price = price

            proposed_position = position + signal
            if proposed_position > 1.0:
                signal = 1.0 - position
                position = 1.0
            elif proposed_position < -1.0:
                signal = -1.0 - position
                position = -1.0
            else:
                position = proposed_position

            last_signal_time = current_time
            if abs(round(position, 2)) < 1e-9:
                entry_price = None

        signals[i] = float(signal)

    return np.array(signals, dtype=float), df


# ============================================================================
# MANAGER LOGIC (4 PRIORITIES) - MODIFIED TO SUPPORT FRACTIONAL POSITIONS
# ============================================================================

def sign_safe(x: float) -> int:
    """Return sign with tolerance for tiny values."""
    if abs(x) < 1e-9:
        return 0
    return 1 if x > 0 else -1


def apply_manager_logic(df: pd.DataFrame, p1_signals: np.ndarray, p2_signals: np.ndarray,
                       p3_signals: np.ndarray, p4_signals: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply manager logic to combine Priority 1, 2, 3, and 4 signals.
    Modified to support fractional signals/positions (P2 choice).
    """
    n = len(df)
    final_signals = np.zeros(n, dtype=float)

    # Manager state
    position = 0.0
    position_owner = None  # None, "P1", "P2", "P3", "P4"
    last_signal_time = -config['UNIVERSAL_COOLDOWN']

    # Priority locks
    priority_2_locked = False
    priority_3_locked = False
    priority_4_locked = False

    # Pending state (for overrides: Exit -> Wait -> Enter)
    pending_signal = 0.0
    pending_owner = None
    pending_time = 0.0

    for i in range(n):
        current_time = df.iloc[i]['Time_sec']
        p1_sig = float(p1_signals[i]) if p1_signals is not None else 0.0
        p2_sig = float(p2_signals[i]) if p2_signals is not None else 0.0
        p3_sig = float(p3_signals[i]) if p3_signals is not None else 0.0
        p4_sig = float(p4_signals[i]) if p4_signals is not None else 0.0
        is_last_bar = (i == n - 1)

        cooldown_ok = (current_time - last_signal_time) >= config['UNIVERSAL_COOLDOWN']
        final_sig = 0.0

        s_pos = sign_safe(position)
        s_p1 = sign_safe(p1_sig)
        s_p2 = sign_safe(p2_sig)
        s_p3 = sign_safe(p3_sig)
        s_p4 = sign_safe(p4_sig)

        # ====================================================================
        # EOD SQUARE-OFF (Unified - overrides everything)
        # ====================================================================
        if is_last_bar:
            if position != 0.0:
                final_sig = -position
                position = 0.0
                position_owner = None
            pending_signal = 0.0  # Cancel pending

        # ====================================================================
        # PENDING ENTRY (Waiting for cooldown after override exit)
        # ====================================================================
        elif pending_signal != 0.0:
            if (current_time - pending_time) >= config['UNIVERSAL_COOLDOWN']:
                # Enter with pending fractional signal
                final_sig = pending_signal
                position = pending_signal
                position_owner = pending_owner

                # Apply locks based on who the pending owner was
                if pending_owner == "P1":
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                elif pending_owner == "P2":
                    priority_3_locked = True
                    priority_4_locked = True
                elif pending_owner == "P3":
                    priority_4_locked = True

                last_signal_time = current_time
                pending_signal = 0.0
                pending_owner = None

        # ====================================================================
        # FLAT POSITION
        # ====================================================================
        elif position == 0.0:
            if cooldown_ok and not is_last_bar:
                # Priority 1 entry
                if p1_sig != 0.0 and not pd.isna(p1_sig):
                    final_sig = p1_sig
                    position = p1_sig
                    position_owner = "P1"
                    priority_2_locked = True
                    priority_3_locked = True
                    priority_4_locked = True
                    last_signal_time = current_time

                # Priority 2 entry
                elif p2_sig != 0.0 and not priority_2_locked and not pd.isna(p2_sig):
                    final_sig = p2_sig
                    position = p2_sig
                    position_owner = "P2"
                    priority_3_locked = True
                    priority_4_locked = True
                    last_signal_time = current_time

                # Priority 3 entry
                elif p3_sig != 0.0 and not priority_3_locked and not pd.isna(p3_sig):
                    final_sig = p3_sig
                    position = p3_sig
                    position_owner = "P3"
                    priority_4_locked = True
                    last_signal_time = current_time

                # Priority 4 entry
                elif p4_sig != 0.0 and not priority_4_locked and not pd.isna(p4_sig):
                    final_sig = p4_sig
                    position = p4_sig
                    position_owner = "P4"
                    last_signal_time = current_time

        # ====================================================================
        # IN POSITION
        # ====================================================================
        else:
            override_triggered = False

            # Utility to test "opposite" taking fractional into account
            def is_opposite(sig_val, pos_val):
                return sign_safe(sig_val) != 0 and sign_safe(pos_val) != 0 and (sign_safe(sig_val) != sign_safe(pos_val))

            # Check P1 Override against P2, P3, P4
            if position_owner in ["P2", "P3", "P4"]:
                if p1_sig != 0.0 and is_opposite(p1_sig, position):
                    # Exit current fractional position
                    final_sig = -position
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                    # Set Pending State for P1 Entry (fractional)
                    pending_signal = p1_sig
                    pending_owner = "P1"
                    pending_time = current_time
                    override_triggered = True

            # Check P2 Override against P3, P4
            if not override_triggered and position_owner in ["P3", "P4"] and not priority_2_locked:
                if p2_sig != 0.0 and is_opposite(p2_sig, position):
                    final_sig = -position
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                    pending_signal = p2_sig
                    pending_owner = "P2"
                    pending_time = current_time
                    override_triggered = True

            # Check P3 Override against P4
            if not override_triggered and position_owner == "P4" and not priority_3_locked:
                if p3_sig != 0.0 and is_opposite(p3_sig, position):
                    final_sig = -position
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                    pending_signal = p3_sig
                    pending_owner = "P3"
                    pending_time = current_time
                    override_triggered = True

            # --- NORMAL EXIT CHECKS (If no override happened) ---
            if not override_triggered and cooldown_ok:
                # Owner-specific exit signals (fractional handled by sign check)
                if position_owner == "P1" and sign_safe(p1_sig) == -sign_safe(position):
                    final_sig = p1_sig
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                elif position_owner == "P2" and sign_safe(p2_sig) == -sign_safe(position):
                    final_sig = p2_sig
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                elif position_owner == "P3" and sign_safe(p3_sig) == -sign_safe(position):
                    final_sig = p3_sig
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

                elif position_owner == "P4" and sign_safe(p4_sig) == -sign_safe(position):
                    final_sig = p4_sig
                    position = 0.0
                    position_owner = None
                    last_signal_time = current_time

        final_signals[i] = float(final_sig)

    return final_signals


# ============================================================================
# DAY PROCESSING
# ============================================================================

def extract_day_num(filepath):
    """Extract day number from filepath"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def extract_day_num_csv(filepath):
    """Extract day number from filepath"""
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(args):
    """
    Process a single day with all four strategies and manager logic.
    Args packed into tuple for multiprocessing compatibility.
    """
    file_path, day_num, temp_dir, config = args

    try:
        # ====================================================================
        # 1. READ DATA (Pandas instead of Dask/CuDF)
        # ====================================================================
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P3_BB4_COLUMN'],
            config['P3_BB6_COLUMN'],
            config['P3_V_PATTERN_FEATURE'],
            config['P1_FEATURE_COLUMN']  # PB9_T1
        ]

        # Remove duplicates
        required_cols = list(dict.fromkeys(required_cols))

        # CPU Load
        df = pd.read_parquet(file_path, columns=required_cols)

        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None

        df = df.reset_index(drop=True)
        # Vectorized time conversion
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)

        # ====================================================================
        # 2. GENERATE STRATEGIES (With Memory Cleanup)
        # ====================================================================

        # --- Priority 1 (NEW) ---
        p1_signals, df_p1 = generate_p1_signals(df.copy(), config)
        del df_p1  # free P1 features immediately if desired (manager may need df for debug; but we delete to match old behavior)

        # --- Priority 4 ---
        p4_signals, df_with_p4_features = generate_p4_signals(df.copy(), config)

        # Extract only what P2 needs from P4 results
        p4_kama_slope = df_with_p4_features['P4_KAMA_Slope'].values.astype(np.float64)

        # Clean up P4 DataFrame immediately
        del df_with_p4_features
        gc.collect()

        # --- Priority 2 ---
        p2_signals = generate_p2_signals(df.copy(), config, p4_kama_slope)
        del p4_kama_slope  # Free slope array

        # --- Priority 3 ---
        p3_signals = generate_p3_signals(df.copy(), config)

        # --- Priority 4 signals were computed earlier; ensure shape alignment ---
        # Note: generate_p4_signals returned p4_signals earlier

        # ====================================================================
        # 3. MANAGER & OUTPUT
        # ====================================================================
        final_signals = apply_manager_logic(df, p1_signals, p2_signals, p3_signals, p4_signals, config)

        # Metrics: count non-zero signals as entries (fractional allowed)
        p1_entries = int(np.sum(np.abs(p1_signals) > 1e-9))
        p2_entries = int(np.sum(np.abs(p2_signals) > 1e-9))
        p3_entries = int(np.sum(np.abs(p3_signals) > 1e-9))
        p4_entries = int(np.sum(np.abs(p4_signals) > 1e-9))
        final_entries = int(np.sum(np.abs(final_signals) > 1e-9))

        # Create Output DataFrame (Signal column will be float)
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': final_signals
        })

        output_path = pathlib.Path(temp_dir) / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)

        # ====================================================================
        # 4. MEMORY CLEANUP
        # ====================================================================
        del df
        del p1_signals
        del p2_signals
        del p3_signals
        del p4_signals
        del final_signals
        del output_df
        gc.collect()  # Force garbage collection before returning to pool

        return {
            'path': str(output_path),
            'day_num': day_num,
            'p1_entries': p1_entries,
            'p2_entries': p2_entries,
            'p3_entries': p3_entries,
            'p4_entries': p4_entries,
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

    print("=" * 80)
    print("STRATEGY MANAGER - CPU MULTIPROCESSING MODE (WITH FRACTIONAL P1)")
    print("=" * 80)

    # Setup temp directory
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    # Find input files
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)

    if not sorted_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        return

    # Prepare arguments for multiprocessing
    tasks = [(f, extract_day_num(f), temp_dir_path, config) for f in sorted_files]

    # Configure Multiprocessing
    num_cores = min(config['MAX_WORKERS'], max(1, mp.cpu_count() - 1))
    print(f"\nProcessing {len(sorted_files)} files using {num_cores} CPU cores...")

    processed_files = []
    total_metrics = {'p1': 0, 'p2': 0, 'p3': 0, 'p4': 0, 'final': 0}

    try:
        with mp.Pool(processes=num_cores, maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(process_single_day, tasks):
                if result:
                    processed_files.append(result['path'])
                    total_metrics['p1'] += result['p1_entries']
                    total_metrics['p2'] += result['p2_entries']
                    total_metrics['p3'] += result['p3_entries']
                    total_metrics['p4'] += result['p4_entries']
                    total_metrics['final'] += result['final_entries']

                    print(f"✓ Processed day{result['day_num']:3d} | Final: {result['final_entries']:3d}")

        # Combine results
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return

        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=extract_day_num_csv)

        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Memory-efficient concat
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i != 0:
                        infile.readline()  # Skip header for subsequent files
                    shutil.copyfileobj(infile, outfile)

        # Final Stats calculation (using chunks to save memory if file is huge)
        print("\nCalculating final statistics...")
        num_long, num_short = 0, 0
        total_rows = 0

        for chunk in pd.read_csv(output_path, chunksize=100000):
            # Count positive and negative signals (fractional included)
            num_long += (chunk['Signal'] > 0).sum()
            num_short += (chunk['Signal'] < 0).sum()
            total_rows += len(chunk)

        print(f"\n{'=' * 80}")
        print("SIGNAL STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total bars:        {total_rows:,}")
        print(f"Total entries:     {(num_long + num_short):,}")
        print(f"P1 raw entries:    {total_metrics['p1']:,}")
        print(f"P2 raw entries:    {total_metrics['p2']:,}")
        print(f"P3 raw entries:    {total_metrics['p3']:,}")
        print(f"P4 raw entries:    {total_metrics['p4']:,}")
        print(f"{'=' * 80}")

    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print("Cleaned up temp directory")


if __name__ == "__main__":
    # Ensure spawn method for consistency (optional but recommended)
    mp.set_start_method('spawn')
    main(CONFIG)
