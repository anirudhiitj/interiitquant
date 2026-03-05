# PART 1/3
import sys
import os
import glob
import re
import shutil
import pathlib
import time
import pandas as pd
import numpy as np
import dask_cudf
import cudf
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

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

#############################################################################
# STRATEGY CONFIGURATION - MODIFY PARAMETERS HERE
#############################################################################

CONFIG = {
    # Data paths
    'DATA_DIR': '/data/quant14/EBY/',  # CHANGE THIS - parquet files directory
    'OUTPUT_DIR': '/data/quant14/signals/',  # CHANGE THIS
    'OUTPUT_FILE': 'ekf_signals.csv',
    'TEMP_DIR': '/tmp/ha_ekf_kama_temp',  # Temporary directory for intermediate files

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # Heikin Ashi parameters
    'ha_1min_tick_window': 60,       # Ticks per 1-min HA bar (60 ticks ≈ 1 min) - for ENTRY
    'ha_3min_tick_window': 240,      # Ticks per 3-min HA bar (180 ticks ≈ 3 min) - for EXIT

    # EKF (Extended Kalman Filter) parameters
    'ekf_process_noise': 0.05,       # Process noise Q
    'ekf_measurement_noise': 0.2,    # Measurement noise R

    # KAMA parameters (for EKF crossover)
    'kama_period': 15,               # KAMA calculation period
    'kama_fast_period': 25,          # Fast smoothing constant period
    'kama_slow_period': 100,         # Slow smoothing constant period
    'pb13_columns': [f'PB13_T{i}' for i in range(1, 11)],  # PB13_T1 to PB13_T10
    'pb13_threshold': 0.0001,
    'min_aligned_features': 9,
    # Position management
    'take_profit_points': 2.0,       # Fixed take profit in points
    'min_signal_gap_seconds': 60,     # Minimum 5 seconds between ANY signals

    # Strategy warmup period
    'strategy_start_tick': 900,     # No signals before this tick (30 min warmup)

    # Execution parameters
    'MAX_WORKERS': 32,               # Number of parallel workers for day processing
}

#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def calculate_ekf_numba(prices):
    """
    Calculate Extended Kalman Filter trend on prices.
    Uses log transformation for better numerical stability.

    Args:
        prices: Price array (numpy array)

    Returns:
        Array of EKF trend values
    """
    n = len(prices)
    ekf_trend = np.zeros(n, dtype=np.float64)

    # EKF parameters
    Q = 0.05  # Process noise
    R = 0.2   # Measurement noise

    # Initialize state
    x = prices[0]  # State estimate
    P = 1.0        # Error covariance

    ekf_trend[0] = x

    for i in range(1, n):
        # Predict
        x_pred = x
        P_pred = P + Q

        # Update (using log transformation)
        if prices[i] > 1e-10:
            z = np.log(prices[i])  # Measurement
            h = np.log(x_pred) if x_pred > 1e-10 else 0.0  # Expected measurement
            H = 1.0 / x_pred if x_pred > 1e-10 else 0.0  # Jacobian

            # Kalman gain
            S = H * P_pred * H + R
            K = P_pred * H / S if S > 1e-10 else 0.0

            # Update estimate
            x = x_pred + K * (z - h)
            P = (1.0 - K * H) * P_pred
        else:
            x = x_pred
            P = P_pred

        ekf_trend[i] = x

    return ekf_trend

@jit(nopython=True, fastmath=True)
def check_momentum_alignment(pb13_features, idx, direction, threshold, min_aligned):
    """
    Check if momentum features align with trade direction.
    
    Args:
        pb13_features: 2D array (n_ticks, 10) of PB13_T1 to PB13_T10 values
        idx: Current tick index
        direction: 1 for long, -1 for short
        threshold: Minimum absolute value for a feature to be considered
        min_aligned: Minimum number of features that must align (e.g., 8)
    
    Returns:
        True if at least min_aligned features align with direction
    """
    if idx >= len(pb13_features):
        return False
    
    aligned_count = 0
    
    for feature_idx in range(10):  # 10 features (T1 to T10)
        feature_val = pb13_features[idx, feature_idx]
        
        # Check if feature value is above threshold
        if abs(feature_val) > threshold:
            # Check if direction aligns
            if direction == 1:  # Long trade
                if feature_val > 0:  # Positive momentum
                    aligned_count += 1
            else:  # Short trade (direction == -1)
                if feature_val < 0:  # Negative momentum
                    aligned_count += 1
    
    return aligned_count >= min_aligned

@jit(nopython=True, fastmath=True)
def calculate_kama_numba(prices, period, fast_period, slow_period):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA) with Numba.

    Args:
        prices: Price array
        period: Lookback period for efficiency ratio
        fast_period: Fast smoothing constant period
        slow_period: Slow smoothing constant period

    Returns:
        Array of KAMA values
    """
    n = len(prices)
    kama = np.zeros(n, dtype=np.float64)

    # Calculate efficiency ratio components
    signal = np.zeros(n, dtype=np.float64)
    noise = np.zeros(n, dtype=np.float64)

    for i in range(period, n):
        signal[i] = abs(prices[i] - prices[i - period])

        noise_sum = 0.0
        for j in range(period):
            noise_sum += abs(prices[i - j] - prices[i - j - 1])
        noise[i] = noise_sum

    # Smoothing constants
    sc_fast = 2.0 / (fast_period + 1)
    sc_slow = 2.0 / (slow_period + 1)

    # Initialize KAMA
    first_valid = period
    if first_valid < n:
        kama[first_valid] = prices[first_valid]

    # Calculate KAMA
    for i in range(first_valid + 1, n):
        if noise[i] > 1e-10:
            er = signal[i] / noise[i]
        else:
            er = 0.0

        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])

    return kama


@jit(nopython=True, fastmath=True)
def calculate_atr_numba(prices, period):
    """
    Calculate Average True Range (ATR) using rolling high-low with EWM smoothing.
    ATR = EWM((rolling_high - rolling_low), span=period)

    Args:
        prices: Price array
        period: Lookback period for rolling high/low and EWM span

    Returns:
        Array of ATR values
    """
    n = len(prices)
    atr = np.zeros(n, dtype=np.float64)

    # Calculate rolling high and low
    rolling_high = np.zeros(n, dtype=np.float64)
    rolling_low = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i < period:
            # Not enough data yet - use what we have
            window_start = 0
        else:
            window_start = i - period + 1

        # Get rolling high and low
        rolling_high[i] = np.max(prices[window_start:i+1])
        rolling_low[i] = np.min(prices[window_start:i+1])

    # Calculate true range (high - low)
    true_range = rolling_high - rolling_low

    # Apply EWM smoothing
    # EWM formula: S_t = α * X_t + (1 - α) * S_{t-1}
    # where α = 2 / (span + 1)
    alpha = 2.0 / (period + 1)

    # Initialize with first value
    if n > 0:
        atr[0] = true_range[0]

    # Calculate EWM
    for i in range(1, n):
        atr[i] = alpha * true_range[i] + (1.0 - alpha) * atr[i-1]

    return atr


@jit(nopython=True, fastmath=True)
def calculate_heikin_ashi_numba(open_prices, high_prices, low_prices, close_prices):
    """
    Calculate Heikin Ashi values from OHLC data.

    Args:
        open_prices: Open array
        high_prices: High array
        low_prices: Low array
        close_prices: Close array

    Returns:
        Tuple of (HA_Open, HA_High, HA_Low, HA_Close)
    """
    n = len(open_prices)
    ha_open = np.zeros(n, dtype=np.float64)
    ha_high = np.zeros(n, dtype=np.float64)
    ha_low = np.zeros(n, dtype=np.float64)
    ha_close = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # HA Close = (O + H + L + C) / 4
        ha_close[i] = (open_prices[i] + high_prices[i] + low_prices[i] + close_prices[i]) / 4

        if i == 0:
            # First bar: HA_Open = (Open + Close) / 2
            ha_open[i] = (open_prices[i] + close_prices[i]) / 2
        else:
            # HA_Open = (prev_HA_Open + prev_HA_Close) / 2
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2

        # HA_High = max(H, HA_Open, HA_Close)
        ha_high[i] = max(high_prices[i], ha_open[i], ha_close[i])

        # HA_Low = min(L, HA_Open, HA_Close)
        ha_low[i] = min(low_prices[i], ha_open[i], ha_close[i])

    return ha_open, ha_high, ha_low, ha_close

# PART 2/3

@jit(nopython=True, fastmath=True)
def generate_ha_ekf_kama_signals(
    timestamps,
    prices,
    ekf_trend,
    kama,
    tick_to_ha_bar_1min,
    ha_1min_open,
    ha_1min_close,
    tick_to_ha_bar_3min,
    ha_3min_open,
    ha_3min_close,
    pb13_features,  # NEW: 2D array of PB13 features
    pb13_threshold,  # NEW
    min_aligned_features,  # NEW
    tp_points,
    min_gap_seconds,
    strategy_start_tick
):
    """
    Generate trading signals using Heikin Ashi + EKF + KAMA crossover strategy.
    Efficiency Ratio REMOVED entirely.

    Strategy Logic (State Machine):

    States: NEUTRAL, LONG, SHORT

    From NEUTRAL:
        * EKF crosses above KAMA AND previous completed 1-MIN HA bar is green → ENTER LONG
        * EKF crosses below KAMA AND previous completed 1-MIN HA bar is red → ENTER SHORT

    From LONG:
        * Price >= entry_price + tp_points → EXIT
        * EKF crosses below KAMA → EXIT
        * 3-MIN HA bar just completed AND it closed red → EXIT

    From SHORT:
        * Price <= entry_price - tp_points → EXIT
        * EKF crosses above KAMA → EXIT
        * 3-MIN HA bar just completed AND it closed green → EXIT

    EOD → mandatory exit.

    Returns:
        Signal array (1=long entry, -1=short entry or exit, 0=no action)
    """

    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)

    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = -1

    state = STATE_NEUTRAL
    entry_idx = -1
    entry_price = 0.0
    tp_price = 0.0
    last_signal_idx = -1

    for i in range(n):
        if i < strategy_start_tick:
            signals[i] = 0
            continue

        is_last_tick = (i == n - 1)

        current_time = timestamps[i]
        current_price = prices[i]

        ekf_val = ekf_trend[i]
        kama_val = kama[i]

        if i > 0:
            ekf_prev = ekf_trend[i-1]
            kama_prev = kama[i-1]
        else:
            ekf_prev = ekf_val
            kama_prev = kama_val

        ekf_crosses_above = (ekf_val > kama_val) and (ekf_prev <= kama_prev)
        ekf_crosses_below = (ekf_val < kama_val) and (ekf_prev >= kama_prev)

        ha_bar_1min_idx = tick_to_ha_bar_1min[i]
        ha_bar_3min_idx = tick_to_ha_bar_3min[i]

        time_since_signal = (
            current_time - timestamps[last_signal_idx]
            if last_signal_idx >= 0 else np.inf
        )

        signal = 0

        # ------------------------------------------------------
        # STATE: NEUTRAL
        # ------------------------------------------------------
        if state == STATE_NEUTRAL:

            if time_since_signal < min_gap_seconds:
                continue

            if is_last_tick:
                continue

            if ha_bar_1min_idx < 0:
                continue

            prev_ha_bar_1min_idx = ha_bar_1min_idx - 1 if ha_bar_1min_idx > 0 else -1

            if prev_ha_bar_1min_idx >= 0:
                prev_o = ha_1min_open[prev_ha_bar_1min_idx]
                prev_c = ha_1min_close[prev_ha_bar_1min_idx]
                prev_is_green = prev_c > prev_o
                prev_is_red = prev_c < prev_o
            else:
                continue

            # LONG ENTRY: EKF crosses above KAMA + prev HA bar is green + momentum aligns
            if ekf_crosses_above and prev_is_green:
                # Check momentum alignment for LONG
                if check_momentum_alignment(pb13_features, i, 1, pb13_threshold, min_aligned_features):
                    signal = 1
                    state = STATE_LONG
                    entry_idx = i
                    entry_price = current_price
                    tp_price = entry_price + tp_points
                    last_signal_idx = i

            # SHORT ENTRY: EKF crosses below KAMA + prev HA bar is red + momentum aligns
            elif ekf_crosses_below and prev_is_red:
                # Check momentum alignment for SHORT
                if check_momentum_alignment(pb13_features, i, -1, pb13_threshold, min_aligned_features):
                    signal = -1
                    state = STATE_SHORT
                    entry_idx = i
                    entry_price = current_price
                    tp_price = entry_price - tp_points
                    last_signal_idx = i

        # ------------------------------------------------------
        # STATE: LONG
        # ------------------------------------------------------
        elif state == STATE_LONG:

            exit_signal = 0

            if is_last_tick:
                exit_signal = -1

            elif current_price >= tp_price and time_since_signal >= min_gap_seconds:
                exit_signal = -1

            elif ekf_crosses_below and time_since_signal >= min_gap_seconds:
                exit_signal = -1

            elif i > 0:
                bar_just_completed = (
                    tick_to_ha_bar_3min[i] != tick_to_ha_bar_3min[i-1] and
                    tick_to_ha_bar_3min[i-1] >= 0
                )
                if bar_just_completed:
                    completed_idx = tick_to_ha_bar_3min[i-1]
                    o = ha_3min_open[completed_idx]
                    c = ha_3min_close[completed_idx]
                    is_red = c < o
                    if is_red and time_since_signal >= min_gap_seconds:
                        exit_signal = -1

            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                last_signal_idx = i

        # ------------------------------------------------------
        # STATE: SHORT
        # ------------------------------------------------------
        elif state == STATE_SHORT:

            exit_signal = 0

            if is_last_tick:
                exit_signal = 1

            elif current_price <= tp_price and time_since_signal >= min_gap_seconds:
                exit_signal = 1

            elif ekf_crosses_above and time_since_signal >= min_gap_seconds:
                exit_signal = 1

            elif i > 0:
                bar_just_completed = (
                    tick_to_ha_bar_3min[i] != tick_to_ha_bar_3min[i-1] and
                    tick_to_ha_bar_3min[i-1] >= 0
                )
                if bar_just_completed:
                    completed_idx = tick_to_ha_bar_3min[i-1]
                    o = ha_3min_open[completed_idx]
                    c = ha_3min_close[completed_idx]
                    is_green = c > o
                    if is_green and time_since_signal >= min_gap_seconds:
                        exit_signal = 1

            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                last_signal_idx = i

        signals[i] = signal

    return signals


#############################################################################
# FILE PROCESSING FUNCTIONS
#############################################################################

def extract_day_num(filepath):
    """Extract day number from filepath."""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    """
    Process a single day file using GPU-accelerated operations + HA + EKF + KAMA.

    NOTE: Efficiency Ratio REMOVED.
    """
    try:
        # ====================================================================
        # 1. READ DATA WITH GPU (dask_cudf -> cudf -> pandas)
        # ====================================================================
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']] + config['pb13_columns']

        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()

        if df.empty:
            print(f"  Warning: Day {day_num} file is empty. Skipping.")
            return None

        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)

        # ====================================================================
        # 2. PREPARE DATA FOR NUMBA
        # ====================================================================
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        pb13_features = df[config['pb13_columns']].values.astype(np.float64)
        # Handle NaN values in PB13 features
        pb13_features = np.where(np.isnan(pb13_features), 0.0, pb13_features)
        prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)

        # ====================================================================
        # 3. CALCULATE EKF TREND
        # ====================================================================
        ekf_trend = calculate_ekf_numba(prices)

        # ====================================================================
        # 4. CALCULATE KAMA
        # ====================================================================
        kama = calculate_kama_numba(
            prices,
            config['kama_period'],
            config['kama_fast_period'],
            config['kama_slow_period']
        )

        # ====================================================================
        # 5. RESAMPLE INTO HEIKIN ASHI BARS
        # ====================================================================
        tick_window_1min = config['ha_1min_tick_window']
        tick_window_3min = config['ha_3min_tick_window']

        n_complete_bars_1min = len(prices) // tick_window_1min
        if n_complete_bars_1min == 0:
            print(f"  Warning: Day {day_num} insufficient 1-min data.")
            return None

        ohlc_data_1min = []
        for bar_idx in range(n_complete_bars_1min):
            s = bar_idx * tick_window_1min
            e = (bar_idx + 1) * tick_window_1min
            pr = prices[s:e]
            ohlc_data_1min.append({
                'Open': pr[0],
                'High': np.max(pr),
                'Low': np.min(pr),
                'Close': pr[-1]
            })
        ohlc_df_1min = pd.DataFrame(ohlc_data_1min)

        n_complete_bars_3min = len(prices) // tick_window_3min
        if n_complete_bars_3min == 0:
            print(f"  Warning: Day {day_num} insufficient 3-min data.")
            return None

        ohlc_data_3min = []
        for bar_idx in range(n_complete_bars_3min):
            s = bar_idx * tick_window_3min
            e = (bar_idx + 1) * tick_window_3min
            pr = prices[s:e]
            ohlc_data_3min.append({
                'Open': pr[0],
                'High': np.max(pr),
                'Low': np.min(pr),
                'Close': pr[-1]
            })
        ohlc_df_3min = pd.DataFrame(ohlc_data_3min)

        # ====================================================================
        # 6. CALCULATE HEIKIN ASHI FOR BOTH TFs
        # ====================================================================
        ha_1min_open, ha_1min_high, ha_1min_low, ha_1min_close = calculate_heikin_ashi_numba(
            ohlc_df_1min['Open'].values,
            ohlc_df_1min['High'].values,
            ohlc_df_1min['Low'].values,
            ohlc_df_1min['Close'].values
        )

        ha_3min_open, ha_3min_high, ha_3min_low, ha_3min_close = calculate_heikin_ashi_numba(
            ohlc_df_3min['Open'].values,
            ohlc_df_3min['High'].values,
            ohlc_df_3min['Low'].values,
            ohlc_df_3min['Close'].values
        )

        # ====================================================================
        # 7. MAP TICKS TO HA BARS
        # ====================================================================
        tick_to_ha_bar_1min = np.full(len(prices), -1, dtype=np.int32)
        n_ticks_truncated_1min = n_complete_bars_1min * tick_window_1min
        for i in range(len(prices)):
            if i < n_ticks_truncated_1min:
                tick_to_ha_bar_1min[i] = i // tick_window_1min

        tick_to_ha_bar_3min = np.full(len(prices), -1, dtype=np.int32)
        n_ticks_truncated_3min = n_complete_bars_3min * tick_window_3min
        for i in range(len(prices)):
            if i < n_ticks_truncated_3min:
                tick_to_ha_bar_3min[i] = i // tick_window_3min

        # ====================================================================
        # 8. GENERATE SIGNALS
        # ====================================================================
        signals = generate_ha_ekf_kama_signals(
            timestamps=timestamps,
            prices=prices,
            ekf_trend=ekf_trend,
            kama=kama,
            tick_to_ha_bar_1min=tick_to_ha_bar_1min,
            ha_1min_open=ha_1min_open,
            ha_1min_close=ha_1min_close,
            tick_to_ha_bar_3min=tick_to_ha_bar_3min,
            ha_3min_open=ha_3min_open,
            ha_3min_close=ha_3min_close,
            pb13_features=pb13_features,  # NEW
            pb13_threshold=config['pb13_threshold'],  # NEW
            min_aligned_features=config['min_aligned_features'],  # NEW
            tp_points=config['take_profit_points'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            strategy_start_tick=config['strategy_start_tick']
        )

        # ====================================================================
        # 9. COUNT SIGNALS
        # ====================================================================
        long_entries = np.sum(signals == 1)
        short_entries = np.sum(signals == -1)
        total_signals = long_entries + short_entries

        if abs(long_entries - short_entries) > 1:
            print(f"  WARNING Day {day_num}: Position imbalance! Long: {long_entries}, Short: {short_entries}")

        # ====================================================================
        # 10. SAVE RESULTS FOR THIS DAY
        # ====================================================================
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': signals
        })

        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)

        return {
            'path': str(output_path),
            'day_num': day_num,
            'total_signals': total_signals,
            'long_entries': long_entries,
            'short_entries': short_entries
        }

    except Exception as e:
        print(f"  ERROR processing day {day_num}: {e}")
        traceback.print_exc()
        return None

# PART 3/3

#############################################################################
# MAIN EXECUTION
#############################################################################

def main(config):
    """Main execution function."""
    start_time = time.time()

    print("\n" + "="*80)
    print("HEIKIN ASHI + EKF + KAMA CROSSOVER STRATEGY")
    print("GPU-ACCELERATED PARALLEL PROCESSING")
    print("FORWARD BIAS FIXED - BAR COMPLETION LOGIC")
    print("="*80)
    print(f"\nCONFIGURATION:")
    print(f"  Data directory: {config['DATA_DIR']}")
    print(f"  Output directory: {config['OUTPUT_DIR']}")
    print(f"  Output file: {config['OUTPUT_FILE']}")
    print(f"  Temp directory: {config['TEMP_DIR']}")
    print(f"  Parallel workers: {config['MAX_WORKERS']}")
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"  1-Min HA tick window: {config['ha_1min_tick_window']} ticks (≈1 minute) - ENTRY")
    print(f"  3-Min HA tick window: {config['ha_3min_tick_window']} ticks (≈3 minutes) - EXIT")
    print(f"  EKF process noise: {config['ekf_process_noise']}")
    print(f"  EKF measurement noise: {config['ekf_measurement_noise']}")
    print(f"  KAMA period: {config['kama_period']}")
    print(f"  KAMA fast period: {config['kama_fast_period']}")
    print(f"  KAMA slow period: {config['kama_slow_period']}")
    print(f"  Take profit: {config['take_profit_points']} points")
    print(f"  Min signal gap: {config['min_signal_gap_seconds']} seconds")
    print(f"  Strategy start tick: {config['strategy_start_tick']} (warmup period)")
    print(f"\nSTRATEGY LOGIC:")
    print(f"  Entry Conditions (ALL must be true):")
    print(f"    LONG:  EKF crosses above KAMA + Previous COMPLETED 1-MIN HA bar is GREEN")
    print(f"    SHORT: EKF crosses below KAMA + Previous COMPLETED 1-MIN HA bar is RED")
    print(f"  Exit Conditions (any of):")
    print(f"    1. Take profit hit (±{config['take_profit_points']} points)")
    print(f"    2. Opposite EKF/KAMA crossover")
    print(f"    3. 3-MIN HA bar JUST COMPLETED with opposite color (NO FORWARD BIAS)")
    print(f"    4. EOD square-off (mandatory)")
    print(f"\nKEY FEATURES:")
    print(f"  ✓ Tick-level crossover detection (every tick)")
    print(f"  ✓ 1-MIN HA bar color confirmation for ENTRY (previous completed bar)")
    print(f"  ✓ 3-MIN HA bar color confirmation for EXIT (on bar completion only)")
    print(f"  ✓ Fixed take profit (no stop loss)")
    print(f"  ✓ State machine prevents invalid transitions")
    print(f"  ✓ {config['min_signal_gap_seconds']}s gap enforcement between ANY signals")
    print(f"  ✓ {config['strategy_start_tick']}-tick warmup period")
    print(f"  ✓ EOD square-off mandatory")
    print(f"  ✓ NO FORWARD BIAS - only uses completed bar data")
    print("="*80)

    # ========================================================================
    # SETUP TEMP DIRECTORY
    # ========================================================================
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    print(f"Created temp directory: {temp_dir_path}")

    # ========================================================================
    # FIND INPUT FILES
    # ========================================================================
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)

    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        sys.exit(1)

    print(f"\nFound {len(sorted_files)} day files to process")
    for f in sorted_files[:5]:
        print(f"  - {os.path.basename(f)}")
    if len(sorted_files) > 5:
        print(f"  ... and {len(sorted_files) - 5} more")

    # ========================================================================
    # PROCESS ALL DAYS IN PARALLEL
    # ========================================================================
    print(f"\n" + "="*80)
    print("PROCESSING FILES IN PARALLEL")
    print("="*80)
    print(f"Using {config['MAX_WORKERS']} workers...\n")

    processed_files = []
    total_signals = 0
    total_long_entries = 0
    total_short_entries = 0

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
                        total_signals += result['total_signals']
                        total_long_entries += result['long_entries']
                        total_short_entries += result['short_entries']
                        print(f"✓ Day {result['day_num']:3d} | Signals: {result['total_signals']:3d} | Long: {result['long_entries']:3d} | Short: {result['short_entries']:3d}")
                except Exception as e:
                    print(f"✗ Error in {os.path.basename(file_path)}: {e}")

        # ====================================================================
        # COMBINE RESULTS
        # ====================================================================
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return

        print(f"\n" + "="*80)
        print("COMBINING RESULTS")
        print("="*80)
        print(f"Combining {len(processed_files)} day files...")

        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))

        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine CSV files
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()  # Skip header
                        shutil.copyfileobj(infile, outfile)

        print(f"✓ Created combined output: {output_path}")

        # ====================================================================
        # FINAL STATISTICS
        # ====================================================================
        final_df = pd.read_csv(output_path)
        num_long = (final_df['Signal'] == 1).sum()
        num_short = (final_df['Signal'] == -1).sum()
        num_hold = (final_df['Signal'] == 0).sum()
        total_bars = len(final_df)

        print(f"\n" + "="*80)
        print("FINAL STATISTICS")
        print("="*80)
        print(f"Total ticks:       {total_bars:,}")
        print(f"Long entries:      {num_long:,}")
        print(f"Short entries:     {num_short:,}")
        print(f"Hold/Wait:         {num_hold:,}")
        print(f"Total signals:     {(num_long + num_short):,}")
        print(f"Position balance:  {abs(num_long - num_short)}")
        print(f"Avg per day:       {(num_long + num_short)/len(processed_files):.1f}")
        print(f"Signal density:    {((num_long + num_short)/total_bars)*100:.4f}%")
        print("="*80)

    finally:
        # Cleanup temp directory
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print(f"\nCleaned up temp directory")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ Processing speed: {len(sorted_files)/elapsed:.2f} days/second")
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("  1. Backtest with your backtesting framework")
    print("  2. Analyze crossover timing vs HA bar color")
    print("  3. Evaluate exit condition hit rates")
    print("  4. Test different HA tick windows (30, 60, 120)")
    print("  5. Optimize EKF and KAMA parameters")
    print("="*80 + "\n")


if __name__ == "__main__":
    main(CONFIG)
