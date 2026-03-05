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
    # Create a dummy decorator if Numba is not available
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
    'MAX_WORKERS': 8,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy 1 (KF) - Priority 1
    'P1_FEATURE': 'PB9_T1',
    
    # Strategy 2 (BB) - Priority 2
    'P2_BB4_COLUMN': 'BB4_T9',
    'P2_BB6_COLUMN': 'BB6_T11',
    'P2_VOLATILITY_FEATURE': 'PB9_T1',
    'P2_MOMENTUM_COLUMNS': [f'PB13_T{i}' for i in range(1, 11)],
    
    # Strategy 2 Parameters
    'P2_ATR_PERIOD': 300,
    'P2_ACTIVATION_ATR_THRESHOLD': 0.01,
    'P2_ACTIVATION_STDDEV_THRESHOLD': 0.06,
    'P2_USE_SMA_FILTER': True,
    'P2_SMA_PERIOD': 100,
    'P2_STDDEV_MULTIPLIER': 0.2,
    'P2_USE_MOMENTUM_FILTER': True,
    'P2_MOMENTUM_ALIGNMENT_THRESHOLD': 0.9,
    'P2_MIN_HOLD_SECONDS': 15.0,
    'P2_MAX_HOLD_SECONDS': 3600*2.2,
    'P2_USE_TAKE_PROFIT': True,
    'P2_TAKE_PROFIT_OFFSET': 0.8,
    'P2_STRATEGY_START_BAR': 2400,
}

# ============================================================================
# STRATEGY 1 (KF) - PRIORITY 1 FUNCTIONS
# ============================================================================

def prepare_derived_features_p1(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for Priority 1 (KF) strategy"""
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.01
    observation_covariance = 1.0
    initial_state_covariance = 1.0
    initial_state_mean = 100
    
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    
    filtered_state_means, _ = kf.filter(df["PB9_T1"].dropna())
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df["PB9_T1"].dropna().index)
    df["KFTrend"] = smoothed_prices
    df["KFTrend_Lagged"] = df["KFTrend"].shift(1)
    
    stl_result = STL(df['PB9_T1'].dropna(), period=30, robust=True).fit()
    df['STL'] = stl_result.trend
    df['STL_Lagged'] = df['STL'].shift(1)
    
    df["SMA_15"] = df["PB9_T1"].rolling(window=15, min_periods=1).mean()
    df["SMA_30"] = df["PB9_T1"].rolling(window=30, min_periods=1).mean()
    
    # RSI
    window = 14
    delta = df["PB9_T1"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = rsi
    
    # KAMA
    kama_signal = df["PB9_T1"].diff(30).abs()
    kama_noise = df["PB9_T1"].diff(1).abs().rolling(window=30).sum()
    er = (kama_signal / kama_noise).fillna(0)
    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow).pow(2)
    
    kama = pd.Series(np.nan, index=df["PB9_T1"].index, dtype=float)
    first_valid_index = sc.first_valid_index()
    if first_valid_index is None:
        return df
    
    kama.loc[first_valid_index] = df["PB9_T1"].loc[first_valid_index]
    sc_values = sc.to_numpy()
    price_values = df["PB9_T1"].to_numpy()
    kama_values = kama.to_numpy()
    
    for i in range(first_valid_index + 1, len(df["PB9_T1"])):
        if np.isnan(kama_values[i-1]):
            kama_values[i] = price_values[i]
        else:
            kama_values[i] = kama_values[i-1] + sc_values[i] * (price_values[i] - kama_values[i-1])
    
    df["KAMA"] = pd.Series(kama_values, index=df["PB9_T1"].index)
    df["KAMA_Slope"] = df["KAMA"] - df["KAMA"].shift(2)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # ATR and STD
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=20).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].cummax()
    df["STD"] = df["PB9_T1"].rolling(window=20, min_periods=1).std()
    df["STD_High"] = df["STD"].cummax()
    
    return df

# ############################################################################
# ## SOLUTION 1: NEW NUMBA-ACCELERATED P1 FUNCTION
# ############################################################################
@jit(nopython=True, fastmath=True)
def generate_p1_signals_numba(
    timestamps,
    prices,
    kf_trends,
    kf_trends_lagged,
    stl_trends,
    stl_trends_lagged,
    kama_slope_abs,
    cooldown_seconds
):
    """
    Numba-accelerated version of Priority 1 signals.
    Returns array of signals: 1=enter long, -1=enter short, 0=hold
    """
    signals = np.zeros(len(prices), dtype=np.int8)
    position = 0
    entry_price = 0.0  # Use 0.0, not None
    last_signal_time = -cooldown_seconds
    
    for i in range(len(prices)):
        current_time = timestamps[i]
        price = prices[i]
        is_last_tick = (i == len(prices) - 1)
        
        signal = 0
        
        # EOD Square Off (will be handled by manager)
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= cooldown_seconds
            
            if cooldown_over:
                if position == 0:
                    # Note: NaNs should be pre-filled (e.g., with 0) before calling this function
                    if kf_trends[i] < stl_trends[i] and kama_slope_abs[i] > 0.001:
                        signal = 1
                    elif kf_trends[i] > stl_trends[i] and kama_slope_abs[i] > 0.001:
                        signal = -1
                elif position == 1:
                    if (kf_trends_lagged[i] < stl_trends_lagged[i] and kf_trends[i] > stl_trends[i]):
                        signal = -1
                elif position == -1:
                    if (kf_trends_lagged[i] > stl_trends_lagged[i] and kf_trends[i] < stl_trends[i]):
                        signal = 1
            
            # Price-based exit
            if cooldown_over and position != 0:
                if position == 1:
                    price_diff = price - entry_price
                    if price_diff >= 0.3:
                        signal = -1
                elif position == -1:
                    price_diff = entry_price - price
                    if price_diff >= 0.3:
                        signal = 1
        
        # Apply signal
        if signal != 0:
            # Prevent pyramiding
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:
                    entry_price = 0.0
                else:
                    entry_price = price
        
        signals[i] = signal
    
    return signals


# ============================================================================
# STRATEGY 2 (BB) - PRIORITY 2 FUNCTIONS
# ============================================================================
# (This section is unchanged as it was already Numba-optimized)

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
def calculate_alignment_score(pb13_values):
    """Calculate momentum alignment score across 10 timeframes"""
    n = pb13_values.shape[0]
    alignment_scores = np.zeros(n, dtype=np.float64)
    noise_threshold = 0.00001
    
    for i in range(n):
        positive_count = 0
        negative_count = 0
        
        for j in range(10):
            if pb13_values[i, j] > noise_threshold:
                positive_count += 1
            elif pb13_values[i, j] < -noise_threshold:
                negative_count += 1
        
        max_agreement = max(positive_count, negative_count)
        alignment_scores[i] = max_agreement / 10.0
    
    return alignment_scores


@jit(nopython=True, fastmath=True)
def generate_p2_signals_internal(
    prices,
    bb4_values,
    bb6_values,
    timestamps,
    atr_values,
    pb9_stddev_values,
    sma_values,
    stddev_values,
    alignment_scores,
    use_sma_filter,
    use_momentum_filter,
    momentum_alignment_threshold,
    activation_atr_threshold,
    activation_stddev_threshold,
    stddev_multiplier,
    min_hold_seconds,
    max_hold_seconds,
    cooldown_seconds,
    use_take_profit,
    take_profit_offset,
    strategy_start_bar
):
    """Generate Priority 2 (BB) signals"""
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    strategy_activated = False
    current_position = 0
    entry_time = 0.0
    entry_price = 0.0
    last_exit_time = 0.0
    
    prev_bb4 = bb4_values[0]
    prev_bb6 = bb6_values[0]
    prev_price = prices[0]
    
    for i in range(1, n):
        current_time = timestamps[i]
        current_price = prices[i]
        current_bb4 = bb4_values[i]
        current_bb6 = bb6_values[i]
        current_atr = atr_values[i]
        current_pb9_stddev = pb9_stddev_values[i]
        current_sma = sma_values[i]
        current_stddev = stddev_values[i]
        current_alignment_score = alignment_scores[i]
        
        upper_band = current_sma + stddev_multiplier * current_stddev
        lower_band = current_sma - stddev_multiplier * current_stddev
        
        prev_sma = sma_values[i-1]
        prev_stddev = stddev_values[i-1]
        prev_upper_band = prev_sma + stddev_multiplier * prev_stddev
        prev_lower_band = prev_sma - stddev_multiplier * prev_stddev
        
        # Check activation
        if not strategy_activated:
            if (current_atr > activation_atr_threshold and 
                current_pb9_stddev > activation_stddev_threshold):
                strategy_activated = True
        
        can_trade = strategy_activated and (i >= strategy_start_bar)
        
        # Detect crossovers
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        price_below_upper = (prev_price >= prev_upper_band) and (current_price < upper_band)
        price_above_lower = (prev_price <= prev_lower_band) and (current_price > lower_band)
        
        if use_sma_filter:
            price_above_upper_band = (current_price > upper_band)
            price_below_lower_band = (current_price < lower_band)
        else:
            price_above_upper_band = True
            price_below_lower_band = True
        
        if use_momentum_filter:
            momentum_filter_pass = (current_alignment_score >= momentum_alignment_threshold)
        else:
            momentum_filter_pass = True
        
        is_last_bar = (i == n - 1)
        
        # Position management
        if current_position == 0:
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)
            
            if (can_trade and cooldown_elapsed and momentum_filter_pass and not is_last_bar):
                if bullish_bb_cross and price_above_upper_band:
                    current_position = 1
                    entry_time = current_time
                    entry_price = current_price
                    signals[i] = 1
                elif bearish_bb_cross and price_below_lower_band:
                    current_position = -1
                    entry_time = current_time
                    entry_price = current_price
                    signals[i] = -1
        
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
                
                if use_sma_filter:
                    both_exit_conditions = bearish_bb_cross and price_below_upper
                else:
                    both_exit_conditions = bearish_bb_cross
                
                should_exit = (both_exit_conditions or max_hold_exceeded or is_last_bar or take_profit_hit)
                
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
                
                if use_sma_filter:
                    both_exit_conditions = bullish_bb_cross and price_above_lower
                else:
                    both_exit_conditions = bullish_bb_cross
                
                should_exit = (both_exit_conditions or max_hold_exceeded or is_last_bar or take_profit_hit)
                
                if should_exit:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1
        
        prev_bb4 = current_bb4
        prev_bb6 = current_bb6
        prev_price = current_price
    
    return signals


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 2 (BB) signals"""
    prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
    bb4_values = df[config['P2_BB4_COLUMN']].values.astype(np.float64)
    bb6_values = df[config['P2_BB6_COLUMN']].values.astype(np.float64)
    pb9_t1_values = df[config['P2_VOLATILITY_FEATURE']].values.astype(np.float64)
    timestamps = df['Time_sec'].values.astype(np.float64)
    
    # Calculate ATR
    true_range = calculate_true_range_from_pb9(pb9_t1_values)
    atr_values = rolling_mean_numba(true_range, config['P2_ATR_PERIOD'])
    
    # Calculate StdDev
    pb9_stddev_values = rolling_std_numba(pb9_t1_values, config['P2_ATR_PERIOD'])
    
    # Calculate SMA and StdDev for filter
    sma_values = rolling_mean_numba(pb9_t1_values, config['P2_SMA_PERIOD'])
    stddev_values = rolling_std_numba(pb9_t1_values, config['P2_SMA_PERIOD'])
    
    # Calculate momentum alignment
    if config['P2_USE_MOMENTUM_FILTER']:
        pb13_values = df[config['P2_MOMENTUM_COLUMNS']].values.astype(np.float64)
        alignment_scores = calculate_alignment_score(pb13_values)
    else:
        alignment_scores = np.ones(len(df), dtype=np.float64)
    
    signals = generate_p2_signals_internal(
        prices=prices,
        bb4_values=bb4_values,
        bb6_values=bb6_values,
        timestamps=timestamps,
        atr_values=atr_values,
        pb9_stddev_values=pb9_stddev_values,
        sma_values=sma_values,
        stddev_values=stddev_values,
        alignment_scores=alignment_scores,
        use_sma_filter=config['P2_USE_SMA_FILTER'],
        use_momentum_filter=config['P2_USE_MOMENTUM_FILTER'],
        momentum_alignment_threshold=config['P2_MOMENTUM_ALIGNMENT_THRESHOLD'],
        activation_atr_threshold=config['P2_ACTIVATION_ATR_THRESHOLD'],
        activation_stddev_threshold=config['P2_ACTIVATION_STDDEV_THRESHOLD'],
        stddev_multiplier=config['P2_STDDEV_MULTIPLIER'],
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

# ############################################################################
# ## SOLUTION 1: NEW NUMBA-ACCELERATED MANAGER FUNCTION
# ############################################################################
@jit(nopython=True, fastmath=True)
def apply_manager_logic_numba(
    timestamps, 
    p1_signals, 
    p2_signals, 
    cooldown_seconds
):
    """
    Apply manager logic to combine Priority 1 and Priority 2 signals
    
    Rules:
    1. Daily reset (handled by processing day-by-day)
    2. Once P1 trades -> P2 locked for entire day
    3. 15s universal cooldown between ANY signals
    4. If in P2 position and P1 wants to reverse -> exit P2, wait 15s, then P1 can enter
    5. Unified EOD square-off
    """
    n = len(timestamps)
    final_signals = np.zeros(n, dtype=np.int8)
    
    # Manager state
    position = 0  # 0=flat, 1=long, -1=short
    # Use integers for Numba: 0 = None, 1 = P1, 2 = P2
    position_owner = 0 
    last_signal_time = -cooldown_seconds
    priority_2_locked = False
    
    for i in range(n):
        current_time = timestamps[i]
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        is_last_bar = (i == n - 1)
        
        # Check cooldown
        cooldown_ok = (current_time - last_signal_time) >= cooldown_seconds
        
        final_sig = 0
        
        # ====================================================================
        # EOD SQUARE-OFF (Unified - overrides everything)
        # ====================================================================
        if is_last_bar and position != 0:
            final_sig = -position
            position = 0
            position_owner = 0 # None
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
                    position_owner = 1 # P1
                    priority_2_locked = True
                    last_signal_time = current_time
                
                # Priority 2 entry (only if P1 hasn't locked it)
                elif p2_sig != 0 and not priority_2_locked:
                    final_sig = p2_sig
                    position = p2_sig
                    position_owner = 2 # P2
                    last_signal_time = current_time
        
        # ====================================================================
        # IN POSITION
        # ====================================================================
        elif position != 0:
            if position_owner == 1: # P1
                # Priority 1 manages its own exit
                if cooldown_ok and p1_sig == -position:
                    final_sig = p1_sig
                    position = 0
                    position_owner = 0 # None
                    last_signal_time = current_time
            
            elif position_owner == 2: # P2
                # Check if Priority 1 wants to override/reverse
                if p1_sig != 0 and p1_sig != position:
                    # Priority 1 wants to reverse - exit P2 position
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = 0 # None
                        priority_2_locked = True
                        last_signal_time = current_time
                        # Note: P1 will enter on next bar after 15s cooldown
                
                # Otherwise check P2 exit
                elif cooldown_ok and p2_sig == -position:
                    final_sig = p2_sig
                    position = 0
                    position_owner = 0 # None
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
            config['P1_FEATURE'],
            config['P2_BB4_COLUMN'],
            config['P2_BB6_COLUMN'],
            config['P2_VOLATILITY_FEATURE'],
        ] + config['P2_MOMENTUM_COLUMNS']
        
        # Remove duplicates (P1_FEATURE and P2_VOLATILITY_FEATURE are both PB9_T1)
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
        # 2. GENERATE PRIORITY 1 SIGNALS
        # ====================================================================
        df_p1 = prepare_derived_features_p1(df.copy())
        
        # ####################################################################
        # ## SOLUTION 1: UPDATED P1 CALL
        # ####################################################################
        
        # Prepare arrays for Numba. 
        # Crucially, fillna(0) to handle NaNs from shift/rolling, which Numba hates.
        p1_input_cols = ['KFTrend', 'KFTrend_Lagged', 'STL', 'STL_Lagged', 'KAMA_Slope_abs']
        df_p1_numba_in = df_p1[p1_input_cols].fillna(0)

        p1_signals = generate_p1_signals_numba(
            df_p1['Time_sec'].values,
            df_p1['Price'].values,
            df_p1_numba_in['KFTrend'].values,
            df_p1_numba_in['KFTrend_Lagged'].values,
            df_p1_numba_in['STL'].values,
            df_p1_numba_in['STL_Lagged'].values,
            df_p1_numba_in['KAMA_Slope_abs'].values,
            config['UNIVERSAL_COOLDOWN']
        )
        
        # ====================================================================
        # 3. GENERATE PRIORITY 2 SIGNALS
        # ====================================================================
        p2_signals = generate_p2_signals(df, config)
        
        # ====================================================================
        # 4. APPLY MANAGER LOGIC
        # ====================================================================

        # ####################################################################
        # ## SOLUTION 1: UPDATED MANAGER CALL
        # ####################################################################
        final_signals = apply_manager_logic_numba(
            df['Time_sec'].values, 
            p1_signals, 
            p2_signals, 
            config['UNIVERSAL_COOLDOWN']
        )
        
        # ====================================================================
        # 5. SAVE RESULT
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
    print("STRATEGY MANAGER - PRIORITY-BASED SIGNAL ROUTING")
    print("="*80)
    print(f"\nPriority 1: KF Strategy (PB9_T1 based)")
    print(f"Priority 2: BB Crossover Strategy (BB4/BB6 based)")
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
        print(f"Total bars:       {total_bars:,}")
        print(f"Long entries:     {num_long:,}")
        print(f"Short entries:    {num_short:,}")
        print(f"Hold/Wait:        {num_hold:,}")
        print(f"Total entries:    {(num_long + num_short):,}")
        print(f"Avg per day:      {(num_long + num_short)/len(processed_files):.1f}")
        print(f"Signal density:   {((num_long + num_short)/total_bars)*100:.3f}%")
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