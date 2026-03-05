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
    'OUTPUT_FILE': 'zscore_signals.csv',
    'TEMP_DIR': '/tmp/zscore_trailing_temp',  # Temporary directory for intermediate files
    
    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',
    
    # Z-Score parameters
    'zscore_window': 1800,           # Rolling window for z-score calculation (30 min)
    
    # SMA Entry Filter parameters (tick-based)
    'sma_fast_window': 420,          # Fast SMA: 7 minutes
    'sma_medium_window': 1500,       # Medium SMA: 25 minutes
    'sma_slow_window': 3600,         # Slow SMA: 60 minutes
    
    # KAMA Volatility Filter parameters (blocks trading in strong trends)
    'kama_period': 15,               # KAMA calculation period
    'kama_fast_period': 30,          # Fast smoothing constant period
    'kama_slow_period': 180,         # Slow smoothing constant period
    'kama_slope_lookback': 2,        # Lookback for slope calculation
    'kama_slope_threshold': 0.0007,   # Block trading if abs(slope) > this
    
    # Sigmoid thresholds (range: -1 to +1)
    'sigmoid_upper_threshold': 0.95,  # Entry short / Exit long
    'sigmoid_lower_threshold': -0.95, # Entry long / Exit short
    
    # Position management
    'min_hold_seconds': 5,          # Minimum position hold time
    'cooldown_seconds': 30,          # Cooldown after any signal (1 minute)
    'min_gap_ticks': 5,             # Minimum ticks between ANY signals
    
    # Take Profit and Trailing Stop parameters
    'absolute_tp_points': 0.1,       # Initial take profit at entry ±0.05 points
    'trailing_stop_distance': 0.05,   # Trailing stop distance after TP hit
    
    # Strategy warmup period
    'strategy_start_bar': 1800,      # No signals before this tick (warmup period)
    
    # Execution parameters
    'MAX_WORKERS': 32,               # Number of parallel workers for day processing
}


#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def calculate_rolling_zscore_raw(prices, window):
    """
    Calculate raw (unsmoothed) rolling z-score.
    
    Args:
        prices: Price array (numpy array)
        window: Rolling window size
    
    Returns:
        Array of raw z-scores
    """
    n = len(prices)
    zscores = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        # Extract window
        window_data = prices[i - window:i]
        
        # Calculate mean and std
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        # Calculate z-score (protect against division by zero)
        if std > 1e-9:
            zscores[i] = (prices[i] - mean) / std
        else:
            zscores[i] = 0.0
    
    return zscores


@jit(nopython=True, fastmath=True)
def calculate_sma_rolling(prices, window):
    """
    Calculate rolling Simple Moving Average (SMA).
    
    Args:
        prices: Price array (numpy array)
        window: Rolling window size
    
    Returns:
        Array of SMA values
    """
    n = len(prices)
    sma = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        # Extract window and calculate mean
        window_data = prices[i - window:i]
        sma[i] = np.mean(window_data)
    
    return sma


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
def calculate_kama_slope_numba(kama, lookback):
    """
    Calculate KAMA slope over lookback period.
    
    Args:
        kama: KAMA values array
        lookback: Number of bars to look back for slope
    
    Returns:
        Array of KAMA slopes
    """
    n = len(kama)
    slope = np.zeros(n, dtype=np.float64)
    
    for i in range(lookback, n):
        slope[i] = kama[i] - kama[i - lookback]
    
    return slope


@jit(nopython=True, fastmath=True)
def check_sma_not_opposite(sma_fast, sma_medium, sma_slow, signal_type):
    """
    Check if SMAs are NOT in opposite alignment to the signal.
    Blocks counter-trend entries, allows aligned and neutral entries.
    
    Args:
        sma_fast: Fast SMA value
        sma_medium: Medium SMA value
        sma_slow: Slow SMA value
        signal_type: 1 for long, -1 for short
    
    Returns:
        True if signal is allowed (not opposite), False if blocked (opposite)
    """
    # Allow if any SMA is zero (not enough data - don't block)
    if sma_fast < 1e-9 or sma_medium < 1e-9 :
        return True
    
    if signal_type == 1:  # Long entry candidate
        # BLOCK if bearish: fast < medium < slow (opposite to long)
        is_bearish = (sma_fast < sma_medium)
        return not is_bearish  # Allow if NOT bearish
    
    elif signal_type == -1:  # Short entry candidate
        # BLOCK if bullish: fast > medium > slow (opposite to short)
        is_bullish = (sma_fast > sma_medium)
        return not is_bullish  # Allow if NOT bullish
    
    return True  # Default: allow


@jit(nopython=True, fastmath=True)
def check_sma_trending_for_trail(sma_fast, sma_medium, position_type):
    """
    Check if SMAs show trending alignment for "Let Winners Run".
    
    Args:
        sma_fast: Fast SMA (7 min)
        sma_medium: Medium SMA (25 min)
        position_type: 1 for long, -1 for short
    
    Returns:
        True if trending in correct direction (should trail)
    """
    # Need both SMAs available
    if sma_fast < 1e-9 or sma_medium < 1e-9:
        return False
    
    if position_type == 1:  # Long position
        # Trending up: fast > medium
        return sma_fast > sma_medium
    
    elif position_type == -1:  # Short position
        # Trending down: fast < medium
        return sma_fast < sma_medium
    
    return False


@jit(nopython=True, fastmath=True)
def apply_sigmoid_transform(zscores):
    """
    Apply sigmoid transformation to z-scores.
    
    Transforms unbounded z-scores into bounded [-1, +1] range.
    Formula: sigmoid = 1 / (1 + exp(-z))
             z_sigmoid = (sigmoid - 0.5) * 2
    
    Args:
        zscores: Raw z-score array
    
    Returns:
        Sigmoid-transformed z-score array (range: -1 to +1)
    """
    n = len(zscores)
    sigmoid_zscores = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        z = zscores[i]
        
        # Sigmoid transformation
        sig = 1.0 / (1.0 + np.exp(-z))
        
        # Transform to [-1, +1] range
        sigmoid_zscores[i] = (sig - 0.5) * 2.0
    
    return sigmoid_zscores


@jit(nopython=True, fastmath=True)
def generate_signals_with_trailing_stop(
    timestamps,
    prices,
    z_sigmoid,
    sma_fast,
    sma_medium,
    sma_slow,
    kama_slope,
    upper_thresh,
    lower_thresh,
    min_hold,
    cooldown,
    min_gap_ticks,
    tp_points,
    trailing_distance,
    strategy_start_bar,
    kama_slope_threshold
):
    """
    Generate trading signals with "Let Winners Run" trailing stop logic and KAMA filter.
    
    Strategy Logic (State Machine with SMA Filter + Trailing Stop + KAMA Filter):
    - States: NEUTRAL, LONG, SHORT, LONG_TRAILING, SHORT_TRAILING
    
    KAMA Filter (ENTRY ONLY):
        * If abs(KAMA_Slope) > kama_slope_threshold → BLOCK NEW ENTRIES
        * Exits are NOT blocked - positions can close normally
        * EOD square-off always executes
    
    From NEUTRAL:
        * z_sigmoid ≤ lower_thresh AND SMA allows AND KAMA allows → ENTER LONG
        * z_sigmoid ≥ upper_thresh AND SMA allows AND KAMA allows → ENTER SHORT
    
    From LONG:
        * Price hits TP (entry + tp_points):
          - Check: SMA_fast > SMA_medium? (trending up)
          - YES → Switch to LONG_TRAILING (trailing stop mode)
          - NO → EXIT at TP
        * z_sigmoid ≥ upper_thresh (after min_hold) → EXIT (opposite extreme)
    
    From LONG_TRAILING:
        * Track highest price reached
        * Trailing stop = highest_price - trailing_distance
        * Price ≤ trailing_stop → EXIT
        * z_sigmoid ≥ upper_thresh → EXIT (opposite extreme)
    
    From SHORT:
        * Price hits TP (entry - tp_points):
          - Check: SMA_fast < SMA_medium? (trending down)
          - YES → Switch to SHORT_TRAILING (trailing stop mode)
          - NO → EXIT at TP
        * z_sigmoid ≤ lower_thresh (after min_hold) → EXIT (opposite extreme)
    
    From SHORT_TRAILING:
        * Track lowest price reached
        * Trailing stop = lowest_price + trailing_distance
        * Price ≥ trailing_stop → EXIT
        * z_sigmoid ≤ lower_thresh → EXIT (opposite extreme)
    
    All positions: EOD square-off mandatory
    
    Args:
        timestamps: Time in seconds (numpy array)
        prices: Price array (numpy array)
        z_sigmoid: Sigmoid-transformed z-score array
        sma_fast: Fast SMA array (7 min)
        sma_medium: Medium SMA array (25 min)
        sma_slow: Slow SMA array (60 min)
        kama_slope: KAMA slope array
        upper_thresh: Upper sigmoid threshold
        lower_thresh: Lower sigmoid threshold
        min_hold: Minimum hold seconds
        cooldown: Cooldown seconds after signal
        min_gap_ticks: Minimum ticks between signals
        tp_points: Take profit distance in price points
        trailing_distance: Trailing stop distance from extreme
        strategy_start_bar: No signals before this tick (warmup period)
        kama_slope_threshold: Block NEW ENTRIES if abs(KAMA slope) > this
    
    Returns:
        Signal array (1=long entry, -1=short entry/long exit, 0=no action)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # State tracking
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = -1
    STATE_LONG_TRAILING = 2
    STATE_SHORT_TRAILING = -2
    
    state = STATE_NEUTRAL
    entry_idx = -1        # Index where position was entered
    entry_price = 0.0     # Price at entry
    tp_price = 0.0        # Take profit price level
    trailing_stop = 0.0   # Current trailing stop level
    highest_price = 0.0   # Highest price in trailing mode (for long)
    lowest_price = 0.0    # Lowest price in trailing mode (for short)
    last_signal_idx = -1  # Index of last signal (for gap enforcement)
    
    for i in range(n):
        # WARMUP PERIOD: No signals before strategy_start_bar
        if i < strategy_start_bar:
            signals[i] = 0
            continue
        
        # KAMA VOLATILITY FILTER: Block NEW ENTRIES if abs(KAMA slope) > threshold
        # (exits are still allowed - only entries blocked)
        kama_blocks_entry = abs(kama_slope[i]) > kama_slope_threshold
        
        # Check if this is the last tick (EOD square-off)
        is_last_tick = (i == n - 1)
        
        # Get current values
        current_time = timestamps[i]
        current_price = prices[i]
        z_sig = z_sigmoid[i]
        
        # Calculate time since last signal (for cooldown)
        time_since_signal = current_time - timestamps[last_signal_idx] if last_signal_idx >= 0 else np.inf
        
        # Calculate ticks since last signal (for gap enforcement)
        ticks_since_signal = i - last_signal_idx if last_signal_idx >= 0 else np.inf
        
        # Calculate hold duration
        hold_duration = current_time - timestamps[entry_idx] if entry_idx >= 0 else 0.0
        
        # Initialize signal
        signal = 0
        
        # === STATE MACHINE LOGIC ===
        
        if state == STATE_NEUTRAL:
            # NO POSITION - CHECK ENTRY CONDITIONS
            
            # Skip if in cooldown period
            if time_since_signal < cooldown:
                continue
            
            # Skip if minimum gap not satisfied
            if ticks_since_signal < min_gap_ticks:
                continue
            
            # Skip entry on last tick (EOD - no new positions)
            if is_last_tick:
                continue
            
            # Skip entry if KAMA filter blocks it (strong trend)
            if kama_blocks_entry:
                continue
            
            # ENTRY LONG: z_sigmoid ≤ lower_thresh (extreme oversold)
            if z_sig <= lower_thresh:
                # Check SMA filter
                sma_allows = check_sma_not_opposite(sma_fast[i], sma_medium[i], sma_slow[i], 1)
                
                if sma_allows:
                    # Enter long
                    signal = 1
                    state = STATE_LONG
                    entry_idx = i
                    entry_price = current_price
                    tp_price = entry_price + tp_points
                    last_signal_idx = i
            
            # ENTRY SHORT: z_sigmoid ≥ upper_thresh (extreme overbought)
            elif z_sig >= upper_thresh:
                # Check SMA filter
                sma_allows = check_sma_not_opposite(sma_fast[i], sma_medium[i], sma_slow[i], -1)
                
                if sma_allows:
                    # Enter short
                    signal = -1
                    state = STATE_SHORT
                    entry_idx = i
                    entry_price = current_price
                    tp_price = entry_price - tp_points
                    last_signal_idx = i
        
        elif state == STATE_LONG:
            # IN LONG POSITION - CHECK TP OR EXIT
            
            exit_signal = 0
            
            # PRIORITY 1: MANDATORY EOD SQUARE-OFF
            if is_last_tick:
                exit_signal = -1
            
            # PRIORITY 2: CHECK TAKE PROFIT (after min_hold)
            elif hold_duration >= min_hold and current_price >= tp_price:
                # TP HIT - Check if we should trail
                should_trail = check_sma_trending_for_trail(sma_fast[i], sma_medium[i], 1)
                
                if should_trail:
                    # Enter trailing mode - DON'T exit yet
                    state = STATE_LONG_TRAILING
                    highest_price = current_price
                    trailing_stop = current_price - trailing_distance
                    # No signal generated - staying in trade
                else:
                    # Exit at TP - trend not strong enough
                    exit_signal = -1
            
            # PRIORITY 3: CHECK EXIT ON OPPOSITE EXTREME (after min_hold)
            if exit_signal == 0 and hold_duration >= min_hold:
                # Exit Long: z_sigmoid ≥ upper_thresh (overcorrection to opposite extreme)
                if z_sig >= upper_thresh:
                    # Also check gap enforcement
                    if ticks_since_signal >= min_gap_ticks:
                        exit_signal = -1
            
            # Execute exit if triggered
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                last_signal_idx = i
        
        elif state == STATE_LONG_TRAILING:
            # IN LONG TRAILING MODE - LET WINNERS RUN
            
            # Update highest price and trailing stop
            if current_price > highest_price:
                highest_price = current_price
                trailing_stop = highest_price - trailing_distance
            
            exit_signal = 0
            
            # PRIORITY 1: MANDATORY EOD SQUARE-OFF
            if is_last_tick:
                exit_signal = -1
            
            # PRIORITY 2: TRAILING STOP HIT
            elif current_price <= trailing_stop:
                exit_signal = -1
            
            # PRIORITY 3: Z-SCORE OPPOSITE EXTREME
            elif z_sig >= upper_thresh:
                # Check gap enforcement
                if ticks_since_signal >= min_gap_ticks:
                    exit_signal = -1
            
            # Execute exit if triggered
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                highest_price = 0.0
                trailing_stop = 0.0
                last_signal_idx = i
        
        elif state == STATE_SHORT:
            # IN SHORT POSITION - CHECK TP OR EXIT
            
            exit_signal = 0
            
            # PRIORITY 1: MANDATORY EOD SQUARE-OFF
            if is_last_tick:
                exit_signal = 1
            
            # PRIORITY 2: CHECK TAKE PROFIT (after min_hold)
            elif hold_duration >= min_hold and current_price <= tp_price:
                # TP HIT - Check if we should trail
                should_trail = check_sma_trending_for_trail(sma_fast[i], sma_medium[i], -1)
                
                if should_trail:
                    # Enter trailing mode - DON'T exit yet
                    state = STATE_SHORT_TRAILING
                    lowest_price = current_price
                    trailing_stop = current_price + trailing_distance
                    # No signal generated - staying in trade
                else:
                    # Exit at TP - trend not strong enough
                    exit_signal = 1
            
            # PRIORITY 3: CHECK EXIT ON OPPOSITE EXTREME (after min_hold)
            if exit_signal == 0 and hold_duration >= min_hold:
                # Exit Short: z_sigmoid ≤ lower_thresh (overcorrection to opposite extreme)
                if z_sig <= lower_thresh:
                    # Also check gap enforcement
                    if ticks_since_signal >= min_gap_ticks:
                        exit_signal = 1
            
            # Execute exit if triggered
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                last_signal_idx = i
        
        elif state == STATE_SHORT_TRAILING:
            # IN SHORT TRAILING MODE - LET WINNERS RUN
            
            # Update lowest price and trailing stop
            if current_price < lowest_price:
                lowest_price = current_price
                trailing_stop = lowest_price + trailing_distance
            
            exit_signal = 0
            
            # PRIORITY 1: MANDATORY EOD SQUARE-OFF
            if is_last_tick:
                exit_signal = 1
            
            # PRIORITY 2: TRAILING STOP HIT
            elif current_price >= trailing_stop:
                exit_signal = 1
            
            # PRIORITY 3: Z-SCORE OPPOSITE EXTREME
            elif z_sig <= lower_thresh:
                # Check gap enforcement
                if ticks_since_signal >= min_gap_ticks:
                    exit_signal = 1
            
            # Execute exit if triggered
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                entry_idx = -1
                entry_price = 0.0
                tp_price = 0.0
                lowest_price = 0.0
                trailing_stop = 0.0
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
    Process a single day file using GPU-accelerated operations + SMA + KAMA Filter + Trailing Stop.
    
    Args:
        file_path: Path to parquet file
        day_num: Day number
        temp_dir: Temporary directory for output
        config: Configuration dictionary
    
    Returns:
        Dictionary with processing results
    """
    try:
        # ====================================================================
        # 1. READ DATA WITH GPU (dask_cudf -> cudf -> pandas)
        # ====================================================================
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        
        # Read with dask_cudf for GPU acceleration
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()  # Compute to get cudf DataFrame
        df = gdf.to_pandas()  # Convert to pandas for CPU processing
        
        if df.empty:
            print(f"  Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        # Reset index and convert time to seconds
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        # ====================================================================
        # 2. PREPARE DATA FOR NUMBA
        # ====================================================================
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        
        # Handle NaN values
        prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
        
        # ====================================================================
        # 3. CALCULATE RAW Z-SCORES WITH NUMBA JIT
        # ====================================================================
        zscores_raw = calculate_rolling_zscore_raw(prices, config['zscore_window'])
        
        # ====================================================================
        # 4. APPLY SIGMOID TRANSFORMATION
        # ====================================================================
        z_sigmoid = apply_sigmoid_transform(zscores_raw)
        
        # ====================================================================
        # 5. CALCULATE SMAs WITH NUMBA JIT
        # ====================================================================
        sma_fast = calculate_sma_rolling(prices, config['sma_fast_window'])
        sma_medium = calculate_sma_rolling(prices, config['sma_medium_window'])
        sma_slow = calculate_sma_rolling(prices, config['sma_slow_window'])
        
        # ====================================================================
        # 6. CALCULATE KAMA AND KAMA SLOPE
        # ====================================================================
        kama = calculate_kama_numba(
            prices,
            config['kama_period'],
            config['kama_fast_period'],
            config['kama_slow_period']
        )
        kama_slope = calculate_kama_slope_numba(kama, config['kama_slope_lookback'])
        
        # ====================================================================
        # 7. GENERATE SIGNALS WITH TRAILING STOP AND KAMA FILTER
        # ====================================================================
        signals = generate_signals_with_trailing_stop(
            timestamps=timestamps,
            prices=prices,
            z_sigmoid=z_sigmoid,
            sma_fast=sma_fast,
            sma_medium=sma_medium,
            sma_slow=sma_slow,
            kama_slope=kama_slope,
            upper_thresh=config['sigmoid_upper_threshold'],
            lower_thresh=config['sigmoid_lower_threshold'],
            min_hold=config['min_hold_seconds'],
            cooldown=config['cooldown_seconds'],
            min_gap_ticks=config['min_gap_ticks'],
            tp_points=config['absolute_tp_points'],
            trailing_distance=config['trailing_stop_distance'],
            strategy_start_bar=config['strategy_start_bar'],
            kama_slope_threshold=config['kama_slope_threshold']
        )
        
        # ====================================================================
        # 8. COUNT SIGNALS
        # ====================================================================
        long_entries = np.sum(signals == 1)
        short_entries = np.sum(signals == -1)
        total_signals = long_entries + short_entries
        
        # Verify position balance
        if abs(long_entries - short_entries) > 1:
            print(f"  WARNING Day {day_num}: Position imbalance! Long: {long_entries}, Short: {short_entries}")
        
        # ====================================================================
        # 9. SAVE RESULT (with SMAs and KAMA for analysis)
        # ====================================================================
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'ZScore_Raw': zscores_raw,
            'Z_Sigmoid': z_sigmoid,
            'SMA_Fast': sma_fast,
            'SMA_Medium': sma_medium,
            'SMA_Slow': sma_slow,
            'KAMA': kama,
            'KAMA_Slope': kama_slope,
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


#############################################################################
# MAIN EXECUTION
#############################################################################

def main(config):
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("Z-SCORE SIGMOID MEAN REVERSION STRATEGY")
    print("WITH SMA FILTER + KAMA FILTER + LET WINNERS RUN (TRAILING STOP)")
    print("GPU-ACCELERATED PARALLEL PROCESSING")
    print("="*80)
    print(f"\nCONFIGURATION:")
    print(f"  Data directory: {config['DATA_DIR']}")
    print(f"  Output directory: {config['OUTPUT_DIR']}")
    print(f"  Output file: {config['OUTPUT_FILE']}")
    print(f"  Temp directory: {config['TEMP_DIR']}")
    print(f"  Parallel workers: {config['MAX_WORKERS']}")
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"  Z-score window: {config['zscore_window']} ticks (30 minutes)")
    print(f"  ✨ Sigmoid transformation: RAW → SIGMOID (no EMA) ✨")
    print(f"  Upper threshold: {config['sigmoid_upper_threshold']} (range: -1 to +1)")
    print(f"  Lower threshold: {config['sigmoid_lower_threshold']} (range: -1 to +1)")
    print(f"  Min hold seconds: {config['min_hold_seconds']}")
    print(f"  Cooldown seconds: {config['cooldown_seconds']}")
    print(f"  Min gap ticks: {config['min_gap_ticks']}")
    print(f"  🔥 Strategy start bar: {config['strategy_start_bar']} ticks (warmup period)")
    print(f"\n💰 TAKE PROFIT + LET WINNERS RUN:")
    print(f"  Initial Take Profit: ±{config['absolute_tp_points']} points from entry")
    print(f"  Trailing Stop Distance: {config['trailing_stop_distance']} points")
    print(f"\n🏃 LET WINNERS RUN LOGIC:")
    print(f"  LONG positions:")
    print(f"    1. Entry at price P")
    print(f"    2. TP hits at P + {config['absolute_tp_points']}")
    print(f"    3. Check: SMA_fast (7min) > SMA_medium (25min)?")
    print(f"       → YES: Enter TRAILING mode")
    print(f"         * Initial stop: TP_price - {config['trailing_stop_distance']}")
    print(f"         * Trailing: highest_price - {config['trailing_stop_distance']}")
    print(f"         * Updates as price makes new highs")
    print(f"       → NO: Exit at TP")
    print(f"  SHORT positions:")
    print(f"    1. Entry at price P")
    print(f"    2. TP hits at P - {config['absolute_tp_points']}")
    print(f"    3. Check: SMA_fast (7min) < SMA_medium (25min)?")
    print(f"       → YES: Enter TRAILING mode")
    print(f"         * Initial stop: TP_price + {config['trailing_stop_distance']}")
    print(f"         * Trailing: lowest_price + {config['trailing_stop_distance']}")
    print(f"         * Updates as price makes new lows")
    print(f"       → NO: Exit at TP")
    print(f"\n🆕 SMA ENTRY FILTER:")
    print(f"  Fast SMA: {config['sma_fast_window']} ticks (7 minutes)")
    print(f"  Medium SMA: {config['sma_medium_window']} ticks (25 minutes)")
    print(f"  Slow SMA: {config['sma_slow_window']} ticks (60 minutes)")
    print(f"  Long Entry: BLOCK if bearish (fast < medium < slow)")
    print(f"  Short Entry: BLOCK if bullish (fast > medium > slow)")
    print(f"  Neutral/Mixed: ALLOW signal through")
    print(f"\n🚫 KAMA VOLATILITY FILTER (ENTRY ONLY):")
    print(f"  Period: {config['kama_period']} ticks")
    print(f"  Fast Period: {config['kama_fast_period']}")
    print(f"  Slow Period: {config['kama_slow_period']}")
    print(f"  Slope Lookback: {config['kama_slope_lookback']} ticks")
    print(f"  Slope Threshold: {config['kama_slope_threshold']}")
    print(f"  ❌ BLOCK RULE: NO NEW ENTRIES when abs(KAMA_Slope) > {config['kama_slope_threshold']} ❌")
    print(f"  ✅ EXITS ALLOWED: Position exits work normally regardless of KAMA filter")
    print(f"\n⏱️  WARMUP PERIOD:")
    print(f"  No signals generated for first {config['strategy_start_bar']} ticks")
    print(f"  All indicators (z-score, sigmoid, SMAs, KAMA) calculate normally from tick 0")
    print(f"  Signal generation starts from tick {config['strategy_start_bar']} onwards")
    print(f"\nSTRATEGY LOGIC (STATE MACHINE):")
    print(f"  States: NEUTRAL, LONG, SHORT, LONG_TRAILING, SHORT_TRAILING")
    print(f"  From NEUTRAL:")
    print(f"    - Long: z_sigmoid ≤ {config['sigmoid_lower_threshold']} AND SMA allows AND KAMA allows")
    print(f"    - Short: z_sigmoid ≥ {config['sigmoid_upper_threshold']} AND SMA allows AND KAMA allows")
    print(f"  From LONG:")
    print(f"    - TP hit AND trending → LONG_TRAILING")
    print(f"    - TP hit AND not trending → EXIT")
    print(f"    - z_sigmoid ≥ {config['sigmoid_upper_threshold']} → EXIT")
    print(f"  From LONG_TRAILING:")
    print(f"    - Price ≤ trailing_stop → EXIT")
    print(f"    - z_sigmoid ≥ {config['sigmoid_upper_threshold']} → EXIT")
    print(f"  From SHORT:")
    print(f"    - TP hit AND trending → SHORT_TRAILING")
    print(f"    - TP hit AND not trending → EXIT")
    print(f"    - z_sigmoid ≤ {config['sigmoid_lower_threshold']} → EXIT")
    print(f"  From SHORT_TRAILING:")
    print(f"    - Price ≥ trailing_stop → EXIT")
    print(f"    - z_sigmoid ≤ {config['sigmoid_lower_threshold']} → EXIT")
    print(f"  All positions: EOD square-off mandatory")
    print(f"\nKEY FEATURES:")
    print(f"  ✓ Sigmoid compression bounds z-scores to [-1, +1]")
    print(f"  ✓ SMA filter blocks counter-trend entries")
    print(f"  ✓ KAMA filter blocks NEW ENTRIES in strong trends (exits still allowed)")
    print(f"  ✓ Let Winners Run: Trails when trend confirms")
    print(f"  ✓ Dynamic trailing stop protects profits")
    print(f"  ✓ Fixed TP for weak trends, trailing for strong trends")
    print(f"  ✓ State machine prevents invalid transitions")
    print(f"  ✓ Exit on opposite extreme (full mean reversion)")
    print(f"  ✓ {config['min_gap_ticks']}-tick gap enforcement")
    print(f"  ✓ {config['cooldown_seconds']}s cooldown (only on actual entries)")
    print(f"  ✓ {config['strategy_start_bar']}-tick warmup period (no early signals)")
    print(f"\nGPU ACCELERATION:")
    print(f"  GPU I/O: dask_cudf + cudf")
    print(f"  CPU Compute: Numba JIT")
    print(f"  Parallelism: ProcessPoolExecutor")
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
            # Submit all tasks
            futures = {
                executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, config): f
                for f in sorted_files
            }
            
            # Collect results as they complete
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
        print(f"Total bars:        {total_bars:,}")
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
    print("  1. Backtest with KAMA filter + trailing stop logic")
    print("  2. Analyze trades blocked by KAMA filter")
    print("  3. Compare performance vs non-KAMA version")
    print("  4. Monitor win rate in different volatility regimes")
    print("  5. Evaluate KAMA threshold sensitivity")
    print("="*80 + "\n")


if __name__ == "__main__":
    main(CONFIG)