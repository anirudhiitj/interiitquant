import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

# Try to import Numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python (install with: pip install numba)")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# ============================================================================
# CONFIGURATION - ALL TUNABLE HYPERPARAMETERS
# ============================================================================

CONFIG = {
    # ========================================================================
    # HARDWARE & PERFORMANCE
    # ========================================================================
    'N_WORKERS': None,  # Number of parallel workers (None = auto: min(cpu_count, 16))
    
    # ========================================================================
    # DATA & FEATURES
    # ========================================================================
    'PRICE_COLUMN': 'Price',
    'BB4_COLUMN': 'BB4_T9',  # Fast Bollinger Band
    'BB6_COLUMN': 'BB6_T11',  # Slow Bollinger Band
    'VOLATILITY_FEATURE': 'PB9_T1',  # Feature used for ATR and StdDev calculation
    'TIMESTAMP_COLUMN_NAMES': ['time', 'timestamp', 'datetime', 'date'],
    
    # ========================================================================
    # STRATEGY ACTIVATION THRESHOLDS
    # ========================================================================
    'ATR_PERIOD': 60,                    # ATR window (300 seconds / 5 minutes)
    'ACTIVATION_ATR_THRESHOLD': 0.01,    # ATR must exceed this to activate
    'ACTIVATION_STDDEV_THRESHOLD': 0.06, # PB9_T1 StdDev must exceed this to activate
    
    # ========================================================================
    # KAMA FILTER (replaces SMA and Momentum filters)
    # ========================================================================
    'USE_KAMA_FILTER': True,      # Toggle KAMA filter on/off
    'KAMA_PERIOD': 30,            # Lookback period for efficiency ratio
    'KAMA_FAST': 60,              # Fast EMA period
    'KAMA_SLOW': 300,             # Slow EMA period
    'KAMA_SLOPE_LOOKBACK': 1,    # Bars to look back for slope calculation
    'KAMA_SLOPE_THRESHOLD': 0.0001,  # Minimum slope magnitude for directional confirmation
    
    # ========================================================================
    # POSITION MANAGEMENT - ENTRY/EXIT RULES
    # ========================================================================
    'MIN_HOLD_SECONDS': 15.0,  # STRICT: Minimum hold time before ANY exit allowed
    'MAX_HOLD_SECONDS': 3600*2,  # Maximum hold time before forced exit
    'COOLDOWN_SECONDS': 15.0,  # STRICT: Cooldown period after exit before re-entry
    
    # --- TAKE PROFIT ---
    'USE_TAKE_PROFIT': True,    # Toggle hard take profit on/off
    'TAKE_PROFIT_OFFSET': 0.8, # Hard take profit offset from entry price
    
    'FORCE_EOD_SQUAREOFF': True,  # Force close all positions at end of day
    
    # ========================================================================
    # DATA PROCESSING
    # ========================================================================
    'MIN_BARS_REQUIRED': 1800,  # Minimum bars needed per day for valid processing
    'STRATEGY_START_BAR': 2400,  # Must wait this many bars before checking activation
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILENAME': 'bb_signals.csv',
}

# ============================================================================
# INDICATOR/FEATURE CALCULATION FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_true_range_from_pb9(pb9_t1_values):
    """
    Calculate True Range using PB9_T1 (previous second price)
    TR = |PB9_T1[i] - PB9_T1[i-1]|
    """
    n = len(pb9_t1_values)
    true_range = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        true_range[i] = abs(pb9_t1_values[i] - pb9_t1_values[i-1])
    
    return true_range


@jit(nopython=True, fastmath=True)
def rolling_mean_numba(arr, window):
    """
    Calculate rolling mean with Numba acceleration
    """
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        result[i] = np.mean(arr[start_idx:i+1])
    
    return result


@jit(nopython=True, fastmath=True)
def rolling_std_numba(arr, window):
    """
    Calculate rolling standard deviation with Numba acceleration
    """
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


# ============================================================================
# BB CROSSOVER SIGNAL GENERATION WITH KAMA FILTER
# ============================================================================

@jit(nopython=True, fastmath=True)
def generate_bb_crossover_signals_with_kama(
    prices,
    bb4_values,
    bb6_values,
    timestamps,
    atr_values,
    pb9_stddev_values,
    kama_slope,
    use_kama_filter,
    kama_slope_threshold,
    activation_atr_threshold,
    activation_stddev_threshold,
    min_hold_seconds,
    max_hold_seconds,
    cooldown_seconds,
    use_take_profit,
    take_profit_offset,
    strategy_start_bar
):
    """
    Generate BB crossover signals with KAMA directional filter
    
    Strategy Logic:
    1. Wait Period: Must wait until strategy_start_bar
    
    2. ACTIVATION CHECK (after wait period):
       - Monitor ATR and PB9_T1 StdDev
       - Strategy activates when BOTH exceed thresholds at any point:
         * ATR > activation_atr_threshold
         * PB9_T1 StdDev > activation_stddev_threshold
       - Once activated, stays active for rest of day
    
    3. ENTRY CONDITIONS (once activated, all must be met):
       a) BB Crossover (BB4 crosses BB6)
       b) KAMA Directional Confirmation (if enabled):
          - Long: KAMA slope > threshold (uptrend)
          - Short: KAMA slope < -threshold (downtrend)
    
    4. EXIT CONDITIONS (any one can trigger, after min_hold):
       - Opposite BB Crossover
       - Max Hold
       - Take Profit
       - EOD
    
    5. Min Hold: STRICT minimum hold before ANY exit
    6. Cooldown: STRICT cooldown after exit before re-entry
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # Strategy activation flag
    strategy_activated = False
    
    # Position tracking
    current_position = 0  # 0=flat, 1=long, -1=short
    entry_time = 0.0
    entry_price = 0.0
    last_exit_time = 0.0
    
    # Crossover tracking (previous bar state)
    prev_bb4 = bb4_values[0]
    prev_bb6 = bb6_values[0]
    
    # Process each bar
    for i in range(1, n):  # Start from 1 (need previous bar for crossover)
        current_time = timestamps[i]
        current_price = prices[i]
        current_bb4 = bb4_values[i]
        current_bb6 = bb6_values[i]
        current_atr = atr_values[i]
        current_pb9_stddev = pb9_stddev_values[i]
        current_kama_slope = kama_slope[i]
        
        # ====================================================================
        # CHECK STRATEGY ACTIVATION (can happen during warmup too)
        # ====================================================================
        if not strategy_activated:
            # Check if both thresholds are exceeded (even during warmup)
            if (current_atr > activation_atr_threshold and 
                current_pb9_stddev > activation_stddev_threshold):
                strategy_activated = True
        
        # ====================================================================
        # CHECK IF STRATEGY CAN GENERATE SIGNALS
        # ====================================================================
        # Strategy must be: (1) activated AND (2) past warmup period
        can_trade = strategy_activated and (i >= strategy_start_bar)
        
        # ====================================================================
        # DETECT BB CROSSOVERS (NO LOOKAHEAD)
        # ====================================================================
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        # ====================================================================
        # KAMA DIRECTIONAL FILTER (for entry, if enabled)
        # ====================================================================
        if use_kama_filter:
            kama_bullish = current_kama_slope > kama_slope_threshold
            kama_bearish = current_kama_slope < -kama_slope_threshold
        else:
            kama_bullish = True
            kama_bearish = True
            
        # ====================================================================
        # CHECK IF LAST BAR (EOD SQUARE-OFF)
        # ====================================================================
        is_last_bar = (i == n - 1)
        
        # ====================================================================
        # POSITION MANAGEMENT
        # ====================================================================
        
        if current_position == 0:
            # ================================================================
            # FLAT - Check for new entry
            # ================================================================
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= cooldown_seconds)
            
            # Entry requires: can_trade + cooldown + KAMA filter + not last bar
            if (can_trade and 
                cooldown_elapsed and
                not is_last_bar):
                
                # LONG ENTRY: BB crossover + KAMA uptrend
                if bullish_bb_cross and kama_bullish:
                    current_position = 1
                    entry_time = current_time
                    entry_price = current_price
                    signals[i] = 1
                
                # SHORT ENTRY: BB crossover + KAMA downtrend
                elif bearish_bb_cross and kama_bearish:
                    current_position = -1
                    entry_time = current_time
                    entry_price = current_price
                    signals[i] = -1
        
        elif current_position == 1:
            # ================================================================
            # IN LONG POSITION - Check exit conditions
            # ================================================================
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                if is_last_bar: # EOD overrides min hold
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1
            else:
                # Min hold satisfied - check all exit conditions
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price >= (entry_price + take_profit_offset))
                
                should_exit = (
                    bearish_bb_cross or 
                    max_hold_exceeded or 
                    is_last_bar or 
                    take_profit_hit
                )
                
                if should_exit:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = -1
        
        elif current_position == -1:
            # ================================================================
            # IN SHORT POSITION - Check exit conditions
            # ================================================================
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                if is_last_bar: # EOD overrides min hold
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1
            else:
                # Min hold satisfied - check all exit conditions
                max_hold_exceeded = (max_hold_seconds > 0) and (hold_time >= max_hold_seconds)
                take_profit_hit = use_take_profit and (current_price <= (entry_price - take_profit_offset))
                
                should_exit = (
                    bullish_bb_cross or 
                    max_hold_exceeded or 
                    is_last_bar or 
                    take_profit_hit
                )
                
                if should_exit:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    entry_price = 0.0
                    signals[i] = 1
        
        # Update previous values for next iteration
        prev_bb4 = current_bb4
        prev_bb6 = current_bb6
    
    return signals


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day_bb_crossover(day_num, data_dir, config):
    """
    Process single day for BB crossover strategy with KAMA filter
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)

        # Check for required columns
        price_col = config['PRICE_COLUMN']
        bb4_col = config['BB4_COLUMN']
        bb6_col = config['BB6_COLUMN']
        volatility_feature = config['VOLATILITY_FEATURE']
        
        required_cols = [price_col, bb4_col, bb6_col, volatility_feature]

        if not all(col in df.columns for col in required_cols):
            return None

        # Handle timestamps
        timestamp_col = next((c for c in df.columns if c.lower() in config['TIMESTAMP_COLUMN_NAMES']), None)
        
        if timestamp_col:
            original_timestamps = df[timestamp_col].values
            try:
                dt_series = pd.to_datetime(df[timestamp_col], format='mixed', errors='coerce')
                df['timestamp_seconds'] = (
                    dt_series.dt.hour * 3600 + 
                    dt_series.dt.minute * 60 + 
                    dt_series.dt.second + 
                    dt_series.dt.microsecond / 1e6
                )
                if df['timestamp_seconds'].isna().any():
                    raise ValueError("Timestamp conversion failed")
            except:
                df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)
        else:
            original_timestamps = np.arange(len(df))
            df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)

        required_cols.append('timestamp_seconds')
        required_cols = sorted(list(set(required_cols))) 
        
        df = df[required_cols].dropna()

        if len(df) < config['MIN_BARS_REQUIRED']:
            return None

        # Get base arrays
        prices = df[price_col].values.astype(np.float64)
        bb4_values = df[bb4_col].values.astype(np.float64)
        bb6_values = df[bb6_col].values.astype(np.float64)
        pb9_t1_values = df[volatility_feature].values.astype(np.float64)
        timestamps = df['timestamp_seconds'].values.astype(np.float64)
        
        # CALCULATE ATR (for activation threshold)
        true_range = calculate_true_range_from_pb9(pb9_t1_values)
        atr_values = rolling_mean_numba(true_range, config['ATR_PERIOD'])
        
        # CALCULATE PB9_T1 STANDARD DEVIATION (for activation threshold)
        pb9_stddev_values = rolling_std_numba(pb9_t1_values, config['ATR_PERIOD'])
        
        # CALCULATE KAMA AND SLOPE
        if config['USE_KAMA_FILTER']:
            kama = calculate_kama(
                pb9_t1_values,
                config['KAMA_PERIOD'],
                config['KAMA_FAST'],
                config['KAMA_SLOW']
            )
            kama_slope = calculate_kama_slope(kama, config['KAMA_SLOPE_LOOKBACK'])
        else:
            kama_slope = np.ones(len(df), dtype=np.float64)
        
        # GENERATE SIGNALS
        signals = generate_bb_crossover_signals_with_kama(
            prices=prices,
            bb4_values=bb4_values,
            bb6_values=bb6_values,
            timestamps=timestamps,
            atr_values=atr_values,
            pb9_stddev_values=pb9_stddev_values,
            kama_slope=kama_slope,
            use_kama_filter=config['USE_KAMA_FILTER'],
            kama_slope_threshold=config['KAMA_SLOPE_THRESHOLD'],
            activation_atr_threshold=config['ACTIVATION_ATR_THRESHOLD'],
            activation_stddev_threshold=config['ACTIVATION_STDDEV_THRESHOLD'],
            min_hold_seconds=config['MIN_HOLD_SECONDS'],
            max_hold_seconds=config['MAX_HOLD_SECONDS'] if config['MAX_HOLD_SECONDS'] is not None else -1.0,
            cooldown_seconds=config['COOLDOWN_SECONDS'],
            use_take_profit=config['USE_TAKE_PROFIT'],
            take_profit_offset=config['TAKE_PROFIT_OFFSET'],
            strategy_start_bar=config['STRATEGY_START_BAR']
        )
        
        if len(signals) == 0:
            return None
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Time': original_timestamps[:len(signals)],
            'Price': prices,
            'Signal': signals
        })

        return result_df

    except Exception as e:
        return None


def bb_crossover_wrapper(args):
    """Wrapper for parallel processing"""
    day_num, data_dir, config = args
    return process_day_bb_crossover(day_num, data_dir, config)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    print("="*80)
    print("BOLLINGER BAND CROSSOVER STRATEGY WITH KAMA FILTER")
    print("="*80)
    print(f"\n=== CONFIGURATION ===")
    print(f"\nHARDWARE & PERFORMANCE:")
    print(f"  JIT Engine: {'Numba' if NUMBA_AVAILABLE else 'Standard Python'}")
    print(f"  Parallel Workers: {CONFIG['N_WORKERS'] if CONFIG['N_WORKERS'] else f'Auto ({min(cpu_count(), 16)})'}")
    
    print(f"\nDATA & FEATURES:")
    print(f"  Price Column: {CONFIG['PRICE_COLUMN']}")
    print(f"  Fast BB: {CONFIG['BB4_COLUMN']}")
    print(f"  Slow BB: {CONFIG['BB6_COLUMN']}")
    print(f"  Volatility Feature: {CONFIG['VOLATILITY_FEATURE']}")
    print(f"  Min Bars Required: {CONFIG['MIN_BARS_REQUIRED']}")
    print(f"  Strategy Start Bar: {CONFIG['STRATEGY_START_BAR']} (wait period)")
    
    print(f"\nSTRATEGY ACTIVATION LOGIC:")
    print(f"  Monitoring starts: Bar 0 (from beginning of day)")
    print(f"  Two metrics checked continuously (ATR period = {CONFIG['ATR_PERIOD']} bars):")
    print(f"    - ATR on {CONFIG['VOLATILITY_FEATURE']}: must exceed {CONFIG['ACTIVATION_ATR_THRESHOLD']}")
    print(f"    - StdDev on {CONFIG['VOLATILITY_FEATURE']}: must exceed {CONFIG['ACTIVATION_STDDEV_THRESHOLD']}")
    print(f"  When BOTH thresholds exceeded → Strategy ACTIVATES")
    print(f"  Once activated → Remains active for entire day")
    print(f"  Warmup Period: {CONFIG['STRATEGY_START_BAR']} bars (signals only after this)")
    print(f"  Note: Activation can happen during warmup, but signals wait until bar {CONFIG['STRATEGY_START_BAR']}")
    
    print(f"\nKAMA DIRECTIONAL FILTER:")
    print(f"  Enabled: {'✓ YES' if CONFIG['USE_KAMA_FILTER'] else '✗ NO (bypassed)'}")
    if CONFIG['USE_KAMA_FILTER']:
        print(f"  KAMA Period: {CONFIG['KAMA_PERIOD']} (efficiency ratio)")
        print(f"  KAMA Fast Period: {CONFIG['KAMA_FAST']} (fast EMA)")
        print(f"  KAMA Slow Period: {CONFIG['KAMA_SLOW']} (slow EMA)")
        print(f"  Slope Lookback: {CONFIG['KAMA_SLOPE_LOOKBACK']} bars")
        print(f"  Slope Threshold: {CONFIG['KAMA_SLOPE_THRESHOLD']}")
        print(f"  Applied to: {CONFIG['VOLATILITY_FEATURE']}")
        print(f"  Logic:")
        print(f"    - Long entry: KAMA slope > {CONFIG['KAMA_SLOPE_THRESHOLD']} (uptrend)")
        print(f"    - Short entry: KAMA slope < -{CONFIG['KAMA_SLOPE_THRESHOLD']} (downtrend)")
    
    print(f"\nPOSITION MANAGEMENT:")
    print(f"  Min Hold (STRICT): {CONFIG['MIN_HOLD_SECONDS']}s")
    if CONFIG['MAX_HOLD_SECONDS'] is not None:
        print(f"  Max Hold (STRICT): {CONFIG['MAX_HOLD_SECONDS']}s ({CONFIG['MAX_HOLD_SECONDS']/60:.1f} min)")
    else:
        print(f"  Max Hold: DISABLED")
    print(f"  Cooldown Period (STRICT): {CONFIG['COOLDOWN_SECONDS']}s")
    
    print(f"  Hard Take Profit: {'✓ YES' if CONFIG['USE_TAKE_PROFIT'] else '✗ NO'}")
    if CONFIG['USE_TAKE_PROFIT']:
        print(f"   └─ Offset: {CONFIG['TAKE_PROFIT_OFFSET']} (from entry price)")

    print(f"  Force EOD Square-off: {CONFIG['FORCE_EOD_SQUAREOFF']}")
    
    print(f"\nREMOVED FEATURES:")
    print(f"  ✗ SMA Filter (replaced by KAMA)")
    print(f"  ✗ Momentum Alignment Filter (replaced by KAMA)")
    print(f"  ✗ ATR Entry Filter (removed)")
    print(f"  ✗ Stop Loss (removed)")
    print(f"  ✗ Time-Based Loss Exit (removed)")
    
    print(f"\nOUTPUT:")
    print(f"  Directory: {CONFIG['OUTPUT_DIR']}")
    print(f"  Filename: {CONFIG['OUTPUT_FILENAME']}")
    print(f"  Columns: Time, Price, Signal")
    
    print("="*80)
    
    # Check data directory
    print(f"\n[1/4] Checking data directory...")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    parquet_files = list(data_path.glob("day*.parquet"))
    print(f"  ✓ Found {len(parquet_files)} day*.parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No day*.parquet files found in {data_dir}")
        return
    
    # Process all days in parallel
    print(f"\n[2/4] Processing {num_days} days with KAMA filter...")
    workers = CONFIG['N_WORKERS'] if CONFIG['N_WORKERS'] else min(cpu_count(), 16)
    print(f"        (Using parallel processing with {workers} workers)")
    start_time = time.time()
    
    all_days = list(range(num_days))
    args_list = [(day_num, data_dir, CONFIG) for day_num in all_days]
    
    with Pool(workers) as pool:
        all_signals_results = pool.map(bb_crossover_wrapper, args_list)
    
    # Filter out None results
    all_signals = [res for res in all_signals_results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f} seconds")
    print(f"  ✓ Successfully processed {len(all_signals)} days (out of {num_days} attempted)")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated (check if data files exist and contain all required columns)")
        return
    
    # Combine all days
    print(f"\n[3/4] Combining results...")
    combined_signals = pd.concat(all_signals, ignore_index=True)
    
    # Count signal statistics
    num_long_signals = (combined_signals['Signal'] == 1).sum()
    num_short_signals = (combined_signals['Signal'] == -1).sum()
    num_holds = (combined_signals['Signal'] == 0).sum()
    total_bars = len(combined_signals)
    
    print(f"\n  === SIGNAL STATISTICS ===")
    print(f"  +1 signals:    {num_long_signals:,} (enter long / exit short)")
    print(f"  -1 signals:    {num_short_signals:,} (enter short / exit long)")
    print(f"   0 signals:    {num_holds:,} (hold / wait / inactive)")
    print(f"  Total bars:    {total_bars:,}")
    print(f"\n  Trade frequency: {((num_long_signals + num_short_signals)/total_bars)*100:.3f}%")
    print(f"  Wait/hold time: {(num_holds/total_bars)*100:.1f}%")
    
    # Estimate trades (entry signals only)
    print(f"\n  Estimated total entries: {(num_long_signals + num_short_signals)//2:,}")
    print(f"  (Avg ~{((num_long_signals + num_short_signals)//2)/len(all_signals):.1f} entries per day)")
    
    # Save to CSV
    print(f"\n[4/4] Saving results...")
    output_path = Path(CONFIG['OUTPUT_DIR']) / CONFIG['OUTPUT_FILENAME']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_signals.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"    Columns: Time, Price, Signal")
    print(f"    Rows: {len(combined_signals):,}")
    
    print("\n" + "="*80)
    print("✓ STRATEGY PROCESSING COMPLETED!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Load signals: pd.read_csv('{output_path}')")
    print(f"  2. Backtest with PnL calculation")
    print(f"  3. Analyze KAMA filter effectiveness")


if __name__ == '__main__':
    main()