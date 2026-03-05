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
# CONFIGURATION
# ============================================================================

CONFIG = {
    # HARDWARE & PERFORMANCE
    'N_WORKERS': None,  # None = auto: min(cpu_count, 16)
    
    # DATA & FEATURES
    'PRICE_COLUMN': 'Price',
    'BB4_COLUMN': 'BB4_T9',      # Fast Bollinger Band
    'BB6_COLUMN': 'BB6_T11',     # Slow Bollinger Band
    'VOLATILITY_FEATURE': 'PB9_T1',  # For ATR, StdDev, and MA calculation
    'TIMESTAMP_COLUMN_NAMES': ['time', 'timestamp', 'datetime', 'date'],
    
    # DAY ACTIVATION THRESHOLDS
    'ATR_ACTIVATION_THRESHOLD': 0.01,    # ATR must exceed this to activate
    'STDDEV_ACTIVATION_THRESHOLD': 0.06,  # StdDev must exceed this to activate
    'ATR_PERIOD': 20,                    # 20-second ATR window
    'STDDEV_PERIOD': 20,                 # 20-second StdDev window
    
    # MOVING AVERAGE EXIT PARAMETERS
    'FAST_MA_PERIOD': 120,        # 120-second (2-minute) SMA on PB9_T1
    'SLOW_MA_PERIOD': 1200,       # 1200-second (20-minute) SMA on PB9_T1
    'MA_WIDTH_THRESHOLD': 0.0,  # Exit when MA crossover AND width > this
    
    # POSITION MANAGEMENT
    'MIN_HOLD_SECONDS': 15.0,      # Minimum hold time
    'COOLDOWN_SECONDS': 15.0,      # Cooldown after exit
    'FORCE_EOD_SQUAREOFF': True,   # Force close at end of day
    
    # DATA PROCESSING
    'MIN_BARS_REQUIRED': 2400,     # Minimum bars needed per day
    'WARMUP_PERIOD': 2400,         # No entry signals before this bar
    
    # OUTPUT
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILENAME': 'bb_signal.csv',
}

# ============================================================================
# INDICATOR CALCULATION FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_true_range_from_pb9(pb9_t1_values):
    """Calculate True Range using PB9_T1: TR = |PB9_T1[i] - PB9_T1[i-1]|"""
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


# ============================================================================
# SIGNAL GENERATION WITH MA EXIT LOGIC
# ============================================================================

@jit(nopython=True, fastmath=True)
def generate_bb_ma_exit_signals(
    prices,
    bb4_values,
    bb6_values,
    timestamps,
    atr_values,
    stddev_values,
    fast_ma_values,
    slow_ma_values,
    atr_threshold,
    stddev_threshold,
    ma_width_threshold,
    warmup_period,
    min_hold_seconds,
    cooldown_seconds
):
    """
    Generate BB crossover signals with MA-based exit logic
    
    ACTIVATION:
    - ATR > threshold → atr_activated = True (permanent)
    - StdDev > threshold → stddev_activated = True (permanent)
    - Strategy active when BOTH = True
    
    ENTRY (after warmup + activation + cooldown):
    - Long: BB4 crosses above BB6
    - Short: BB4 crosses below BB6
    
    EXIT SCENARIOS:
    
    1. MA CROSSOVER EXIT (directional):
       - Long Exit: fast_MA crosses below slow_MA AND |fast_MA - slow_MA| > 0.015
       - Short Exit: fast_MA crosses above slow_MA AND |fast_MA - slow_MA| > 0.015
       - Action: Exit → Wait for next BB crossover entry signal (NO auto-reversal)
    
    2. EOD SQUARE-OFF:
       - Force close at last bar
    
    CONSTRAINTS:
    - Min hold: 15s before any exit (priority)
    - Cooldown: 15s after exit before new entry
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # Activation flags (persistent for entire day)
    atr_activated = False
    stddev_activated = False
    
    # Position tracking
    current_position = 0  # 0=flat, 1=long, -1=short
    entry_time = 0.0
    last_exit_time = 0.0
    
    # Previous bar values (for crossover detection)
    prev_bb4 = bb4_values[0]
    prev_bb6 = bb6_values[0]
    prev_fast_ma = fast_ma_values[0]
    prev_slow_ma = slow_ma_values[0]
    
    # Process each bar
    for i in range(1, n):
        current_time = timestamps[i]
        current_price = prices[i]
        current_bb4 = bb4_values[i]
        current_bb6 = bb6_values[i]
        current_atr = atr_values[i]
        current_stddev = stddev_values[i]
        current_fast_ma = fast_ma_values[i]
        current_slow_ma = slow_ma_values[i]
        
        # ====================================================================
        # CHECK ACTIVATION FLAGS (from bar 0, even during warmup)
        # ====================================================================
        if not atr_activated and current_atr > atr_threshold:
            atr_activated = True
        
        if not stddev_activated and current_stddev > stddev_threshold:
            stddev_activated = True
        
        strategy_activated = atr_activated and stddev_activated
        
        # ====================================================================
        # CHECK IF PAST WARMUP PERIOD
        # ====================================================================
        past_warmup = (i >= warmup_period)
        can_trade = strategy_activated and past_warmup
        
        # ====================================================================
        # DETECT CROSSOVERS
        # ====================================================================
        # BB crossovers for entry
        bullish_bb_cross = (prev_bb4 <= prev_bb6) and (current_bb4 > current_bb6)
        bearish_bb_cross = (prev_bb4 >= prev_bb6) and (current_bb4 < current_bb6)
        
        # MA crossovers for exit
        fast_crosses_below_slow = (prev_fast_ma >= prev_slow_ma) and (current_fast_ma < current_slow_ma)
        fast_crosses_above_slow = (prev_fast_ma <= prev_slow_ma) and (current_fast_ma > current_slow_ma)
        
        # Calculate MA width at crossover
        ma_width = abs(current_fast_ma - current_slow_ma)
        
        # ====================================================================
        # CHECK IF LAST BAR (EOD)
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
            
            if can_trade and not is_last_bar and cooldown_elapsed:
                if bullish_bb_cross and atr_values[i] > 0.003 and prices[i] > prices[i-1]:
                    current_position = 1
                    entry_time = current_time
                    signals[i] = 1
                
                elif bearish_bb_cross and atr_values[i] > 0.003 and prices[i] < prices[i-1]:
                    current_position = -1
                    entry_time = current_time
                    signals[i] = -1
        
        elif current_position == 1:
            # ================================================================
            # IN LONG POSITION - Check exits
            # ================================================================
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                # Min hold not satisfied - only EOD can force exit
                if is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = -1
            else:
                # Min hold satisfied - check exit conditions
                
                # EXIT 1: MA CROSSOVER EXIT (fast crosses below slow with sufficient width)
                if (fast_ma_values[i] < slow_ma_values[i]) and ma_width > ma_width_threshold:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = -1
                
                # EXIT 2: EOD SQUARE-OFF
                elif is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = -1
        
        elif current_position == -1:
            # ================================================================
            # IN SHORT POSITION - Check exits
            # ================================================================
            hold_time = current_time - entry_time
            
            if hold_time < min_hold_seconds:
                # Min hold not satisfied - only EOD can force exit
                if is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = 1
            else:
                # Min hold satisfied - check exit conditions
                
                # EXIT 1: MA CROSSOVER EXIT (fast crosses above slow with sufficient width)
                if (fast_ma_values[i] > slow_ma_values[i]) and ma_width > ma_width_threshold:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = 1
                
                # EXIT 2: EOD SQUARE-OFF
                elif is_last_bar:
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    signals[i] = 1
        
        # Update previous values for next iteration
        prev_bb4 = current_bb4
        prev_bb6 = current_bb6
        prev_fast_ma = current_fast_ma
        prev_slow_ma = current_slow_ma
    
    return signals


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day(day_num, data_dir, config):
    """Process single day with MA exit logic"""
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
        df = df[required_cols].dropna()

        if len(df) < config['MIN_BARS_REQUIRED']:
            return None

        # Extract arrays
        prices = df[price_col].values.astype(np.float64)
        bb4_values = df[bb4_col].values.astype(np.float64)
        bb6_values = df[bb6_col].values.astype(np.float64)
        pb9_t1_values = df[volatility_feature].values.astype(np.float64)
        timestamps = df['timestamp_seconds'].values.astype(np.float64)
        
        # CALCULATE INDICATORS
        # 1. ATR (20-period rolling mean of True Range)
        true_range = calculate_true_range_from_pb9(pb9_t1_values)
        atr_values = rolling_mean_numba(true_range, config['ATR_PERIOD'])
        
        # 2. StdDev (20-period rolling std of PB9_T1)
        stddev_values = rolling_std_numba(pb9_t1_values, config['STDDEV_PERIOD'])
        
        # 3. Fast MA (120-period SMA on PB9_T1)
        fast_ma_values = rolling_mean_numba(pb9_t1_values, config['FAST_MA_PERIOD'])
        
        # 4. Slow MA (1200-period SMA on PB9_T1)
        slow_ma_values = rolling_mean_numba(pb9_t1_values, config['SLOW_MA_PERIOD'])
        
        # GENERATE SIGNALS
        signals = generate_bb_ma_exit_signals(
            prices=prices,
            bb4_values=bb4_values,
            bb6_values=bb6_values,
            timestamps=timestamps,
            atr_values=atr_values,
            stddev_values=stddev_values,
            fast_ma_values=fast_ma_values,
            slow_ma_values=slow_ma_values,
            atr_threshold=config['ATR_ACTIVATION_THRESHOLD'],
            stddev_threshold=config['STDDEV_ACTIVATION_THRESHOLD'],
            ma_width_threshold=config['MA_WIDTH_THRESHOLD'],
            warmup_period=config['WARMUP_PERIOD'],
            min_hold_seconds=config['MIN_HOLD_SECONDS'],
            cooldown_seconds=config['COOLDOWN_SECONDS']
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


def process_wrapper(args):
    """Wrapper for parallel processing"""
    day_num, data_dir, config = args
    return process_day(day_num, data_dir, config)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    print("="*80)
    print("BB CROSSOVER STRATEGY WITH MA EXIT")
    print("="*80)
    print(f"\n=== CONFIGURATION ===")
    print(f"\nHARDWARE:")
    print(f"  JIT Engine: {'Numba' if NUMBA_AVAILABLE else 'Standard Python'}")
    print(f"  Workers: {CONFIG['N_WORKERS'] if CONFIG['N_WORKERS'] else f'Auto ({min(cpu_count(), 16)})'}")
    
    print(f"\nDATA:")
    print(f"  Price: {CONFIG['PRICE_COLUMN']}")
    print(f"  Fast BB: {CONFIG['BB4_COLUMN']}")
    print(f"  Slow BB: {CONFIG['BB6_COLUMN']}")
    print(f"  Volatility Feature: {CONFIG['VOLATILITY_FEATURE']}")
    
    print(f"\nDAY ACTIVATION SYSTEM:")
    print(f"  ATR Threshold: > {CONFIG['ATR_ACTIVATION_THRESHOLD']} (using {CONFIG['ATR_PERIOD']}s period)")
    print(f"  StdDev Threshold: > {CONFIG['STDDEV_ACTIVATION_THRESHOLD']} (using {CONFIG['STDDEV_PERIOD']}s period)")
    print(f"  Logic: Strategy active when BOTH flags = True (permanent for day)")
    
    print(f"\nMOVING AVERAGES (on {CONFIG['VOLATILITY_FEATURE']}):")
    print(f"  Fast MA: {CONFIG['FAST_MA_PERIOD']}s ({CONFIG['FAST_MA_PERIOD']/60:.1f} min) SMA")
    print(f"  Slow MA: {CONFIG['SLOW_MA_PERIOD']}s ({CONFIG['SLOW_MA_PERIOD']/60:.1f} min) SMA")
    print(f"  Width Threshold: {CONFIG['MA_WIDTH_THRESHOLD']}")
    
    print(f"\nENTRY RULES (all required):")
    print(f"  1. Strategy activated (ATR & StdDev flags = True)")
    print(f"  2. Past warmup period ({CONFIG['WARMUP_PERIOD']} bars)")
    print(f"  3. BB4 crosses above/below BB6")
    print(f"  4. Cooldown elapsed ({CONFIG['COOLDOWN_SECONDS']}s)")
    print(f"  5. Not last bar of day")
    
    print(f"\nEXIT RULES (priority order):")
    print(f"  1. MIN HOLD: Must hold for {CONFIG['MIN_HOLD_SECONDS']}s minimum")
    print(f"  2. MA CROSSOVER EXIT:")
    print(f"     - Long: fast_MA crosses below slow_MA AND width > {CONFIG['MA_WIDTH_THRESHOLD']}")
    print(f"     - Short: fast_MA crosses above slow_MA AND width > {CONFIG['MA_WIDTH_THRESHOLD']}")
    print(f"     → Exit & wait for next BB crossover signal (NO reversal)")
    print(f"  3. EOD: Force square-off at last bar")
    
    print(f"\nCONSTRAINTS:")
    print(f"  Min Hold: {CONFIG['MIN_HOLD_SECONDS']}s (strict priority)")
    print(f"  Cooldown: {CONFIG['COOLDOWN_SECONDS']}s (after exit)")
    print(f"  Warmup: {CONFIG['WARMUP_PERIOD']} bars ({CONFIG['WARMUP_PERIOD']/60:.1f} min)")
    
    print(f"\nOUTPUT:")
    print(f"  {CONFIG['OUTPUT_DIR']}{CONFIG['OUTPUT_FILENAME']}")
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
        print(f"ERROR: No day*.parquet files found")
        return
    
    # Process all days
    print(f"\n[2/4] Processing {num_days} days...")
    workers = CONFIG['N_WORKERS'] if CONFIG['N_WORKERS'] else min(cpu_count(), 16)
    print(f"  Using {workers} parallel workers")
    start_time = time.time()
    
    args_list = [(day_num, data_dir, CONFIG) for day_num in range(num_days)]
    
    with Pool(workers) as pool:
        results = pool.map(process_wrapper, args_list)
    
    all_signals = [res for res in results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f}s")
    print(f"  ✓ Processed {len(all_signals)}/{num_days} days")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated")
        return
    
    # Combine results
    print(f"\n[3/4] Combining results...")
    combined = pd.concat(all_signals, ignore_index=True)
    
    # Statistics
    longs = (combined['Signal'] == 1).sum()
    shorts = (combined['Signal'] == -1).sum()
    holds = (combined['Signal'] == 0).sum()
    total = len(combined)
    
    print(f"\n  === STATISTICS ===")
    print(f"  Long entries:  {longs:,}")
    print(f"  Short entries: {shorts:,}")
    print(f"  Hold/Wait:     {holds:,}")
    print(f"  Total bars:    {total:,}")
    print(f"\n  Signal frequency: {((longs + shorts)/total)*100:.3f}%")
    print(f"  Total signals: {longs + shorts:,} (~{(longs + shorts)/len(all_signals):.1f}/day)")
    
    # Save
    print(f"\n[4/4] Saving...")
    output_path = Path(CONFIG['OUTPUT_DIR']) / CONFIG['OUTPUT_FILENAME']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    print(f"    Rows: {len(combined):,}")
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()