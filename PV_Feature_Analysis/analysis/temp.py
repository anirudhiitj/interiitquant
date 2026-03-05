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
    # Create a dummy decorator if Numba is not available
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# Try to import GPU libraries (from your reference code)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - GPU acceleration disabled (install with: pip install cupy-cuda12x)")

try:
    import dask_cudf
    import dask.dataframe as dd
    DASK_CUDF_AVAILABLE = True
    print("✓ Dask-cuDF detected - GPU-accelerated dataframe loading enabled")
except ImportError:
    DASK_CUDF_AVAILABLE = False
    print("✗ Dask-cudf not found - using pandas for loading (install with: pip install dask-cudf)")


# ============================================================================
# LOGIC 1: CUSUM FUNCTIONS (from your CUSUM code)
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_directional_cusum(changes, drift=0.02, decay=0.9):
    """
    Calculate directional CUSUM on a pre-calculated 'changes' series.
    The opposing CUSUM now DECAYS instead of instantly resetting.
    """
    n = len(changes)
    
    # Initialize CUSUM tracking
    cusum_up = 0.0
    cusum_down = 0.0
    
    cusum_up_values = np.zeros(n, dtype=np.float64)
    cusum_down_values = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = changes[i] # Use the pre-calculated change
        
        # Upward move
        if change > drift:
            cusum_up += change
            cusum_down = max(0.0, cusum_down * decay) # Opposing side now decays
        
        # Downward move
        elif change < -drift:
            cusum_down += abs(change)
            cusum_up = max(0.0, cusum_up * decay) # Opposing side now decays
        
        # Small move (noise/consolidation) - both decay
        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)
        
        # Store values
        cusum_up_values[i] = cusum_up
        cusum_down_values[i] = cusum_down
    
    return cusum_up_values, cusum_down_values


def classify_regime_by_difference(cusum_up, cusum_down, 
                                  choppy_up_thresh, choppy_down_thresh,
                                  trending_up_thresh, trending_down_thresh):
    """
    Classifies every time step based on the CUSUM difference.
    (This function is from your CUSUM code)
    """
    n = len(cusum_up)
    cusum_diff = cusum_up - cusum_down
    
    # Default to 'Transition'
    regimes = np.full(n, 'Transition', dtype='<U15')
    
    # Prioritized Classification:
    
    # 1. Trending (Highest priority)
    is_trending_up = cusum_diff > trending_up_thresh
    is_trending_down = cusum_diff < trending_down_thresh
    
    regimes[is_trending_up] = 'Trending Up'
    regimes[is_trending_down] = 'Trending Down'

    # 2. Choppy (Second priority)
    is_choppy = (cusum_diff <= choppy_up_thresh) & (cusum_diff >= choppy_down_thresh)
    is_not_trending = ~is_trending_up & ~is_trending_down
    
    regimes[is_choppy & is_not_trending] = 'Choppy'

    # Convert to integer representation for Numba
    # 1 = Trend Up, -1 = Trend Down, 0 = Choppy/Transition
    regime_map = {'Trending Up': 1, 'Trending Down': -1, 'Choppy': 0, 'Transition': 0}
    regime_int = np.array([regime_map[r] for r in regimes], dtype=np.int8)

    return regime_int, cusum_diff


# ============================================================================
# LOGIC 2: SIGNAL GENERATION (Modified Framework)
# ============================================================================

# *** THIS IS THE UPDATED FUNCTION ***
@jit(nopython=True, fastmath=True)
def generate_clean_signals(timestamps, prices, 
                           is_stagnant,          # Master Filter (ATR)
                           cusum_regime_int,     # Entry Filter (CUSUM)
                           momentum,             # Entry & Exit Filter (Momentum)
                           momentum_avg,         # Exit Filter (Momentum Average)
                           start_delay_seconds=1200.0,
                           min_hold_seconds=15.0,
                           min_cooldown_seconds=15.0,
                           momentum_threshold=0.0):
    """
    Generate trading signals based on the 3-stage hierarchical filter:
    1. MASTER FILTER: ATR (is_stagnant) - If true, force exit and block entry
    2. ENTRY:         CUSUM (cusum_regime_int) + Momentum (momentum)
    3. EXIT:          Momentum crosses its rolling average OR ATR Stagnation
    """
    n = len(prices)
    
    # Find the first timestamp after start_delay_seconds
    start_timestamp = timestamps[0] + start_delay_seconds
    
    clean_signals = np.zeros(n, dtype=np.int8)
    
    current_position = 0  # 0 = flat, 1 = long, -1 = short
    entry_time = 0.0 
    last_exit_time = 0.0
    
    for i in range(n):
        current_time = timestamps[i]
        
        # Skip signal generation before start_delay_seconds
        if current_time < start_timestamp:
            clean_signals[i] = 0
            continue
            
        # 1. GET ALL CURRENT-STATE VALUES
        is_stagnant_now = is_stagnant[i]
        cusum_signal = cusum_regime_int[i]
        momentum_val = momentum[i]
        momentum_avg_val = momentum_avg[i] # New value

        # 2. --- MASTER FILTER: ATR STAGNATION (EXIT) ---
        # If the market is stagnant, we force exit and do not allow entries.
        if is_stagnant_now:
            if current_position != 0: # We are in a position
                if current_time - entry_time >= min_hold_seconds:
                    if current_position == 1:
                        clean_signals[i] = -1 # Exit long
                    else:
                        clean_signals[i] = 1  # Exit short
                    
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
            
            # Whether we just exited or were already flat, do nothing else.
            continue # Skip to the next bar

        # 3. --- MARKET IS VOLATILE ---
        # If we are here, is_stagnant_now is False (market is volatile).
        
        # --- Entry Signal (CUSUM + Momentum) ---
        # "Keep the entry condition the same as before"
        raw_entry_signal = 0
        if cusum_signal == 1:
            if momentum_val > momentum_threshold:
                raw_entry_signal = 1
        elif cusum_signal == -1:
            if momentum_val < -momentum_threshold:
                raw_entry_signal = -1
        
        # --- Exit Signal (Momentum vs. its Average) ---
        # "exit when... momentum falling below its rolling average"
        exit_long_signal = momentum_val < momentum_avg_val 
        exit_short_signal = momentum_val > momentum_avg_val
        
        # 4. STATE MACHINE LOGIC
        
        if current_position == 0:
            # --- Handle Entries ---
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= min_cooldown_seconds)
            
            if raw_entry_signal != 0 and cooldown_elapsed:
                current_position = raw_entry_signal
                entry_time = current_time
                clean_signals[i] = raw_entry_signal
        
        elif current_position == 1:
            # --- Handle Long Exit (based on new Momentum logic) ---
            if exit_long_signal:
                if current_time - entry_time >= min_hold_seconds:
                    clean_signals[i] = -1  # Exit long
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
        
        elif current_position == -1:
            # --- Handle Short Exit (based on new Momentum logic) ---
            if exit_short_signal:
                if current_time - entry_time >= min_hold_seconds:
                    clean_signals[i] = 1   # Exit short
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
    
    # EOD Square-off: Force close any open position at the last bar
    if current_position != 0:
        if current_position == 1:
            clean_signals[n-1] = -1
        else:
            clean_signals[n-1] = 1
    
    return timestamps, prices, clean_signals


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day(day_num, data_dir, config):
    """
    Process single day and return signals based on the new strategy
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        # *** GPU loading logic ***
        if config['USE_GPU'] and DASK_CUDF_AVAILABLE:
            try:
                ddf = dask_cudf.read_parquet(str(file_path))
                df = ddf.compute().to_pandas()
            except:
                df = pd.read_parquet(file_path) # Fallback
        else:
            df = pd.read_parquet(file_path)


        required_cols = ['Price', 'PB9_T1'] 
        if not all(col in df.columns for col in required_cols):
            return None 

        # Handle timestamps
        timestamp_col = next((c for c in df.columns if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
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

        all_cols = required_cols + ['timestamp_seconds']
        df = df[all_cols].dropna()

        # *** UPDATED: Added MOMENTUM_AVG_WINDOW to check ***
        if len(df) < max(config['CUSUM_EWM_SPAN'], config['ATR_WINDOW'], config['MOMENTUM_WINDOW'], config['MOMENTUM_AVG_WINDOW']):
            return None # Need enough data for all rolling/ewm windows

        # --- STRATEGY CALCULATIONS ---
        
        # 1. ATR Proxy (Stagnation Filter)
        df['atr_proxy'] = df['PB9_T1'].diff().abs().rolling(window=config['ATR_WINDOW']).mean().bfill()
        is_stagnant = (df['atr_proxy'] < config['ATR_STAGNANT_THRESHOLD']).values
        
        # 2. Momentum (Entry & Exit Filter)
        df['momentum'] = df['PB9_T1'].diff(config['MOMENTUM_WINDOW']).bfill()
        
        # *** NEW: 2b. Momentum Average (Exit Filter) ***
        df['momentum_avg'] = df['momentum'].rolling(window=config['MOMENTUM_AVG_WINDOW']).mean().bfill()

        # 3. CUSUM Input (Entry Filter)
        df['ci'] = df['PB9_T1'].diff().bfill()
        
        # 4. CUSUM Calculation
        changes_array = df['ci'].values.astype(np.float64)
        cusum_up, cusum_down = calculate_directional_cusum(
            changes_array, config['CUSUM_DRIFT'], config['CUSUM_DECAY']
        )
        
        # 5. CUSUM Thresholds & Regime Classification
        cusum_diff_series = pd.Series(cusum_up - cusum_down)
        ewm_center = cusum_diff_series.ewm(span=config['CUSUM_EWM_SPAN']).mean()
        ewm_std = cusum_diff_series.ewm(span=config['CUSUM_EWM_SPAN']).std().bfill()
        
        trending_up_thresh = (ewm_center + (config['CUSUM_TREND_STD_MULT'] * ewm_std)).values
        trending_down_thresh = (ewm_center - (config['CUSUM_TREND_STD_MULT'] * ewm_std)).values
        choppy_up_thresh = (ewm_center + (config['CUSUM_CHOPPY_STD_MULT'] * ewm_std)).values
        choppy_down_thresh = (ewm_center - (config['CUSUM_CHOPPY_STD_MULT'] * ewm_std)).values
        
        cusum_regime_int, _ = classify_regime_by_difference(
            cusum_up, cusum_down, 
            choppy_up_thresh, choppy_down_thresh,
            trending_up_thresh, trending_down_thresh
        )

        # 6. Get remaining arrays for signal generator
        momentum = df['momentum'].values.astype(np.float64)
        momentum_avg = df['momentum_avg'].values.astype(np.float64) # *** NEW ***
        timestamps = df['timestamp_seconds'].values.astype(np.float64)
        prices = df['Price'].values.astype(np.float64) 
        
        # 7. Generate final signals
        signal_times, signal_prices, signals = generate_clean_signals(
            timestamps, prices, is_stagnant, cusum_regime_int, momentum,
            momentum_avg, # *** NEW ***
            config['START_DELAY_SECONDS'],
            config['MIN_HOLD_SECONDS'],
            config['MIN_COOLDOWN_SECONDS'],
            config['MOMENTUM_THRESHOLD']
        )
        
        if len(signals) == 0:
            return None
            
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Time': original_timestamps[:len(signals)],
            'Price': signal_prices,
            'Signal': signals
        })

        return result_df

    except Exception as e:
        # print(f"Error processing day {day_num}: {e}") # Uncomment for debugging
        return None


def signal_wrapper(args):
    """Wrapper for parallel processing"""
    day_num, data_dir, config = args
    return process_day(day_num, data_dir, config)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    # ========================================================================
    # STRATEGY CONFIGURATION
    # ========================================================================
    strategy_config = {
        "USE_GPU": False,
    
        # 1. Primary Filter (ATR / Stagnation)
        "ATR_WINDOW": 20,
        "ATR_STAGNANT_THRESHOLD": 0.01,
        
        # 2. Secondary Filter (CUSUM Trend)
        "CUSUM_DRIFT": 0.005,
        "CUSUM_DECAY": 0.7,
        "CUSUM_EWM_SPAN": 180,
        "CUSUM_TREND_STD_MULT": 1.0,
        "CUSUM_CHOPPY_STD_MULT": 1.0,
        
        # 3. Tertiary Filter (Momentum)
        "MOMENTUM_WINDOW": 7,
        "MOMENTUM_THRESHOLD": 0.0,
        "MOMENTUM_AVG_WINDOW": 50,
        
        # Trading Rules
        "START_DELAY_SECONDS": 600.0,
        "MIN_HOLD_SECONDS": 15.0,
        "MIN_COOLDOWN_SECONDS": 15.0,
    }
    
    # --- Determine effective acceleration ---
    use_gpu_loading = strategy_config['USE_GPU'] and DASK_CUDF_AVAILABLE
    jit_engine = 'Numba' if NUMBA_AVAILABLE else 'Standard Python'
    
    # *** UPDATED PRINT STATEMENTS ***
    print("="*80)
    print("NEW STRATEGY: (1) ATR Master -> (2) 3-Filter Entry -> (3) Mom-Avg Exit")
    print("="*80)
    print(f"Configuration:")
    print(f" 1. Data directory: {data_dir} (Processing {num_days} days)")
    print(f" 2. All features derived from PB9_T1")
    print(f" 3. Loading Engine: {'GPU (Dask-cuDF)' if use_gpu_loading else 'CPU (Pandas)'}")
    print(f" 4. JIT Engine: {jit_engine}")

    print(f"\n--- HIERARCHICAL LOGIC ---")
    print(f" 1. [MASTER FILTER]: IF (ATR_Proxy(PB9_T1, {strategy_config['ATR_WINDOW']}) < {strategy_config['ATR_STAGNANT_THRESHOLD']})")
    print(f"                 THEN: FORCE EXIT position and BLOCK new entries.")
    print(f" 2. [ENTRY]:         IF (Market is VOLATILE) AND (CUSUM == 'Trend') AND (Momentum confirms)")
    print(f"                 THEN: Enter position.")
    print(f" 3. [EXIT]:          IF (Market is VOLATILE) AND (Momentum crosses its {strategy_config['MOMENTUM_AVG_WINDOW']}-period avg)")
    print(f"                 THEN: Exit position.")
    print(f"\n--- TRADING RULES ---")
    print(f" ‣ Start delay: {strategy_config['START_DELAY_SECONDS']}s ({strategy_config['START_DELAY_SECONDS']/60:.0f} min)")
    print(f" ‣ Min Hold: {strategy_config['MIN_HOLD_SECONDS']}s | Min Cooldown: {strategy_config['MIN_COOLDOWN_SECONDS']}s")
    print(f" ‣ EOD Square-off: Enabled")
    print("="*80)
    
    # Check data directory
    print(f"\n[1/4] Checking data directory...")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    parquet_files = list(data_path.glob("day*.parquet"))
    print(f"   ✓ Found {len(parquet_files)} day*.parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No day*.parquet files found in {data_dir}")
        return
    
    # Process all days in parallel
    print(f"\n[2/4] Generating signals for {num_days} days...")
    print(f"   (Using parallel processing with {min(cpu_count(), 16)} workers)")
    start_time = time.time()
    
    workers = min(cpu_count(), 16)
    all_days = list(range(num_days))
    args_list = [(day_num, data_dir, strategy_config) for day_num in all_days]
    
    with Pool(workers) as pool:
        all_signals_results = pool.map(signal_wrapper, args_list)
    
    # Filter out None results
    all_signals = [res for res in all_signals_results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"   ✓ Completed in {elapsed:.1f} seconds")
    print(f"   ✓ Successfully processed {len(all_signals)} days")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated. Check data files and parameters.")
        return
    
    # Combine all days
    print(f"\n[3/4] Combining results...")
    combined_signals = pd.concat(all_signals, ignore_index=True)
    
    # Count signal statistics
    num_long_signals = (combined_signals['Signal'] == 1).sum()
    num_short_signals = (combined_signals['Signal'] == -1).sum()
    num_holds = (combined_signals['Signal'] == 0).sum()
    total_bars = len(combined_signals)
    
    print(f"\n   === SIGNAL STATISTICS ===")
    print(f"   +1 signals:  {num_long_signals:,} (enter long / exit short)")
    print(f"   -1 signals:  {num_short_signals:,} (enter short / exit long)")
    print(f"   0 signals:   {num_holds:,} (hold / no trade)")
    print(f"   Total rows:  {total_bars:,}")
    
    # Save to CSV
    print(f"\n[4/4] Saving results...")
    save_path = '/data/quant14/signals/trade_signalsp2.csv'
    combined_signals.to_csv(save_path, index=False)
    print(f"   ✓ Saved: {save_path} ({len(combined_signals):,} rows)")
    print(f"     Columns: Time, Price, Signal")
    
    print("\n" + "="*80)
    print("✓ SIGNAL GENERATION COMPLETED!")
    print("="*80)


if __name__ == '__main__':
    main()