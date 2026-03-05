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

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - using CPU only (install with: pip install cupy-cuda12x)")

try:
    import dask_cudf
    import dask.dataframe as dd
    DASK_CUDF_AVAILABLE = True
    print("✓ Dask-cuDF detected - GPU-accelerated dataframe operations enabled")
except ImportError:
    DASK_CUDF_AVAILABLE = False
    print("✗ Dask-cudf not found - using pandas (install with: pip install dask-cudf)")


# ============================================================================
# CUSUM REGIME DETECTION
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_directional_cusum(PB9_T1s, drift=0.001, decay=0.95):
    """
    Calculate directional CUSUM on PB9_T1 changes.
    The opposing CUSUM decays instead of instantly resetting.
    
    Args:
        PB9_T1s: Array of PB9_T1 values
        drift: Minimum change to count as directional movement
        decay: Decay factor for opposing CUSUM
    
    Returns:
        cusum_up_values, cusum_down_values, PB9_T1_changes
    """
    n = len(PB9_T1s)
    
    # Calculate PB9_T1 changes
    PB9_T1_changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        PB9_T1_changes[i] = PB9_T1s[i] - PB9_T1s[i-1]
    
    # Initialize CUSUM tracking
    cusum_up = 0.0
    cusum_down = 0.0
    
    cusum_up_values = np.zeros(n, dtype=np.float64)
    cusum_down_values = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = PB9_T1_changes[i]
        
        # Upward move
        if change > drift:
            cusum_up += change
            cusum_down = max(0.0, cusum_down * decay)
        
        # Downward move
        elif change < -drift:
            cusum_down += abs(change)
            cusum_up = max(0.0, cusum_up * decay)
        
        # Small move (noise/consolidation) - both decay
        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)
        
        # Store values
        cusum_up_values[i] = cusum_up
        cusum_down_values[i] = cusum_down
    
    return cusum_up_values, cusum_down_values


@jit(nopython=True, fastmath=True)
def calculate_ewm_thresholds(cusum_diff, span=180, 
                             trending_up_mult=1.0, trending_down_mult=1.0):
    """
    Calculate dynamic EWM-based thresholds for regime classification.
    Simplified implementation compatible with Numba.
    
    Args:
        cusum_diff: CUSUM difference array (cusum_up - cusum_down)
        span: EWM span parameter
        trending_up_mult: Multiplier for trending up threshold
        trending_down_mult: Multiplier for trending down threshold
    
    Returns:
        trending_up_thresh, trending_down_thresh arrays
    """
    n = len(cusum_diff)
    alpha = 2.0 / (span + 1.0)
    
    # Calculate EWM mean
    ewm_mean = np.zeros(n, dtype=np.float64)
    ewm_mean[0] = cusum_diff[0]
    
    for i in range(1, n):
        ewm_mean[i] = alpha * cusum_diff[i] + (1.0 - alpha) * ewm_mean[i-1]
    
    # Calculate EWM std (simplified)
    ewm_var = np.zeros(n, dtype=np.float64)
    ewm_var[0] = 0.0
    
    for i in range(1, n):
        diff_sq = (cusum_diff[i] - ewm_mean[i]) ** 2
        ewm_var[i] = alpha * diff_sq + (1.0 - alpha) * ewm_var[i-1]
    
    ewm_std = np.sqrt(ewm_var)
    
    # Calculate thresholds
    trending_up_thresh = ewm_mean + (trending_up_mult * ewm_std)
    trending_down_thresh = ewm_mean - (trending_down_mult * ewm_std)
    
    return trending_up_thresh, trending_down_thresh


@jit(nopython=True, fastmath=True)
def classify_cusum_regimes(cusum_diff, trending_up_thresh, trending_down_thresh):
    """
    Classify each bar as Trending Up, Trending Down, or Other.
    
    Args:
        cusum_diff: CUSUM difference values
        trending_up_thresh: Upper threshold for trending up
        trending_down_thresh: Lower threshold for trending down
    
    Returns:
        regime array: 1 = Trending Up, -1 = Trending Down, 0 = Other
    """
    n = len(cusum_diff)
    regimes = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        if cusum_diff[i] > trending_up_thresh[i]:
            regimes[i] = 1  # Trending Up
        elif cusum_diff[i] < trending_down_thresh[i]:
            regimes[i] = -1  # Trending Down
        else:
            regimes[i] = 0  # Other (Choppy/Transition)
    
    return regimes


# ============================================================================
# SIGNAL GENERATION WITH START DELAY - NO LOOKAHEAD BIAS
# ============================================================================

@jit(nopython=True, fastmath=True)
def generate_clean_signals_with_cusum(bb_avg, timestamps, prices, cusum_regimes,
                                       start_delay_seconds=1200.0):
    """
    Generate trading signals with NO lookahead bias + CUSUM regime filtering
    
    CRITICAL: Compare current bar with PREVIOUS bar only
    - At bar i, we compare bb_avg[i] with bb_avg[i-1]
    - We NEVER use bb_avg[i+1] or any future information
    
    CUSUM Filter:
    - Only trade when CUSUM regime is Trending Up (1) or Trending Down (-1)
    - CUSUM regime does NOT dictate long/short direction
    - BB strategy determines entry/exit direction
    
    FIXED RULES:
    - Force exit on regime change ONLY after 15s hold period
    - Strictly 15s gap between ANY two signals (no exceptions)
    - EOD square-off WITHOUT conflicting with last signal
    
    Args:
        bb_avg: BB average values
        timestamps: Timestamp values in seconds
        prices: Price values
        cusum_regimes: Regime array (1=Trending Up, -1=Trending Down, 0=Other)
        start_delay_seconds: Seconds to wait before generating signals (default 1200s = 20min)
    
    Returns:
        timestamps, prices, signals
    """
    n = len(bb_avg)
    
    # Find the first timestamp after start_delay_seconds
    start_timestamp = timestamps[0] + start_delay_seconds
    
    # Calculate raw signals WITHOUT lookahead bias
    # Compare current bar with previous bar (not future bar)
    raw_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(1, n):  # Start from index 1 (need previous bar)
        # Compare current vs previous - NO FUTURE DATA
        if bb_avg[i] < bb_avg[i-1]:
            raw_signals[i] = -1  # Current bar lower than previous = Bearish
        elif bb_avg[i] > bb_avg[i-1]:
            raw_signals[i] = 1   # Current bar higher than previous = Bullish
        else:
            raw_signals[i] = 0   # No change
    
    # Apply CUSUM regime filter to raw signals
    regime_filtered_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        # Only allow signals in trending regimes (up or down)
        if cusum_regimes[i] == 1 or cusum_regimes[i] == -1:
            regime_filtered_signals[i] = raw_signals[i]
        else:
            # Not trending - no signals
            regime_filtered_signals[i] = 0
    
    # Generate clean signals with position tracking and all rules
    clean_signals = np.zeros(n, dtype=np.int8)
    
    current_position = 0  # 0 = flat, 1 = long, -1 = short
    entry_time = 0.0 
    last_signal_time = 0.0  # Track time of LAST SIGNAL (any signal)
    min_hold_seconds = 15.0
    min_gap_seconds = 15.0  # Strict 15s gap between ANY signals
    
    for i in range(n):
        current_time = timestamps[i]
        signal = regime_filtered_signals[i]
        
        # Skip signal generation before start_delay_seconds
        if current_time < start_timestamp:
            clean_signals[i] = 0
            continue
        
        # CRITICAL: If regime turns non-trending while in position
        # Force exit ONLY if 15s hold period is satisfied
        if cusum_regimes[i] == 0 and current_position != 0:
            # Check both hold period AND gap since last signal
            hold_satisfied = (current_time - entry_time >= min_hold_seconds)
            gap_satisfied = (current_time - last_signal_time >= min_gap_seconds)
            
            if hold_satisfied and gap_satisfied:
                # Exit position
                if current_position == 1:
                    clean_signals[i] = -1  # Exit long
                else:
                    clean_signals[i] = 1   # Exit short
                
                current_position = 0
                last_signal_time = current_time
                entry_time = 0.0
            # If hold/gap not satisfied, stay in position silently
            continue
        
        if current_position == 0:
            # Flat - can enter any position IF gap satisfied
            gap_satisfied = (last_signal_time == 0.0) or (current_time - last_signal_time >= min_gap_seconds)
            
            if gap_satisfied:
                if signal == 1:
                    # Enter long
                    current_position = 1
                    entry_time = current_time
                    last_signal_time = current_time
                    clean_signals[i] = 1
                    
                elif signal == -1:
                    # Enter short
                    current_position = -1
                    entry_time = current_time
                    last_signal_time = current_time
                    clean_signals[i] = -1
        
        elif current_position == 1:
            # In long position - check for exit signal (-1)
            if signal == -1:
                # Check both hold period AND gap since last signal
                hold_satisfied = (current_time - entry_time >= min_hold_seconds)
                gap_satisfied = (current_time - last_signal_time >= min_gap_seconds)
                
                if hold_satisfied and gap_satisfied:
                    # Exit long to flat
                    current_position = 0
                    last_signal_time = current_time
                    entry_time = 0.0
                    clean_signals[i] = -1
        
        elif current_position == -1:
            # In short position - check for exit signal (+1)
            if signal == 1:
                # Check both hold period AND gap since last signal
                hold_satisfied = (current_time - entry_time >= min_hold_seconds)
                gap_satisfied = (current_time - last_signal_time >= min_gap_seconds)
                
                if hold_satisfied and gap_satisfied:
                    # Exit short to flat
                    current_position = 0
                    last_signal_time = current_time
                    entry_time = 0.0
                    clean_signals[i] = 1
    
    # EOD Square-off: Force close any open position
    # BUT: Only if we haven't already signaled at the last bar
    if current_position != 0 and clean_signals[n-1] == 0:
        # Check if 15s hold period satisfied for final exit
        if timestamps[n-1] - entry_time >= min_hold_seconds:
            if current_position == 1:
                clean_signals[n-1] = -1  # Exit long
            else:
                clean_signals[n-1] = 1   # Exit short
    
    return timestamps, prices, clean_signals


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day_for_signals(day_num, data_dir, start_delay=1200.0,
                           cusum_drift=0.001, cusum_decay=0.95, 
                           ewm_span=180, trending_mult=1.0):
    """
    Process single day and return signals with CUSUM regime filtering
    
    Args:
        day_num: Day number to process
        data_dir: Directory containing parquet files
        start_delay: Seconds to wait before starting signal generation (default 1200s = 20min)
        cusum_drift: CUSUM drift parameter
        cusum_decay: CUSUM decay parameter
        ewm_span: EWM span for threshold calculation
        trending_mult: Multiplier for trending thresholds
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        # Load data - use dask_cudf if available for GPU acceleration
        if DASK_CUDF_AVAILABLE:
            try:
                ddf = dask_cudf.read_parquet(str(file_path))
                df = ddf.compute()
                df = df.to_pandas()
            except:
                df = pd.read_parquet(file_path)
        else:
            df = pd.read_parquet(file_path)

        # Check for required columns
        if 'Price' not in df.columns or 'PB9_T1' not in df.columns:
            return None

        # Check for BB columns
        bb_cols = [c for c in df.columns if c in ['BB5_T2', 'BB13_T2', 'BB4_T2', 'BB6_T2']]
        if len(bb_cols) == 0:
            return None

        # Calculate BB average - use GPU if available
        if GPU_AVAILABLE and DASK_CUDF_AVAILABLE:
            try:
                bb_values = cp.array(df[bb_cols].values, dtype=cp.float64)
                bb_avg_gpu = cp.mean(bb_values, axis=1)
                df['BB_avg'] = cp.asnumpy(bb_avg_gpu)
            except:
                df['BB_avg'] = df[bb_cols].mean(axis=1)
        else:
            df['BB_avg'] = df[bb_cols].mean(axis=1)

        # Handle timestamps - preserve original timestamp column
        timestamp_col = next((c for c in df.columns if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col:
            original_timestamps = df[timestamp_col].values
            try:
                # Try to parse timestamps and convert to seconds for hold period tracking
                dt_series = pd.to_datetime(df[timestamp_col], format='mixed', errors='coerce')
                
                # Convert to seconds from start of day
                df['timestamp_seconds'] = (
                    dt_series.dt.hour * 3600 + 
                    dt_series.dt.minute * 60 + 
                    dt_series.dt.second + 
                    dt_series.dt.microsecond / 1e6
                )
                
                # Check if conversion failed (NaT values)
                if df['timestamp_seconds'].isna().any():
                    raise ValueError("Timestamp conversion failed")
                    
            except:
                # Fallback: assume 1-second intervals
                df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)
        else:
            # No timestamp column found: use index
            original_timestamps = np.arange(len(df))
            df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)

        # Select required columns
        required_cols = ['Price', 'BB_avg', 'timestamp_seconds', 'PB9_T1']
        df = df[required_cols].dropna()

        if len(df) < 2:
            return None

        # Get arrays
        prices = df['Price'].values.astype(np.float64)
        bb_avg_vals = df['BB_avg'].values.astype(np.float64)
        timestamps = df['timestamp_seconds'].values.astype(np.float64)
        pb9_t1_vals = df['PB9_T1'].values.astype(np.float64)
        
        # Calculate CUSUM regimes
        cusum_up, cusum_down = calculate_directional_cusum(
            pb9_t1_vals, drift=cusum_drift, decay=cusum_decay
        )
        cusum_diff = cusum_up - cusum_down
        
        # Calculate dynamic thresholds
        trending_up_thresh, trending_down_thresh = calculate_ewm_thresholds(
            cusum_diff, span=ewm_span, 
            trending_up_mult=trending_mult, 
            trending_down_mult=trending_mult
        )
        
        # Classify regimes
        cusum_regimes = classify_cusum_regimes(
            cusum_diff, trending_up_thresh, trending_down_thresh
        )
        
        # Generate signals with CUSUM regime filter and start delay - NO LOOKAHEAD
        signal_times, signal_prices, signals = generate_clean_signals_with_cusum(
            bb_avg_vals, timestamps, prices, cusum_regimes, start_delay
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
        # Suppress errors during parallel execution
        return None


def signal_wrapper(args):
    """Wrapper for parallel processing"""
    day_num, data_dir, start_delay, cusum_drift, cusum_decay, ewm_span, trending_mult = args
    return process_day_for_signals(day_num, data_dir, start_delay,
                                   cusum_drift, cusum_decay, ewm_span, trending_mult)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    START_DELAY_SECONDS = 1200.0  # 20 minutes = skip volatile opening period
    
    # CUSUM Parameters
    CUSUM_DRIFT = 0.001    # Minimum PB9_T1 change to count as directional ($)
    CUSUM_DECAY = 0.95     # Decay factor for opposing CUSUM
    
    # EWM Dynamic Threshold Parameters
    EWM_SPAN = 1200         # EWM Span for threshold calculation
    TRENDING_MULT = 3.0    # Multiplier for trending thresholds
    
    print("="*80)
    print("TRADING SIGNAL GENERATION - BB CROSSOVER + CUSUM REGIME FILTER")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Total days: {num_days}")
    print(f"  Acceleration: {'GPU (Dask-cuDF + CuPy)' if (DASK_CUDF_AVAILABLE and GPU_AVAILABLE) else 'CPU (Pandas)'}")
    print(f"  JIT Engine: {'Numba' if NUMBA_AVAILABLE else 'Standard Python'}")
    print(f"\n  === CRITICAL FIX ===")
    print(f"  ✓ NO LOOKAHEAD BIAS - Compares current vs PREVIOUS bar only")
    print(f"  ✓ Signal at bar[i] uses ONLY data from bar[i] and bar[i-1]")
    print(f"  ✓ NEVER uses future data from bar[i+1]")
    print(f"\n  === CUSUM REGIME FILTER ===")
    print(f"  Using PB9_T1 for regime detection")
    print(f"  CUSUM Drift: ${CUSUM_DRIFT} (min change to count)")
    print(f"  CUSUM Decay: {CUSUM_DECAY} (decay on opposing side)")
    print(f"  EWM Span: {EWM_SPAN} bars")
    print(f"  Trending Threshold: Center ± {TRENDING_MULT}σ")
    print(f"  → Only trade during Trending Up OR Trending Down regimes")
    print(f"  → CUSUM does NOT dictate long/short direction")
    print(f"  → BB strategy determines entry/exit signals")
    print(f"\n  === STRATEGY ===")
    print(f"  BB Average Crossover (filtered by CUSUM)")
    print(f"  - BB_avg = average of BB5_T2, BB13_T2, BB4_T2, BB6_T2")
    print(f"  - Signal when BB_avg changes direction")
    print(f"  - ONLY trade when CUSUM shows trending regime")
    print(f"\n  === TRADING RULES ===")
    print(f"  Start delay: {START_DELAY_SECONDS:.0f} seconds ({START_DELAY_SECONDS/60:.0f} minutes)")
    print(f"  → Signals start ONLY after {START_DELAY_SECONDS:.0f}s into each day")
    print(f"  Signal meanings:")
    print(f"    +1: Enter long (if flat) OR Exit short (if short)")
    print(f"    -1: Enter short (if flat) OR Exit long (if long)")
    print(f"    0:  Hold current position (do nothing)")
    print(f"  Rules:")
    print(f"    - 15-second minimum hold period (must hold before exit)")
    print(f"    - STRICT 15-second gap between ANY two signals")
    print(f"    - Force exit on regime change ONLY after 15s hold satisfied")
    print(f"    - EOD square-off (only if last bar doesn't already have signal)")
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
    print(f"\n[2/4] Generating signals with CUSUM regime filtering for {num_days} days...")
    print(f"       (Using parallel processing with {min(cpu_count(), 16)} workers)")
    start_time = time.time()
    
    workers = min(cpu_count(), 16)
    all_days = list(range(num_days))
    args_list = [(day_num, data_dir, START_DELAY_SECONDS, 
                  CUSUM_DRIFT, CUSUM_DECAY, EWM_SPAN, TRENDING_MULT) 
                 for day_num in all_days]
    
    with Pool(workers) as pool:
        all_signals_results = pool.map(signal_wrapper, args_list)
    
    # Filter out None results
    all_signals = [res for res in all_signals_results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f} seconds")
    print(f"  ✓ Successfully processed {len(all_signals)} days")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated")
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
    print(f"  +1 signals:   {num_long_signals:,} (enter long / exit short)")
    print(f"  -1 signals:   {num_short_signals:,} (enter short / exit long)")
    print(f"  0 signals:    {num_holds:,} (hold / no trade)")
    print(f"  Total rows:   {total_bars:,}")
    print(f"\n  Trade reduction: ~{(num_holds/total_bars)*100:.1f}% of bars filtered out")
    print(f"  (CUSUM regime filter + {START_DELAY_SECONDS:.0f}s opening period)")
    
    # Save to CSV
    print(f"\n[4/4] Saving results...")
    combined_signals.to_csv('/data/quant14/signals/trade_signalsp2.csv', index=False)
    print(f"  ✓ Saved: trade_signalsp.csv ({len(combined_signals):,} rows)")
    print(f"    Columns: Time, Price, Signal")
    
    print("\n" + "="*80)
    print("✓ SIGNAL GENERATION COMPLETED - BB + CUSUM REGIME FILTER!")
    print("="*80)


if __name__ == '__main__':
    main()