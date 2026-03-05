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
# MOMENTUM ALIGNMENT REGIME DETECTION
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_alignment_score(pb13_values):
    """
    Calculate momentum alignment score across 10 timeframes (PB13_T1 to PB13_T10)
    
    Logic:
    - For each timestamp, count how many timeframes agree on direction
    - Agreement = majority pointing same way (positive or negative)
    - Score = (number agreeing) / 10
    - Excludes T11 and T12 (1500s and 3600s) as too slow for intraday
    
    Args:
        pb13_values: 2D array of shape (n_timestamps, 10) 
                     containing PB13_T1 through PB13_T10
    
    Returns:
        alignment_scores: 1D array of shape (n_timestamps,)
                         values from 0.0 to 1.0
                         1.0 = all 10 timeframes agree (strong trend)
                         0.5 = split 5-5 (choppy)
    """
    n = pb13_values.shape[0]
    alignment_scores = np.zeros(n, dtype=np.float64)
    
    # Noise threshold - ignore very small momentum values
    noise_threshold = 0.00001  # 0.005% - adjust if needed
    
    for i in range(n):
        # Count positive, negative, and near-zero values
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for j in range(10):  # Only use first 10 timeframes
            if pb13_values[i, j] > noise_threshold:
                positive_count += 1
            elif pb13_values[i, j] < -noise_threshold:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Alignment = how many agree with the majority direction
        max_agreement = max(positive_count, negative_count)
        
        # Calculate alignment score (0 to 1)
        alignment_scores[i] = max_agreement / 10.0
    
    return alignment_scores


@jit(nopython=True, fastmath=True)
def calculate_regime_stats(alignment_scores, threshold):
    """
    Calculate regime statistics for reporting
    
    Returns: (trending_count, choppy_count)
    """
    trending = 0
    choppy = 0
    
    for i in range(len(alignment_scores)):
        if alignment_scores[i] >= threshold:
            trending += 1
        else:
            choppy += 1
    
    return trending, choppy


# ============================================================================
# SIGNAL GENERATION WITH REGIME FILTER AND START DELAY - NO LOOKAHEAD BIAS
# ============================================================================

@jit(nopython=True, fastmath=True)
def generate_clean_signals_with_regime(bb_avg, timestamps, prices, alignment_scores, 
                                        alignment_threshold=0.80,
                                        start_delay_seconds=1200.0):
    """
    Generate trading signals with NO lookahead bias
    
    CRITICAL FIX: Compare current bar with PREVIOUS bar only
    - At bar i, we compare bb_avg[i] with bb_avg[i-1]
    - We NEVER use bb_avg[i+1] or any future information
    """
    n = len(bb_avg)
    
    # Find the first timestamp after start_delay_seconds
    start_timestamp = timestamps[0] + start_delay_seconds
    
    # FIXED: Calculate raw signals WITHOUT lookahead bias
    # Compare current bar with previous bar (not future bar)
    raw_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(1, n):  # Start from index 1 (need previous bar)
        # Compare current vs previous - NO FUTURE DATA
        if bb_avg[i] - bb_avg[i-1] < -0.002:
            raw_signals[i] = -1  # Current bar lower than previous = Bearish
        elif bb_avg[i] - bb_avg[i-1] > 0.002:
            raw_signals[i] = 1   # Current bar higher than previous = Bullish
        else:
            raw_signals[i] = 0   # No change
    
    # Apply regime filter to raw signals
    regime_filtered_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(n):
        # Only allow signals in trending regimes
        if alignment_scores[i] >= alignment_threshold:
            regime_filtered_signals[i] = raw_signals[i]
        else:
            # Choppy regime - no signals
            regime_filtered_signals[i] = 0
    
    # Generate clean signals with position tracking and all rules
    clean_signals = np.zeros(n, dtype=np.int8)
    
    current_position = 0  # 0 = flat, 1 = long, -1 = short
    entry_time = 0.0 
    last_exit_time = 0.0
    min_hold_seconds = 15.0
    min_cooldown_seconds = 15.0
    
    for i in range(n):
        current_time = timestamps[i]
        signal = regime_filtered_signals[i]
        
        # Skip signal generation before start_delay_seconds
        if current_time < start_timestamp:
            clean_signals[i] = 0
            continue
        
        # CRITICAL: If regime turns choppy while in position, force exit
        if alignment_scores[i] < alignment_threshold and current_position != 0:
            # Regime turned choppy - exit immediately if hold period satisfied
            if current_time - entry_time >= min_hold_seconds:
                if current_position == 1:
                    clean_signals[i] = -1  # Exit long
                else:
                    clean_signals[i] = 1   # Exit short
                
                current_position = 0
                last_exit_time = current_time
                entry_time = 0.0
            # If hold period not satisfied, stay in position but don't take new signals
            continue
        
        if current_position == 0:
            # Flat - can enter any position IF cooldown period has passed
            cooldown_elapsed = (last_exit_time == 0.0) or (current_time - last_exit_time >= min_cooldown_seconds)
            
            if cooldown_elapsed:
                if signal == 1:
                    # Enter long
                    current_position = 1
                    entry_time = current_time
                    clean_signals[i] = 1
                    
                elif signal == -1:
                    # Enter short
                    current_position = -1
                    entry_time = current_time
                    clean_signals[i] = -1
        
        elif current_position == 1:
            # In long position - check for exit signal (-1)
            if signal == -1:
                # Check if 15s hold period satisfied
                if current_time - entry_time >= min_hold_seconds:
                    # Exit long to flat
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    clean_signals[i] = -1
        
        elif current_position == -1:
            # In short position - check for exit signal (+1)
            if signal == 1:
                # Check if 15s hold period satisfied
                if current_time - entry_time >= min_hold_seconds:
                    # Exit short to flat
                    current_position = 0
                    last_exit_time = current_time
                    entry_time = 0.0
                    clean_signals[i] = 1
    
    # EOD Square-off: Force close any open position
    if current_position != 0:
        if current_position == 1:
            clean_signals[n-1] = -1
        else:
            clean_signals[n-1] = 1
    
    return timestamps, prices, clean_signals


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day_for_signals(day_num, data_dir, alignment_threshold=0.80, start_delay=1200.0):
    """
    Process single day and return signals with regime filtering
    
    Args:
        day_num: Day number to process
        data_dir: Directory containing parquet files
        alignment_threshold: Trending threshold (default 8/10 = 0.80)
        start_delay: Seconds to wait before starting signal generation (default 1200s = 20min)
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
        if 'Price' not in df.columns:
            return None

        # Check for BB columns
        bb_cols = [c for c in df.columns if c in ['BB5_T2', 'BB13_T2', 'BB4_T2', 'BB6_T2']]
        if len(bb_cols) == 0:
            return None
        
        # Check for PB13 features T1-T10 only (excluding slow T11 and T12)
        pb13_cols = [f'PB13_T{i}' for i in range(1, 11)]  # Only T1 through T10
        available_pb13 = [c for c in pb13_cols if c in df.columns]
        
        if len(available_pb13) < 10:
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
                # Suppress the UserWarning by inferring format automatically
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

        # Select required columns including PB13 features
        required_cols = ['Price', 'BB_avg', 'timestamp_seconds'] + available_pb13
        df = df[required_cols].dropna()

        if len(df) < 2:
            return None

        # Get arrays
        prices = df['Price'].values.astype(np.float64)
        bb_avg_vals = df['BB_avg'].values.astype(np.float64)
        timestamps = df['timestamp_seconds'].values.astype(np.float64)
        
        # Get PB13 features as 2D array (n_timestamps x 10)
        pb13_values = df[available_pb13].values.astype(np.float64)
        
        # Calculate momentum alignment scores
        alignment_scores = calculate_alignment_score(pb13_values)
        
        # Generate signals with regime filter and start delay - NO LOOKAHEAD
        signal_times, signal_prices, signals = generate_clean_signals_with_regime(
            bb_avg_vals, timestamps, prices, alignment_scores, alignment_threshold, start_delay
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
    day_num, data_dir, threshold, start_delay = args
    return process_day_for_signals(day_num, data_dir, threshold, start_delay)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    # ========================================================================
    # REGIME DETECTION CONFIGURATION
    # ========================================================================
    # Two regimes: Trending vs Choppy
    # Using only T1-T10 timeframes (5s to 900s), excluding slow T11 & T12
    ALIGNMENT_THRESHOLD = 1.0  # 8/10 timeframes agree = Trending, below = Choppy
    START_DELAY_SECONDS = 1200.0  # 20 minutes = skip volatile opening period
    
    print("="*80)
    print("TRADING SIGNAL GENERATION - NO LOOKAHEAD BIAS VERSION")
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
    print(f"\n  === REGIME DETECTION ===")
    print(f"  Using PB13_T1 to PB13_T10 only (excluding slow T11, T12)")
    print(f"  Trending Threshold: {ALIGNMENT_THRESHOLD:.2f} ({int(ALIGNMENT_THRESHOLD*10)}/10 timeframes)")
    print(f"  Choppy: < {ALIGNMENT_THRESHOLD:.2f} → NO TRADING")
    print(f"\n  === TRADING RULES ===")
    print(f"  Start delay: {START_DELAY_SECONDS:.0f} seconds ({START_DELAY_SECONDS/60:.0f} minutes)")
    print(f"  → Signals start ONLY after {START_DELAY_SECONDS:.0f}s into each day")
    print(f"  Signal meanings:")
    print(f"    +1: Enter long (if flat) OR Exit short (if short)")
    print(f"    -1: Enter short (if flat) OR Exit long (if long)")
    print(f"    0:  Hold current position (do nothing)")
    print(f"  Rules:")
    print(f"    - 15-second minimum hold period")
    print(f"    - 15-second cooldown after exit")
    print(f"    - Force exit if regime turns choppy")
    print(f"    - EOD square-off (no overnight)")
    print(f"    - BB strategy ONLY in trending regimes")
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
    print(f"\n[2/4] Generating signals with regime filtering for {num_days} days...")
    print(f"       (Using parallel processing with {min(cpu_count(), 16)} workers)")
    start_time = time.time()
    
    workers = min(cpu_count(), 16)
    all_days = list(range(num_days))
    args_list = [(day_num, data_dir, ALIGNMENT_THRESHOLD, START_DELAY_SECONDS) for day_num in all_days]
    
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
    print(f"  (Includes {START_DELAY_SECONDS:.0f}s opening period skipped per day)")
    
    # Save to CSV
    print(f"\n[4/4] Saving results...")
    combined_signals.to_csv('/data/quant14/signals/trade_signalsp.csv', index=False)
    print(f"  ✓ Saved: trade_signalsp.csv ({len(combined_signals):,} rows)")
    print(f"    Columns: Time, Price, Signal")
    
    print("\n" + "="*80)
    print("✓ SIGNAL GENERATION COMPLETED - NO LOOKAHEAD BIAS!")
    print("="*80)


if __name__ == '__main__':
    main()