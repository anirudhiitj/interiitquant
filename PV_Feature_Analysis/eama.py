import sys
import os
import glob
import re
import shutil
import pathlib
import time
import math
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
# STRATEGY CONFIGURATION
#############################################################################

CONFIG = {
    # Data paths
    'DATA_DIR': '/data/quant14/EBY/',  
    'OUTPUT_DIR': '/data/quant14/signals/',  
    'OUTPUT_FILE': 'eama_signals.csv',
    'TEMP_DIR': '/tmp/eama_slope_temp',

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # Ehlers Super Smoother parameters
    'super_smoother_period': 25,

    # EAMA parameters
    'eama_period': 15,           
    'eama_fast_period': 30,      
    'eama_slow_period': 120,
    
    # NEW: Slope MA Period
    'slope_ma_period': 5,        # Explicitly set to 3 ticks as requested

    # Entry thresholds (Applied to the Slope MA)
    'long_entry_slope': 0.0002,   
    'short_entry_slope': -0.0002, 

    # Exit conditions
    'take_profit_points': 2.0,    
    'use_slope_exit': True,       
    'min_signal_gap_seconds': 5,  

    # Strategy warmup
    'strategy_start_tick': 600,   

    # Execution
    'MAX_WORKERS': 24,            
}

#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    """
    Calculate Ehlers Super Smoother filter.
    """
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return ss
    
    # Calculate coefficients
    pi = 3.14159265358979323846
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    
    # Initialize first two values
    ss[0] = prices[0]
    if n > 1:
        ss[1] = prices[1]
    
    # Recursive calculation
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    
    return ss


@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, eama_period, eama_fast, eama_slow):
    """
    Calculate EAMA, Raw Slope, and 3-Tick Slope MA.
    """
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    eama_slope = np.zeros(n, dtype=np.float64)
    eama_slope_ma = np.zeros(n, dtype=np.float64) # The 3-tick MA of the slope
    
    if n == 0:
        return eama, eama_slope, eama_slope_ma
    
    # Initialize
    eama[0] = ss_values[0]
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (eama_fast + 1.0)
    slow_sc = 2.0 / (eama_slow + 1.0)
    
    for i in range(1, n):
        # 1. Calculate Efficiency Ratio (ER)
        if i >= eama_period:
            direction = abs(ss_values[i] - ss_values[i - eama_period])
            volatility = 0.0
            for j in range(eama_period):
                if i - j > 0:
                    volatility += abs(ss_values[i - j] - ss_values[i - j - 1])
            
            if volatility > 1e-10:
                er = direction / volatility
            else:
                er = 0.0
        else:
            er = 0.0
        
        # 2. Update EAMA
        sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
        
        # 3. Calculate Raw Slope
        eama_slope[i] = eama[i] - eama[i-1]
        
        # 4. Calculate 3-Tick Slope MA
        # We average the current slope and the previous 2 slopes
        if i >= 2:
            eama_slope_ma[i] = (eama_slope[i] + eama_slope[i-1] + eama_slope[i-2]) / 3.0
        elif i == 1:
            # Fallback for index 1 (average of 2)
            eama_slope_ma[i] = (eama_slope[i] + eama_slope[i-1]) / 2.0
        else:
            eama_slope_ma[i] = eama_slope[i]
    
    return eama, eama_slope, eama_slope_ma


@jit(nopython=True, fastmath=True)
def generate_eama_slope_signals(
    timestamps,
    prices,
    eama_slope_ma, # This must be the MA array, not raw slope
    long_entry_slope,
    short_entry_slope,
    tp_points,
    use_slope_exit,
    min_gap_seconds,
    strategy_start_tick
):
    """
    Generate signals based on the Slope MA.
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = -1
    
    state = STATE_NEUTRAL
    entry_price = 0.0
    tp_price = 0.0
    last_signal_idx = -1
    
    for i in range(n):
        # Skip warmup
        if i < strategy_start_tick:
            continue
        
        is_last_tick = (i == n - 1)
        current_time = timestamps[i]
        current_price = prices[i]
        
        # KEY LOGIC: Use the MA of the slope
        current_slope_val = eama_slope_ma[i]
        
        time_since_signal = (
            current_time - timestamps[last_signal_idx]
            if last_signal_idx >= 0 else 999999.0
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
            
            # Entry Logic using MA
            if current_slope_val > long_entry_slope:
                signal = 1
                state = STATE_LONG
                entry_price = current_price
                tp_price = entry_price + tp_points
                last_signal_idx = i
            
            elif current_slope_val < short_entry_slope:
                signal = -1
                state = STATE_SHORT
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
            # Exit based on Slope MA crossing opposite threshold
            elif use_slope_exit and current_slope_val < short_entry_slope and time_since_signal >= min_gap_seconds:
                exit_signal = -1
            
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
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
            # Exit based on Slope MA crossing opposite threshold
            elif use_slope_exit and current_slope_val > long_entry_slope and time_since_signal >= min_gap_seconds:
                exit_signal = 1
            
            if exit_signal != 0:
                signal = exit_signal
                state = STATE_NEUTRAL
                last_signal_idx = i
        
        signals[i] = signal
    
    return signals


#############################################################################
# FILE PROCESSING FUNCTIONS
#############################################################################

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    try:
        # 1. Read Data (GPU)
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        # Convert Time to seconds for gap calculations
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        prices = np.where(np.isnan(prices) | (np.abs(prices) < 1e-10), 1e-10, prices)
        
        # 2. Super Smoother
        super_smoother = calculate_super_smoother_numba(prices, config['super_smoother_period'])
        
        # 3. EAMA + Slope + Slope MA (3 ticks)
        # Returns 3 arrays now
        eama, eama_slope, eama_slope_ma = calculate_eama_numba(
            super_smoother,
            config['eama_period'],
            config['eama_fast_period'],
            config['eama_slow_period']
        )
        
        # 4. Generate Signals (Passing eama_slope_ma)
        signals = generate_eama_slope_signals(
            timestamps=timestamps,
            prices=prices,
            eama_slope_ma=eama_slope_ma,  # <-- CRITICAL: Passing the MA array
            long_entry_slope=config['long_entry_slope'],
            short_entry_slope=config['short_entry_slope'],
            tp_points=config['take_profit_points'],
            use_slope_exit=config['use_slope_exit'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            strategy_start_tick=config['strategy_start_tick']
        )
        
        # 5. Stats
        long_entries = np.sum(signals == 1)
        short_entries = np.sum(signals == -1)
        total_signals = long_entries + short_entries
        
        # 6. Save
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'SuperSmoother': super_smoother,
            'EAMA': eama,
            'EAMA_Slope': eama_slope,     # Raw Slope
            'EAMA_Slope_MA3': eama_slope_ma, # The 3-tick MA used for strategy
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
    start_time = time.time()
    
    print("\n" + "="*80)
    print("EAMA SLOPE MA (3-TICK) STRATEGY")
    print("="*80)
    print(f"Data Dir: {config['DATA_DIR']}")
    print(f"Output:   {config['OUTPUT_FILE']}")
    print(f"Entry:    Slope MA > {config['long_entry_slope']} or < {config['short_entry_slope']}")
    
    # Setup Temp
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    # Find Files
    files_pattern = os.path.join(config['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print("No files found.")
        sys.exit(1)
        
    print(f"Processing {len(sorted_files)} files using {config['MAX_WORKERS']} workers.")

    processed_files = []
    
    with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
        futures = {
            executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, config): f
            for f in sorted_files
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                processed_files.append(res['path'])
                print(f"Day {res['day_num']} | Signals: {res['total_signals']}")

    # Combine
    if processed_files:
        print("Combining files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i != 0: infile.readline() 
                    shutil.copyfileobj(infile, outfile)
                    
        print(f"Done. Saved to {output_path}")
        
    shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    main(CONFIG)