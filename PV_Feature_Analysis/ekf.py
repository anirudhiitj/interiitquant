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
import multiprocessing
import gc
import traceback

# Try to import Numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
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
    'OUTPUT_FILE': 'eama_ekf_signals_pure.csv', 
    'TEMP_DIR': '/tmp/eama_ekf_pure_temp', 

    # Required columns
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',

    # --- INDICATOR PARAMETERS ---
    'super_smoother_period': 30,
    'eama_er_period': 15,
    'eama_fast': 30,
    'eama_slow': 120,
    'ekf_Q': 0.07,
    'ekf_R': 0.2,

    # --- TRADING LOGIC ---
    'take_profit_points': 2.0,    
    'min_signal_gap_seconds': 5,  
    'strategy_start_tick': 120,
    
    'warmup_seconds': 900.0, 
    
    'MAX_WORKERS': 24,            
}

#############################################################################
# NUMBA-ACCELERATED CORE FUNCTIONS
#############################################################################

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    if n == 0: return ss
    
    pi = 3.14159265358979323846
    arg = 1.414 * pi / period
    a1 = np.exp(-arg)
    b1 = 2.0 * a1 * np.cos(arg)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    
    ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, er_period, fast, slow):
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    if n == 0: return eama
    
    eama[0] = ss_values[0]
    fast_sc_limit = 2.0 / (fast + 1.0)
    slow_sc_limit = 2.0 / (slow + 1.0)
    
    for i in range(1, n):
        er = 0.0
        if i >= er_period:
            direction = abs(ss_values[i] - ss_values[i - er_period])
            volatility = 0.0
            for j in range(er_period):
                volatility += abs(ss_values[i - j] - ss_values[i - j - 1])
            if volatility > 1e-10:
                er = direction / volatility
        
        sc = ((er * (fast_sc_limit - slow_sc_limit)) + slow_sc_limit) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
    return eama

@jit(nopython=True, fastmath=True)
def detect_crossover(series1, series2, i):
    if i < 1: return 0
    curr_diff = series1[i] - series2[i]
    prev_diff = series1[i-1] - series2[i-1]
    
    if prev_diff <= 0 and curr_diff > 0: return 1
    if prev_diff >= 0 and curr_diff < 0: return -1
    return 0

@jit(nopython=True, fastmath=True)
def generate_crossover_signals(timestamps, prices, eama, ekf, 
                               tp_points, min_gap_seconds, 
                               strategy_start_tick, warmup_seconds):
    """
    MODIFIED LOGIC:
    1. Check Warmup (Time based)
    2. Detect Crossover (EKF vs EAMA)
    3. Execute Trade until end of data (is_last_tick)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    STATE_NEUTRAL = 0
    STATE_LONG = 1
    STATE_SHORT = -1
    
    state = STATE_NEUTRAL
    tp_price = 0.0
    last_signal_time = -999999.0
    
    start_time_of_data = timestamps[0]
    
    for i in range(n):
        if i < strategy_start_tick: continue
        
        current_time = timestamps[i]
        
        # --- WARMUP CHECK ---
        if (current_time - start_time_of_data) < warmup_seconds:
            continue
            
        current_price = prices[i]
        time_since_last = current_time - last_signal_time
        is_last_tick = (i == n - 1)
        
        crossover = detect_crossover(ekf, eama, i)
        
        # ==========================================
        # ENTRY LOGIC
        # ==========================================
        if state == STATE_NEUTRAL:
            if time_since_last >= min_gap_seconds and not is_last_tick:
                
                potential_sig = 0
                if crossover == 1: potential_sig = 1
                elif crossover == -1: potential_sig = -1
                
                if potential_sig != 0:
                    if potential_sig == 1:
                        signals[i] = 1
                        state = STATE_LONG
                        tp_price = current_price + tp_points
                    else:
                        signals[i] = -1
                        state = STATE_SHORT
                        tp_price = current_price - tp_points
                    
                    last_signal_time = current_time

        # ==========================================
        # EXIT LOGIC
        # ==========================================
        elif state == STATE_LONG:
            exit_sig = 0
            # Always close on the very last tick of the dataset
            if is_last_tick: exit_sig = -1
            elif current_price >= tp_price and time_since_last >= min_gap_seconds: exit_sig = -1
            elif crossover == -1 and time_since_last >= min_gap_seconds: exit_sig = -1
            
            if exit_sig != 0:
                signals[i] = exit_sig
                state = STATE_NEUTRAL
                last_signal_time = current_time

        elif state == STATE_SHORT:
            exit_sig = 0
            # Always close on the very last tick of the dataset
            if is_last_tick: exit_sig = 1
            elif current_price <= tp_price and time_since_last >= min_gap_seconds: exit_sig = 1
            elif crossover == 1 and time_since_last >= min_gap_seconds: exit_sig = 1
            
            if exit_sig != 0:
                signals[i] = exit_sig
                state = STATE_NEUTRAL
                last_signal_time = current_time

    return signals

#############################################################################
# FILE PROCESSING FUNCTIONS
#############################################################################

def calculate_ekf_trend_exact(prices, Q, R):
    n = len(prices)
    ekf_values = np.zeros(n, dtype=np.float64)
    x = np.array([[prices[0]]])
    P = np.array([[1.0]])
    Q_mat = np.array([[Q]])
    R_mat = np.array([[R]])
    
    for i in range(n):
        p = prices[i]
        if np.isnan(p) or p <= 0:
            ekf_values[i] = x[0,0]
            continue
            
        x_pred = x
        P_pred = P + Q_mat
        
        z = np.array([[np.log(p)]])
        H = np.array([[1.0 / x_pred[0, 0]]])
        y = z - np.array([[np.log(x_pred[0, 0])]])
        S = H @ P_pred @ H.T + R_mat
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(1) - K @ H) @ P_pred
        
        ekf_values[i] = x[0, 0]
        
    return ekf_values

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: str, config: dict) -> dict:
    try:
        temp_dir_path = pathlib.Path(temp_dir)
        
        # 1. READ DATA
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        del ddf
        del gdf
        gc.collect() 
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(float)
        
        timestamps = df['Time_sec'].values.astype(np.float64)
        prices = df[config['PRICE_COLUMN']].values.astype(np.float64)
        prices = np.where(np.isnan(prices) | (prices <= 0), 1e-10, prices)
        
        # 3. CPU CALCULATIONS
        
        # A. Core Indicators (SuperSmoother, EAMA, EKF)
        super_smoother = calculate_super_smoother_numba(prices, config['super_smoother_period'])
        
        eama = calculate_eama_numba(
            super_smoother,
            config['eama_er_period'],
            config['eama_fast'],
            config['eama_slow']
        )
        
        ekf_raw = calculate_ekf_trend_exact(prices, config['ekf_Q'], config['ekf_R'])
        ekf_trend = np.roll(ekf_raw, 1)
        ekf_trend[0] = ekf_raw[0]
        
        # B. Generate Signals
        signals = generate_crossover_signals(
            timestamps=timestamps,
            prices=prices,
            eama=eama,
            ekf=ekf_trend,
            tp_points=config['take_profit_points'],
            min_gap_seconds=config['min_signal_gap_seconds'],
            strategy_start_tick=config['strategy_start_tick'],
            warmup_seconds=config['warmup_seconds']
        )
        
        long_entries = np.sum(signals == 1)
        short_entries = np.sum(signals == -1)
        total_signals = long_entries + short_entries
        
        print(f"  [Day {day_num}] Sig: {total_signals}")
        
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'SuperSmoother': super_smoother,
            'EAMA': eama,
            'EKF_Trend': ekf_trend,
            'Signal': signals
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
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

def process_wrapper(args):
    return process_single_day(*args)

#############################################################################
# MAIN EXECUTION
#############################################################################

def main(config):
    start_time = time.time()
    
    print("\n" + "="*80)
    print("STRATEGY: EAMA-EKF (NO DURATION LIMIT)")
    print("="*80)
    print(f"Data Dir:        {config['DATA_DIR']}")
    print(f"Output:          {config['OUTPUT_FILE']}")
    print(f"Warmup:          {config['warmup_seconds']} seconds")
    print("="*80 + "\n")
    
    # Setup Temp Directory
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    # Find Files
    files_pattern = os.path.join(config['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print("❌ No parquet files found.")
        sys.exit(1)
    
    print(f"📁 Found {len(sorted_files)} files")
    
    tasks = [
        (f, extract_day_num(f), str(temp_dir_path), config) 
        for f in sorted_files
    ]
    
    processed_files = []
    total_signals = 0
    total_long = 0
    total_short = 0

    with multiprocessing.Pool(processes=config['MAX_WORKERS'], maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(process_wrapper, tasks):
            if res:
                processed_files.append(res['path'])
                total_signals += res['total_signals']
                total_long += res['long_entries']
                total_short += res['short_entries']
    
    if processed_files:
        print(f"\n{'='*80}")
        print("📊 COMBINING RESULTS...")
        
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i != 0:
                        infile.readline()  
                    shutil.copyfileobj(infile, outfile)
        
        elapsed = time.time() - start_time
        
        print(f"{'='*80}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Output File:     {output_path}")
        print(f"Total Signals:   {total_signals:,}")
        print(f"Execution Time:  {elapsed:.2f}s")
        print(f"{'='*80}\n")
    else:
        print("❌ No data processed successfully.")
    
    shutil.rmtree(temp_dir_path)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass
    main(CONFIG)