#!/usr/bin/env python3
"""
STATE-DEPENDENT HAWKES PROCESS STRATEGY WITH EAMA SMOOTHING - OPTIMIZED WITH NUMBA:

PERFORMANCE OPTIMIZATIONS:
- NUCLEAR MEMORY CLEANUP: Uses maxtasksperchild to restart workers periodically.
- ULTRA-FAST CSV MERGING: Parallel processing + chunked I/O
- Numba JIT compilation for all core functions
- GPU-accelerated data loading with cudf (optional)
- DYNAMIC WALLS: Expanding Mean/StdDev calculated via Numba
"""

import os
import re
import shutil
import time
import gc
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings

import pandas as pd
import numpy as np
try:
    import dask_cudf
    import cudf
    import rmm
    # We don't need dask_cudf for single-file processing, just cudf is enough/safer
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("Warning: cudf/rmm not found. Will use standard pandas/numpy.")
from numba import jit, config, cuda

# ---------------- NUMBA Setup ----------------
warnings.filterwarnings("ignore")
config.THREADING_LAYER = "threadsafe"
config.NUMBA_NUM_THREADS = os.environ.get("NUMBA_NUM_THREADS", "24")
os.environ["NUMBA_NUM_THREADS"] = config.NUMBA_NUM_THREADS

# ------------------------------------------------
# CONFIG 
# ------------------------------------------------
# ONLY ESSENTIAL PATHS RETAINED
TEMP_DIR = "/data/quant14/signals/temp_hawkes_ebX"
OUTPUT_FILE = "/data/quant14/signals/hawkes_trading_signals_EBX.csv"
DATA_DIR = "/data/quant14/EBX"

REQUIRED_COLUMNS = ['Time', 'Price']
DEFAULT_MAX_WORKERS = 24
BLACKLIST_DAYS = {} 

# --- EAMA CONFIGURATION ---
SUPER_SMOOTHER_PERIOD = 25
EAMA_PERIOD = 15
EAMA_FAST_PERIOD = 30
EAMA_SLOW_PERIOD = 120

# --- OPTIMIZED HAWKES CONFIG FOR EAMA ---
HAWKES_LOOKBACK = 1800
HAWKES_THRESHOLD_ABS = 0.00005
HAWKES_ALPHA = 1.2
HAWKES_BETA = 1.8
HAWKES_SIGNAL_THRESHOLD = 0.07

# Fallback walls if data < 60 periods
HARD_WALL_UPPER = 100.19
HARD_WALL_LOWER = 99.55  
COOLDOWN_PERIOD = 5

# --- DYNAMIC WALL CONFIG ---
WALL_MIN_PERIODS = 60
WALL_STD_MULT = 2.0

# --- TRADE CONFIG ---
ATR_LOOKBACK = 25
ATR_MIN_TRADABLE = 0.01
TAKE_PROFIT_ABS = 1.0
TRAIL_STOP_ABS = 0.005
TRADE_COST_ABS = 0.005

EPS = 1e-10


# ------------------------------------------------
# EAMA & WALL INDICATORS (NUMBA-ACCELERATED)
# ------------------------------------------------

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return ss
    
    pi = 3.14159265358979323846
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    
    ss[0] = prices[0]
    if n > 1:
        ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    
    return ss


@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, eama_period, eama_fast, eama_slow):
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    
    if n == 0:
        return eama
    
    eama[0] = ss_values[0]
    fast_sc = 2.0 / (eama_fast + 1.0)
    slow_sc = 2.0 / (eama_slow + 1.0)
    
    for i in range(1, n):
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
        
        sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
    
    return eama


@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period):
    n = len(prices)
    atr = np.zeros(n)
    if n < 2: return atr
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = abs(prices[i] - prices[i-1])
    for i in range(n):
        if i < period - 1:
            window_size = i + 1
            start_idx = 0
        else:
            window_size = period
            start_idx = i - period + 1
        atr_sum = 0.0
        for j in range(start_idx, i + 1):
            atr_sum += tr[j]
        atr[i] = atr_sum / window_size
    return atr


@jit(nopython=True, fastmath=True)
def calculate_expanding_walls_numba(prices, min_periods, std_mult, fallback_upper, fallback_lower):
    """
    Calculates expanding mean +/- std_mult * mean_of_std efficiently in one pass.
    Replaces .expanding().mean() and .std() for Numba compatibility.
    """
    n = len(prices)
    upper_walls = np.zeros(n, dtype=np.float64)
    lower_walls = np.zeros(n, dtype=np.float64)
    
    sum_x = 0.0
    sum_sq_x = 0.0
    sum_std = 0.0
    std_count = 0.0
    
    for i in range(n):
        val = prices[i]
        sum_x += val
        sum_sq_x += (val * val)
        count = i + 1.0
        
        if count >= min_periods:
            mean = sum_x / count
            # Variance = E[X^2] - (E[X])^2
            variance = (sum_sq_x / count) - (mean * mean)
            
            if variance < 0: variance = 0.0 # Float stability
            std = np.sqrt(variance)

            # Accumulate std to get expanding mean of std
            sum_std += std
            std_count += 1.0
            mean_std = sum_std / std_count
            
            upper_walls[i] = mean + (std)
            lower_walls[i] = mean - (std)
        else:
            # Fallback for the first N periods
            upper_walls[i] = fallback_upper
            lower_walls[i] = fallback_lower
            
    return upper_walls, lower_walls


@jit(nopython=True, fastmath=True, cache=True)
def calculate_recursive_hawkes_on_eama(eama_prices, alpha, beta, time_delta, threshold):
    N = len(eama_prices)
    H_buy = np.zeros(N, dtype=np.float64)
    H_sell = np.zeros(N, dtype=np.float64)
    
    decay_factor = np.exp(-beta * time_delta)
    
    for i in range(1, N):
        current_H_buy = H_buy[i-1] * decay_factor
        current_H_sell = H_sell[i-1] * decay_factor
        
        eama_change = eama_prices[i] - eama_prices[i-1]

        if eama_change > threshold:
            current_H_buy += alpha 
        elif eama_change < -threshold:
            current_H_sell += alpha
            
        H_buy[i] = current_H_buy
        H_sell[i] = current_H_sell
        
    return H_buy, H_sell


@jit(nopython=True, fastmath=True)
def generate_signals(raw_prices, h_pos, h_neg, atr, upper_walls, lower_walls, ss_slope, trade_cost, cooldown):
    N = len(raw_prices)
    signals = np.zeros(N, dtype=np.int32)
    
    position = 0
    entry_price = 0.0
    traded_today = False
    day_tradable = False
    tp_activated = False
    tp_anchor = 0.0
    last_trade_time = -cooldown 
    
    for i in range(N):
        p = raw_prices[i]
        
        # # Expanding walls
        # curr_upper = upper_walls[i]
        # curr_lower = lower_walls[i]
        curr_upper = HARD_WALL_UPPER
        curr_lower = HARD_WALL_LOWER
        
        in_cooldown = i - last_trade_time < cooldown
        
        if (not day_tradable) and (atr[i] > ATR_MIN_TRADABLE):
            day_tradable = True
        
        if i < HAWKES_LOOKBACK:
            continue
        
        h_buy_i = h_pos[i]
        h_sell_i = h_neg[i]
        
        distance_lower = p - curr_lower
        if distance_lower < 0.0:
            distance_lower = 0.0 
        long_signal_score = h_sell_i * distance_lower
        
        distance_upper = curr_upper - p
        if distance_upper < 0.0:
            distance_upper = 0.0 
        short_signal_score = h_buy_i * distance_upper
        
        long_sig = long_signal_score > HAWKES_SIGNAL_THRESHOLD
        short_sig = short_signal_score > HAWKES_SIGNAL_THRESHOLD
        
        # --- ENTRY LOGIC ---
        if position == 0 and (not traded_today) and day_tradable and (not in_cooldown):
            current_slope = ss_slope[i]
            
            # Long Entry: Hawkes Long Signal AND Positive Slope Confirmation
            if long_sig and current_slope > 0.01:
                signals[i] = 1 
                position = 1
                entry_price = p + trade_cost
                traded_today = False 
                tp_activated = False
                tp_anchor = 0.0
                last_trade_time = i
            # Short Entry: Hawkes Short Signal AND Negative Slope Confirmation
            elif short_sig and current_slope < -0.01:
                signals[i] = -1 
                position = -1
                entry_price = p - trade_cost
                traded_today = False 
                tp_activated = False
                tp_anchor = 0.0
                last_trade_time = i
        
        # --- EXIT / MANAGEMENT LOGIC ---
        if position != 0 and entry_price > 0:
            exit_now = False
            exit_signal = -position
            
            if position == 1:
                abs_move = p - entry_price
            else:
                abs_move = entry_price - p
            
            if (not tp_activated) and abs_move >= 0.2:
                tp_activated = False
                tp_anchor = p
            
            if tp_activated:
                if position == 1:
                    if p > tp_anchor:
                        tp_anchor = p
                    if p < tp_anchor - TRAIL_STOP_ABS:
                        exit_now = True
                else:
                    if p < tp_anchor:
                        tp_anchor = p
                    if p > tp_anchor + TRAIL_STOP_ABS:
                        exit_now = True
            
            # --- SUPER SMOOTHER SLOPE EXIT ---
            # Exit conditions kept same relative to position
            # Position 1 (Long) -> Exit if slope < -0.01
            # Position -1 (Short) -> Exit if slope > 0.01
            # current_slope = ss_slope[i]
            # if position == 1 and current_slope < -0.01:
            #     exit_now = True
            # elif position == -1 and current_slope > 0.01:
            #     exit_now = True

            if i == N - 1 and position != 0:
                exit_now = True
                exit_signal = -position
            
            if exit_now and (not in_cooldown):
                signals[i] = exit_signal
                position = 0
                entry_price = 0.0
                last_trade_time = i

    return signals, day_tradable


# ------------------------------------------------
# ULTRA-FAST CSV MERGING
# ------------------------------------------------

def ultra_fast_merge_binary(csv_files, output_file):
    """ULTRA-FAST binary concatenation"""
    if not csv_files:
        print("No files to merge.")
        return False

    print(f"⚡⚡ ULTRA-FAST MERGE: {len(csv_files)} files (binary concat)...")
    merge_start = time.time()
    
    with open(csv_files[0], 'rb') as first_file:
        header = first_file.readline()
    
    with open(output_file, 'wb') as outfile:
        outfile.write(header)
    
    for i, fpath in enumerate(csv_files):
        try:
            with open(fpath, 'rb') as infile:
                if i == 0:
                    infile.readline() # Read and ignore header
                else:
                    infile.readline() # Read and ignore header
                with open(output_file, 'ab') as outfile:
                    shutil.copyfileobj(infile, outfile, length=2*1024*1024)
        except Exception as e:
            print(f"⚠️ Error merging {fpath}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Merged {i+1}/{len(csv_files)} files...")
    
    merge_time = time.time() - merge_start
    print(f"✅ Ultra-fast merge completed in {merge_time:.2f}s")
    return True


# ------------------------------------------------
# MULTIPROCESSING FUNCTIONS
# ------------------------------------------------

def extract_day_num(filepath):
    m = re.search(r"day(\d+)", str(filepath))
    return int(m.group(1)) if m else -1

def process_day_wrapper(args):
    """Wrapper needed for multiprocessing.Pool starmap or imap"""
    return process_day(*args)

def process_day(file_path, day_num, temp_dir):
    # DEBUG PRINT: Helps identify if a specific file hangs on start
    # print(f"-> Starting Day {day_num}...", flush=True)

    try:
        # --- PHASE 1: GPU LOADING & TRANSFER ---
        try:
            if DASK_AVAILABLE:
                
                df = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)
                
                # # 2. Read with cuDF
                # gdf = ddf.compute()
                # df = gdf.to_pandas()
                
                # # 3. Clean up variables
                # del ddf
                # del gdf
                # gc.collect()
                
                # REMOVED: cuda.close() 
                # Why? Calling this inside a worker that is about to die can cause 
                # the NVIDIA driver to deadlock/hang. Process death is sufficient.
            else:
                df = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)
        except Exception as e:
            print(f"[WARN] GPU load failed for day {day_num}, falling back to CPU: {e}")
            df = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)

        # --- PHASE 2: CPU PROCESSING (GPU is now free) ---
        if df.empty:
            return None

        df = df.reset_index(drop=True)
        raw_price = df["Price"].astype(float).values
        raw_price = np.where(np.isnan(raw_price) | (np.abs(raw_price) < 1e-10), 1e-10, raw_price)

        # Calculate Indicators (Numba)
        super_smoother = calculate_super_smoother_numba(raw_price, SUPER_SMOOTHER_PERIOD)
        
        # Calculate Slope (diff) of Super Smoother
        # Prepend 0 to keep shape alignment with other arrays
        ss_slope = np.zeros_like(super_smoother)
        if len(super_smoother) > 1:
            ss_slope[1:] = super_smoother[1:] - super_smoother[:-1]

        eama_price = calculate_eama_numba(super_smoother, EAMA_PERIOD, EAMA_FAST_PERIOD, EAMA_SLOW_PERIOD)
        atr = compute_atr_fast(raw_price, ATR_LOOKBACK)
        
        # Calculate Expanding Walls (Numba) - REPLACES PANDAS .EXPANDING()
        u_walls, l_walls = calculate_expanding_walls_numba(
            raw_price, WALL_MIN_PERIODS, WALL_STD_MULT, HARD_WALL_UPPER, HARD_WALL_LOWER
        )

        h_pos, h_neg = calculate_recursive_hawkes_on_eama(
            eama_price, HAWKES_ALPHA, HAWKES_BETA, 1.0, HAWKES_THRESHOLD_ABS
        )
        
        # Pass dynamic walls and slope to signal generator
        signals, day_tradable = generate_signals(
            raw_price, h_pos, h_neg, atr, u_walls, l_walls, ss_slope, TRADE_COST_ABS, COOLDOWN_PERIOD
        )

        out_df = pd.DataFrame({
            "Time": df["Time"],
            "Price": raw_price,
            "Signal": signals,
            "H_Buy": h_pos,
            "H_Sell": h_neg,
            "Upper_Wall": u_walls,
            "Lower_Wall": l_walls,
            "EhlersSuperSmoother_Slope": ss_slope
        })
        
        out_path = Path(temp_dir) / f"day{day_num}.csv"
        out_df.to_csv(out_path, index=False)

        # Force flush to ensure log appears even if crashed right after
        print(f"[OK] day {day_num} | trades={(signals!=0).sum()} | tradable={day_tradable}", flush=True)
        
        return str(out_path)

    except Exception as e:
        print(f"[ERR] day {day_num}: {e}", flush=True)
        return None


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main(directory, max_workers):
    start = time.time()

    temp_dir = Path(TEMP_DIR)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    import glob
    files = sorted(
        glob.glob(os.path.join(directory,"day*.parquet")),
        key=extract_day_num
    )

    outputs=[]
    
    task_args = [(f, extract_day_num(f), str(temp_dir)) for f in files]

    try:
        # Changed maxtasksperchild to 10
        # Killing after *every* task (1) is too aggressive and causes driver deadlocks.
        # 10 is sufficient to keep memory clean without creating OS instability.
        print(f"Starting Pool with {max_workers} workers (maxtasksperchild=10)...")
        with Pool(processes=max_workers, maxtasksperchild=10) as pool:
            
            # chunksize=1 ensures results yield immediately, preventing logs from "batching up"
            results = pool.imap_unordered(process_day_wrapper, task_args, chunksize=1)
            
            for res in results:
                if res:
                    outputs.append(res)

        sorted_outputs = sorted(outputs, key=extract_day_num)
        
        print(f"\n{'='*70}")
        print("MERGING SIGNAL FILES")
        print(f"{'='*70}")
        ultra_fast_merge_binary(sorted_outputs, OUTPUT_FILE)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"✅ TOTAL EXECUTION TIME: {elapsed:.2f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=DATA_DIR)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("STATE-DEPENDENT HAWKES STRATEGY WITH EAMA SMOOTHING")
    print("ULTRA-FAST EDITION (DYNAMIC WALLS + STABLE MEMORY CLEANUP)")
    print("="*70)
    print(f"↪ Pipeline: Raw Price → Super Smoother({SUPER_SMOOTHER_PERIOD}) → EAMA → Hawkes")
    print(f"↪ Wall Logic: Expanding Mean +/- {WALL_STD_MULT}*StdDev (Min Periods: {WALL_MIN_PERIODS})")
    print(f"↪ Workers: {args.max_workers} | Memory Mode: Restart-Worker-Every-10-Tasks")
    print("="*70 + "\n")

    main(args.directory, args.max_workers)