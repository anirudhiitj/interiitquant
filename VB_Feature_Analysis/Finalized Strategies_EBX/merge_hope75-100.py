import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, int64, float64

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing_MERGED"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_MERGED.csv"
REQUIRED_COLUMNS = ['Time', 'Price']

# Blacklist
BLACKLIST_DAYS = {87, 311, 273, 446, 5, 358, 14, 331, 136, 389, 504, 323, 96, 205, 230, 477, 456,
 236, 181, 197, 372, 173, 28, 186, 367, 98, 410, 161, 107, 124, 208, 290, 239,
 408, 137, 343, 108, 36, 448, 298, 85, 421,
 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}

# Strategy Parameters
STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1", 
}

# --- GLOBAL PARAMETERS ---
DECISION_WINDOW_SECONDS = 30 * 60
SIGMA_THRESHOLD = 0.11
MU_THRESHOLD_SPLIT = 100.0 # The dividing line between S1 and S2
COOLDOWN_PERIOD_SECONDS = 5

# ==========================================
# STRATEGY 1 PARAMETERS (Low Mu < 100)
# ==========================================
S1_ENTRY_DROP_THRESHOLD = 0.8  
S1_PRE_DROP_SHORT_SL = 0.4      
S1_LONG_STATIC_SL = 0.7
S1_LONG_TP_ACTIVATION = 1.0
S1_LONG_TRAIL_CALLBACK = 0.1
S1_REVERSAL_RISE_TRIGGER = 0.2
S1_SHORT_STATIC_SL = 4.4
S1_SHORT_TP_ACTIVATION = 6.2
S1_SHORT_TRAIL_CALLBACK = 0.04

# ==========================================
# STRATEGY 2 PARAMETERS (High Mu > 100)
# ==========================================
S2_ENTRY_RISE_THRESHOLD = 0.9 
S2_PRE_RISE_SL = 0.9
S2_PRIMARY_SHORT_SL = 0.6
S2_PRIMARY_TP_ACTIVATION = 1.0
S2_PRIMARY_TRAIL_CALLBACK = 0.1
S2_REVERSAL_DROP_TRIGGER = 0.2
S2_SECONDARY_LONG_SL = 0.15
S2_SECONDARY_TP_ACTIVATION = 0.8
S2_SECONDARY_TRAIL_CALLBACK = 0.1


# --- Numba Core 1: Strategy 1 (Mu < 100) ---
@njit(cache=True, nogil=True)
def strategy_1_core(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_long,   
    cooldown_seconds,
    entry_drop_threshold,
    pre_drop_short_sl
):
    """
    Strategy 1 Logic: Pre-Drop Short -> Flip Long -> Reversal Short
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # Pre-Drop Logic State
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    # Trailing Stop State (For Longs)
    trailing_active = False
    trailing_peak_price = 0.0 

    # Trailing Stop State (For Shorts)
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    # Reversal State
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: 30 MIN TRIGGER ---
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 # Enter Pre-Drop Short
                    pre_drop_short_active = True
            
            # --- A. PRE-DROP SHORT LOGIC ---
            elif pre_drop_short_active:
                if position == -1:
                    short_pnl = entry_price - curr_p
                    # 1. Check STOP LOSS
                    if short_pnl <= -pre_drop_short_sl:
                        signal = 1 # Buy to Close (SL Hit)
                        pre_drop_short_active = False
                    # 2. Check Target
                    elif curr_p <= pre_drop_target:
                        signal = 1 # Close Short (Target Hit)
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING LONG FLIP ---
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1 # Enter Long
                    pending_long_flip = False
                    waiting_for_short = False 
            
            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- LONG POSITION LOGIC ---
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -S1_LONG_STATIC_SL:
                        exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= S1_LONG_TP_ACTIVATION:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - S1_LONG_TRAIL_CALLBACK):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1  # Close Long
                        waiting_for_short = True
                        short_target_price = curr_p + S1_REVERSAL_RISE_TRIGGER

                # --- SHORT POSITION LOGIC (REVERSAL SHORTS) ---
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -S1_SHORT_STATIC_SL:
                        signal = 1 # Buy to Close (SL Hit)
                    else:
                        if not short_trailing_active:
                            if pnl >= S1_SHORT_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S1_SHORT_TRAIL_CALLBACK):
                                signal = 1 # Buy to Close (Trailing Hit)

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1  # Enter Short
                            waiting_for_short = False 
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1 # Enter Long
                                waiting_for_short = False 
        
        # 4. Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        positions[i] = position
        
    return signals, positions


# --- Numba Core 2: Strategy 2 (Mu > 100) ---
@njit(cache=True, nogil=True)
def strategy_2_core(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_short,   
    cooldown_seconds,
    entry_rise_threshold,
    pre_rise_sl
):
    """
    Strategy 2 Logic: Pre-Rise Long -> Flip Short -> Reversal Long
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # --- Pre-Rise (Gap Fill) State ---
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0

    # Execution Flags
    short_trade_executed = False 
    
    # Trailing State (Short - Primary)
    short_trailing_active = False
    short_trailing_valley_price = 0.0

    # Trailing State (Long - Reversal)
    long_trailing_active = False
    long_trailing_peak_price = 0.0
    
    # Reversal Trigger State
    waiting_for_long_reversal = False
    long_reversal_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: 30 MIN TRIGGER (GAP FILL LONG) ---
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1 # Enter Pre-Rise Long
                    pre_rise_long_active = True

            # --- A. PRE-RISE LONG LOGIC ---
            elif pre_rise_long_active:
                if position == 1:
                    # SL CHECK
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1 # Close Long (Hit SL)
                        pre_rise_long_active = False
                    # TARGET CHECK
                    elif curr_p >= pre_rise_target:
                        signal = -1 # Close Long (Square off)
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING SHORT FLIP ---
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1 # Enter Main Short
                    pending_short_flip = False
                    short_trade_executed = True
                    waiting_for_long_reversal = False

            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- PRIMARY POSITION (SHORT) ---
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    
                    if pnl <= -S2_PRIMARY_SHORT_SL:
                        exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= S2_PRIMARY_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S2_PRIMARY_TRAIL_CALLBACK):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = 1  # Buy to Close (Short Exit)
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - S2_REVERSAL_DROP_TRIGGER

                # --- SECONDARY POSITION (LONG - REVERSAL) ---
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -S2_SECONDARY_LONG_SL:
                        exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= S2_SECONDARY_TP_ACTIVATION:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - S2_SECONDARY_TRAIL_CALLBACK):
                                exit_signal = True

                    if exit_signal:
                        signal = -1 # Sell to Close

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1  # Enter Long (Reversal)
                            waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1 # Enter Short
                                short_trade_executed = True
                                waiting_for_long_reversal = False 
        
        # 4. Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        positions[i] = position
        
    return signals, positions

# --- Trade Reporting ---

def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        print("Error: File not found:", output_file)
        return

    if "Signal" not in df.columns or "Day" not in df.columns:
        print("Error: Columns missing.")
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        if pd.isna(signal): continue

        if position is None:
            # Entry (1 for Long, -1 for Short)
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            # Exit
            if signal == -position:
                direction = "LONG" if position == 1 else "SHORT"
                pnl = (float(row["Price"]) - float(entry["Price"])) * position
                
                trades.append({
                    "day": entry["Day"],
                    "direction": direction,
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": pnl
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)
    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

# --- Processing ---

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        df = pd.read_parquet(file_path, columns=columns)
        
        if df.empty:
            return None

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)
        df["Day"] = day_num

        # Blacklist Logic
        if day_num in BLACKLIST_DAYS:
            df["Signal"] = 0
            df["Position"] = 0
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day"]
            df[final_columns].to_csv(output_path, index=False)
            return str(output_path)

        # Valid Day Logic
        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        
        if len(price_arr) < 10:
            return None
            
        day_open_price = price_arr[0]
        
        # --- Volatility & Mu Gating ---
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        
        if not np.any(mask_30min):
             # Not enough data
             return None
        else:
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            
            # Common Condition: Sigma must be > 0.11
            is_volatile = sigma > SIGMA_THRESHOLD
            start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')

            signals = np.zeros(len(price_arr), dtype=np.int64)
            positions = np.zeros(len(price_arr), dtype=np.int64)

            if is_volatile:
                # === DECISION LOGIC: MU THRESHOLD ===
                if mu < MU_THRESHOLD_SPLIT:
                    # RUN STRATEGY 1
                    signals, positions = strategy_1_core(
                        time_sec_arr, 
                        price_arr, 
                        day_open_price,
                        int(start_idx),
                        True, # can_trade_long flag from S1
                        COOLDOWN_PERIOD_SECONDS,
                        S1_ENTRY_DROP_THRESHOLD,
                        S1_PRE_DROP_SHORT_SL
                    )
                elif mu > MU_THRESHOLD_SPLIT:
                    # RUN STRATEGY 2
                    signals, positions = strategy_2_core(
                        time_sec_arr, 
                        price_arr, 
                        day_open_price,
                        int(start_idx),
                        True, # can_trade_short flag from S2
                        COOLDOWN_PERIOD_SECONDS,
                        S2_ENTRY_RISE_THRESHOLD,
                        S2_PRE_RISE_SL
                    )
                # If exactly 100 or not volatile, defaults to zeros

        df["Signal"] = signals
        df["Position"] = positions
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day"]
        df[final_columns].to_csv(output_path, index=False)
        
        del df
        gc.collect()
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params):
    if sys.platform != 'win32':
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass

    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files:
        print("No parquet files found.")
        return

    processed_files = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params): f
            for f in files
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    processed_files.append(res)
            except Exception as e:
                print(f"Error in worker: {e}")

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    print(f"Merging {len(processed_sorted)} files...")
    
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline() 
                    shutil.copyfileobj(inp, out)

    print("Generating trade reports...")
    generate_trade_reports_csv(OUTPUT_FILE)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)