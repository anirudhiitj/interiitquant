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
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing_FULL_MERGE"
OUTPUT_FILE = "/data/quant14/signals/trading_signals_MERGED_ALL.csv"
REQUIRED_COLUMNS = ['Time', 'Price']

# Blacklist (Union of both scripts - they appeared identical)
BLACKLIST_DAYS = {87, 311, 273, 446, 5, 358, 14, 331, 136, 389, 504, 323, 96, 205, 230, 477, 456,
 236, 181, 197, 372, 173, 28, 186, 367, 98, 410, 161, 107, 124, 208, 290, 239,
 408, 137, 343, 108, 36, 448, 298, 85, 421,
 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}

# Strategy Parameters Column Mapping
STRATEGY_PARAMS_COLS = {
    "pb9t1": "PB9_T1", 
}

# --- GLOBAL SETTINGS ---
DECISION_WINDOW_SECONDS = 30 * 60
MU_SPLIT_THRESHOLD = 100.0    
COOLDOWN_PERIOD_SECONDS = 5

# ==========================================
#      STRATEGY 1: LOW MU (< 100)
# ==========================================

REVERSAL_RISE_TRIGGER_LOW_MU = 0.2

# Strategy A: High Sigma (> 0.13)
PARAMS_STRATEGY_A = {
    "ENTRY_DROP_THRESHOLD": 0.8,
    "PRE_DROP_SHORT_SL": 0.4,
    "STATIC_SL": 0.7,
    "TP_ACTIVATION": 1.0,
    "TRAIL_CALLBACK": 0.1,
    "SHORT_STATIC_SL": 4.4,
    "SHORT_TP_ACTIVATION": 6.2,
    "SHORT_TRAIL_CALLBACK": 0.04
}

# Strategy B: Mid Sigma (0.07 to 0.1)
PARAMS_STRATEGY_B = {
    "ENTRY_DROP_THRESHOLD": 0.5,
    "PRE_DROP_SHORT_SL": 0.4,
    "STATIC_SL": 0.6,
    "TP_ACTIVATION": 1.2,
    "TRAIL_CALLBACK": 0.1,
    "SHORT_STATIC_SL": 0.1,
    "SHORT_TP_ACTIVATION": 999.0,
    "SHORT_TRAIL_CALLBACK": 0.04
}

# ==========================================
#      STRATEGY 2: HIGH MU (> 100)
# ==========================================
# Prefixed with HM_ to avoid conflicts with Strategy 1 variables

HM_SIGMA_THRESHOLD = 0.12  

# 1. Primary Entry (Short)
HM_ENTRY_RISE_THRESHOLD = 0.9  # Price must RISE 0.8 above Open

# 2. Primary Exit (Short)
HM_PRIMARY_STATIC_SL = 0.6
HM_PRIMARY_TP_ACTIVATION = 1.0
HM_PRIMARY_TRAIL_CALLBACK = 0.1

# 3. Reversal Trigger (Long)
HM_REVERSAL_DROP_TRIGGER = 0.2  # Price must FALL 0.2 below Short exit price

# 4. Reversal Exit (Long)
HM_SECONDARY_STATIC_SL = 0.15
HM_SECONDARY_TP_ACTIVATION = 0.8
HM_SECONDARY_TRAIL_CALLBACK = 0.1

# 5. Pre-Rise Long Stop Loss
HM_PRE_RISE_SL = 0.9


# ==========================================
#      NUMBA CORES
# ==========================================

@njit(cache=True, nogil=True)
def backtest_core_low_mu(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_long,    
    cooldown_seconds,
    # Dynamic Parameters
    entry_drop_threshold,
    pre_drop_short_sl,
    static_sl,
    tp_activation,
    trail_callback,
    reversal_rise_trigger,
    short_static_sl,
    short_tp_activation,
    short_trail_callback
):
    """
    Core logic for Low Mu (< 100) strategies (A & B).
    Logic: Pre-Drop Short -> Long -> Reversal Short
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
        
        # FORCED SQUARE OFF
        if is_last_tick and position != 0:
            signal = -position 
        
        else:
            # --- 30 MIN TRIGGER ---
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 # Enter Pre-Drop Short
                    pre_drop_short_active = True
            
            # --- A. PRE-DROP SHORT LOGIC ---
            elif pre_drop_short_active:
                if position == -1:
                    short_pnl = entry_price - curr_p
                    if short_pnl <= -pre_drop_short_sl:
                        signal = 1 # Buy to Close (SL Hit)
                        pre_drop_short_active = False
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
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -static_sl:
                        exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1  # Close Long
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger

                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl:
                        signal = 1 
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback):
                                signal = 1 

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
        
        # Apply Signal
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


@njit(cache=True, nogil=True)
def backtest_core_high_mu(
    time_sec, price, 
    day_open_price,
    start_index,      
    can_trade_short,   
    cooldown_seconds,
    entry_rise_threshold,
    pre_rise_sl 
):
    """
    Core logic for High Mu (> 100) strategy.
    Logic: Pre-Rise Long -> Short -> Reversal Long
    """
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # Pre-Rise (Gap Fill) State
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
            # --- 30 MIN TRIGGER (GAP FILL LONG) ---
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1 # Enter Pre-Rise Long
                    pre_rise_long_active = True

            # --- A. PRE-RISE LONG LOGIC ---
            elif pre_rise_long_active:
                if position == 1:
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1 # Close Long (Hit SL)
                        pre_rise_long_active = False
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
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    
                    if pnl <= -HM_PRIMARY_STATIC_SL:
                        exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= HM_PRIMARY_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + HM_PRIMARY_TRAIL_CALLBACK):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = 1  # Buy to Close (Short Exit)
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - HM_REVERSAL_DROP_TRIGGER

                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -HM_SECONDARY_STATIC_SL:
                        exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= HM_SECONDARY_TP_ACTIVATION:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - HM_SECONDARY_TRAIL_CALLBACK):
                                exit_signal = True

                    if exit_signal:
                        signal = -1 # Sell to Close (Long Exit)

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
        
        # Apply Signal
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
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports_MERGED.csv")

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
                    "pnl": pnl,
                    "regime": entry["Regime"] if "Regime" in entry else "Unknown"
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

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params_cols) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params_cols.values())
        
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
            df["Regime"] = "Blacklisted"
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Regime"]
            df[final_columns].to_csv(output_path, index=False)
            return str(output_path)

        # Valid Day Logic
        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        
        if len(price_arr) < 10:
            return None
            
        day_open_price = price_arr[0]
        
        # --- Volatility Gating & Regime Detection ---
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        
        # Initialize Variables
        can_trade_low_mu = False
        can_trade_high_mu = False
        active_params_low_mu = None
        
        regime_name = "NoTrade"
        start_idx = 0

        if np.any(mask_30min):
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            
            # --- SPLIT LOGIC BASED ON MU ---
            
            # 1. LOW MU STRATEGY (< 100)
            if mu < MU_SPLIT_THRESHOLD:
                
                # Strategy A (High Sigma)
                if sigma > 0.13:
                    active_params_low_mu = PARAMS_STRATEGY_A
                    can_trade_low_mu = True
                    regime_name = "LowMu_StratA_HighSig"
                
                # Strategy B (Mid Sigma)
                elif 0.07 <= sigma <= 0.1:
                    active_params_low_mu = PARAMS_STRATEGY_B
                    can_trade_low_mu = True
                    regime_name = "LowMu_StratB_MidSig"
                    
            # 2. HIGH MU STRATEGY (> 100)
            else: # mu > 100
                if sigma > HM_SIGMA_THRESHOLD:
                    can_trade_high_mu = True
                    regime_name = "HighMu_Volatile"

            start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')

        # --- EXECUTION ROUTING ---

        if can_trade_low_mu and active_params_low_mu is not None:
            # CALL LOW MU CORE (Strategies A/B)
            signals, positions = backtest_core_low_mu(
                time_sec_arr, 
                price_arr, 
                day_open_price,
                int(start_idx),
                bool(can_trade_low_mu),
                COOLDOWN_PERIOD_SECONDS,
                # Dynamic Parameters
                active_params_low_mu["ENTRY_DROP_THRESHOLD"],
                active_params_low_mu["PRE_DROP_SHORT_SL"],
                active_params_low_mu["STATIC_SL"],
                active_params_low_mu["TP_ACTIVATION"],
                active_params_low_mu["TRAIL_CALLBACK"],
                REVERSAL_RISE_TRIGGER_LOW_MU,
                active_params_low_mu["SHORT_STATIC_SL"],
                active_params_low_mu["SHORT_TP_ACTIVATION"],
                active_params_low_mu["SHORT_TRAIL_CALLBACK"]
            )
            
        elif can_trade_high_mu:
            # CALL HIGH MU CORE
            signals, positions = backtest_core_high_mu(
                time_sec_arr, 
                price_arr, 
                day_open_price,
                int(start_idx),
                bool(can_trade_high_mu),
                COOLDOWN_PERIOD_SECONDS,
                HM_ENTRY_RISE_THRESHOLD,
                HM_PRE_RISE_SL
            )
            
        else:
            # NO TRADE
            signals = np.zeros(len(price_arr), dtype=np.int64)
            positions = np.zeros(len(price_arr), dtype=np.int64)

        df["Signal"] = signals
        df["Position"] = positions
        df["Regime"] = regime_name
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Regime"]
        df[final_columns].to_csv(output_path, index=False)
        
        del df
        gc.collect()
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params_cols):
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
            executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params_cols): f
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

    main(args.directory, args.max_workers, STRATEGY_PARAMS_COLS)



# === Backtest Summary ===
# Initial Capital               : 100.00
# Final Capital                 : 114.80
# Total PnL                     : 14.80
# Total Transaction Cost         : 7.03
# Penalty Counts                : 0
# Final Returns                 : 14.80%
# CAGR                          : 7.0575%
# Annualized Returns            : 7.3127%
# Sharpe Ratio                  : 2.0727
# Calmar Ratio                  : 5.8224
# Maximum Drawdown              : -1.2560%
# No. of Days                   : 510
# Winning Days                  : 58
# Losing Days                   : 49
# Best Day                      : 298
# Worst Day                     : 342
# Best Day PnL                  : 2.94
# Worst Day PnL                 : -0.96
# Total Trades                  : 352
# Winning Trades                : 166
# Losing Trades                 : 186
# Win Rate (%)                  : 47.16%
# Average Winning Trade         : 0.4277
# Average Losing Trade          : -0.2050
# Average Hold Period (seconds) : 9,017.01
# Average Hold Period (minutes) : 150.28
# ========================