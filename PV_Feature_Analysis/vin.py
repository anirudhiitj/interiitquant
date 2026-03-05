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
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing_ALL"
OUTPUT_FILE = "/data/quant14/signals/vin_signals.csv"
REQUIRED_COLUMNS = ['Time', 'Price']
BLACKLIST_DAYS = {104, 165, 110, 115, 6, 209, 135}

# --- GLOBAL CONSTANTS ---
DECISION_WINDOW_SECONDS = 30 * 60
MU_THRESHOLD = 100.0  
COOLDOWN_PERIOD_SECONDS = 5
REVERSAL_RISE_TRIGGER = 0.2

# ==============================================================================
# 1. CONFIGURATION: LOW MEAN (Mu < 100)
# ==============================================================================
LOW_MU_HIGH_VOL_PARAMS = {
    "entry_drop": 0.8, "static_sl": 0.7, "tp_activation": 1.0, "trail_callback": 0.1,
    "short_static_sl": 999.0, "short_tp_activation": 999.0, "short_trail_callback": 0.04,
    "pre_drop_sl": 999.0 
}

LOW_MU_MID_VOL_PARAMS = {
    "entry_drop": 0.9, "static_sl": 999.0, "tp_activation": 1.0, "trail_callback": 0.1,
    "short_static_sl": 0.15, "short_tp_activation": 999.0, "short_trail_callback": 0.04,
    "pre_drop_sl": 0.6    
}

# ==============================================================================
# 2. CONFIGURATION: HIGH MEAN (Mu >= 100)
# ==============================================================================
# High Vol Params (Short First)
S3_ENTRY_RISE_THRESHOLD = 0.8
S3_PRIMARY_STATIC_SL = 0.7
S3_PRIMARY_TP_ACTIVATION = 0.65
S3_PRIMARY_TRAIL_CALLBACK = 0.15
S3_REVERSAL_DROP_TRIGGER = 0.2
S3_SECONDARY_STATIC_SL = 0.15
S3_SECONDARY_TP_ACTIVATION = 0.8
S3_SECONDARY_TRAIL_CALLBACK = 0.1
S3_PRE_RISE_SL = 0.4

# Mid Vol Params (Long First)
S4_ENTRY_DROP_THRESHOLD = 0.9
S4_STATIC_SL = 4.0
S4_TP_ACTIVATION = 1.0
S4_TRAIL_CALLBACK = 0.1
S4_REVERSAL_RISE_TRIGGER = 0.2
S4_SHORT_STATIC_SL = 0.15
S4_SHORT_TP_ACTIVATION = 5.2
S4_SHORT_TRAIL_CALLBACK = 0.04


# ==============================================================================
# BACKTEST FUNCTION 1: Unified Core for Low Mu (Strategies 1 & 2)
# ==============================================================================
@njit(cache=True, nogil=True)
def backtest_core_low_mu(time_sec, price, day_open_price, start_index, can_trade_long, cooldown_seconds, reversal_rise_trigger, entry_drop_threshold, static_sl, tp_activation, trail_callback, short_static_sl, short_tp_activation, short_trail_callback, pre_drop_sl):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    position = 0
    entry_price = 0.0
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    trailing_active = False
    trailing_peak_price = 0.0 
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    waiting_for_short = False
    short_target_price = 0.0
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue
        curr_t = time_sec[i]
        curr_p = price[i]
        signal = 0
        if i == n - 1 and position != 0: signal = -position 
        else:
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1; pre_drop_short_active = True
            elif pre_drop_short_active:
                if position == -1:
                    pnl_short = entry_price - curr_p
                    if pnl_short <= -pre_drop_sl: signal = 1; pre_drop_short_active = False
                    elif curr_p <= pre_drop_target: signal = 1; pre_drop_short_active = False; pending_long_flip = True; flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds: signal = 1; pending_long_flip = False; waiting_for_short = False 
            else:
                if position == 1:
                    pnl = curr_p - entry_price; exit_signal = False
                    if pnl <= -static_sl: exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation: trailing_active = True; trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price: trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback): exit_signal = True
                    if exit_signal: signal = -1; waiting_for_short = True; short_target_price = curr_p + reversal_rise_trigger
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl: signal = 1 
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation: short_trailing_active = True; short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback): signal = 1 
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price: signal = -1; waiting_for_short = False 
                    elif can_trade_long:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p <= pre_drop_target: signal = 1; waiting_for_short = False 
        
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1): signal = 0
            else:
                if position != 0 and signal == -position:
                    entry_price = 0.0; trailing_active = False; trailing_peak_price = 0.0; short_trailing_active = False; short_trailing_valley_price = 0.0
                elif position == 0:
                    entry_price = curr_p; trailing_active = False; trailing_peak_price = 0.0; short_trailing_active = False; short_trailing_valley_price = 0.0
                position += signal; last_signal_time = curr_t
        signals[i] = signal; positions[i] = position
    return signals, positions

# ==============================================================================
# BACKTEST FUNCTION 2: High Mu / High Vol (Strategy 3)
# ==============================================================================
@njit(cache=True, nogil=True)
def backtest_high_mu_high_vol(time_sec, price, day_open_price, start_index, can_trade_short, cooldown_seconds, entry_rise_threshold, pre_rise_sl):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    position = 0; entry_price = 0.0
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False; pending_short_flip = False; flip_wait_start_time = 0
    short_trade_executed = False
    short_trailing_active = False; short_trailing_valley_price = 0.0
    long_trailing_active = False; long_trailing_peak_price = 0.0
    waiting_for_long_reversal = False; long_reversal_target_price = 0.0
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue
        curr_t = time_sec[i]; curr_p = price[i]
        signal = 0
        if i == n - 1 and position != 0: signal = -position 
        else:
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target: signal = 1; pre_rise_long_active = True
            elif pre_rise_long_active:
                if position == 1:
                    if (curr_p - entry_price) <= -pre_rise_sl: signal = -1; pre_rise_long_active = False
                    elif curr_p >= pre_rise_target: signal = -1; pre_rise_long_active = False; pending_short_flip = True; flip_wait_start_time = curr_t
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds: signal = -1; pending_short_flip = False; short_trade_executed = True; waiting_for_long_reversal = False
            else:
                if position == -1:
                    pnl = entry_price - curr_p; exit_signal = False
                    if pnl <= -S3_PRIMARY_STATIC_SL: exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= S3_PRIMARY_TP_ACTIVATION: short_trailing_active = True; short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S3_PRIMARY_TRAIL_CALLBACK): exit_signal = True
                    if exit_signal: signal = 1; waiting_for_long_reversal = True; long_reversal_target_price = curr_p - S3_REVERSAL_DROP_TRIGGER
                elif position == 1:
                    pnl = curr_p - entry_price; exit_signal = False
                    if pnl <= -S3_SECONDARY_STATIC_SL: exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= S3_SECONDARY_TP_ACTIVATION: long_trailing_active = True; long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price: long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - S3_SECONDARY_TRAIL_CALLBACK): exit_signal = True
                    if exit_signal: signal = -1
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price: signal = 1; waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p >= pre_rise_target: signal = -1; short_trade_executed = True; waiting_for_long_reversal = False 
        
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1): signal = 0
            else:
                if position != 0 and signal == -position: entry_price = 0.0; short_trailing_active = False; short_trailing_valley_price = 0.0; long_trailing_active = False; long_trailing_peak_price = 0.0
                elif position == 0: entry_price = curr_p; short_trailing_active = False; short_trailing_valley_price = 0.0; long_trailing_active = False; long_trailing_peak_price = 0.0
                position += signal; last_signal_time = curr_t
        signals[i] = signal; positions[i] = position
    return signals, positions

# ==============================================================================
# BACKTEST FUNCTION 3: High Mu / Mid Vol (Strategy 4)
# ==============================================================================
@njit(cache=True, nogil=True)
def backtest_high_mu_mid_vol(time_sec, price, day_open_price, start_index, can_trade_long, cooldown_seconds, entry_drop_threshold):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    position = 0; entry_price = 0.0
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False; pending_long_flip = False; flip_wait_start_time = 0
    trailing_active = False; trailing_peak_price = 0.0 
    short_trailing_active = False; short_trailing_valley_price = 0.0
    waiting_for_short = False; short_target_price = 0.0
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue
        curr_t = time_sec[i]; curr_p = price[i]
        signal = 0
        if i == n - 1 and position != 0: signal = -position
        else:
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target: signal = -1; pre_drop_short_active = True
            elif pre_drop_short_active:
                if position == -1:
                    if curr_p <= pre_drop_target: signal = 1; pre_drop_short_active = False; pending_long_flip = True; flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds: signal = 1; pending_long_flip = False; waiting_for_short = False
            else:
                if position == 1:
                    pnl = curr_p - entry_price; exit_signal = False
                    if pnl <= -S4_STATIC_SL: exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= S4_TP_ACTIVATION: trailing_active = True; trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price: trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - S4_TRAIL_CALLBACK): exit_signal = True
                    if exit_signal: signal = -1; waiting_for_short = True; short_target_price = curr_p + S4_REVERSAL_RISE_TRIGGER
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -S4_SHORT_STATIC_SL: signal = 1
                    else:
                        if not short_trailing_active:
                            if pnl >= S4_SHORT_TP_ACTIVATION: short_trailing_active = True; short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price: short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + S4_SHORT_TRAIL_CALLBACK): signal = 1
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price: signal = -1; waiting_for_short = False 
                    elif can_trade_long:
                        if (curr_t - last_signal_time) >= cooldown_seconds:
                            if curr_p <= pre_drop_target: signal = 1; waiting_for_short = False 
        
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1): signal = 0
            else:
                if position != 0 and signal == -position: entry_price = 0.0; trailing_active = False; trailing_peak_price = 0.0; short_trailing_active = False; short_trailing_valley_price = 0.0
                elif position == 0: entry_price = curr_p; trailing_active = False; trailing_peak_price = 0.0; short_trailing_active = False; short_trailing_valley_price = 0.0
                position += signal; last_signal_time = curr_t
        signals[i] = signal; positions[i] = position
    return signals, positions

# --- Trade Reporting ---
def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports_combined_all.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    try: df = pd.read_csv(output_file)
    except FileNotFoundError: print("Error: File not found:", output_file); return
    if "Signal" not in df.columns or "Day" not in df.columns: print("Error: Columns missing."); return
    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0: print("No trades found."); return
    trades = []; position = None; entry = None
    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        if pd.isna(signal): continue
        if position is None:
            if signal in [1, -1]: position = signal; entry = row
        else:
            if signal == -position:
                direction = "LONG" if position == 1 else "SHORT"
                pnl = (float(row["Price"]) - float(entry["Price"])) * position
                trades.append({
                    "day": entry["Day"], "strategy_type": row.get("Strat_Type", "Unknown"), "direction": direction,
                    "entry_time": entry["Time"], "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"], "exit_price": float(row["Price"]), "pnl": pnl
                })
                position = None; entry = None
    if len(trades) == 0: print("No completed trades."); return
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)
    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

# --- Processing (THE REQUESTED LOGIC BLOCK) ---
def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params_cols) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params_cols.values())
        df = pd.read_parquet(file_path, columns=columns)
        if df.empty: return None

        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)
        df["Day"] = day_num

        if day_num in BLACKLIST_DAYS:
            df["Signal"] = 0; df["Position"] = 0; df["Strat_Type"] = 0
            output_path = temp_dir / f"day{day_num}.csv"
            df[REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strat_Type"]].to_csv(output_path, index=False)
            return str(output_path)

        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        if len(price_arr) < 10: return None
        day_open_price = price_arr[0]
        
        # 1. COMPUTE FIRST 30 MIN STATS
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        signals = np.zeros(len(price_arr), dtype=np.int64)
        positions = np.zeros(len(price_arr), dtype=np.int64)
        strat_type = 0

        if np.any(mask_30min):
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            start_idx = int(np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right'))

            # 2. LOGIC BLOCK TO DECIDE STRATEGY
            if mu < MU_THRESHOLD:
                # --- LOW MEAN STRATEGIES ---
                if sigma > 0.12:
                    # Strategy 1: Low Mu / High Vol
                    strat_type = 1
                    signals, positions = backtest_core_low_mu(
                        time_sec_arr, price_arr, day_open_price, start_idx, True, COOLDOWN_PERIOD_SECONDS, REVERSAL_RISE_TRIGGER,
                        float(LOW_MU_HIGH_VOL_PARAMS["entry_drop"]), float(LOW_MU_HIGH_VOL_PARAMS["static_sl"]),
                        float(LOW_MU_HIGH_VOL_PARAMS["tp_activation"]), float(LOW_MU_HIGH_VOL_PARAMS["trail_callback"]),
                        float(LOW_MU_HIGH_VOL_PARAMS["short_static_sl"]), float(LOW_MU_HIGH_VOL_PARAMS["short_tp_activation"]),
                        float(LOW_MU_HIGH_VOL_PARAMS["short_trail_callback"]), float(LOW_MU_HIGH_VOL_PARAMS["pre_drop_sl"])
                    )
                elif 0.09 <= sigma <= 0.12:
                    # Strategy 2: Low Mu / Mid Vol
                    strat_type = 2
                    signals, positions = backtest_core_low_mu(
                        time_sec_arr, price_arr, day_open_price, start_idx, True, COOLDOWN_PERIOD_SECONDS, REVERSAL_RISE_TRIGGER,
                        float(LOW_MU_MID_VOL_PARAMS["entry_drop"]), float(LOW_MU_MID_VOL_PARAMS["static_sl"]),
                        float(LOW_MU_MID_VOL_PARAMS["tp_activation"]), float(LOW_MU_MID_VOL_PARAMS["trail_callback"]),
                        float(LOW_MU_MID_VOL_PARAMS["short_static_sl"]), float(LOW_MU_MID_VOL_PARAMS["short_tp_activation"]),
                        float(LOW_MU_MID_VOL_PARAMS["short_trail_callback"]), float(LOW_MU_MID_VOL_PARAMS["pre_drop_sl"])
                    )
            
            elif mu >= MU_THRESHOLD:
                # --- HIGH MEAN STRATEGIES ---
                if sigma > 0.12:
                    # Strategy 3: High Mu / High Vol
                    strat_type = 3
                    signals, positions = backtest_high_mu_high_vol(
                        time_sec_arr, price_arr, day_open_price, start_idx, True,
                        COOLDOWN_PERIOD_SECONDS, S3_ENTRY_RISE_THRESHOLD, S3_PRE_RISE_SL
                    )
                elif 0.09 <= sigma <= 0.12:
                    # Strategy 4: High Mu / Mid Vol
                    strat_type = 4
                    signals, positions = backtest_high_mu_mid_vol(
                        time_sec_arr, price_arr, day_open_price, start_idx, True,
                        COOLDOWN_PERIOD_SECONDS, S4_ENTRY_DROP_THRESHOLD
                    )

        df["Signal"] = signals
        df["Position"] = positions
        df["Strat_Type"] = strat_type
        
        output_path = temp_dir / f"day{day_num}.csv"
        df[REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Strat_Type"]].to_csv(output_path, index=False)
        del df; gc.collect()
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params_cols):
    if sys.platform != 'win32':
        try: mp.set_start_method('fork', force=True)
        except RuntimeError: pass

    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files: print("No parquet files found."); return

    processed_files = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params_cols): f for f in files}
        for fut in as_completed(futures):
            res = fut.result()
            if res: processed_files.append(res)

    if not processed_files: print("No files processed."); return
    processed_sorted = sorted(processed_files, key=extract_day_num)
    print(f"Merging {len(processed_sorted)} files...")
    
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0: shutil.copyfileobj(inp, out)
                else: inp.readline(); shutil.copyfileobj(inp, out)

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
    main(args.directory, args.max_workers, {"pb9t1": "PB9_T1"})