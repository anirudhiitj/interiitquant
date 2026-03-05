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
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/VB2_temp_signal_processing"
OUTPUT_FILE = "/data/quant14/signals/temp_trading_signals_EBX.csv"
COOLDOWN_PERIOD_SECONDS = 5
MIN_TRADE_DURATION_SECONDS = 5
REQUIRED_COLUMNS = ['Time', 'Price']

STRATEGY_PARAMS = {
    "pb9t1": "PB9_T1",
}

# ==========================================
# SECTION 1: NUMBA OPTIMIZED CALCULATIONS
# ==========================================

@njit
def numba_super_smoother(prices, period):
    """
    Calculates Ehlers Super Smoother Filter using Numba for speed.
    """
    n = len(prices)
    out = np.zeros(n)
    
    # Constants
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # Initialization
    out[0] = prices[0]
    if n > 1:
        out[1] = prices[1]
        
    for i in range(2, n):
        out[i] = c1 * prices[i] + c2 * out[i-1] + c3 * out[i-2]
        
    return out

@njit
def numba_kama_array(price, window, fast, slow):
    """
    Calculates KAMA array using Numba.
    """
    n = len(price)
    kama = np.full(n, np.nan)
    
    # Constants
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    
    # Find first valid index
    start_idx = 0
    for i in range(n):
        if not np.isnan(price[i]):
            start_idx = i
            break
            
    # Initialize
    kama[start_idx] = price[start_idx]
    
    # We need a rolling sum for volatility. 
    # To keep it purely iterative in one pass is complex, 
    # so we'll calculate ER iteratively.
    
    for i in range(start_idx + 1, n):
        if i < window:
            kama[i] = kama[i-1] # Not enough data yet
            continue
            
        # Calculate ER components over 'window'
        change = abs(price[i] - price[i - window])
        volatility = 0.0
        for j in range(window):
            volatility += abs(price[i-j] - price[i-j-1])
            
        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility
            
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        
        # Recursive filter
        kama[i] = kama[i-1] + sc * (price[i] - kama[i-1])
        
    return kama

@njit
def numba_eama_calc(ss_values, period, fast, slow):
    """
    Calculates EAMA based on SuperSmoother values.
    """
    n = len(ss_values)
    eama = np.zeros(n)
    
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    
    # Initialize
    eama[0] = ss_values[0]
    
    for i in range(1, n):
        if i < period:
            eama[i] = eama[i-1] # Not enough data
            continue
            
        # ER Calculation on SuperSmoother
        change = abs(ss_values[i] - ss_values[i - period])
        volatility = 0.0
        for j in range(period):
            volatility += abs(ss_values[i-j] - ss_values[i-j-1])
            
        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility
            
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        
        # Filter
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
        
    return eama

# ==========================================
# SECTION 2: FEATURE ENGINEERING
# ==========================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    if strategy_params["pb9t1"] not in df.columns:
        raise KeyError(f"Required feature {strategy_params['pb9t1']} not found.")

    # Convert to numpy for Numba
    price_arr = df[strategy_params["pb9t1"]].to_numpy(dtype=float)
    clean_price_arr = np.nan_to_num(price_arr, nan=price_arr[0] if len(price_arr) > 0 else 0)

    # 1. KAMA (Existing Logic, Numba Optimized)
    df["KAMA"] = numba_kama_array(clean_price_arr, window=30, fast=60, slow=300)
    df["KAMA_Slope"] = df["KAMA"].diff(2).shift(1)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()

    # 2. Volatility (ATR & STD)
    tr = df[strategy_params["pb9t1"]].diff().abs()
    atr = tr.ewm(span=30, adjust=False).mean()
    df["ATR"] = atr.shift(1)
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()

    std = df[strategy_params["pb9t1"]].rolling(window=120, min_periods=1).std()
    df["STD"] = std.shift(1)
    df["STD_High"] = df["STD"].expanding(min_periods=1).max()

    # 3. NEW TREND INDICATORS
    # A. EMA Fast (Span 20)
    df["EMA_Fast"] = df["Price"].ewm(span=20, adjust=False).mean()
    
    # B. EMA Slow (Span 130)
    df["EMA_Slow"] = df["Price"].ewm(span=130, adjust=False).mean()
    
    # C. SMA 240 (Lagged by 1)
    df["SMA_240"] = df["Price"].rolling(240).mean().shift(1)
    
    # D. EAMA (Ehlers Adaptive Moving Average)
    # Step D1: Super Smoother (Period 30)
    ss_values = numba_super_smoother(clean_price_arr, period=30)
    # Step D2: EAMA (Period=15, Fast=30, Slow=120)
    df["EAMA"] = numba_eama_calc(ss_values, period=15, fast=30, slow=120)

    return df

# ==========================================
# SECTION 3: TRADING LOGIC
# ==========================================

def generate_signal(row, position, cooldown_over, strategy_params):
    signal = 0
    # Parameters
    ATR_THRESHOLD = 0.0
    STD_THRESHOLD = 0.08
    KAMA_SLOPE_ENTRY = 0.000

    if cooldown_over:
        volatility_confirmed = (row["ATR_High"] > ATR_THRESHOLD and row["STD_High"] > STD_THRESHOLD)
        
        if position == 0 and volatility_confirmed:
            # --- UPDATED TREND LOGIC ---
            # Long: EMAfast > EMAslow > SMA240 > EAMA
            trend_long = (
                row["EMA_Fast"] > row["EMA_Slow"] and 
                row["EMA_Slow"] > row["SMA_240"] and 
                row["SMA_240"] > row["EAMA"]
            )
            
            # Short: EMAfast < EMAslow < SMA240 < EAMA
            trend_short = (
                row["EMA_Fast"] < row["EMA_Slow"] and 
                row["EMA_Slow"] < row["SMA_240"] and 
                row["SMA_240"] < row["EAMA"]
            )

            # Combine with KAMA Slope confirmation
            if trend_long and row["KAMA_Slope"] > KAMA_SLOPE_ENTRY:
                signal = 1
            elif trend_short and row["KAMA_Slope"] < -KAMA_SLOPE_ENTRY:
                signal = -1

    return signal

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params.values())
        
        # SWITCHED TO PANDAS (CPU)
        df = pd.read_parquet(file_path, columns=columns)
        
        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)

        df = prepare_derived_features(df, strategy_params)

        # Trading Loop
        position = 0
        entry_price = None
        entry_time_sec = None
        trailing_active = False
        trailing_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals = np.zeros(len(df), dtype=int)
        positions = np.zeros(len(df), dtype=int)

        TAKE_PROFIT = 0.35
        TRAIL_STOP = 0.15

        time_sec_arr = df["Time_sec"].values
        price_arr = df["Price"].values
        
        # Feature dict for fast access
        features = {
            "ATR_High": df["ATR_High"].values,
            "STD_High": df["STD_High"].values,
            "EMA_Fast": df["EMA_Fast"].values,
            "EMA_Slow": df["EMA_Slow"].values,
            "SMA_240": df["SMA_240"].values,
            "EAMA": df["EAMA"].values,
            "KAMA_Slope": df["KAMA_Slope"].values
        }

        len_df = len(df)
        
        for i in range(len_df):
            current_time = time_sec_arr[i]
            price = price_arr[i]
            is_last_tick = (i == len_df - 1)
            signal = 0

            # Exit on last tick
            if is_last_tick and position != 0:
                signal = -position
            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                
                desired_signal = 0
                if cooldown_over:
                    # Volatility check
                    if features["ATR_High"][i] > 0.0 and features["STD_High"][i] > 0.08:
                        if position == 0:
                            # Trend Logic
                            ema_f = features["EMA_Fast"][i]
                            ema_s = features["EMA_Slow"][i]
                            sma_240 = features["SMA_240"][i]
                            eama = features["EAMA"][i]
                            kama_slope = features["KAMA_Slope"][i]

                            # Long
                            if (ema_f > ema_s > sma_240 > eama) and (kama_slope > 0.0008):
                                desired_signal = 1
                            # Short
                            elif (ema_f < ema_s < sma_240 < eama) and (kama_slope < -0.0008):
                                desired_signal = -1

                # Signal Processing Logic
                if position == 0 and desired_signal != 0:
                    signal = desired_signal
                elif position != 0:
                    # TP / Trailing Stop Logic
                    price_diff = price - entry_price if position == 1 else entry_price - price

                    if not trailing_active and price_diff >= TAKE_PROFIT:
                        trailing_active = True
                        trailing_price = price

                    if trailing_active:
                        if position == 1:
                            if price > trailing_price:
                                trailing_price = price
                            elif price <= trailing_price - TRAIL_STOP:
                                signal = -1
                                trailing_active = False
                        elif position == -1:
                            if price < trailing_price:
                                trailing_price = price
                            elif price >= trailing_price + TRAIL_STOP:
                                signal = 1
                                trailing_active = False
                    
                    if signal == 0 and desired_signal != 0 and desired_signal == -position:
                        signal = -position

                # Min Trade Duration
                if signal != 0 and position != 0:
                    time_in_trade = current_time - (entry_time_sec if entry_time_sec is not None else -999999)
                    if time_in_trade < MIN_TRADE_DURATION_SECONDS:
                        signal = 0

            # State Update
            if signal != 0:
                if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                    signal = 0
                else:
                    if position == 0 and signal != 0:
                        entry_price = price
                        entry_time_sec = int(current_time)
                        trailing_active = False
                        trailing_price = None

                    if position != 0 and signal == -position:
                        entry_price = None
                        entry_time_sec = None
                        trailing_price = None
                        trailing_active = False

                    position += signal
                    last_signal_time = int(current_time)

                    if position != 0 and entry_time_sec is None:
                        entry_time_sec = int(current_time)
                        entry_price = price

                    if position == 0:
                        entry_time_sec = None
                        entry_price = None
                        trailing_price = None
                        trailing_active = False

            signals[i] = signal
            positions[i] = position

        df["Signal"] = signals
        df["Position"] = positions
        df["Day"] = day_num  # ADDED: Essential for report generation

        output_path = temp_dir / f"day{day_num}.csv"
        # ADDED "Day" to final_columns
        final_columns = ["Day"] + REQUIRED_COLUMNS + list(strategy_params.values()) + [
            "Signal", "Position", "KAMA", "KAMA_Slope", "ATR", "STD",
            "EMA_Fast", "EMA_Slow", "SMA_240", "EAMA"
        ]

        df[final_columns].to_csv(output_path, index=False)
        print(f"✅ Processed Day {day_num}")
        
        del df
        gc.collect()
        
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# SECTION 4: REPORT GENERATION
# ==========================================

def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)

    signals_file = output_file

    try:
        df = pd.read_csv(signals_file)
    except FileNotFoundError:
        print("Error: Combined signals file not found:", signals_file)
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found in final signal file.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        
        if pd.isna(signal):
            continue

        if position is None:
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            if signal == -position:
                trades.append({
                    "day": entry["Day"],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": (float(row["Price"]) - float(entry["Price"])) * position
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades found.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(SAVE_PATH, index=False)

    print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")


def main(directory: str, max_workers: int, strategy_params):
    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    print(f"Created temp directory: {temp_dir_path}")

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    if not files:
        print("No parquet files found.")
        return

    processed_files = []
    # CPU bound, so multiprocessing is good. Numba releases GIL often, but separate processes are safer for stability.
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
                print(f"Error: {e}")

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    
    print("Merging files...")
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline()
                    shutil.copyfileobj(inp, out)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    
    # GENERATE REPORTS
    print("Generating trade reports...")
    generate_trade_reports_csv(OUTPUT_FILE)
    
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS)