import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pykalman import KalmanFilter

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

# ==========================================
# CONFIGURATION (STRATEGY 1 ONLY)
# ==========================================

EBY_Config = {
    # Data Settings
    'DATA_DIR': '/data/quant14/EBY/', 
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'EBY_P1_Kalman_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/EBY_temp_signals_p1',
    
    # Execution Settings
    'MAX_WORKERS': 32,
    'UNIVERSAL_COOLDOWN': 5.0,  # seconds between signals (used as min hold here)
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy 1 (Kalman Filter) Parameters
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_KF_TRANSITION_COV': 0.24,
    'P1_KF_OBSERVATION_COV': 0.7,
    'P1_KF_INITIAL_STATE_COV': 100,
    'P1_TAKE_PROFIT': 0.4,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_UCM_ALPHA': 0.1,
    'P1_UCM_BETA': 0.02,
    'P1_ENTRY_KAMA_SLOPE_THRESHOLD': 0.001,
    'P1_EXIT_KAMA_SLOPE_THRESHOLD': 0.0005,
    'P1_MIN_HOLD_SECONDS': 15.0,
}

# ==========================================
# STRATEGY 1 CORE FUNCTIONS
# ==========================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates Kalman Filter, KAMA, and UCM features required for Strategy 1.
    """
    # 1. Kalman Filter Trend
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_TRANSITION_COV'],
        observation_covariance=config['P1_KF_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_INITIAL_STATE_COV']
    )
    
    # Ensure no NaNs before passing to Kalman Filter
    feature_series = df[config['P1_FEATURE_COLUMN']].dropna()
    if feature_series.empty:
        return df # Return empty if no valid data
        
    filtered_state_means, _ = kf.filter(feature_series.values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=feature_series.index)
    
    df["KF_Trend"] = smoothed_prices.shift(1)
    df["KF_Trend_Lagged"] = df["KF_Trend"].shift(1)
    
    df["Take_Profit"] = config['P1_TAKE_PROFIT']
    
    # 2. KAMA (Kaufman Adaptive Moving Average)
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    
    # Calculate ER (Efficiency Ratio)
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    # Calculate SC (Smoothing Constant)
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    # Calculate KAMA recursively
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    
    if len(valid_start) > 0:
        start = valid_start[0]
        kama[start] = price[start]
        
        for i in range(start + 1, len(price)):
            if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
    
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # 3. UCM (Unobserved Components Model)
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    
    if first_valid_idx is None:
        raise ValueError(f"{config['P1_FEATURE_COLUMN']} has no valid values")
    
    alpha = config['P1_UCM_ALPHA']
    beta = config['P1_UCM_BETA']
    n = len(series)
    
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    # Initialize UCM
    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
    
    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue
        
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]
        
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]
    
    df["UCM"] = filtered
    df["UCM_Lagged"] = df["UCM"].shift(1)
    
    return df


def generate_p1_signal_single(row, position, cooldown_over, config):
    """
    Determines entry/exit signal for a single row based on KF, UCM, and KAMA slope.
    """
    signal = 0
    
    if cooldown_over:
        # ENTRY LOGIC
        if position == 0:
            # Long Entry: Trend > UCM and Volatility/Slope confirms
            if row["KF_Trend"] <= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = 1
            # Short Entry: Trend < UCM and Volatility/Slope confirms
            elif row["KF_Trend"] >= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = -1
        
        # EXIT LOGIC
        # elif position == 1:
        #     # Exit Long: Trend crosses below UCM with sufficient slope
        #     if (row["KF_Trend_Lagged"] < row["UCM_Lagged"] and 
        #         row["KF_Trend"] > row["UCM"] and 
        #         row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
        #         signal = -1
                
        # elif position == -1:
        #     # Exit Short: Trend crosses above UCM with sufficient slope
        #     if (row["KF_Trend_Lagged"] > row["UCM_Lagged"] and 
        #         row["KF_Trend"] < row["UCM"] and 
        #         row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
        #         signal = 1
    
    # Return signal (-1 to close long, 1 to close short, or new entry direction)
    return signal

def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Iterates through the dataframe to generate signals with state management.
    """
    try:
        df = prepare_p1_features(df, config)
    except Exception as e:
        print(f"Error generating features: {e}")
        return np.zeros(len(df), dtype=np.int8)
    
    # Initialize state
    position = 0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = np.zeros(len(df), dtype=np.int8)
    
    # Process row by row
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)
        
        signal = 0
        
        # EOD Square Off
        if is_last_tick and position != 0:
            signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P1_MIN_HOLD_SECONDS']
            
            # Get raw logic signal
            # Note: The original helper returned specific logic, we implement the switch here
            
            # Check logic
            logic_signal = 0
            if cooldown_over:
                if position == 0:
                    if row["KF_Trend"] <= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                        logic_signal = 1
                    elif row["KF_Trend"] >= row["UCM"] and row["KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                        logic_signal = -1
                # elif position == 1:
                #     if (row["KF_Trend_Lagged"] < row["UCM_Lagged"] and 
                #         row["KF_Trend"] > row["UCM"] and 
                #         row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                #         logic_signal = -1
                # elif position == -1:
                #     if (row["KF_Trend_Lagged"] > row["UCM_Lagged"] and 
                #         row["KF_Trend"] < row["UCM"] and 
                #         row["KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                #         logic_signal = 1

            signal = logic_signal

            # Entry Logic
            if position == 0 and signal != 0:
                entry_price = price
                take_profit = row["Take_Profit"]
            
            # Exit Logic (Price-based TP confirmation overrides technicals if hit)
            elif cooldown_over and position != 0:
                tp_hit = False
                if position == 1:
                    price_diff = price - entry_price
                    if price_diff >= take_profit:
                        signal = -1
                        tp_hit = True
                elif position == -1:
                    price_diff = entry_price - price
                    if price_diff >= take_profit:
                        signal = 1
                        tp_hit = True
                
                # If TP not hit, use the logic signal calculated above
                if not tp_hit and logic_signal != 0:
                     signal = logic_signal

        # Apply Signal to State
        if signal != 0:
            # Prevent redundant signals
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0 
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:  # position closed
                    entry_price = None
        
        signals[i] = signal
    
    return signals

# ==========================================
# FILE PROCESSING
# ==========================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    try:
        # 1. READ DATA
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['P1_FEATURE_COLUMN']
        ]
        
        # Use dask_cudf if available, otherwise pandas could be used here
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        # Calculate time in seconds
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # 2. GENERATE STRATEGY 1 SIGNALS
        p1_signals = generate_p1_signals(df.copy(), config)
        
        # 3. STATISTICS
        entries = np.sum(np.abs(p1_signals) == 1)
        
        # 4. SAVE RESULT
        # === FIX: Added 'Day': day_num to the DataFrame ===
        output_df = pd.DataFrame({
            'Day': day_num, 
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': p1_signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'entries': entries
        }
    
    except Exception as e:
        print(f"Error processing day {day_num} ({file_path}): {e}")
        return None


def generate_trade_reports_csv(output_file):
    """
    Generate a CSV of completed trades using the combined signals file.
    Adds a 'Day' column automatically.
    Saves to: /home/raid/Quant14/VB_Feature_Analysis/Histogram/trade_reports.csv
    """
    import pandas as pd
    import os

    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports.csv")

    # Ensure folder exists
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

        if position is None:  # expecting entry
            if signal in [1, -1]:
                position = signal
                entry = row
        else:                 # expecting exit
            if signal == -position:
                trades.append({
                    "day": entry["Day"],   # include day column
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

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # Argparse Setup
    parser = argparse.ArgumentParser(description="EBY Priority 1 (Kalman Filter) Strategy")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=EBY_Config['DATA_DIR'],
        help=f"Directory containing the 'day*.parquet' files. Default: {EBY_Config['DATA_DIR']}"
    )
    args = parser.parse_args()
    
    # Update Config
    EBY_Config['DATA_DIR'] = args.data_dir
    
    start_time = time.time()
    
    print("="*80)
    print("EBY STRATEGY 1 - KALMAN FILTER EXECUTION")
    print("="*80)
    print(f"\nProcessing data from: {EBY_Config['DATA_DIR']}")
    print(f"Feature Column: {EBY_Config['P1_FEATURE_COLUMN']}")
    print(f"Parameters:")
    print(f"  - KF Transition Cov: {EBY_Config['P1_KF_TRANSITION_COV']}")
    print(f"  - KF Obs Cov: {EBY_Config['P1_KF_OBSERVATION_COV']}")
    print(f"  - Entry Slope Threshold: {EBY_Config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Exit Slope Threshold: {EBY_Config['P1_EXIT_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Take Profit: {EBY_Config['P1_TAKE_PROFIT']}")
    print("="*80)
    
    # Setup temp directory
    temp_dir_path = pathlib.Path(EBY_Config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    # Find input files
    files_pattern = os.path.join(EBY_Config['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No Parquet files found in {EBY_Config['DATA_DIR']}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    print(f"Processing with {EBY_Config['MAX_WORKERS']} workers...")
    
    processed_files = []
    total_entries = 0
    
    # Parallel Processing
    try:
        with ProcessPoolExecutor(max_workers=EBY_Config['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, EBY_Config): f
                for f in sorted_files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result['path'])
                        total_entries += result['entries']
                        print(f"✓ Processed day{result['day_num']:3d} | Entries: {result['entries']:3d}")
                except Exception as e:
                    print(f"✗ Error in {file_path}: {e}")
                    
        # Combine Results
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
            
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        
        output_path = pathlib.Path(EBY_Config['OUTPUT_DIR']) / EBY_Config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()  # Skip header
                        shutil.copyfileobj(infile, outfile)
                        
        print(f"\n✓ Created combined output: {output_path}")
        
        # Final Statistics
        final_df = pd.read_csv(output_path)
        num_long = (final_df['Signal'] == 1).sum()
        num_short = (final_df['Signal'] == -1).sum()

        print("\nGenerating trade_reports.csv ...")
        generate_trade_reports_csv(output_path)

        
        print(f"\n{'='*80}")
        print("STATISTICS (KALMAN FILTER STRATEGY)")
        print(f"{'='*80}")
        print(f"Total Entries:      {total_entries:,}")
        print(f"Long Trades:        {num_long:,}")
        print(f"Short Trades:       {num_short:,}")
        print(f"{'='*80}")
        
    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print(f"Cleaned up temp directory")

    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")

if __name__ == "__main__":
    main()