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

# CONFIGURATION

EBX_Config = {
    # Data
    'DATA_DIR': './EBX/', # This is now the default
    'NUM_DAYS': 510,
    'OUTPUT_DIR': './',
    'OUTPUT_FILE': 'EBX_signals.csv',
    'TEMP_DIR': './EBX_temp_signals',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 15.0,
    'MAX_WORKERS': 64,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    'FEATURE_COLUMN': 'PB9_T1',
    
    # Strategy 1 (Kalman Filter) - Priority 1
    'P1_KF_TRANSITION_COV': 0.24,
    'P1_KF_OBSERVATION_COV': 0.7,
    'P1_KF_INITIAL_STATE_COV': 100,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_UCM_ALPHA': 0.8,
    'P1_UCM_BETA': 0.08,
    'P1_ENTRY_KAMA_SLOPE_THRESHOLD': 0.001,
    'P1_EXIT_KAMA_SLOPE_THRESHOLD': 0.0005,
    'P1_TAKE_PROFIT': 0.3,
    'P1_MIN_HOLD_SECONDS': 15.0,
    
    # Strategy 2 (KAMA + Volatility with Trailing Stop) - Priority 2
    'P2_KAMA_WINDOW': 30,
    'P2_KAMA_FAST_PERIOD': 60,
    'P2_KAMA_SLOW_PERIOD': 300,
    'P2_KAMA_SLOPE_DIFF': 2,
    'P2_ATR_SPAN': 30,
    'P2_ATR_THRESHOLD': 0.01,
    'P2_STD_PERIOD': 20,
    'P2_STD_THRESHOLD': 0.06,
    'P2_KAMA_SLOPE_ENTRY': 0.0008,
    'P2_TAKE_PROFIT': 0.35,
    'P2_TRAIL_STOP': 0.1,
    'P2_MIN_HOLD_SECONDS': 15.0,
}

# STRATEGY 1 (KALMAN FILTER) - PRIORITY 1 FUNCTIONS

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    
    # KF_Trend
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_TRANSITION_COV'],
        observation_covariance=config['P1_KF_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf.filter(df[config['FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['FEATURE_COLUMN']].dropna().index)
    df["P1_KF_Trend"] = smoothed_prices.shift(1)
    df["P1_KF_Trend_Lagged"] = df["P1_KF_Trend"].shift(1)
    
    # KAMA
    price = df[config['FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]
    
    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
    
    df["P1_KAMA"] = kama
    df["P1_KAMA_Slope"] = df["P1_KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["P1_KAMA_Slope_abs"] = df["P1_KAMA_Slope"].abs()
    
    # UCM (Unobserved Components Model)
    series = df[config['FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError(f"{config['FEATURE_COLUMN']} has no valid values")
    
    alpha = config['P1_UCM_ALPHA']
    beta = config['P1_UCM_BETA']
    n = len(series)
    
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
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
    
    df["P1_UCM"] = filtered
    df["P1_UCM_Lagged"] = df["P1_UCM"].shift(1)
    
    return df


def generate_p1_signal_single(row, position, cooldown_over, config):
    signal = 0
    
    if cooldown_over:
        if position == 0:
            if row["P1_KF_Trend"] >= row["P1_UCM"] and row["P1_KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = 1
            elif row["P1_KF_Trend"] <= row["P1_UCM"] and row["P1_KAMA_Slope_abs"] > config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']:
                signal = -1
        elif position == 1:
            if (row["P1_KF_Trend_Lagged"] > row["P1_UCM_Lagged"] and 
                row["P1_KF_Trend"] < row["P1_UCM"] and 
                row["P1_KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                signal = -1
        elif position == -1:
            if (row["P1_KF_Trend_Lagged"] < row["P1_UCM_Lagged"] and 
                row["P1_KF_Trend"] > row["P1_UCM"] and 
                row["P1_KAMA_Slope_abs"] > config['P1_EXIT_KAMA_SLOPE_THRESHOLD']):
                signal = 1
    
    return -1 * signal


def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # Prepare features
    df = prepare_p1_features(df, config)
    
    # Initialize state
    position = 0
    entry_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = [0] * len(df)
    
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
            signal = generate_p1_signal_single(row, position, cooldown_over, config)
            
            # Entry Logic
            if position == 0 and signal != 0:
                entry_price = price
            
            # Exit Logic (Price-based confirmation)
            elif cooldown_over and position != 0:
                if position == 1:
                    price_diff = price - entry_price
                    if price_diff >= config['P1_TAKE_PROFIT']:
                        signal = -1
                elif position == -1:
                    price_diff = entry_price - price
                    if price_diff >= config['P1_TAKE_PROFIT']:
                        signal = 1
        
        # Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0  # redundant signal
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:  # position closed
                    entry_price = None
        
        signals[i] = signal
    
    return np.array(signals, dtype=np.int8), df

# STRATEGY 2 (KAMA + VOLATILITY WITH TRAILING STOP) - PRIORITY 2 FUNCTION

def prepare_p2_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    
    # KAMA calculation
    price = df[config['FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P2_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (config['P2_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P2_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["P2_KAMA"] = kama
    df["P2_KAMA_Slope"] = df["P2_KAMA"].diff(config['P2_KAMA_SLOPE_DIFF']).shift(1)
    df["P2_KAMA_Slope_abs"] = df["P2_KAMA_Slope"].abs()

    # ATR calculation
    tr = df[config['FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P2_ATR_SPAN'], adjust=False).mean()
    df["P2_ATR"] = atr.shift(1)
    df["P2_ATR_High"] = df["P2_ATR"].expanding(min_periods=1).max()

    # Standard Deviation
    std = df[config['FEATURE_COLUMN']].rolling(window=config['P2_STD_PERIOD'], min_periods=1).std()
    df["P2_STD"] = std.shift(1)
    df["P2_STD_High"] = df["P2_STD"].expanding(min_periods=1).max()
    
    return df


def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    # Prepare features
    df = prepare_p2_features(df, config)
    
    # Initialize state
    position = 0
    entry_price = None
    trailing_active = False
    trailing_price = None
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    signals = [0] * len(df)
    
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
            cooldown_over = (current_time - last_signal_time) >= config['P2_MIN_HOLD_SECONDS']
            
            # Entry conditions
            if position == 0 and cooldown_over and not is_last_tick:
                volatility_confirmed = (row["P2_ATR_High"] > config['P2_ATR_THRESHOLD'] and 
                                        row["P2_STD_High"] > config['P2_STD_THRESHOLD'])
                
                if volatility_confirmed:
                    if row["P2_KAMA_Slope"] > config['P2_KAMA_SLOPE_ENTRY']:
                        signal = 1
                    elif row["P2_KAMA_Slope"] < -config['P2_KAMA_SLOPE_ENTRY']:
                        signal = -1
            
            # Exit conditions with trailing stop
            elif cooldown_over and position != 0:
                price_diff = price - entry_price if position == 1 else entry_price - price
                
                # Activate trailing stop after TP reached
                if not trailing_active and price_diff >= config['P2_TAKE_PROFIT']:
                    trailing_active = True
                    trailing_price = price
                
                # Manage trailing stop
                if trailing_active:
                    if position == 1:
                        if price > trailing_price:
                            trailing_price = price
                        elif price <= trailing_price - config['P2_TRAIL_STOP']:
                            signal = -1
                    elif position == -1:
                        if price < trailing_price:
                            trailing_price = price
                        elif price >= trailing_price + config['P2_TRAIL_STOP']:
                            signal = 1
        
        # Apply Signal
        if signal != 0:
            if position == 0 and signal != 0:
                entry_price = price
                trailing_active = False
                trailing_price = None
            
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                position += signal
                last_signal_time = current_time
                if position == 0:
                    entry_price = None
                    trailing_active = False
                    trailing_price = None
        
        signals[i] = signal
    
    return np.array(signals, dtype=np.int8), df

# MANAGER LOGIC (2 PRIORITIES)

def apply_manager_logic(df: pd.DataFrame, p1_signals: np.ndarray, p2_signals: np.ndarray, 
                        config: dict) -> np.ndarray:
    n = len(df)
    final_signals = np.zeros(n, dtype=np.int8)
    
    # Manager state
    position = 0
    position_owner = None  # None, "P1", "P2"
    last_signal_time = -config['UNIVERSAL_COOLDOWN']
    priority_2_locked = False
    
    for i in range(n):
        current_time = df.iloc[i]['Time_sec']
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        is_last_bar = (i == n - 1)
        
        cooldown_ok = (current_time - last_signal_time) >= config['UNIVERSAL_COOLDOWN']
        
        final_sig = 0

        # EOD SQUARE-OFF (overrides everything)
        if is_last_bar and position != 0:
            final_sig = -position
            position = 0
            position_owner = None
            last_signal_time = current_time
        
        # FLAT POSITION
        elif position == 0:
            if cooldown_ok and not is_last_bar:
                # Priority 1 entry (highest priority)
                if p1_sig != 0:
                    final_sig = p1_sig
                    position = p1_sig
                    position_owner = "P1"
                    priority_2_locked = True
                    last_signal_time = current_time
                
                # Priority 2 entry (only if P1 hasn't locked it)
                elif p2_sig != 0 and not priority_2_locked:
                    final_sig = p2_sig
                    position = p2_sig
                    position_owner = "P2"
                    last_signal_time = current_time
        
        # IN POSITION
        elif position != 0:
            if position_owner == "P1":
                # Priority 1 manages its own exit
                if cooldown_ok and p1_sig == -position:
                    final_sig = p1_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
            
            elif position_owner == "P2":
                # Check if Priority 1 wants to override/reverse
                if p1_sig != 0 and p1_sig != position:
                    if cooldown_ok:
                        final_sig = -position
                        position = 0
                        position_owner = None
                        priority_2_locked = True
                        last_signal_time = current_time
                
                # Otherwise check P2 exit
                elif cooldown_ok and p2_sig == -position:
                    final_sig = p2_sig
                    position = 0
                    position_owner = None
                    last_signal_time = current_time
        
        final_signals[i] = final_sig
    
    return final_signals

# DAY PROCESSING

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    try:
        # 1. READ DATA
        required_cols = [
            config['TIME_COLUMN'],
            config['PRICE_COLUMN'],
            config['FEATURE_COLUMN'],
        ]
        
        # Note: This script defaults to using dask_cudf (GPU)
        # The user requested *no* GPU toggle for this version.
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # 2. GENERATE PRIORITY 1 SIGNALS (Kalman Filter)
        p1_signals, df_with_p1 = generate_p1_signals(df.copy(), config)
        
        # 3. GENERATE PRIORITY 2 SIGNALS (KAMA + Volatility)
        p2_signals, df_with_p2 = generate_p2_signals(df.copy(), config)
        
        # 4. APPLY MANAGER LOGIC
        final_signals = apply_manager_logic(df, p1_signals, p2_signals, config)
        
        # 5. COUNT SIGNALS PER STRATEGY
        p1_entries = np.sum(np.abs(p1_signals) == 1)
        p2_entries = np.sum(np.abs(p2_signals) == 1)
        final_entries = np.sum(np.abs(final_signals) == 1)
        
        # 6. SAVE RESULT
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': final_signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'p1_entries': p1_entries,
            'p2_entries': p2_entries,
            'final_entries': final_entries
        }
    
    except Exception as e:
        print(f"Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # --- Start Argparse Setup ---
    parser = argparse.ArgumentParser(description="EBX 2-Priority Signal Generator")
    parser.add_argument(
        '--data-dir',
        type=str,
        default=EBX_Config['DATA_DIR'], # Use default from the config dict
        help=f"Directory containing the 'day*.parquet' files. Default: {EBX_Config['DATA_DIR']}"
    )
    args = parser.parse_args()
    
    # Override the config value with the one from the command line
    EBX_Config['DATA_DIR'] = args.data_dir
    config = EBX_Config # Use 'config' variable as used in the rest of the function
    # --- End Argparse Setup ---

    start_time = time.time()
    
    print("="*80)
    print("EBX STRATEGY MANAGER - 2-PRIORITY SIGNAL ROUTING")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}") # Show the directory being used
    print(f"\nPriority 1: Kalman Filter Strategy")
    print(f"  - KF_Trend vs UCM crossovers with KAMA_Slope_abs filter")
    print(f"  - Entry threshold: KAMA_Slope_abs > {config['P1_ENTRY_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Exit threshold: KAMA_Slope_abs > {config['P1_EXIT_KAMA_SLOPE_THRESHOLD']}")
    print(f"  - Take Profit: {config['P1_TAKE_PROFIT']} absolute")
    print(f"  - Min Hold: {config['P1_MIN_HOLD_SECONDS']}s")
    print(f"\nPriority 2: KAMA + Volatility with Trailing Stop")
    print(f"  - KAMA Slope Entry: >{config['P2_KAMA_SLOPE_ENTRY']}")
    print(f"  - Volatility Filter: ATR>{config['P2_ATR_THRESHOLD']}, StdDev>{config['P2_STD_THRESHOLD']}")
    print(f"  - Take Profit: {config['P2_TAKE_PROFIT']}, Trailing Stop: {config['P2_TRAIL_STOP']}")
    print(f"  - Min Hold: {config['P2_MIN_HOLD_SECONDS']}s")
    print(f"\nUniversal Cooldown: {config['UNIVERSAL_COOLDOWN']}s between ANY signals")
    print(f"Priority Locking: P1→P2")
    print(f"Daily Reset: Yes (priority locks reset each day)")
    print(f"EOD Handling: Unified manager-level square-off")
    print("="*80)
    
    # Setup temp directory
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        print(f"\nRemoving existing temp directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    print(f"Created temp directory: {temp_dir_path}")
    
    # Find input files
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    filtered_files = [f for f in sorted_files if extract_day_num(f) > 79 or extract_day_num(f) < 60]
    
    if not sorted_files:
        print(f"\nERROR: No csv files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(filtered_files)} day files to process")
    
    # Process all days in parallel
    print(f"\nProcessing with {config['MAX_WORKERS']} workers...")
    processed_files = []
    total_p1_entries = 0
    total_p2_entries = 0
    total_final_entries = 0
    
    try:
        with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
            futures = {
                executor.submit(process_single_day, f, extract_day_num(f), temp_dir_path, config): f
                for f in filtered_files
            }
            
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result['path'])
                        total_p1_entries += result['p1_entries']
                        total_p2_entries += result['p2_entries']
                        total_final_entries += result['final_entries']
                        print(f"✓ Processed day{result['day_num']:3d} | P1: {result['p1_entries']:3d} | P2: {result['p2_entries']:3d} | Final: {result['final_entries']:3d}")
                except Exception as e:
                    print(f"✗ Error in {file_path}: {e}")
        
        # Combine results
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
        
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num(p))
        
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
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
        
        # Statistics
        final_df = pd.read_csv(output_path)
        num_long = (final_df['Signal'] == 1).sum()
        num_short = (final_df['Signal'] == -1).sum()
        num_hold = (final_df['Signal'] == 0).sum()
        total_bars = len(final_df)
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:         {total_bars:,}")
        print(f"Long entries:       {num_long:,}")
        print(f"Short entries:      {num_short:,}")
        print(f"Hold/Wait:          {num_hold:,}")
        print(f"Total entries:      {(num_long + num_short):,}")
        print(f"\nPer-Strategy Breakdown (before manager):")
        print(f"P1 entries:         {total_p1_entries:,}")
        print(f"P2 entries:         {total_p2_entries:,}")
        print(f"Final entries:      {total_final_entries:,}")
        if len(processed_files) > 0:
            print(f"\nAvg per day:        {(num_long + num_short)/len(processed_files):.1f}")
        else:
            print(f"\nAvg per day:        0.0")
        if total_bars > 0:
            print(f"Signal density:     {((num_long + num_short)/total_bars)*100:.3f}%")
        else:
            print(f"Signal density:     0.000%")
        print(f"{'='*80}")
        
    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print(f"\nCleaned up temp directory")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"✓ Output saved to: {output_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main() 