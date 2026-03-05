import os
import glob
import re
import shutil
import pathlib
import pandas as pd
import numpy as np
import dask_cudf
import time
import math
import gc
from multiprocessing import Pool
from typing import Tuple

# ==========================================
# CONFIGURATION
# ==========================================
Strategy_Config = {
    # Data & I/O
    'DATA_DIR': '/data/quant14/EBY/',
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'hydra_sma_kama_slope_pure_signals.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_hydra_sma_pure',
    
    # Processing
    'MAX_WORKERS': 32,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,         # 30 minutes
    'SIGMA_THRESHOLD': 0.12,            # 0.12 sigma threshold
    'RESAMPLE_PERIOD': 45,              # 45 seconds
    'MIN_TRADE_DURATION': 5,            # Minimum duration in seconds
    
    # Entry Confirmation
    'KAMA_SLOPE_THRESHOLD': 0.0008,     # Slope must be > X for Long, < -X for Short
    
    # Signal Exit Parameters
    'EXIT_THRESHOLD': 0.0378,           # Price distance from SMA for signal exit
    
    # Indicator Parameters
    'SMA_WINDOW_MINUTES': 75,           # Target window for expanding SMA
    'EAMA_SS_PERIOD': 25,
    'EAMA_ER_PERIOD': 15,
    'EAMA_FAST_PERIOD': 30,
    'EAMA_SLOW_PERIOD': 120,
    
    # EAMA Slow (for HYDRA)
    'EAMA_SLOW_FAST_PERIOD': 60,
    'EAMA_SLOW_SLOW_PERIOD': 180,
    
    # KAMA
    'KAMA_ER_PERIOD': 30,
    'KAMA_FAST_PERIOD': 60,
    'KAMA_SLOW_PERIOD': 300,
    'SMA_PERIOD': 100,
}


# ==========================================
# RESAMPLING
# ==========================================

def resample_to_candles(data_df, period_seconds=45):
    """Resample tick data to OHLC candles with strict right-edge labeling"""
    data_df = data_df.copy()
    
    if 'Time' in data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(data_df['Time']):
            data_df['Time'] = pd.to_timedelta(data_df['Time'])
            data_df['DateTime'] = pd.Timestamp('2024-01-01') + data_df['Time']
        else:
            data_df['DateTime'] = data_df['Time']
    
    data_df = data_df.set_index('DateTime')
    
    # NO LOOKAHEAD FIX: label='right', closed='right'
    ohlc = data_df['Price'].resample(f'{period_seconds}s', label='right', closed='right').ohlc()
    ohlc = ohlc.dropna()
    
    return ohlc


# ==========================================
# CORE MATH & INDICATOR HELPERS
# ==========================================

def apply_recursive_filter(series: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """Generic recursive filter"""
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float32)
    
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
        else:
            return out 
            
    out[start_idx] = series[start_idx]
    
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        prev_out = out[i-1] if not np.isnan(out[i-1]) else target
        
        if not np.isnan(target):
            out[i] = prev_out + c * (target - prev_out)
        else:
            out[i] = prev_out
            
    return out


def get_super_smoother_array(prices: np.ndarray, period: int) -> np.ndarray:
    """Returns the numpy array for SuperSmoother (Ehlers Filter)."""
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    
    ss = np.zeros(n, dtype=np.float32)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    
    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss


def calculate_eama(prices: np.ndarray, ss_period: int, er_period: int, fast: int, slow: int) -> np.ndarray:
    """Calculates EAMA."""
    ss = get_super_smoother_array(prices, ss_period)
    
    ss_series = pd.Series(ss)
    direction = ss_series.diff(er_period).abs()
    volatility = ss_series.diff(1).abs().rolling(er_period).sum()
    er = (direction / volatility).fillna(0).values

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2

    eama = apply_recursive_filter(ss, sc)
    return eama


def calculate_kama(prices: np.ndarray, er_period: int, fast: int, slow: int) -> np.ndarray:
    """Calculates KAMA."""
    price_series = pd.Series(prices)
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    sc = (((direction/volatility).fillna(0) * (2/(fast+1) - 2/(slow+1))) + (2/(slow+1)))**2
    
    kama_vals = apply_recursive_filter(prices, sc.values)
    return kama_vals


def calculate_hydra_ma(df: pd.DataFrame, eama_col: str, config: dict) -> pd.DataFrame:
    """Calculates HYDRA_MA by blending EAMA and EAMA_Slow."""
    # 1. Calculate EAMA Slow
    prices = df[config['PRICE_COLUMN']].values
    eama_slow = calculate_eama(
        prices, 
        config['EAMA_SS_PERIOD'],
        config['EAMA_ER_PERIOD'],
        config['EAMA_SLOW_FAST_PERIOD'],
        config['EAMA_SLOW_SLOW_PERIOD']
    )
    df['EAMA_Slow'] = eama_slow

    # 2. Calculate Volatility Thresholds
    df['Diff_STD'] = df[config['PRICE_COLUMN']].diff().rolling(120).std()
    df['Diff_STD_STD'] = df['Diff_STD'].rolling(120).std()

    # 3. Determine Threshold
    if not pd.api.types.is_timedelta64_dtype(df[config['TIME_COLUMN']]):
        times = pd.to_timedelta(df[config['TIME_COLUMN']])
    else:
        times = df[config['TIME_COLUMN']]
    
    start_time = times.iloc[0]
    cutoff = start_time + pd.Timedelta(minutes=60)
    first_hour_data = df.loc[times < cutoff, "Diff_STD_STD"].dropna()
    
    if first_hour_data.empty:
        df['HYDRA_MA'] = df[eama_col]
        return df
        
    threshold = np.percentile(first_hour_data, 99) / 2.5
    
    # 4. Calculate Weight
    raw_sig = np.where(df["Diff_STD_STD"] > threshold, 1, 0)
    weight = pd.Series(raw_sig).rolling(300).sum().fillna(0) / 300
    
    # 5. Calculate HYDRA
    df["HYDRA_MA"] = (weight * df[eama_col]) + ((1 - weight) * df["EAMA_Slow"])
    
    # Override first hour with Slow MA
    mask_first_hour = times < cutoff
    df.loc[mask_first_hour, "HYDRA_MA"] = df.loc[mask_first_hour, "EAMA_Slow"]
    
    return df


# ==========================================
# STRATEGY LOGIC
# ==========================================

def calculate_mu_sigma_condition(df: pd.DataFrame, config: dict) -> bool:
    """Check if volatility condition is met in the first 30 minutes."""
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    
    if len(first_30_data) < 10:
        return False
    
    prices = first_30_data[config['PRICE_COLUMN']]
    avg_sigma = prices.std()
    
    return avg_sigma > config['SIGMA_THRESHOLD']


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate signals based on EAMA/HYDRA crossover with SMA filter + KAMA Slope.
    Removes SL/TP logic. Only Signal Exit.
    """
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals
    
    # 1. Volatility Filter
    if not calculate_mu_sigma_condition(df.copy(), config):
        return signals
    
    # 2. Prepare data
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[config['TIME_COLUMN']]):
        df_copy[config['TIME_COLUMN']] = pd.to_timedelta(df_copy[config['TIME_COLUMN']])
        df_copy['DateTime'] = pd.Timestamp('2024-01-01') + df_copy[config['TIME_COLUMN']]
    else:
        df_copy['DateTime'] = df_copy[config['TIME_COLUMN']]
    
    # 3. Calculate indicators on TICK data
    prices = df_copy[config['PRICE_COLUMN']].values
    
    # EAMA (Fast)
    eama_tick = calculate_eama(
        prices, 
        config['EAMA_SS_PERIOD'],
        config['EAMA_ER_PERIOD'],
        config['EAMA_FAST_PERIOD'],
        config['EAMA_SLOW_PERIOD']
    )
    df_copy['EAMA'] = eama_tick
    
    # KAMA (Used for Slope check)
    kama_tick = calculate_kama(
        prices, 
        config['KAMA_ER_PERIOD'],
        config['KAMA_FAST_PERIOD'],
        config['KAMA_SLOW_PERIOD']
    )
    df_copy['KAMA'] = kama_tick
    
    # HYDRA MA
    df_copy = calculate_hydra_ma(df_copy, 'EAMA', config)
    
    # 4. Resample to 45s candles
    period_seconds = config['RESAMPLE_PERIOD']
    
    data_for_candles = df_copy[['DateTime', config['PRICE_COLUMN']]].copy()
    data_for_candles = data_for_candles.rename(columns={'DateTime': 'Time'})
    
    ohlc = resample_to_candles(data_for_candles, period_seconds)
    
    if len(ohlc) < 10:
        return signals
    
    # 5. Resample indicators to candle times
    df_indexed = df_copy.set_index('DateTime')
    strat_df = pd.DataFrame(index=ohlc.index)
    strat_df['Price'] = ohlc['close'] 
    
    for col in ['EAMA', 'HYDRA_MA', 'KAMA']:
        resampled = df_indexed[col].resample(f'{period_seconds}s', label='right', closed='right').last()
        strat_df[col] = resampled
    
    # --- KAMA SLOPE CALCULATION ---
    strat_df['KAMA_Slope'] = strat_df['KAMA'].diff(2).shift(1)
    
    # --- EXPANDING SMA LOGIC ---
    window_bars = int((config['SMA_WINDOW_MINUTES'] * 60) / config['RESAMPLE_PERIOD'])
    
    start_time = strat_df.index[0]
    cutoff_time_30m = start_time + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    mask_trading = strat_df.index > cutoff_time_30m
    
    trading_prices = strat_df.loc[mask_trading, 'Price']
    sma_expanding = trading_prices.rolling(window=window_bars, min_periods=1).mean()
    
    strat_df['SMA'] = np.nan
    strat_df.loc[mask_trading, 'SMA'] = sma_expanding
    strat_df = strat_df.ffill()
    
    # 6. Previous Values
    strat_df['EAMA_prev'] = strat_df['EAMA'].shift(1)
    strat_df['HYDRA_prev'] = strat_df['HYDRA_MA'].shift(1)
    
    # 7. Loop Variables
    resampled_signals = np.zeros(len(strat_df), dtype=np.float32)
    position = 0.0 
    last_trade_time_idx = -999
    
    # Data Arrays
    eama = strat_df['EAMA'].values
    hydra = strat_df['HYDRA_MA'].values
    eama_prev = strat_df['EAMA_prev'].values
    hydra_prev = strat_df['HYDRA_prev'].values
    sma = strat_df['SMA'].values
    price = strat_df['Price'].values
    kama_slope = strat_df['KAMA_Slope'].values
    
    cutoff_idx = strat_df.index.get_indexer([cutoff_time_30m], method='nearest')[0]
    first_tradable_idx = cutoff_idx + 1
    cooldown_indices = 1 
    
    # Config Shortcuts
    SIG_EXIT_THRESH = config['EXIT_THRESHOLD']
    SLOPE_THRESH = config['KAMA_SLOPE_THRESHOLD']
    
    for i in range(first_tradable_idx, len(strat_df)):
        if (np.isnan(eama[i]) or np.isnan(hydra[i]) or 
            np.isnan(eama_prev[i]) or np.isnan(hydra_prev[i]) or 
            np.isnan(sma[i]) or np.isnan(price[i]) or np.isnan(kama_slope[i])):
            continue
            
        E = eama[i]; H = hydra[i]
        Ep = eama_prev[i]; Hp = hydra_prev[i]
        P = price[i]; S = sma[i]
        KS = kama_slope[i]
        
        cross_long = (Ep < Hp) and (E > H)
        cross_short = (Ep > Hp) and (E < H)
        
        # ============================
        # EXIT LOGIC (Risk + Signal)
        # ============================
        if position != 0.0:
            # 1. DURATION EXIT
            duration_ok = (i > last_trade_time_idx) 
            if duration_ok:
                if position == 1.0: 
                    # Price < SMA - Thresh OR Bearish Cross
                    stop_hit = P < (S - SIG_EXIT_THRESH)
                    reversal = cross_short
                    if stop_hit or reversal:
                        resampled_signals[i] = -1.0 
                        position = 0.0
                        last_trade_time_idx = i
                        continue
                        
                elif position == -1.0: 
                    # Price > SMA + Thresh OR Bullish Cross
                    stop_hit = P > (S + SIG_EXIT_THRESH)
                    reversal = cross_long
                    if stop_hit or reversal:
                        resampled_signals[i] = 1.0
                        position = 0.0
                        last_trade_time_idx = i
                        continue

        # ============================
        # ENTRY LOGIC
        # ============================
        if position == 0.0:
            cooldown_ok = (i - last_trade_time_idx) >= cooldown_indices
            
            if cooldown_ok:
                # Long: Price > SMA & Cross Up & Positive Slope
                if (P > S) and cross_long and (KS > SLOPE_THRESH):
                    resampled_signals[i] = 1.0
                    position = 1.0
                    last_trade_time_idx = i
                    
                # Short: Price < SMA & Cross Down & Negative Slope
                elif (P < S) and cross_short and (KS < -SLOPE_THRESH):
                    resampled_signals[i] = -1.0
                    position = -1.0
                    last_trade_time_idx = i

    # EOD Square-off
    if position != 0.0:
        resampled_signals[-1] = -position
        
    sig_df_temp = pd.DataFrame(resampled_signals, index=strat_df.index, columns=['Signal'])
    df_indexed['Signal'] = sig_df_temp['Signal'].reindex(df_indexed.index).ffill().fillna(0)
    
    return df_indexed['Signal'].values


# ==========================================
# FILE UTILITIES
# ==========================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, 
                        config: dict) -> dict:
    
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    
    try:
        # GPU loading
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        del ddf, gdf
        gc.collect()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate signals
        signals = generate_signals(df.copy(), config)
        
        entry_signal = signals[signals.astype(bool)][0] if len(signals[signals.astype(bool)]) > 0 else 0
        direction = 'LONG' if entry_signal > 0 else ('SHORT' if entry_signal < 0 else 'FLAT')
        
        output_df = pd.DataFrame({
            'Day': day_num,
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': signals
        })
        
        output_path = temp_dir / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        return {
            'path': str(output_path),
            'day_num': day_num,
            'direction': direction
        }
    
    except Exception as e:
        print(f"Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_trade_reports_csv(output_file):
    import pandas as pd
    import os

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


def main():
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print("HYDRA/SMA STRATEGY - PURE SIGNAL LOGIC")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"\nStrategy Parameters:")
    print(f"  Vol Filter Window: {config['DECISION_WINDOW']}s")
    print(f"  Resample Period: {config['RESAMPLE_PERIOD']}s")
    print(f"  Min Duration/Cooldown: > {config['MIN_TRADE_DURATION']}s")
    print(f"\nEntry Confirmation:")
    print(f"  - KAMA Slope > {config['KAMA_SLOPE_THRESHOLD']} (Long) or < -{config['KAMA_SLOPE_THRESHOLD']} (Short)")
    print(f"\nExit Logic:")
    print(f"  - SMA Threshold: SMA ± {config['EXIT_THRESHOLD']}")
    print(f"  - Reverse Crossover (EAMA vs Hydra)")
    print("="*80)
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    
    pool_args = [
        (f, extract_day_num(f), temp_dir_path, config) 
        for f in sorted_files
    ]
    
    processed_files = []
    direction_counts = {'LONG': 0, 'SHORT': 0, 'FLAT': 0}
    
    try:
        with Pool(processes=config['MAX_WORKERS']) as pool:
            results = pool.starmap(process_single_day, pool_args)
        
        for result in results:
            if result:
                processed_files.append(result['path'])
                direction_counts[result['direction']] += 1
                print(f"✓ Processed day {result['day_num']}: {result['direction']}")

        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
        
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0:
                        shutil.copyfileobj(infile, outfile)
                    else:
                        infile.readline()
                        shutil.copyfileobj(infile, outfile)
        
        print(f"\n✓ Created combined output: {output_path}")
        
        print("\nGenerating trade_reports.csv ...")
        generate_trade_reports_csv(output_path)
        
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