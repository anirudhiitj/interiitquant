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
    'OUTPUT_FILE': 'range_separation_pattern_signals.csv', 
    'TEMP_DIR': '/data/quant14/signals/temp_range_sep',
    
    # Processing
    'MAX_WORKERS': 32,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,         
    'SIGMA_THRESHOLD': 0.2,            
    'RESAMPLE_PERIOD': 300,             # 300s (5 Mins) as per your snippet
    'REENTRY_WAIT_CANDLES': 1,          
    'STATIC_STOP_LOSS': 1.0,            # Exit if price drops 1.0 below entry
    
    # Indicator Parameters (Used for Filtering/Exits)
    'EAMA_SS_PERIOD': 25, 'EAMA_ER_PERIOD': 15, 'EAMA_FAST_PERIOD': 30, 'EAMA_SLOW_PERIOD': 120,
    'KAMA_ER_PERIOD': 15, 'KAMA_FAST_PERIOD': 25, 'KAMA_SLOW_PERIOD': 100,
    'KAMA_REGIME_THRESHOLD': 0.30,      
    'SMA_PERIOD': 80,
}


# ==========================================
# RESAMPLING & HEIKIN-ASHI
# ==========================================

def resample_to_candles(data_df, period_seconds=300):
    """
    Resample tick data to Normal OHLC candles.
    Uses label='right' to ensure the candle at timestamp T contains data from T-period to T.
    """
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


def calculate_heikin_ashi(ohlc_df):
    """
    Calculate Heikin-Ashi candles from OHLC data.
    (Used mainly for visualization or secondary indicators, NOT for the main pattern logic)
    """
    ha_df = pd.DataFrame(index=ohlc_df.index)
    
    ha_df['HA_Close'] = (ohlc_df['open'] + ohlc_df['high'] + 
                         ohlc_df['low'] + ohlc_df['close']) / 4
    
    ha_open = np.zeros(len(ohlc_df), dtype=np.float64)
    ha_open[0] = (ohlc_df['open'].iloc[0] + ohlc_df['close'].iloc[0]) / 2
    
    ha_close_values = ha_df['HA_Close'].values
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close_values[i-1]) / 2
    
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = pd.concat([ohlc_df['high'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).max(axis=1)
    ha_df['HA_Low'] = pd.concat([ohlc_df['low'], ha_df['HA_Open'], ha_df['HA_Close']], axis=1).min(axis=1)
    
    return ha_df


# ==========================================
# CORE MATH & INDICATOR HELPERS
# ==========================================
def apply_recursive_filter(series: np.ndarray, sc: np.ndarray) -> np.ndarray:
    n = len(series)
    out = np.full(n, np.nan, dtype=np.float32)
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0: start_idx = valid_indices[0]
        else: return out 
    out[start_idx] = series[start_idx]
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        prev_out = out[i-1] if not np.isnan(out[i-1]) else target
        if not np.isnan(target): out[i] = prev_out + c * (target - prev_out)
        else: out[i] = prev_out
    return out

def get_super_smoother_array(prices: np.ndarray, period: int) -> np.ndarray:
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
    for i in range(2, n): ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss

def calculate_eama(prices: np.ndarray, ss_period: int, er_period: int, fast: int, slow: int) -> Tuple[np.ndarray, np.ndarray]:
    ss = get_super_smoother_array(prices, ss_period)
    ss_series = pd.Series(ss)
    direction = ss_series.diff(er_period).abs()
    volatility = ss_series.diff(1).abs().rolling(er_period).sum()
    er = (direction / volatility).fillna(0).values
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    eama = apply_recursive_filter(ss, sc)
    return eama, ss

def calculate_kama(prices: np.ndarray, er_period: int, fast: int, slow: int) -> np.ndarray:
    price_series = pd.Series(prices)
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    sc = (((direction/volatility).fillna(0) * (2/(fast+1) - 2/(slow+1))) + (2/(slow+1)))**2
    kama_vals = apply_recursive_filter(prices, sc.values)
    return kama_vals

def calculate_kama_regime(prices: np.ndarray, er_period: int, threshold: float) -> np.ndarray:
    price_series = pd.Series(prices)
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    er = (direction / volatility).fillna(0).values
    regime = np.where(er > threshold, 1.0, 0.0)
    return regime

def calculate_mu_sigma_condition(df: pd.DataFrame, config: dict) -> bool:
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    if len(first_30_data) < 10: return False
    prices = first_30_data[config['PRICE_COLUMN']]
    avg_sigma = prices.std()
    return avg_sigma > config['SIGMA_THRESHOLD']


# ==========================================
# STRATEGY LOGIC
# ==========================================

def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate signals based on NORMAL CANDLESTICK PATTERN with KAMA Regime Filter
    and Static Stop Loss.
    """
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals
    
    # 1. Volatility Filter
    if not calculate_mu_sigma_condition(df.copy(), config):
        return signals
    
    # 2. Data Prep
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[config['TIME_COLUMN']]):
        df_copy[config['TIME_COLUMN']] = pd.to_timedelta(df_copy[config['TIME_COLUMN']])
        df_copy['DateTime'] = pd.Timestamp('2024-01-01') + df_copy[config['TIME_COLUMN']]
    else:
        df_copy['DateTime'] = df_copy[config['TIME_COLUMN']]
    
    # 3. Tick Indicators (Calculated on Raw Price)
    prices = df_copy[config['PRICE_COLUMN']].values
    eama_tick, _ = calculate_eama(prices, config['EAMA_SS_PERIOD'], config['EAMA_ER_PERIOD'], config['EAMA_FAST_PERIOD'], config['EAMA_SLOW_PERIOD'])
    kama_tick = calculate_kama(prices, config['KAMA_ER_PERIOD'], config['KAMA_FAST_PERIOD'], config['KAMA_SLOW_PERIOD'])
    kama_regime_tick = calculate_kama_regime(prices, config['KAMA_ER_PERIOD'], config['KAMA_REGIME_THRESHOLD'])
    
    df_copy['EAMA'] = eama_tick
    df_copy['KAMA'] = kama_tick
    df_copy['KAMA_Regime'] = kama_regime_tick 
    
    # 4. Resample
    period_seconds = config['RESAMPLE_PERIOD']
    data_for_candles = df_copy[['DateTime', config['PRICE_COLUMN']]].copy()
    data_for_candles = data_for_candles.rename(columns={'DateTime': 'Time'})
    
    # Get NORMAL OHLC (Required for the Pattern Logic & Stop Loss)
    ohlc = resample_to_candles(data_for_candles, period_seconds)
    
    if len(ohlc) < 10:
        return signals
    
    # Heikin-Ashi (Calculated only to maintain dataframe structure if needed, but not used for signals)
    ha_df = calculate_heikin_ashi(ohlc)
    
    # CRITICAL: Add Normal OHLC columns to the dataframe we loop through
    # The user explicitly requested pattern logic on Normal Candlesticks
    ha_df['Raw_High'] = ohlc['high']
    ha_df['Raw_Low'] = ohlc['low']
    ha_df['Raw_Open'] = ohlc['open']
    ha_df['Raw_Close'] = ohlc['close']
    
    # 5. Resample Indicators
    df_indexed = df_copy.set_index('DateTime')
    for col in ['EAMA', 'KAMA', 'KAMA_Regime', config['PRICE_COLUMN']]:
        resampled = df_indexed[col].resample(f'{period_seconds}s', label='right', closed='right').last()
        ha_df[col] = resampled.reindex(ha_df.index).ffill()
    
    # 6. SMA
    ha_df['SMA'] = ha_df[config['PRICE_COLUMN']].rolling(window=config['SMA_PERIOD'], min_periods=1).mean()
    
    # 7. Previous Indicators (For Exit Cross logic)
    ha_df['EAMA_prev'] = ha_df['EAMA'].shift(1)
    ha_df['KAMA_prev'] = ha_df['KAMA'].shift(1)
    
    # 8. Setup Loop
    start_time = ha_df.index[0]
    cutoff_time = start_time + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    resampled_signals = np.zeros(len(ha_df), dtype=np.float32)
    position = 0.0
    entry_price = 0.0
    exit_candle_idx = -999
    
    eama_vals = ha_df['EAMA'].values
    kama_vals = ha_df['KAMA'].values
    kama_regime_vals = ha_df['KAMA_Regime'].values
    sma_vals = ha_df['SMA'].values
    
    # RAW OHLC VALUES (Used for Pattern & Stop Loss)
    raw_high_vals = ha_df['Raw_High'].values
    raw_low_vals = ha_df['Raw_Low'].values
    raw_close_vals = ha_df['Raw_Close'].values # Used for price check
    
    cutoff_idx = ha_df.index.get_indexer([cutoff_time], method='nearest')[0]
    # Ensure we have enough history for the 4-candle pattern (i-3)
    first_tradable_idx = max(cutoff_idx + 1, 4) 
    
    stop_loss_dist = config['STATIC_STOP_LOSS']
    
    for i in range(first_tradable_idx, len(ha_df)):
        if (np.isnan(eama_vals[i]) or np.isnan(kama_vals[i]) or np.isnan(sma_vals[i])):
            continue
        
        # --- 1. GATHER DATA ---
        kama_regime_cur = kama_regime_vals[i]
        price_cur = raw_close_vals[i] # Current Price is the Close of the normal candle
        sma_cur = sma_vals[i]
        
        # Normal Candle Data for Pattern
        # Note: i is current completed candle. i-1 is previous, etc.
        h_i, l_i = raw_high_vals[i], raw_low_vals[i]
        h_i1, l_i1 = raw_high_vals[i-1], raw_low_vals[i-1]
        h_i2, l_i2 = raw_high_vals[i-2], raw_low_vals[i-2]
        h_i3, l_i3 = raw_high_vals[i-3], raw_low_vals[i-3]
        
        # Exit Crossovers (EAMA/KAMA) - Still used as trailing exit
        bullish_crossover = (eama_vals[i] > kama_vals[i]) and (eama_vals[i-1] <= kama_vals[i-1])
        bearish_crossover = (eama_vals[i] < kama_vals[i]) and (eama_vals[i-1] >= kama_vals[i-1])

        # ============================================
        # 0. STATIC STOP LOSS (PRIORITY)
        # ============================================
        # Implements: "if the price goes down the entry price then square off"
        # We use strict Raw Low/High to detect if price touched the SL level.
        stop_loss_hit = False
        if position == 1.0:
            if l_i <= (entry_price - stop_loss_dist):
                resampled_signals[i] = -1.0; position = 0.0; entry_price = 0.0; exit_candle_idx = i; stop_loss_hit = True
        elif position == -1.0:
            if h_i >= (entry_price + stop_loss_dist):
                resampled_signals[i] = +1.0; position = 0.0; entry_price = 0.0; exit_candle_idx = i; stop_loss_hit = True
        if stop_loss_hit: continue

        # ============================================
        # EXIT LOGIC (Trailing via EAMA/KAMA Crossover)
        # ============================================
        if position == 1.0 and bearish_crossover:
            resampled_signals[i] = -1.0; position = 0.0; entry_price = 0.0; exit_candle_idx = i; continue
        elif position == -1.0 and bullish_crossover:
            resampled_signals[i] = +1.0; position = 0.0; entry_price = 0.0; exit_candle_idx = i; continue

        # ============================================
        # ENTRY LOGIC: NORMAL CANDLESTICK PATTERN
        # ============================================
        if position == 0.0:
            
            # 1. KAMA Regime Gatekeeper (Must be 1 to enter)
            if kama_regime_cur > 0.5:
                
                # 2. Re-entry Wait
                if (i - exit_candle_idx) >= config['REENTRY_WAIT_CANDLES']:
                    
                    # --- PATTERN RECOGNITION (RAW OHLC) ---
                    # Using the specific logic provided for range separation [Image of strong bearish trend chart]
                    
                    # Buy Pattern: Climbing Stairs (Separated Range)
                    # high[i] > high[i-1] AND high[i-1] > low[i] AND low[i] > high[i-2] ...
                    pattern_buy = (
                        h_i > h_i1 and         
                        h_i1 > l_i and         
                        l_i > h_i2 and         # ACCELERATION (Range Separation 1)
                        h_i2 > l_i1 and        
                        l_i1 > h_i3 and        # ACCELERATION (Range Separation 2)
                        h_i3 > l_i2 and        
                        l_i2 > l_i3            
                    )
                    
                    # Sell Pattern: Falling Cliff (Separated Range)
                    pattern_sell = (
                        l_i < l_i1 and         
                        l_i1 < h_i and         
                        h_i < l_i2 and         # ACCELERATION (Range Separation 1)
                        l_i2 < h_i1 and        
                        h_i1 < l_i3 and        # ACCELERATION (Range Separation 2)
                        l_i3 < h_i2 and        
                        h_i2 < h_i3            
                    )
                    
                    # 3. Execution (With SMA Filter)
                    # If Bullish Pattern matches -> Go Long
                    if pattern_buy and price_cur > sma_cur:
                        resampled_signals[i] = +1.0
                        position = 1.0
                        entry_price = price_cur
                        
                    # If Bearish Pattern matches -> Go Short
                    elif pattern_sell and price_cur < sma_cur:
                        resampled_signals[i] = -1.0
                        position = -1.0
                        entry_price = price_cur
    
    # Square off at end of day
    if position != 0.0:
        resampled_signals[-1] = -position
    
    ha_df['Signal'] = resampled_signals
    signal_series = ha_df['Signal']
    df_indexed['Signal'] = signal_series.reindex(df_indexed.index).ffill().fillna(0)
    
    return df_indexed['Signal'].values


# ==========================================
# FILE/EXECUTION UTILITIES
# ==========================================
def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    try:
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        del ddf, gdf
        gc.collect()
        if df.empty: return None
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
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
        return {'path': str(output_path), 'day_num': day_num, 'direction': direction}
    except Exception as e:
        print(f"Error processing day {day_num} ({file_path}): {e}")
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
        print("No trades found.")
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
    print("RANGE SEPARATION PATTERN STRATEGY (NORMAL CANDLES)")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        return
    
    pool_args = [(f, extract_day_num(f), temp_dir_path, config) for f in sorted_files]
    processed_files = []
    
    try:
        with Pool(processes=config['MAX_WORKERS']) as pool:
            results = pool.starmap(process_single_day, pool_args)
        for result in results:
            if result:
                processed_files.append(result['path'])
                print(f"✓ Processed day {result['day_num']}: {result['direction']}")
        
        if processed_files:
            print(f"\nCombining {len(processed_files)} day files...")
            sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
            output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as outfile:
                for i, csv_file in enumerate(sorted_csvs):
                    with open(csv_file, 'rb') as infile:
                        if i == 0: shutil.copyfileobj(infile, outfile)
                        else:
                            infile.readline()
                            shutil.copyfileobj(infile, outfile)
            print(f"\n✓ Created combined output: {output_path}")
            generate_trade_reports_csv(output_path)
    finally:
        if temp_dir_path.exists(): shutil.rmtree(temp_dir_path)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()