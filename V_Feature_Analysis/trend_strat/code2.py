import os
import pandas as pd
import numpy as np
import dask_cudf
import time
import math
import gc
import pathlib

# ==========================================
# CONFIGURATION
# ==========================================
Strategy_Config = {
    # Data & I/O
    'DATA_DIR': '/data/quant14/EBY/',
    'OUTPUT_DIR': '/home/raid/Quant14/V_Feature_Analysis/trend_strat/',
    'OUTPUT_FILE': 'day10_signals_truncated.csv',
    'DAY_NUM': 10,  # Process only day 5
    'TRUNCATE_TIME': '03:34:00',  # Remove data after this time for forward bias check
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # Strategy Parameters
    'DECISION_WINDOW': 30 * 60,         # 30 minutes = 1800 seconds
    'SIGMA_THRESHOLD': 0.12,            # 0.12 σ threshold
    'RESAMPLE_PERIOD': 120,             # 120 seconds for HA candles
    'REENTRY_WAIT_CANDLES': 1,          # Wait 1 candle before re-entry
    
    # Indicator Parameters
    'EAMA_SS_PERIOD': 25,
    'EAMA_ER_PERIOD': 15,
    'EAMA_FAST_PERIOD': 30,
    'EAMA_SLOW_PERIOD': 120,
    
    'KAMA_ER_PERIOD': 15,
    'KAMA_FAST_PERIOD': 25,
    'KAMA_SLOW_PERIOD': 100,
    
    'SMA_PERIOD': 100,
}


# ==========================================
# RESAMPLING & HEIKIN-ASHI
# ==========================================

def resample_to_candles(data_df, period_seconds=120):
    """Resample tick data to OHLC candles"""
    data_df = data_df.copy()
    
    if 'Time' in data_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(data_df['Time']):
            data_df['Time'] = pd.to_timedelta(data_df['Time'])
            data_df['DateTime'] = pd.Timestamp('2024-01-01') + data_df['Time']
        else:
            data_df['DateTime'] = data_df['Time']
    
    data_df = data_df.set_index('DateTime')
    ohlc = data_df['Price'].resample(f'{period_seconds}s').ohlc()
    ohlc = ohlc.dropna()
    
    return ohlc


def calculate_heikin_ashi(ohlc_df):
    """Calculate Heikin-Ashi candles from OHLC data"""
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
    """Generic recursive filter: out[i] = out[i-1] + sc[i] * (series[i] - out[i-1])"""
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


def calculate_eama(prices: np.ndarray, ss_period: int, er_period: int, fast: int, slow: int):
    """Calculates EAMA and returns both EAMA and the SuperSmoother source."""
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
    """Calculates KAMA (ER on Price)."""
    price_series = pd.Series(prices)
    direction = price_series.diff(er_period).abs()
    volatility = price_series.diff(1).abs().rolling(window=er_period).sum()
    sc = (((direction/volatility).fillna(0) * (2/(fast+1) - 2/(slow+1))) + (2/(slow+1)))**2
    
    kama_vals = apply_recursive_filter(prices, sc.values)
    return kama_vals


# ==========================================
# STRATEGY LOGIC
# ==========================================

def calculate_mu_sigma_condition(df: pd.DataFrame, config: dict) -> bool:
    """
    Check if volatility condition is met in the first 30 minutes.
    Returns: True if tradable (avg sigma > threshold), False otherwise.
    """
    decision_window = config['DECISION_WINDOW']
    first_30_data = df[df['Time_sec'] <= decision_window]
    
    if len(first_30_data) < 10:
        return False
    
    prices = first_30_data[config['PRICE_COLUMN']]
    avg_sigma = prices.std()
    
    return avg_sigma > config['SIGMA_THRESHOLD']


def generate_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Generate signals based on EAMA/KAMA crossover with SMA filter.
    
    Strategy Rules:
    1. First 30 min: No trading, check volatility (σ > 0.12)
    2. At 30min + 1 candle: Initial entry based on EAMA vs KAMA + SMA confirmation
    3. Exit: On crossover (NO SMA confirmation needed)
    4. Re-entry: Wait 1 candle, then check SMA confirmation
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
    
    eama_tick, _ = calculate_eama(
        prices, 
        config['EAMA_SS_PERIOD'],
        config['EAMA_ER_PERIOD'],
        config['EAMA_FAST_PERIOD'],
        config['EAMA_SLOW_PERIOD']
    )
    kama_tick = calculate_kama(
        prices, 
        config['KAMA_ER_PERIOD'],
        config['KAMA_FAST_PERIOD'],
        config['KAMA_SLOW_PERIOD']
    )
    
    df_copy['EAMA'] = eama_tick
    df_copy['KAMA'] = kama_tick
    
    # 4. Resample to candles
    period_seconds = config['RESAMPLE_PERIOD']
    
    data_for_candles = df_copy[['DateTime', config['PRICE_COLUMN']]].copy()
    data_for_candles = data_for_candles.rename(columns={'DateTime': 'Time'})
    ohlc = resample_to_candles(data_for_candles, period_seconds)
    
    if len(ohlc) < 10:
        return signals
    
    # Calculate Heikin-Ashi
    ha_df = calculate_heikin_ashi(ohlc)
    
    # 5. Resample indicators to candle times
    df_indexed = df_copy.set_index('DateTime')
    
    for col in ['EAMA', 'KAMA', config['PRICE_COLUMN']]:
        resampled = df_indexed[col].resample(f'{period_seconds}s').last()
        ha_df[col] = resampled.reindex(ha_df.index).ffill()
    
    # 6. Calculate SMA on resampled price
    ha_df['SMA'] = ha_df[config['PRICE_COLUMN']].rolling(
        window=config['SMA_PERIOD'], 
        min_periods=1
    ).mean()
    
    # 7. SHIFT candle-level data by 1 period
   # 7. SHIFT candle-level data by 1 period (avoid lookahead bias)
    ha_df['Price_shifted'] = ha_df[config['PRICE_COLUMN']].shift(1)
    ha_df['SMA_shifted'] = ha_df['SMA'].shift(1)
    
    # Previous values for crossover detection
    ha_df['EAMA_prev'] = ha_df['EAMA'].shift(2)
    ha_df['KAMA_prev'] = ha_df['KAMA'].shift(2)

    ha_df['EAMA'] = ha_df['EAMA'].shift(1)
    ha_df['KAMA'] = ha_df['KAMA'].shift(1)
    
    # 8. Mark 30-minute cutoff
    start_time = ha_df.index[0]
    cutoff_time = start_time + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    
    # 9. Generate signals
    resampled_signals = np.zeros(len(ha_df), dtype=np.float32)
    position = 0.0
    exit_candle_idx = -999
    
    eama_vals = ha_df['EAMA'].values
    kama_vals = ha_df['KAMA'].values
    eama_prev_vals = ha_df['EAMA_prev'].values
    kama_prev_vals = ha_df['KAMA_prev'].values
    price_vals = ha_df['Price_shifted'].values
    sma_vals = ha_df['SMA_shifted'].values
    
    cutoff_idx = ha_df.index.get_indexer([cutoff_time], method='nearest')[0]
    first_tradable_idx = cutoff_idx + 1
    
    for i in range(first_tradable_idx, len(ha_df)):
        if (np.isnan(eama_vals[i]) or np.isnan(kama_vals[i]) or 
            np.isnan(eama_prev_vals[i]) or np.isnan(kama_prev_vals[i]) or 
            np.isnan(sma_vals[i]) or np.isnan(price_vals[i])):
            continue
        
        eama_cur = eama_vals[i]
        kama_cur = kama_vals[i]
        eama_prev = eama_prev_vals[i]
        kama_prev = kama_prev_vals[i]
        price_cur = price_vals[i]
        sma_cur = sma_vals[i]
        
        bullish_crossover = (eama_prev <= kama_prev) and (eama_cur > kama_cur)
        bearish_crossover = (eama_prev >= kama_prev) and (eama_cur < kama_cur)
        
        # EXIT LOGIC
        if position == 1.0 and bearish_crossover:
            resampled_signals[i] = -1.0
            position = 0.0
            exit_candle_idx = i
            continue
            
        elif position == -1.0 and bullish_crossover:
            resampled_signals[i] = +1.0
            position = 0.0
            exit_candle_idx = i
            continue
        
        # ENTRY LOGIC
        if position == 0.0:
            if i == first_tradable_idx:
                if eama_cur > kama_cur and price_cur > sma_cur:
                    resampled_signals[i] = +1.0
                    position = 1.0
                elif eama_cur < kama_cur and price_cur < sma_cur:
                    resampled_signals[i] = -1.0
                    position = -1.0
            else:
                candles_since_exit = i - exit_candle_idx
                if candles_since_exit >= config['REENTRY_WAIT_CANDLES']:
                    if bullish_crossover and price_cur > sma_cur:
                        resampled_signals[i] = +1.0
                        position = 1.0
                    elif bearish_crossover and price_cur < sma_cur:
                        resampled_signals[i] = -1.0
                        position = -1.0
    
    # EOD SQUARE-OFF
    if position != 0.0:
        resampled_signals[-1] = -position
    
    # 10. Map signals back to tick data
    ha_df['Signal'] = resampled_signals
    signal_series = ha_df['Signal']
    
    df_indexed['Signal'] = signal_series.reindex(df_indexed.index).ffill().fillna(0)
    
    return df_indexed['Signal'].values


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    config = Strategy_Config
    
    start_time = time.time()
    
    print("="*80)
    print(f"EAMA/KAMA CROSSOVER STRATEGY - DAY {config['DAY_NUM']} (FORWARD BIAS CHECK)")
    print("="*80)
    print(f"\nProcessing data from: {config['DATA_DIR']}")
    print(f"TRUNCATING DATA AFTER: {config['TRUNCATE_TIME']}")
    print(f"\nStrategy Parameters:")
    print(f"  Vol Filter Window: {config['DECISION_WINDOW']}s ({config['DECISION_WINDOW']/60:.0f} min)")
    print(f"  Sigma Threshold: {config['SIGMA_THRESHOLD']}")
    print(f"  Resample Period: {config['RESAMPLE_PERIOD']}s")
    print(f"  SMA Period: {config['SMA_PERIOD']}")
    print(f"  Re-entry Wait: {config['REENTRY_WAIT_CANDLES']} candle(s)")
    print("="*80)
    
    # Load day 5 file
    day_file = os.path.join(config['DATA_DIR'], f"day{config['DAY_NUM']}.parquet")
    
    if not os.path.exists(day_file):
        print(f"\nERROR: File not found: {day_file}")
        return
    
    print(f"\nLoading day {config['DAY_NUM']} from: {day_file}")
    
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    
    try:
        # Load with GPU
        ddf = dask_cudf.read_parquet(day_file, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        del ddf, gdf
        gc.collect()
        
        if df.empty:
            print(f"ERROR: Day {config['DAY_NUM']} file is empty.")
            return
        
        print(f"Loaded {len(df):,} rows")
        
        # TRUNCATE DATA AFTER SPECIFIED TIME
        df = df.reset_index(drop=True)
        
        # Convert Time column to timedelta if needed
        if not pd.api.types.is_datetime64_any_dtype(df[config['TIME_COLUMN']]):
            df['Time_td'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str))
        else:
            df['Time_td'] = df[config['TIME_COLUMN']]
        
        # Parse truncate time
        truncate_time = pd.to_timedelta(config['TRUNCATE_TIME'])
        
        # Filter data - keep only rows BEFORE truncate time
        df_before_truncate = df[df['Time_td'] <= truncate_time].copy()
        
        print(f"Original rows: {len(df):,}")
        print(f"After truncating at {config['TRUNCATE_TIME']}: {len(df_before_truncate):,} rows")
        print(f"Removed {len(df) - len(df_before_truncate):,} rows for forward bias check")
        
        if df_before_truncate.empty:
            print(f"ERROR: No data remaining after truncation at {config['TRUNCATE_TIME']}")
            return
        
        # Use truncated data
        df = df_before_truncate
        
        df['Time_sec'] = df['Time_td'].dt.total_seconds().astype(int)
        
        # Generate signals
        print("\nGenerating signals...")
        signals = generate_signals(df.copy(), config)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'Day': config['DAY_NUM'],
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'Signal': signals
        })
        
        # Save to CSV
        output_dir = pathlib.Path(config['OUTPUT_DIR'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / config['OUTPUT_FILE']
        output_df.to_csv(output_path, index=False)
        
        # Print statistics
        entry_signals = output_df[output_df['Signal'].abs() == 1.0]
        direction = 'LONG' if (signals[signals != 0][0] if len(signals[signals != 0]) > 0 else 0) > 0 else 'SHORT'
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS (TRUNCATED DATA)")
        print(f"{'='*80}")
        print(f"Data range:        00:00:00 to {config['TRUNCATE_TIME']}")
        print(f"Total bars:        {len(output_df):,}")
        print(f"Entry signals:     {len(entry_signals):,}")
        print(f"Direction:         {direction}")
        print(f"{'='*80}")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Processing time: {elapsed:.2f}s")
        print(f"✓ Output saved to: {output_path}")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"ERROR processing day {config['DAY_NUM']}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()