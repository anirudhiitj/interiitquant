import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import math
import gc
from filterpy.kalman import ExtendedKalmanFilter
import warnings
import sys

warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBX")
EXTERNAL_SIGNAL_PATH = "/data/quant14/signals/temp_trading_signals_EBX.csv" 
DAY_TO_PLOT = 87
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"

# Strategy Config
STRATEGY_CONFIG = {
    'DECISION_WINDOW': 30 * 60,         # 30 minutes
    'SIGMA_THRESHOLD': 0.12,
    'RESAMPLE_PERIOD': 30,             # 300s (5 Mins) - Pattern works better here
    'REENTRY_WAIT_CANDLES': 1,
    
    # SMA Config
    'SMA_WINDOW_MINUTES': 75,           
    
    # Indicator Config
    'EAMA_SS_PERIOD': 25, 'EAMA_ER_PERIOD': 15, 'EAMA_FAST_PERIOD': 30, 'EAMA_SLOW_PERIOD': 120,
    'KAMA_ER_PERIOD': 15, 'KAMA_FAST_PERIOD': 25, 'KAMA_SLOW_PERIOD': 100,
    # EAMA Slow (for HYDRA)
    'EAMA_SLOW_FAST_PERIOD': 60, 'EAMA_SLOW_SLOW_PERIOD': 180,
    'MIN_TRADE_DURATION': 5,
    'EXIT_THRESHOLD': 0.0378
}

# ==========================================================
# PART 1: CORE MATH HELPERS
# ==========================================================
def apply_recursive_filter(series, sc):
    n = len(series)
    out = np.full(n, np.nan)
    start_idx = 0
    if np.isnan(series[0]):
        valid_indices = np.where(~np.isnan(series))[0]
        if len(valid_indices) > 0: start_idx = valid_indices[0]
        else: return out 
    out[start_idx] = series[start_idx]
    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        target = series[i]
        if not np.isnan(target): out[i] = out[i-1] + c * (target - out[i-1])
        else: out[i] = out[i-1]
    return out

def get_super_smoother_array(prices, period):
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1
    ss = np.zeros(n)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    for i in range(2, n): ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]
    return ss

def rolling_fft_reconstruction(series, window=120, top_k=3):
    n = len(series)
    out = np.full(n, np.nan)
    series = np.array(series, dtype=float)
    for i in range(window, n):
        segment = series[i-window:i]
        fft_vals = np.fft.fft(segment)
        idx = np.argsort(np.abs(fft_vals))[::-1]
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[idx[:top_k]] = fft_vals[idx[:top_k]]
        reconstructed = np.fft.ifft(fft_filtered).real
        out[i] = reconstructed[-1]
    return out

# ==========================================================
# PART 2: EXTENSIVE FEATURE LIBRARY
# ==========================================================
def add_basic_super_smoothers(df):
    prices = df["Price"].values
    df["EhlersSuperSmoother"] = get_super_smoother_array(prices, period=30)
    df["EhlersSuperSmoother_Slow"] = get_super_smoother_array(prices, period=600)
    return df

def add_kama_suite(df):
    prices = df["Price"].values
    period, fast, slow = 15, 60, 180
    direction = df["Price"].diff(period).abs()
    volatility = df["Price"].diff(1).abs().rolling(period).sum()
    er = (direction / volatility).fillna(0)
    df['ER'] = er 
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (((er * (fast_sc - slow_sc)) + slow_sc)**2).values
    df['Kama'] = apply_recursive_filter(prices, sc)
    df['KAMA_Slope'] = df['Kama'].diff().fillna(0).abs()
    return df

def add_eama_suite(df):
    ss_series = df["EhlersSuperSmoother"]
    period, fast, slow = 15, 30, 120
    direction = ss_series.diff(period).abs()
    volatility = ss_series.diff(1).abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
    df["EAMA"] = apply_recursive_filter(ss_series.values, sc)
    df["EAMA_Slope"] = df["EAMA"].diff().fillna(0)
    df["EAMA_Slope_MA"] = df["EAMA_Slope"].rolling(5).mean()

    # Slow variants
    period2, fast2, slow2 = 15, 60, 180
    er2 = (ss_series.diff(period2).abs() / ss_series.diff(1).abs().rolling(period2).sum()).fillna(0).values
    sc2 = ((er2 * (2/(fast2+1) - 2/(slow2+1))) + 2/(slow2+1)) ** 2
    df["EAMA_Slow"] = apply_recursive_filter(ss_series.values, sc2)

    period3, fast3, slow3 = 30, 60, 300
    er3 = (ss_series.diff(period3).abs() / ss_series.diff(1).abs().rolling(period3).sum()).fillna(0).values
    sc3 = ((er3 * (2/(fast3+1) - 2/(slow3+1))) + 2/(slow3+1)) ** 2
    df["EAMA_Slowest"] = apply_recursive_filter(ss_series.values, sc3)
    return df

def add_ftama_suite(df):
    fft_vals = rolling_fft_reconstruction(df["Price"], window=120, top_k=3)
    df["FFT"] = fft_vals
    period, fast, slow = 15, 30, 120
    fft_s = pd.Series(fft_vals)
    er = (fft_s.diff(period).abs() / fft_s.diff(1).abs().rolling(period).sum()).fillna(0).values
    sc = ((er * (2/(fast+1) - 2/(slow+1))) + 2/(slow+1)) ** 2
    df["FTAMA"] = apply_recursive_filter(fft_vals, sc)
    return df

def add_hybrid_ham(df):
    prices = df["Price"].values
    fft_slow = rolling_fft_reconstruction(prices, window=120, top_k=3)
    er = (df["Price"].diff(15).abs() / df["Price"].diff().abs().rolling(15).sum()).fillna(0).values
    sc = ((er * (2/31 - 2/121)) + 2/121) ** 2
    ham_vals = np.nan_to_num(sc, nan=0.0) * df["EhlersSuperSmoother"].values + (1.0 - np.nan_to_num(sc, nan=0.0)) * fft_slow
    df["HAM"] = pd.Series(ham_vals).bfill().ffill().values
    
    ham_smooth = np.zeros(len(prices))
    ham_smooth[0] = df["HAM"].iloc[0]
    threshold = 0.02
    for i in range(1, len(prices)):
        if abs(df["HAM"].iloc[i] - ham_smooth[i-1]) > threshold:
            ham_smooth[i] = df["HAM"].iloc[i]
        else:
            ham_smooth[i] = ham_smooth[i-1]
    df["HAM_Smooth"] = ham_smooth
    return df

def add_zlama_suite(df):
    z_per = 30
    lag = int((z_per - 1) / 2)
    z_dat = 2 * df["Price"] - df["Price"].shift(lag)
    zlema_source = z_dat.ewm(span=z_per, adjust=False).mean()
    er = (zlema_source.diff(15).abs() / zlema_source.diff(1).abs().rolling(15).sum()).fillna(0).values
    sc = ((er * (2/31 - 2/121)) + 2/121) ** 2
    df["ZLAMA"] = apply_recursive_filter(zlema_source.values, sc)
    return df

def add_hamaz(df):
    z_per = 30
    lag = int((z_per - 1) / 2)
    z_dat = 2 * df["Price"] - df["Price"].shift(lag)
    zlema_fast = z_dat.ewm(span=z_per, adjust=False).mean()
    er = (df["Price"].diff(15).abs() / df["Price"].diff().abs().rolling(15).sum()).fillna(0)
    sc = ((er * (2/31 - 2/121)) + 2/121) ** 2
    df["HAMAZ"] = (sc * zlema_fast) + ((1 - sc) * df["EAMA"])
    return df

def add_jma(df):
    prices = df["Price"].values
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(1500, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(1500, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    alpha = 2.0 / (1501)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    df["JMA"] = out
    return df

def add_hama(df):
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        dt_index = pd.to_datetime("2024-01-01") + df['Time']
    else:
        try:
            dt_index = pd.to_datetime("2024-01-01") + pd.to_timedelta(df['Time'])
        except:
             dt_index = pd.to_datetime(df['Time'], unit='s', origin='unix')
    
    price_series = pd.Series(df["Price"].values, index=dt_index)
    ohlc = price_series.resample('60s').ohlc()
    ohlc['close'] = ohlc['close'].ffill()
    for c in ['open','high','low']: ohlc[c] = ohlc[c].fillna(ohlc['close'])
    
    ha_close = (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4
    hama_calc = ha_close.rolling(window=50, min_periods=1).mean().shift(1) 
    
    mapped_hama = hama_calc.reindex(dt_index, method='ffill')
    df["HAMA"] = mapped_hama.values
    df["HAMA"] = df["HAMA"].fillna(df["Price"].iloc[0])
    return df

def add_PAMA_suite(df):
    sma_source = df["Price"].rolling(30).mean()
    er = (sma_source.diff(15).abs() / sma_source.diff(1).abs().rolling(15).sum()).fillna(0).values
    sc = ((er * (2/31 - 2/121)) + 2/121) ** 2
    df["PAMA"] = apply_recursive_filter(sma_source.values, sc)
    return df

def add_ekf(df):
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[df["Price"].iloc[0]]])
    ekf.F = np.array([[1.0]])
    ekf.Q = np.array([[0.07]])
    ekf.R = np.array([[0.2]])
    def h(x): return np.log(x)
    def H_jac(x): return np.array([[1.0/x[0][0]]])
    vals = []
    for p in df["Price"].values:
        ekf.predict()
        if not np.isnan(p): ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
        vals.append(ekf.x.item())
    df["EKFTrend"] = pd.Series(vals).shift(1)
    return df

def add_ucm(df):
    n = len(df)
    alpha_ucm, beta_ucm = 0.005, 0.0001
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    mu[0] = df['Price'].iloc[0]
    filtered[0] = mu[0]
    for t in range(1, n):
        p = df['Price'].iloc[t]
        if np.isnan(p):
            mu[t], beta_slope[t], filtered[t] = mu[t-1], beta_slope[t-1], filtered[t-1]
        else:
            mu_pred = mu[t-1] + beta_slope[t-1]
            mu[t] = mu_pred + alpha_ucm * (p - mu_pred)
            beta_slope[t] = beta_slope[t-1] + beta_ucm * (p - mu_pred)
            filtered[t] = mu[t] + beta_slope[t]
    df["UCM"] = filtered
    return df

def add_eama_ekf(df):
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    source = df["EAMA"].fillna(method='bfill')
    ekf.x = np.array([[source.iloc[0]]])
    ekf.F, ekf.Q, ekf.R = np.array([[1.0]]), np.array([[0.07]]), np.array([[0.2]])
    def h(x): return np.log(x)
    def H_jac(x): return np.array([[1.0/x[0][0]]])
    vals = []
    for v in source.values:
        ekf.predict()
        ekf.update(np.array([[np.log(v)]]), HJacobian=H_jac, Hx=h)
        vals.append(ekf.x.item())
    df["EAMA_EKF"] = pd.Series(vals).shift(1)
    return df

def add_vhf(df):
    period = 30
    num = (df["Price"].rolling(period).max() - df["Price"].rolling(period).min()).abs()
    den = df["Price"].diff().abs().rolling(period).sum()
    df["VHF"] = num / den.replace(0, np.nan)
    return df

def add_standard_indicators(df):
    df['Slow_MA'] = df['Price'].rolling(1200).mean()
    df['Slow_MA_Slope'] = df['Slow_MA'].diff()
    df['Fast_MA'] = df['Price'].rolling(180).mean()
    df['EMA_Fast'] = df['Price'].ewm(span=25, adjust=False).mean()
    df['EMA_Slow'] = df['Price'].ewm(span=1200, adjust=False).mean()
    z_dat = 2 * df['Price'] - df['Price'].shift(599)
    df['ZLEMA'] = z_dat.ewm(span=1200, adjust=False).mean()
    df['Diff_STD'] = df['Price'].diff().rolling(120).std()
    df['Diff_STD_STD'] = df['Diff_STD'].rolling(120).std()
    return df

def add_std_threshold(df):
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        times = df['Time']
    else:
        try:
            times = pd.to_timedelta(df['Time'])
        except:
            return df
    cutoff = times.iloc[0] + pd.Timedelta(minutes=60)
    first_hour = df.loc[times < cutoff, "Diff_STD_STD"].dropna()
    thresh = np.percentile(first_hour, 99) / 2.5 if not first_hour.empty else np.nan
    df["STD_Threshold"] = thresh
    return df

def add_HYDRA_MA(df):
    if "Diff_STD_STD" not in df.columns or "STD_Threshold" not in df.columns: return df
    
    # 1. Calculate HYDRA_MA (Same as before)
    raw_sig = np.where(df["Diff_STD_STD"] > df["STD_Threshold"], 1, 0)
    weight = pd.Series(raw_sig).rolling(300).sum().fillna(0) / 300
    df["HYDRA_MA"] = (weight * df["EAMA"]) + ((1 - weight) * df["EAMA_Slow"])
    
    # 2. Calculate Bands based on HYDRA_MA Volatility
    # We calculate rolling StdDev of the HYDRA_MA ITSELF.
    std_dev = df["HYDRA_MA"].rolling(window=450).std()
    
    # Note: Since MAs are smoother than Price, the StdDev will be smaller.
    # You might want to increase the multiplier (e.g., to 2.0 or 3.0) if bands are too tight.
    multiplier = 2.0 
    
    df["HYDRA_MA_Upper"] = df["HYDRA_MA"] + (multiplier * std_dev)
    df["HYDRA_MA_Lower"] = df["HYDRA_MA"] - (multiplier * std_dev)
    
    return df

def add_regimes(df):
    # SS Regime
    pos = np.where(df["Price"] > df["EhlersSuperSmoother_Slow"], 1, 0)
    cross = pd.Series(pos).diff().abs().rolling(300).sum().fillna(0)
    df["SS_Regime"] = np.where(cross > 8, 0, 1)
    
    # Slope Ratio
    slope = df["EhlersSuperSmoother_Slow"].diff().fillna(0)
    pos_roll = pd.Series(np.where(slope > 0, slope, 0)).rolling(300).sum()
    neg_roll = pd.Series(np.where(slope < 0, np.abs(slope), 0)).rolling(300).sum()
    df["SS_Slope_Ratio"] = np.maximum(pos_roll, neg_roll) / (np.minimum(pos_roll, neg_roll) + 1e-9)
    
    # 30s VHF Regime
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        t_idx = pd.to_datetime("2024-01-01") + df['Time']
    else:
        try:
            t_idx = pd.to_datetime("2024-01-01") + pd.to_timedelta(df['Time'])
        except:
             t_idx = pd.to_datetime(df['Time'], unit='s', origin='unix')
    
    p_series = pd.Series(df["Price"].values, index=t_idx)
    res_30 = p_series.resample('30s').ohlc()['close']
    vhf_30 = (res_30.rolling(20).max() - res_30.rolling(20).min()) / res_30.diff().abs().rolling(20).sum()
    mapped_vhf = vhf_30.reindex(t_idx, method='ffill').values
    df["VHF_30s_Regime"] = np.where(mapped_vhf > 0.3, 1, 0)
    return df

def add_kama_regime(df):
    """
    KAMA regime with rolling-window and 60-tick lockout (signal extension).
    OPTIMIZED: Replaced explicit loop with vectorized rolling max.
    """
    # 1. Column Check (Handle case sensitivity)
    if 'KAMA_Slope' in df.columns:
        slope = df['KAMA_Slope']
    elif 'Kama_Slope' in df.columns:
        slope = df['Kama_Slope']
    else:
        print("Warning: KAMA_Slope not found, skipping Regime.")
        return df

    threshold = 0.0003

    # --- Stage 1: Raw KAMA regime from rolling window ---
    # Identify high slope moments
    mag = np.where(slope > threshold, 1, 0)
    
    # Rolling sum of high slope moments (120 window)
    pos_series = pd.Series(mag, index=df.index)
    window = 120
    r_pos = pos_series.rolling(window=window).sum().fillna(0)
    
    # Raw Regime Trigger
    raw_regime = np.where(r_pos > 20, 1, 0)

    # --- Stage 2: Apply 60-tick lockout (Vectorized) ---
    # Logic: If a '1' appeared anywhere in the last 60 ticks, 
    # the current state is locked to '1'.
    # This is mathematically equivalent to a rolling Maximum over 60 ticks.
    
    locked = pd.Series(raw_regime, index=df.index).rolling(window=60, min_periods=1).max().fillna(0).astype(int)

    df['KAMA_Regime'] = locked.values
    return df

# ==========================================================
# PART 3: SIGNAL GENERATION & PATTERN VISUALIZATION
# ==========================================================
def generate_signals_for_plot(df, config):
    if pd.api.types.is_timedelta64_dtype(df[TIME_COLUMN]):
        df['Time_sec'] = df[TIME_COLUMN].dt.total_seconds().astype(int)
    else:
        df['Time_sec'] = pd.to_timedelta(df[TIME_COLUMN]).dt.total_seconds().astype(int)

    # ... inside generate_signals_for_plot ...

    # OLD BROKEN CODE:
    # vol_pass = df.loc[df['Time_sec'] <= config['DECISION_WINDOW'], PRICE_COLUMN].std() > config['SIGMA_THRESHOLD']
    # if not vol_pass: return df, False, 0.0, pd.DataFrame()

    # NEW FIXED CODE:
    vol_pass = df.loc[df['Time_sec'] <= config['DECISION_WINDOW'], PRICE_COLUMN].std() > config['SIGMA_THRESHOLD']
    if not vol_pass:
        # We must return 6 values to match the success path.
        # We use empty Series for SMA and Patterns so .values calls downstream don't crash.
        empty_sma = pd.Series(np.nan, index=df.index)
        empty_pattern_series = pd.Series(0, index=df.index)
        
        # Returns: df, vol_passed, sma_tick, events, patterns, pattern_series_tick
        return df, False, empty_sma, pd.DataFrame(), pd.DataFrame(), empty_pattern_series

    p = df[PRICE_COLUMN].values
    
    # EAMA / KAMA / HYDRA (Standard Setup)
    # ... (Calculations identical to previous scripts) ...
    ss = get_super_smoother_array(p, config['EAMA_SS_PERIOD'])
    ss_s = pd.Series(ss)
    er = (ss_s.diff(config['EAMA_ER_PERIOD']).abs() / ss_s.diff(1).abs().rolling(config['EAMA_ER_PERIOD']).sum()).fillna(0).values
    sc = ((er * (2/(config['EAMA_FAST_PERIOD']+1) - 2/(config['EAMA_SLOW_PERIOD']+1))) + 2/(config['EAMA_SLOW_PERIOD']+1))**2
    eama = apply_recursive_filter(ss, sc)
    
    ps = pd.Series(p)
    er_k = (ps.diff(config['KAMA_ER_PERIOD']).abs() / ps.diff(1).abs().rolling(config['KAMA_ER_PERIOD']).sum()).fillna(0).values
    sc_k = ((er_k * (2/(config['KAMA_FAST_PERIOD']+1) - 2/(config['KAMA_SLOW_PERIOD']+1))) + 2/(config['KAMA_SLOW_PERIOD']+1))**2
    kama = apply_recursive_filter(p, sc_k)
    
    eama_slow = apply_recursive_filter(ss, ((er * (2/(config['EAMA_SLOW_FAST_PERIOD']+1) - 2/(config['EAMA_SLOW_SLOW_PERIOD']+1))) + 2/(config['EAMA_SLOW_SLOW_PERIOD']+1))**2)
    
    diff_std = ps.diff().rolling(120).std()
    diff_std_std = diff_std.rolling(120).std()
    cutoff_idx = np.searchsorted(df['Time_sec'], 3600)
    threshold = np.percentile(diff_std_std.iloc[:cutoff_idx].dropna(), 99) / 2.5
    raw_sig = np.where(diff_std_std > threshold, 1, 0)
    weight = pd.Series(raw_sig).rolling(300).sum().fillna(0) / 300
    hydra = (weight * eama) + ((1 - weight) * eama_slow)
    hydra[:cutoff_idx] = eama_slow[:cutoff_idx]

    df['Str_EAMA'] = eama
    df['Str_KAMA'] = kama
    df['Str_HYDRA'] = hydra

    # Resample (Standard Candles)
    if pd.api.types.is_timedelta64_dtype(df[TIME_COLUMN]):
        df['DateTime'] = pd.Timestamp('2024-01-01') + df[TIME_COLUMN]
    else:
        df['DateTime'] = pd.Timestamp('2024-01-01') + pd.to_timedelta(df[TIME_COLUMN])

    df_idx = df.set_index('DateTime')
    
    # We need Raw OHLC for the Pattern
    ohlc = df_idx[PRICE_COLUMN].resample(f"{config['RESAMPLE_PERIOD']}s", label='right', closed='right').ohlc().dropna()
    
    # Resample Indicators
    res_e = df_idx['Str_EAMA'].resample(f"{config['RESAMPLE_PERIOD']}s", label='right', closed='right').last()
    res_k = df_idx['Str_KAMA'].resample(f"{config['RESAMPLE_PERIOD']}s", label='right', closed='right').last()
    res_h = df_idx['Str_HYDRA'].resample(f"{config['RESAMPLE_PERIOD']}s", label='right', closed='right').last()
    
    # SMA Expanding
    window_bars = int((config['SMA_WINDOW_MINUTES'] * 60) / config['RESAMPLE_PERIOD'])
    start_time_limit = df_idx.index[0] + pd.Timedelta(seconds=config['DECISION_WINDOW'])
    mask_trading = ohlc.index > start_time_limit
    sma_expanding = ohlc['close'][mask_trading].rolling(window=window_bars, min_periods=1).mean()
    res_sma = pd.Series(np.nan, index=ohlc.index)
    res_sma.loc[mask_trading] = sma_expanding
    
    # Setup working DF
    res_df = pd.DataFrame({
        'high': ohlc['high'], 'low': ohlc['low'], 'close': ohlc['close'],
        'SMA': res_sma, 'E': res_e, 'K': res_k, 'H': res_h
    }).dropna()
    
    # --- PATTERN DETECTION LOGIC ---
    h = res_df['high'].values
    l = res_df['low'].values
    c = res_df['close'].values
    
    sigs = np.zeros(len(res_df))
    pattern_sigs = np.zeros(len(res_df)) # Just for visualization
    
    pos = 0.0
    idx_exit = -999
    
    for i in range(4, len(res_df)):
        # Pattern Logic
        h_i, l_i = h[i], l[i]
        h_i1, l_i1 = h[i-1], l[i-1]
        h_i2, l_i2 = h[i-2], l[i-2]
        h_i3, l_i3 = h[i-3], l[i-3]
        
        # Buy Pattern
        buy_cond = (
            h_i > h_i1 and         # Higher High
            h_i1 > l_i and         # Continuity
            l_i > h_i2 and         # ACCELERATION (Range Separation 1)
            h_i2 > l_i1 and        # Continuity
            l_i1 > h_i3 and        # ACCELERATION (Range Separation 2)
            h_i3 > l_i2 and        # Continuity
            l_i2 > l_i3            # Trend Start
        )
        
        # Sell Pattern
        sell_cond = (
            l_i < l_i1 and         # Lower Low
            l_i1 < h_i and         # Continuity
            h_i < l_i2 and         # ACCELERATION (Range Separation 1)
            l_i2 < h_i1 and        # Continuity
            h_i1 < l_i3 and        # ACCELERATION (Range Separation 2)
            l_i3 < h_i2 and        # Continuity
            h_i2 < h_i3            # Trend Start
        )
        
        if buy_cond: pattern_sigs[i] = 1
        elif sell_cond: pattern_sigs[i] = -1
        
        # Entry/Exit Logic (Simplified for Visualizing the PATTERN primarily)
        # Assuming you want to see where the pattern fires trades
        # (This uses simplified logic matching your signal generation script)
        
        # ... [Standard Cross/Exit Logic Omitted for brevity, focusing on Pattern Vis] ...
        
        if pos == 0:
            if buy_cond and c[i] > res_df['SMA'].iloc[i]:
                sigs[i] = 1; pos = 1
            elif sell_cond and c[i] < res_df['SMA'].iloc[i]:
                sigs[i] = -1; pos = -1
        elif pos == 1 and (res_df['E'].iloc[i] < res_df['K'].iloc[i]): # Simple cross exit
            sigs[i] = -1; pos = 0
        elif pos == -1 and (res_df['E'].iloc[i] > res_df['K'].iloc[i]): # Simple cross exit
            sigs[i] = 1; pos = 0

            
    res_df['Signal'] = sigs
    res_df['Pattern_Signal'] = pattern_sigs
    
    events = res_df[res_df['Signal'] != 0].copy().reset_index()
    patterns = res_df[res_df['Pattern_Signal'] != 0].copy().reset_index() # For markers
    
    # Map pattern signal back to ticks for the Oscillator panel
    # We reindex to the original index to plot line
    pattern_series_tick = res_df['Pattern_Signal'].reindex(df_idx.index).ffill().fillna(0)
    
    sma_tick = res_sma.reindex(df_idx.index).ffill()
    
    return df, vol_pass, sma_tick, events, patterns, pattern_series_tick

# ==========================================================
# MAIN
# ==========================================================
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"Loading {file_path}...")
df_pd = cudf.read_parquet(file_path).to_pandas()

if not pd.api.types.is_timedelta64_dtype(df_pd[TIME_COLUMN]):
    df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

df_pd = df_pd.dropna(subset=[PRICE_COLUMN, TIME_COLUMN]).reset_index(drop=True)

print("Generating Strategy Signals...")
# Note: Now unpacking patterns and pattern_series
_, vol_passed, sma_series, entry_exits, pattern_events, pattern_series = generate_signals_for_plot(df_pd.copy(), STRATEGY_CONFIG)

# Assign series to main dataframe
df_pd['SMA'] = sma_series.values
df_pd['Pattern_Signal'] = pattern_series.values

print("Calculating All Feature Library...")
df_pd = add_basic_super_smoothers(df_pd)
df_pd = add_kama_suite(df_pd)
df_pd = add_eama_suite(df_pd) 
df_pd = add_ftama_suite(df_pd)
df_pd = add_hybrid_ham(df_pd)
df_pd = add_zlama_suite(df_pd)
df_pd = add_hamaz(df_pd)
df_pd = add_jma(df_pd)
df_pd = add_hama(df_pd)
df_pd = add_PAMA_suite(df_pd)
df_pd = add_ekf(df_pd)
df_pd = add_ucm(df_pd)
df_pd = add_eama_ekf(df_pd)
df_pd = add_vhf(df_pd)
df_pd = add_standard_indicators(df_pd)
df_pd = add_std_threshold(df_pd)
df_pd = add_HYDRA_MA(df_pd)
df_pd = add_regimes(df_pd)
df_pd = add_kama_regime(df_pd)

base_dt = pd.Timestamp("2024-01-01")
df_pd['PlotTime'] = base_dt + df_pd[TIME_COLUMN]

# ==========================================================
# NEW: LOAD & PROCESS EXTERNAL SIGNALS CSV
# ==========================================================
print(f"Loading External Signals from {EXTERNAL_SIGNAL_PATH}...")
try:
    ext_sigs = pd.read_csv(EXTERNAL_SIGNAL_PATH)
    time_col_candidates = ['timestamp', 'time', 'datetime', 'Date']
    sig_time_col = next((c for c in time_col_candidates if c in ext_sigs.columns), None)
    sig_val_col_candidates = ['signal', 'side', 'Signal']
    sig_val_col = next((c for c in sig_val_col_candidates if c in ext_sigs.columns), None)

    if sig_time_col and sig_val_col:
        ext_sigs[sig_time_col] = pd.to_datetime(ext_sigs[sig_time_col])
        min_t, max_t = df_pd['PlotTime'].min(), df_pd['PlotTime'].max()
        ext_sigs = ext_sigs[(ext_sigs[sig_time_col] >= min_t) & (ext_sigs[sig_time_col] <= max_t)]
        
        df_pd_sorted = df_pd[['PlotTime', PRICE_COLUMN]].sort_values('PlotTime')
        ext_sigs_sorted = ext_sigs.sort_values(sig_time_col)
        
        merged_ext = pd.merge_asof(ext_sigs_sorted, df_pd_sorted, 
                                   left_on=sig_time_col, right_on='PlotTime', 
                                   direction='nearest', tolerance=pd.Timedelta('1min'))
        merged_ext = merged_ext.dropna(subset=[PRICE_COLUMN])
        
        ext_longs = merged_ext[merged_ext[sig_val_col] == 1]
        ext_shorts = merged_ext[merged_ext[sig_val_col] == -1]
        
    else:
        print("Warning: CSV columns not found.")
        ext_longs, ext_shorts = pd.DataFrame(), pd.DataFrame()
        
except Exception as e:
    print(f"Error loading external signals: {e}")
    ext_longs, ext_shorts = pd.DataFrame(), pd.DataFrame()

# ==========================================================
# PLOTTING SETUP
# ==========================================================
overlays = ['Kama', 'SMA', 'EMA_Fast', 'EMA_Slow', 'ZLEMA', 'HAMA', 'EAMA_EKF', 
            'EAMA_Slow', 'EAMA_Slowest', 'HAM', 'HAM_Smooth', 'PAMA', 'FTAMA', 
            'Slow_MA', 'Fast_MA', 'UCM', 'EAMA', 'JMA', 
            'EhlersSuperSmoother_Slow', 'HAMAZ', 'EKFTrend']

# Added 'Pattern_Signal' to oscillators
oscillators = ['ER', 'SS_Regime', 'VHF_30s_Regime', 'Slow_MA_Slope', 'VHF', 
               'EAMA_Slope', 'EAMA_Slope_MA', 'KAMA_Slope', 'SS_Slope_Ratio', 
               'KAMA_Regime', 'Pattern_Signal']

# ==========================================================
# PLOTTING EXECUTION
# ==========================================================
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

# --- ROW 1: PRICE & OVERLAYS ---
fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd[PRICE_COLUMN], mode='lines', 
                         line=dict(color='gray', width=1), name='Price'), row=1, col=1)

# Pattern Markers (Squares)
if not pattern_events.empty:
    pat_longs = pattern_events[pattern_events['Pattern_Signal']==1]
    pat_shorts = pattern_events[pattern_events['Pattern_Signal']==-1]
    fig.add_trace(go.Scatter(x=pat_longs['DateTime'], y=pat_longs['close'], mode='markers',
                             marker=dict(symbol='square', color='purple', size=10, line=dict(color='white', width=1)),
                             name='Pattern: Bullish'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pat_shorts['DateTime'], y=pat_shorts['close'], mode='markers',
                             marker=dict(symbol='square', color='orange', size=10, line=dict(color='white', width=1)),
                             name='Pattern: Bearish'), row=1, col=1)

# External CSV Signals (Diamonds)
if not ext_longs.empty:
    fig.add_trace(go.Scatter(x=ext_longs['PlotTime'], y=ext_longs[PRICE_COLUMN], mode='markers',
                             marker=dict(symbol='diamond', color='blue', size=8),
                             name='CSV Signal: BUY'), row=1, col=1)

if not ext_shorts.empty:
    fig.add_trace(go.Scatter(x=ext_shorts['PlotTime'], y=ext_shorts[PRICE_COLUMN], mode='markers',
                             marker=dict(symbol='diamond', color='black', size=8),
                             name='CSV Signal: SELL'), row=1, col=1)

# HYDRA Bands
if 'HYDRA_MA' in df_pd.columns:
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd['HYDRA_MA_Upper'], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd['HYDRA_MA_Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.2)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd['HYDRA_MA'], mode='lines', line=dict(color='cyan', width=2), name='HYDRA_MA'), row=1, col=1)

# Overlays
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
for i, feat in enumerate(overlays):
    if feat in df_pd.columns:
        is_strat = feat in ['EAMA', 'Kama', 'SMA'] 
        viz = True if is_strat else 'legendonly'
        width = 2 if is_strat else 1.5
        fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd[feat], mode='lines', line=dict(color=colors[i%len(colors)], width=width), visible=viz, name=feat), row=1, col=1)

# --- ROW 2: OSCILLATORS ---
for i, osc in enumerate(oscillators):
    if osc in df_pd.columns:
        # Pattern Signal is prominent
        width = 2 if osc == 'Pattern_Signal' else 1.5
        viz = True if osc == 'Pattern_Signal' else 'legendonly'
        fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd[osc], mode='lines', line=dict(width=width), visible=viz, name=osc), row=2, col=1)

status = "PASSED" if vol_passed else "FAILED (Low Vol)"
fig.update_layout(title=f"Day {DAY_TO_PLOT} Feature Analysis | Strategy Volatility Check: {status}", height=900, template="plotly_white", hovermode="x unified")
fig.show()