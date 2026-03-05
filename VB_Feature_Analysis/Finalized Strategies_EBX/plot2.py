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
import dask_cudf

warnings.filterwarnings('ignore')

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBY")
DAY_TO_PLOT = 61
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"
SIGNALS_FILE_PATH = '/data/quant14/signals/combined_mu_gated_signals.csv'

# --- STRATEGY PARAMETERS ---
DECISION_WINDOW_SECONDS = 30 * 60
MU_SPLIT_THRESHOLD = 100.0 

# Low Mu (< 100) Params
L_MU_STRAT_1_PARAMS = {"entry_drop": 0.8, "pre_drop_sl": 999.0, "label": "LowMu_Strat1 (High Vol)"}
L_MU_STRAT_2_PARAMS = {"entry_drop": 0.9, "pre_drop_sl": 0.6, "label": "LowMu_Strat2 (Mid Vol)"}

# High Mu (> 100) Params
H_MU_S1_PARAMS = {"entry_rise_threshold": 0.8, "pre_rise_sl": 0.4, "label": "HighMu_Strat1 (Short First)"}
H_MU_S2_PARAMS = {"entry_drop_threshold": 0.9, "label": "HighMu_Strat2 (Long First)"}

# Thresholds
H_MU_SIGMA_HIGH_THRESHOLD = 0.12 
H_MU_SIGMA_MID_MIN = 0.09     
H_MU_SIGMA_MID_MAX = 0.12

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
# PART 2: FEATURE LIBRARY
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

    period2, fast2, slow2 = 15, 60, 180
    er2 = (ss_series.diff(period2).abs() / ss_series.diff(1).abs().rolling(period2).sum()).fillna(0).values
    sc2 = ((er2 * (2/(fast2+1) - 2/(slow2+1))) + 2/(slow2+1)) ** 2
    df["EAMA_Slow"] = apply_recursive_filter(ss_series.values, sc2)

    period3, fast3, slow3 = 30, 60, 300
    er3 = (ss_series.diff(period3).abs() / ss_series.diff(1).abs().rolling(period3).sum()).fillna(0).values
    sc3 = ((er3 * (2/(fast3+1) - 2/(slow3+1))) + 2/(slow3+1)) ** 3
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

def add_jma_with_period(prices, period):
    """Calculate JMA with specified period"""
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(period, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(period, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    alpha = 2.0 / (period + 1)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    return out

def add_jma(df):
    prices = df["Price"].values
    df["JMA60"] = add_jma_with_period(prices, 60)
    df["JMA300"] = add_jma_with_period(prices, 300)
    df["JMA900"] = add_jma_with_period(prices, 900)
    df["JMA"] = add_jma_with_period(prices, 1500)
    return df

def add_hama(df):
    if pd.api.types.is_timedelta64_dtype(df[TIME_COLUMN]):
        dt_index = pd.to_datetime("2024-01-01") + df[TIME_COLUMN]
    else:
        try:
            dt_index = pd.to_datetime("2024-01-01") + pd.to_timedelta(df[TIME_COLUMN])
        except:
             dt_index = pd.to_datetime(df[TIME_COLUMN], unit='s', origin='unix')
    
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
    if pd.api.types.is_timedelta64_dtype(df[TIME_COLUMN]):
        times = df[TIME_COLUMN]
    else:
        try:
            times = pd.to_timedelta(df[TIME_COLUMN])
        except:
            return df
            
    cutoff = times.iloc[0] + pd.Timedelta(minutes=60)
    first_hour = df.loc[times < cutoff, "Diff_STD_STD"].dropna()
    thresh = np.percentile(first_hour, 99) / 2.5 if not first_hour.empty else np.nan
    df["STD_Threshold"] = thresh
    return df

def add_HYDRA_MA(df):
    if "Diff_STD_STD" not in df.columns or "STD_Threshold" not in df.columns: return df
    raw_sig = np.where(df["Diff_STD_STD"] > df["STD_Threshold"], 1, 0)
    weight = pd.Series(raw_sig).rolling(300).sum().fillna(0) / 300
    df["HYDRA_MA"] = (weight * df["EAMA"]) + ((1 - weight) * df["EAMA_Slow"])
    return df

def add_regimes(df):
    pos = np.where(df["Price"] > df["EhlersSuperSmoother_Slow"], 1, 0)
    cross = pd.Series(pos).diff().abs().rolling(300).sum().fillna(0)
    df["SS_Regime"] = np.where(cross > 8, 0, 1)
    
    slope = df["EhlersSuperSmoother_Slow"].diff().fillna(0)
    pos_roll = pd.Series(np.where(slope > 0, slope, 0)).rolling(300).sum()
    neg_roll = pd.Series(np.where(slope < 0, np.abs(slope), 0)).rolling(300).sum()
    df["SS_Slope_Ratio"] = np.maximum(pos_roll, neg_roll) / (np.minimum(pos_roll, neg_roll) + 1e-9)
    
    if pd.api.types.is_timedelta64_dtype(df[TIME_COLUMN]):
        t_idx = pd.to_datetime("2024-01-01") + df[TIME_COLUMN]
    else:
        try:
            t_idx = pd.to_datetime("2024-01-01") + pd.to_timedelta(df[TIME_COLUMN])
        except:
             t_idx = pd.to_datetime(df[TIME_COLUMN], unit='s', origin='unix')
    
    p_series = pd.Series(df["Price"].values, index=t_idx)
    res_30 = p_series.resample('30s').ohlc()['close']
    vhf_30 = (res_30.rolling(20).max() - res_30.rolling(20).min()) / res_30.diff().abs().rolling(20).sum()
    mapped_vhf = vhf_30.reindex(t_idx, method='ffill').values
    df["VHF_30s_Regime"] = np.where(mapped_vhf > 0.3, 1, 0)
    return df

def add_zsigmoid(df):
    """Add Z-Sigmoid indicator (30-min window)"""
    price = df["Price"].astype(float)
    window_30min = 1800  # 30 minutes = 1800 seconds
    roll_mean = price.rolling(window=window_30min).mean()
    roll_std = price.rolling(window=window_30min).std(ddof=1)
    z_score = (price - roll_mean) / roll_std
    sig = 1 / (1 + np.exp(-z_score.fillna(0)))
    df["Z-Sigmoid"] = (sig - 0.5) * 2
    return df

# ==========================================================
# PART 3: SIGNAL LOADING AND STRATEGY RECONSTRUCTION
# ==========================================================
def load_and_filter_signals(target_day):
    try:
        print(f"Reading signals from {SIGNALS_FILE_PATH} with Dask/cuDF...")
        ddf = dask_cudf.read_csv(SIGNALS_FILE_PATH)
        df = ddf.compute()

        df = df.reset_index(drop=True)
        time_pd = df[TIME_COLUMN].to_pandas()
        time_pd = pd.to_timedelta(time_pd)
        time_diff = time_pd.diff().fillna(pd.Timedelta(seconds=1))
        day_transitions = (time_diff < pd.Timedelta(0)).astype(int).cumsum()
        df["Day"] = cudf.Series(day_transitions.values)

        if target_day > df['Day'].max() or target_day < 0:
            print(f"Day {target_day} does not exist in signals file.")
            return False, None

        day_data = df[df['Day'] == target_day].reset_index(drop=True)
        
        if len(day_data) == 0: return False, None

        df_pandas = day_data.to_pandas()
        if not pd.api.types.is_timedelta64_dtype(df_pandas[TIME_COLUMN]):
             df_pandas[TIME_COLUMN] = pd.to_timedelta(df_pandas[TIME_COLUMN])
             
        df_pandas[PRICE_COLUMN] = df_pandas[PRICE_COLUMN].astype(float)
        return True, df_pandas

    except Exception as e:
        print(f"Error loading signals: {e}")
        return False, None

def get_day_regime_params(df):
    """
    Reconstructs the Volatility Gating Logic.
    """
    df_sorted = df.sort_values(TIME_COLUMN)
    # Convert timedelta to seconds for filtering
    times_sec = df_sorted[TIME_COLUMN].dt.total_seconds().values
    prices = df_sorted[PRICE_COLUMN].values
    
    # Filter first 30 mins
    mask_30min = times_sec <= DECISION_WINDOW_SECONDS
    
    if not np.any(mask_30min):
        return "Unknown", {}, 0, 0

    prices_30min = prices[mask_30min]
    mu = np.mean(prices_30min)
    sigma = np.std(prices_30min)
    
    regime_name = "None"
    params = {}
    
    if mu < MU_SPLIT_THRESHOLD:
        # LOW MU
        if sigma > 0.12:
            regime_name = "LowMu_Strat1 (HighVol)"
            params = L_MU_STRAT_1_PARAMS
        elif 0.09 <= sigma <= 0.12:
            regime_name = "LowMu_Strat2 (MidVol)"
            params = L_MU_STRAT_2_PARAMS
    else:
        # HIGH MU
        if sigma > H_MU_SIGMA_HIGH_THRESHOLD:
            regime_name = "HighMu_Strat1 (ShortFirst)"
            params = H_MU_S1_PARAMS
        elif H_MU_SIGMA_MID_MIN <= sigma <= H_MU_SIGMA_MID_MAX:
             regime_name = "HighMu_Strat2 (LongFirst)"
             params = H_MU_S2_PARAMS
    
    return regime_name, params, mu, sigma

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    
    day_to_use = DAY_TO_PLOT
    
    file_path = DATA_DIR / f"day{day_to_use}.parquet"
    print(f"Loading base data {file_path}...")
    try:
        df_pd = cudf.read_parquet(file_path).to_pandas()
    except Exception as e:
        print(f"Error loading base parquet file: {e}")
        sys.exit()

    if not pd.api.types.is_timedelta64_dtype(df_pd[TIME_COLUMN]):
        df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

    df_pd = df_pd.dropna(subset=[PRICE_COLUMN, TIME_COLUMN]).reset_index(drop=True)

    success, signals_df = load_and_filter_signals(day_to_use)
    
    if not success:
        print(f"FATAL: Could not load signals for day {day_to_use}. Exiting.")
        sys.exit()
    
    # Merge Logic
    if len(df_pd) != len(signals_df):
        print("Merging signals by Time...")
        df_pd['Time_Key'] = df_pd[TIME_COLUMN].astype(str)
        signals_df['Time_Key'] = signals_df[TIME_COLUMN].astype(str)
        df_pd = pd.merge(df_pd, signals_df[['Time_Key', 'Signal']], on='Time_Key', how='left').drop(columns=['Time_Key'])
    else:
        df_pd['Signal'] = signals_df['Signal']

    # --- FEATURE CALCULATION ---
    print("Calculating Feature Library...")
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
    df_pd = add_zsigmoid(df_pd) 
    
    # --- STRATEGY REGIME DETECTION ---
    regime_name, strat_params, mu_val, sigma_val = get_day_regime_params(df_pd)
    day_open = df_pd[PRICE_COLUMN].iloc[0]
    
    base_dt = pd.Timestamp("2024-01-01")
    df_pd['PlotTime'] = base_dt + df_pd[TIME_COLUMN]
    
    entry_exits = df_pd[df_pd['Signal'] != 0].copy().reset_index()

    overlays = ['Kama', 'EMA_Fast', 'EMA_Slow', 'ZLEMA', 'HAMA', 'EAMA_EKF', 
                'EAMA_Slow', 'EAMA_Slowest', 'HAM', 'HAM_Smooth', 'PAMA', 'FTAMA', 
                'HYDRA_MA', 'Slow_MA', 'Fast_MA', 'UCM', 'EAMA', 'JMA', 
                'EhlersSuperSmoother_Slow', 'HAMAZ', 'EKFTrend', 
                'JMA60', 'JMA300', 'JMA900']

    # --- PLOTTING ---
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, 
        specs=[[{"secondary_y": True}]]
    )

    # 1. Price
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd[PRICE_COLUMN], mode='lines', 
                             line=dict(color='black', width=1), name='Price'), 
                  row=1, col=1, secondary_y=False)

    # 2. Strategy Threshold Lines (Dashed)
    if "LowMu" in regime_name:
        if "entry_drop" in strat_params:
            trigger_price = day_open - strat_params['entry_drop']
            fig.add_hline(y=trigger_price, line_dash="dash", line_color="gray", 
                          annotation_text=f"Entry Trigger (Open - {strat_params['entry_drop']})", 
                          row=1, col=1)

    elif "HighMu_Strat1" in regime_name:
        if "entry_rise_threshold" in strat_params:
            trigger_price = day_open + strat_params['entry_rise_threshold']
            fig.add_hline(y=trigger_price, line_dash="dash", line_color="orange", 
                          annotation_text=f"Short Entry Trigger (Open + {strat_params['entry_rise_threshold']})", 
                          row=1, col=1)

    elif "HighMu_Strat2" in regime_name:
        if "entry_drop_threshold" in strat_params:
            trigger_price = day_open - strat_params['entry_drop_threshold']
            fig.add_hline(y=trigger_price, line_dash="dash", line_color="purple", 
                          annotation_text=f"Long Entry Trigger (Open - {strat_params['entry_drop_threshold']})", 
                          row=1, col=1)

    # CHANGED: 30 Minute Dashed Line using STRING format to prevent Timezone shift
    decision_time_str = (df_pd['PlotTime'].min() + pd.Timedelta(minutes=30)).isoformat()
    
    fig.add_vline(x=decision_time_str, line_dash="dash", line_color="blue", 
                  annotation_text="Decision Window (30m)", 
                  row=1, col=1)

    # 4. Signals (Green/Red) - LARGER MARKERS
    if not entry_exits.empty:
        longs = entry_exits[entry_exits['Signal']==1]
        shorts = entry_exits[entry_exits['Signal']==-1]
        
        fig.add_trace(go.Scatter(x=longs['PlotTime'], y=longs[PRICE_COLUMN], mode='markers',
                                 marker=dict(symbol='triangle-up', color='green', size=25, 
                                             line=dict(color='darkgreen', width=1)),
                                 name='Signal: BUY'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=shorts['PlotTime'], y=shorts[PRICE_COLUMN], mode='markers',
                                 marker=dict(symbol='triangle-down', color='red', size=25,
                                             line=dict(color='darkred', width=1)),
                                 name='Signal: SELL'), row=1, col=1)

    # 5. Z-Sigmoid
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd['Z-Sigmoid'], mode='lines',
                             line=dict(color='#ff7f0e', width=2), name='Z-Sigmoid'), 
                  row=1, col=1, secondary_y=True)

    # 6. JMA Overlay
    fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd['JMA60'], mode='lines',
                             line=dict(color='cyan', width=2), name='JMA60'), row=1, col=1)
    
    # 7. Other Overlays
    colors = ['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
    for i, feat in enumerate(overlays):
        if feat in df_pd.columns and feat not in ['JMA60']:
            fig.add_trace(go.Scatter(x=df_pd['PlotTime'], y=df_pd[feat], mode='lines',
                                     line=dict(color=colors[i%len(colors)], width=1.5),
                                     visible='legendonly', name=feat), row=1, col=1)

    # Layout Updates
    title_text = (f"Day {DAY_TO_PLOT} | Regime: {regime_name} | "
                  f"Mu: {mu_val:.2f} | Sigma: {sigma_val:.4f}")
    
    fig.update_layout(
        title=title_text,
        height=700, 
        template="plotly_white", hovermode="x unified",
        legend=dict(x=1.01, y=1, bordercolor="Black", borderwidth=1)
    )
    
    fig.update_yaxes(title="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title="Z-Sigmoid", range=[-1, 1], tickformat='.2f', showgrid=False, row=1, col=1, secondary_y=True)

    fig.show()