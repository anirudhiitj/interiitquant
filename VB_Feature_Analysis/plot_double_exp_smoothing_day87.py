import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import pywt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from filter import CONFIG, get_pairs_for_day

# ================================================================
# CAUSAL LOWESS FILTER (NO FORWARD BIAS)
# ================================================================
def causal_lowess_filter(series, frac=0.05, min_points=None):
    """
    Apply LOWESS filter using only past data at each point (no forward bias).
    
    Parameters:
    -----------
    series : array-like or pd.Series
        Input data to smooth
    frac : float, default=0.05
        Fraction of data to use for smoothing at each point
    min_points : int, optional
        Minimum number of points needed before smoothing starts
    
    Returns:
    --------
    np.ndarray
        Smoothed trend values (same length as input)
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = np.array(series)
    
    valid_mask = ~np.isnan(values)
    valid_indices = np.where(valid_mask)[0]
    valid_values = values[valid_mask]
    
    trend = np.full(len(values), np.nan)
    
    if min_points is None:
        min_points = max(10, int(len(valid_values) * frac))
    
    for i in range(len(valid_indices)):
        current_idx = valid_indices[i]
        
        if i + 1 < min_points:
            trend[current_idx] = np.mean(valid_values[:i+1])
        else:
            past_values = valid_values[:i+1]
            past_indices = np.arange(len(past_values))
            
            smoothed = lowess(
                past_values,
                past_indices,
                frac=frac,
                it=0,
                is_sorted=True,
                return_sorted=False
            )
            
            trend[current_idx] = smoothed[-1]
    
    return trend

# ================================================================
# TEMA (Triple Exponential Moving Average) - No forward bias
# ================================================================
def tema(series, length):
    series = pd.Series(series)
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3*ema1 - 3*ema2 + ema3


# ================================================================
# HMA (Hull Moving Average) - Causal Implementation
# ================================================================
def wma(arr, window):
    weights = np.arange(1, window+1)
    out = np.full(len(arr), np.nan)
    for i in range(window-1, len(arr)):
        window_slice = arr[i-window+1:i+1]
        out[i] = np.dot(window_slice, weights) / weights.sum()
    return out

def hma(series, length):
    half = int(length / 2)
    sqrt_l = int(np.sqrt(length))
    wma_full = wma(series, length)
    wma_half = wma(series, half)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_l)


# ================================================================
# 1. Load data
# ================================================================
day_to_plot = 87
success, price_pairs, df_day = get_pairs_for_day(day_to_plot)
if not success or df_day is None:
    raise FileNotFoundError(f"Failed to load data for day {day_to_plot}")

print(f"Loaded Day {day_to_plot}: {len(df_day)} rows")
print(f"  Found {len(price_pairs)} price jump pairs")

# ================================================================
# 2. Prepare data
# ================================================================
price_col = CONFIG["PRICE_COLUMN"]
PB9_T1_col = CONFIG["VOLATILITY_FEATURE"]

df_day = df_day.dropna(subset=[price_col, PB9_T1_col]).reset_index(drop=True)
df_day = df_day.sort_index()
df_day["Index"] = np.arange(len(df_day))

prices = df_day[price_col].astype(float).values
PB9_T1_values = df_day[PB9_T1_col].astype(float).values

# ================================================================
# ⭐ NEW FEATURES: TEMA & HMA (Low-Lag Crossovers)
# ================================================================
df_day["TEMA_fast"] = tema(prices, 8)
df_day["TEMA_slow"] = tema(prices, 21)

df_day["HMA_fast"] = hma(prices, 12)
df_day["HMA_slow"] = hma(prices, 28)

print("Added crossover features: TEMA(8/21), HMA(12/28)")


# ================================================================
# 3. Double Exponential Smoothing (DES)
# ================================================================
alpha = 0.05
beta = 0.001
model = ExponentialSmoothing(prices, trend='add', seasonal=None)
fit = model.fit(smoothing_level=alpha, smoothing_trend=beta)
df_day["DES_Smoothed"] = fit.fittedvalues
initial_level = fit.params.get("initial_level", prices[0])
initial_trend = fit.params.get("initial_trend", 0)
df_day["DES_Trend"] = initial_level + np.arange(len(prices)) * initial_trend
print("Applied DES")

# ================================================================
# 3½ BAXTER-KING (Ultra-Smooth)
# ================================================================
N = len(prices)
BK_LOW, BK_HIGH, BK_K = 60, 180, min(80, N // 6)
idx_bk = pd.Index([])

if 2 * BK_K < N:
    try:
        result = bkfilter(prices, low=BK_LOW, high=BK_HIGH, K=BK_K)
        cycle_bk = result['cycle'].values if isinstance(result, pd.DataFrame) else np.asarray(result)
        trend_bk = prices[BK_K:-BK_K] - cycle_bk
        idx_bk = df_day["Index"][BK_K:-BK_K]
        df_day.loc[idx_bk, "BK_Trend"] = trend_bk
        df_day.loc[idx_bk, "BK_Cycle"] = cycle_bk
        print(f"BK applied: low={BK_LOW}, high={BK_HIGH}, K={BK_K}")
    except Exception as e:
        print(f"BK failed: {e}")
else:
    print("BK skipped")

# ================================================================
# 4. HODRICK-PRESCOTT (HP)
# ================================================================
try:
    cycle_hp, trend_hp = hpfilter(prices, lamb=1600)
    df_day["HP_Cycle"] = cycle_hp
    df_day["HP_Trend"] = trend_hp
    print("HP applied (λ=1600)")
except Exception as e:
    print(f"HP failed: {e}")

# ================================================================
# 5. GAUSSIAN KERNEL
# ================================================================
sigma_gauss = 35
trend_gauss = gaussian_filter1d(prices, sigma=sigma_gauss)
cycle_gauss = prices - trend_gauss
df_day["Gauss_Cycle"] = cycle_gauss
df_day["Gauss_Trend"] = trend_gauss
print(f"Gaussian applied (σ={sigma_gauss})")

# ================================================================
# 6. SAVITZKY-GOLAY
# ================================================================
window_sg = 101
cycle_sg = prices - savgol_filter(prices, window_length=window_sg, polyorder=3)
df_day["SG_Cycle"] = cycle_sg
df_day["SG_Trend"] = savgol_filter(prices, window_length=window_sg, polyorder=3)
print(f"Savitzky-Golay applied (window={window_sg})")

# ================================================================
# 7. WAVELET DENOISING
# ================================================================
coeffs = pywt.wavedec(prices, 'db4', level=5)
threshold = 0.015 * np.max(np.abs(coeffs[-1]))
coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
trend_wave = pywt.waverec(coeffs_thresh, 'db4')[:len(prices)]
cycle_wave = prices - trend_wave
df_day["Wave_Cycle"] = cycle_wave
df_day["Wave_Trend"] = trend_wave
print("Wavelet denoising applied")

# ================================================================
# 8. THETA METHOD
# ================================================================
try:
    model_theta = ThetaModel(prices, deseasonalize=False)
    res = model_theta.fit()
    theta_forecast = res.forecast(len(prices))
    trend_theta = lowess(theta_forecast, np.arange(len(prices)), frac=0.1, return_sorted=False)
    cycle_theta = prices - trend_theta
    df_day["Theta_Trend"] = trend_theta
    df_day["Theta_Cycle"] = cycle_theta
    print("Theta method applied")
except Exception as e:
    print(f"THETA failed: {e}")
    df_day["Theta_Trend"] = np.nan
    df_day["Theta_Cycle"] = np.nan

# ================================================================
# 9. KZ FILTER (CAUSAL)
# ================================================================
def kz_filter_causal(y, m=101, k=3):
    """KZ filter using only past data (no forward bias)."""
    if m % 2 == 0: 
        m += 1
    
    kernel = np.ones(m) / m
    filtered = np.full(len(y), np.nan)
    
    for _ in range(k):
        for i in range(len(y)):
            if i >= m - 1:
                window = y[i-m+1:i+1] if _ == 0 else filtered[i-m+1:i+1]
                filtered[i] = np.mean(window)
            elif i > 0:
                window = y[:i+1] if _ == 0 else filtered[:i+1]
                filtered[i] = np.mean(window)
        
        if _ < k - 1:
            y = filtered.copy()
    
    return filtered

try:
    trend_kz = kz_filter_causal(prices, m=101, k=3)
    df_day["KZ_Trend"] = pd.Series(trend_kz).shift(1)
    df_day["KZ_Cycle"] = prices - df_day["KZ_Trend"]
    print("Causal KZ filter applied")
except Exception as e:
    print(f"KZ failed: {e}")
    df_day["KZ_Trend"] = np.nan
    df_day["KZ_Cycle"] = np.nan

# ================================================================
# 10. FRACTAL DIMENSION INDICATOR (FDI)
# ================================================================
def hurst_exponent(ts, max_lag=40):
    lags = range(2, max_lag)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    tau = np.array(tau)
    tau = tau[tau > 0]
    if len(tau) == 0:
        return np.nan
    poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
    return poly[0]

def fdi_series(price_series, window=500, max_lag=40):
    fdi = np.full_like(price_series, np.nan)
    half = window // 2
    for i in range(half, len(price_series) - half):
        segment = price_series[i - half:i + half]
        H = hurst_exponent(segment, max_lag=max_lag)
        fdi[i] = 2.0 - H if not np.isnan(H) else np.nan
    return fdi

try:
    fdi_vals = fdi_series(prices, window=500, max_lag=40)
    df_day["FDI"] = fdi_vals
    print("FDI computed")
except Exception as e:
    print(f"FDI failed: {e}")
    df_day["FDI"] = np.nan

# ================================================================
# 11. CAUSAL LOWESS on PB9_T1 (NO FORWARD BIAS)
# ================================================================
try:
    print("Applying causal LOWESS on PB9_T1...")
    PB9_T1_trend = causal_lowess_filter(PB9_T1_values, frac=0.05)
    
    # No shift needed if PB9_T1 is already lagged
    # Add .shift(1) if PB9_T1 uses current price values
    df_day["LOWESS_PB9_T1_Trend"] = PB9_T1_trend
    df_day["LOWESS_PB9_T1_Cycle"] = PB9_T1_values - PB9_T1_trend
    
    print("Causal LOWESS PB9_T1 applied (no forward bias)")
except Exception as e:
    print(f"LOWESS PB9_T1 failed: {e}")
    df_day["LOWESS_PB9_T1_Trend"] = np.nan
    df_day["LOWESS_PB9_T1_Cycle"] = np.nan

# ================================================================
# 12. Plot all filters + FDI Zones + LOWESS
# ================================================================
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df_day["Index"], y=prices, name="Price", line=dict(color="black", width=2)))

# --- NEW CROSSOVER FEATURES ---
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["TEMA_fast"], 
                         name="TEMA Fast (8)", line=dict(color="red")))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["TEMA_slow"], 
                         name="TEMA Slow (21)", line=dict(color="blue")))

fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["HMA_fast"], 
                         name="HMA Fast (12)", line=dict(color="purple")))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["HMA_slow"], 
                         name="HMA Slow (28)", line=dict(color="green")))


# DES
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["DES_Smoothed"], name="DES Smoothed", line=dict(color="blue", dash="dot")))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["DES_Trend"], name="DES Trend", line=dict(color="orange", dash="dash")))

# BK
if len(idx_bk) > 0:
    fig.add_trace(go.Scatter(x=idx_bk, y=df_day.loc[idx_bk, "BK_Cycle"], name="BK Cycle"))
    fig.add_trace(go.Scatter(x=idx_bk, y=df_day.loc[idx_bk, "BK_Trend"], name="BK Trend"))

# HP
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["HP_Cycle"], name="HP Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["HP_Trend"], name="HP Trend"))

# Gaussian
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Gauss_Cycle"], name="Gauss Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Gauss_Trend"], name="Gauss Trend"))

# Savitzky-Golay
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["SG_Cycle"], name="SG Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["SG_Trend"], name="SG Trend"))

# Wavelet
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Wave_Cycle"], name="Wave Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Wave_Trend"], name="Wave Trend"))

# Theta
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Theta_Cycle"], name="Theta Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["Theta_Trend"], name="Theta Trend"))

# KZ
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["KZ_Cycle"], name="KZ Cycle"))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["KZ_Trend"], name="KZ Trend"))

# LOWESS PB9_T1 (NEW - NO FORWARD BIAS)
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["LOWESS_PB9_T1_Trend"], 
                         name="LOWESS PB9_T1 Trend", 
                         line=dict(color="darkgreen", width=3)))
fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["LOWESS_PB9_T1_Cycle"], 
                         name="LOWESS PB9_T1 Cycle", 
                         line=dict(color="lightgreen", dash="dot")))

# FDI
# fig.add_trace(go.Scatter(x=df_day["Index"], y=df_day["FDI"], name="FDI", line=dict(color="gray")), secondary_y=True)

# --- Jump Pairs ---
if price_pairs:
    for i, (s, e) in enumerate(price_pairs[:5]):
        fig.add_trace(go.Scatter(x=[s], y=[prices[s]], mode='markers',
                                 marker=dict(size=8, color='green'),
                                 name='Jump Start' if i == 0 else None, showlegend=i == 0))
        fig.add_trace(go.Scatter(x=[e], y=[prices[e]], mode='markers',
                                 marker=dict(size=8, color='red'),
                                 name='Jump End' if i == 0 else None, showlegend=i == 0))
        fig.add_trace(go.Scatter(x=[s, e], y=[prices[s], prices[e]],
                                 mode='lines', line=dict(color='gray', dash='dot'), showlegend=False))

# ================================================================
# 13. Styling
# ================================================================
y_min, y_max = prices.min() - 0.02, prices.max() + 0.02
fig.update_layout(
    title=f"Day {day_to_plot} — All Filters + KZ + FDI + Causal LOWESS PB9_T1",
    xaxis_title="Time Index",
    yaxis=dict(range=[y_min, y_max]),
    yaxis2=dict(title="FDI", overlaying="y", side="right", range=[1.0, 2.0]),
    plot_bgcolor="white",
    width=1800,
    height=800
)

# ================================================================
# 14. Save & Show
# ================================================================
output_file = f"ALL_FILTERS_KZ_FDI_CAUSAL_LOWESS_day{day_to_plot}.html"
fig.write_html(output_file)
fig.show()

print(f"Plot saved → {output_file}")
print(f"\n✓ Causal LOWESS applied - NO FORWARD BIAS")