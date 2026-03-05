import pandas as pd
import math
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import ExtendedKalmanFilter

from filter import CONFIG, get_pairs_for_day


# ================================================================
# ⭐⭐ FOURIER TRANSFORM MODULE (Rolling FFT With No Forward Bias)
# ================================================================

def zscore_mmap_style(prices, period, eps=1e-10):
    """
    EXACT match of your compute_zscore_fast logic,
    but vectorized for plotting scripts.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    z = np.zeros(n)
    mean_arr = np.zeros(n)
    std_arr = np.zeros(n)

    for i in range(n):
        if i < period - 1:
            window = prices[:i+1]
        else:
            window = prices[i-period+1:i+1]

        mean_val = window.mean()
        std_val = window.std()

        if std_val < eps:
            std_val = eps

        mean_arr[i] = mean_val
        std_arr[i] = std_val
        z[i] = (prices[i] - mean_val) / std_val

    return z, mean_arr, std_arr


# ================================================================
# ⭐⭐ NEW: KALMAN β FOR PAIRS SPREAD + ROLLING Z-SCORE
# ================================================================

class KalmanBeta:
    """Online Kalman filter for hedge ratio β(t)."""
    def __init__(self, q=1e-5, r=0.001, beta0=1.0):
        self.beta = beta0
        self.P = 1.0
        self.Q = q
        self.R = r

    def update(self, priceA, priceB):
        # Prediction
        self.P += self.Q

        # Measurement update
        y = priceA - self.beta * priceB      # innovation
        S = self.P + self.R
        K = self.P / S

        self.beta += K * y
        self.P *= (1 - K)

        return self.beta


def compute_pairs_zscore(priceA, priceB, window=1800, eps=1e-10):
    priceA = np.asarray(priceA, float)
    priceB = np.asarray(priceB, float)
    n = len(priceA)

    beta_filt = np.zeros(n)
    spread = np.zeros(n)

    # Kalman adaptive hedge ratio
    kf = KalmanBeta(q=1e-5, r=0.001)
    for i in range(n):
        beta_filt[i] = kf.update(np.log(priceA[i]), np.log(priceB[i]))
        spread[i] = np.log(priceA[i]) - beta_filt[i] * np.log(priceB[i])

    # ⭐ Smooth spread with a second Kalman (reduces spikes massively)
    spread_smooth = np.zeros(n)
    kf2 = KalmanBeta(q=1e-6, r=0.005)
    for i in range(n):
        spread_smooth[i] = kf2.update(spread[i], 1.0)

    # Rolling Z-score
    z = np.zeros(n)
    for i in range(n):
        win = spread_smooth[max(0, i-window+1):i+1]
        m = win.mean()
        s = win.std()
        if s < eps:
            s = eps
        z[i] = (spread_smooth[i] - m) / s

    # Sigmoid → (-1..1)
    z_sig = 2 * (1 / (1 + np.exp(-z))) - 1

    # ⭐ Smooth final ZScore for stability
    z_final = pd.Series(z_sig).ewm(span=50, adjust=False).mean().values

    return z_final, beta_filt, spread_smooth



def rolling_fft_reconstruction(series, window=120, top_k=3):
    """
    Rolling Fourier Transform reconstruction (no forward bias).
    - Uses last `window` samples only
    - Takes top_k dominant frequency components
    - Reconstructs smoothed price
    """
    n = len(series)
    out = np.full(n, np.nan)
    series = np.array(series, dtype=float)

    for i in range(window, n):
        segment = series[i-window:i]

        # Apply FFT to the window
        fft_vals = np.fft.fft(segment)
        freqs = np.fft.fftfreq(window)

        # Sort frequencies by amplitude (except DC component)
        idx = np.argsort(np.abs(fft_vals))[::-1]

        # Keep only top_k components
        fft_filtered = np.zeros_like(fft_vals)
        fft_filtered[idx[:top_k]] = fft_vals[idx[:top_k]]

        # Inverse FFT to get smoothed signal
        reconstructed = np.fft.ifft(fft_filtered).real
        out[i] = reconstructed[-1]  # last point as the current smoothed value

    return out


def rolling_fft_spectrum(series, window=120):
    """
    Returns spectrum features for each rolling FFT window:
    - Dominant frequency
    - Dominant amplitude
    - High-frequency energy
    - SNR = lowFreqEnergy / highFreqEnergy
    """
    n = len(series)
    dom_freq = np.full(n, np.nan)
    dom_amp  = np.full(n, np.nan)
    snr_arr  = np.full(n, np.nan)

    series = np.array(series, dtype=float)

    for i in range(window, n):
        seg = series[i-window:i]
        fft_vals = np.fft.fft(seg)
        freqs = np.fft.fftfreq(window)

        amps = np.abs(fft_vals)

        # dominant component excluding DC
        idx = np.argmax(amps[1:]) + 1
        dom_freq[i] = abs(freqs[idx])
        dom_amp[i] = amps[idx]

        # Energy split
        hf_energy = np.sum(amps[int(window*0.25):])  
        lf_energy = np.sum(amps[1:int(window*0.1)])  

        snr_arr[i] = lf_energy / (hf_energy + 1e-9)

    return dom_freq, dom_amp, snr_arr


# ================================================================
# LAYER 1: NOISE FILTERS (Heavy Smoothing)
# ================================================================

class KalmanFilter1D:
    """1D Kalman Filter - Very smooth for heavy noise reduction."""
    def __init__(self, process_variance=0.0001, measurement_variance=0.1, initial_value=None, initial_error=1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.initial_value = initial_value
        self.initial_error = initial_error
        
    def filter(self, measurements):
        n = len(measurements)
        filtered = np.full(n, np.nan)
        
        valid_idx = np.where(~np.isnan(measurements))[0]
        if len(valid_idx) == 0:
            return filtered
            
        estimate = measurements[valid_idx[0]] if self.initial_value is None else self.initial_value
        error_estimate = self.initial_error
        filtered[valid_idx[0]] = estimate
        
        for i in range(valid_idx[0] + 1, n):
            if np.isnan(measurements[i]):
                filtered[i] = estimate
                continue
                
            predicted_estimate = estimate
            predicted_error = error_estimate + self.process_variance
            
            kalman_gain = predicted_error / (predicted_error + self.measurement_variance)
            estimate = predicted_estimate + kalman_gain * (measurements[i] - predicted_estimate)
            error_estimate = (1 - kalman_gain) * predicted_error
            
            filtered[i] = estimate
            
        return filtered

def kalman_smooth(series, Q=0.0001, R=0.1):
    """Heavy Kalman smoothing for noisy data."""
    kf = KalmanFilter1D(process_variance=Q, measurement_variance=R)
    return kf.filter(np.array(series))


def supersmoother(series, length=10):
    """
    Supersmoother - John Ehlers' noise reduction filter.
    Extremely smooth with minimal lag.
    """
    series = pd.Series(series)
    n = len(series)
    smooth = np.full(n, np.nan)
    
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / length)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for i in range(n):
        if i < 2:
            smooth[i] = series.iloc[i]
        else:
            smooth[i] = c1 * (series.iloc[i] + series.iloc[i-1]) / 2 + c2 * smooth[i-1] + c3 * smooth[i-2]
    
    return smooth


def adaptive_ema(series, fast_period=2, slow_period=30):
    """
    Adaptive EMA based on volatility.
    Faster in trends, slower in consolidation.
    """
    series = pd.Series(series)
    
    # Calculate efficiency ratio
    change = abs(series.diff(slow_period))
    volatility = series.diff().abs().rolling(slow_period).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate adaptive EMA
    aema = np.full(len(series), np.nan)
    aema[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        aema[i] = aema[i-1] + sc.iloc[i] * (series.iloc[i] - aema[i-1])
    
    return aema


def ehlers_filter(series, length=20):
    """
    Ehlers' Smoothing Filter - designed specifically for reducing market noise.
    """
    series = pd.Series(series)
    alpha = 2 / (length + 1)
    
    # Two-pole filter
    smooth = np.full(len(series), np.nan)
    smooth[0] = series.iloc[0]
    smooth[1] = series.iloc[1]
    
    for i in range(2, len(series)):
        smooth[i] = (alpha - alpha**2/4) * series.iloc[i] + \
                    (alpha**2/2) * series.iloc[i-1] - \
                    (alpha - 3*alpha**2/4) * series.iloc[i-2] + \
                    2 * (1 - alpha) * smooth[i-1] - \
                    (1 - alpha)**2 * smooth[i-2]
    
    return smooth


# ================================================================
# LAYER 2: CROSSOVER SYSTEMS (Applied to Smoothed Data)
# ================================================================

def t3(series, length, volume_factor=0.7):
    """T3 Moving Average."""
    series = pd.Series(series)
    
    a = volume_factor
    c1 = -(a**3)
    c2 = 3*(a**2) + 3*(a**3)
    c3 = -6*(a**2) - 3*a - 3*(a**3)
    c4 = 1 + 3*a + a**3 + 3*(a**2)
    
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    ema4 = ema3.ewm(span=length, adjust=False).mean()
    ema5 = ema4.ewm(span=length, adjust=False).mean()
    ema6 = ema5.ewm(span=length, adjust=False).mean()
    
    t3_values = c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3
    
    return t3_values.values


def tema(series, length):
    """Triple EMA."""
    series = pd.Series(series)
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return (3*ema1 - 3*ema2 + ema3).values


def ema(series, length):
    """Standard EMA."""
    return pd.Series(series).ewm(span=length, adjust=False).mean().values


def sma(series, length):
    """Standard Simple Moving Average."""
    return pd.Series(series).rolling(window=length, min_periods=1).mean().values


# ================================================================
# ⭐ FINAL CLEAN Z-SCORE (ONLY RAW PRICE, WINDOW=1800, SIGMOID, RANGE -1..1)
# ================================================================

print("✓ ZScore_Final created (range: -1 to +1)")



# ================================================================
# ⭐ NEW CROSSOVER INDICATORS (HMA, ZLEMA, RSI)
# ================================================================

def wma(arr, window):
    """Weighted Moving Average - causal implementation."""
    weights = np.arange(1, window+1)
    out = np.full(len(arr), np.nan)
    for i in range(window-1, len(arr)):
        window_slice = arr[i-window+1:i+1]
        out[i] = np.dot(window_slice, weights) / weights.sum()
    return out

def hma(series, length):
    """Hull Moving Average - extremely low lag."""
    series = np.array(series)
    half = int(length / 2)
    sqrt_l = int(np.sqrt(length))
    
    wma_full = wma(series, length)
    wma_half = wma(series, half)
    diff = 2 * wma_half - wma_full
    
    return wma(diff, sqrt_l)


def zlema(series, length):
    """
    Zero-Lag Exponential Moving Average.
    Reduces lag by using (price + price - price[lag]) instead of just price.
    """
    series = pd.Series(series)
    lag = int((length - 1) / 2)
    
    # Create de-lagged series
    delagged = series.copy()
    for i in range(lag, len(series)):
        delagged.iloc[i] = 2 * series.iloc[i] - series.iloc[i - lag]
    
    # Apply EMA to de-lagged series
    zlema_values = delagged.ewm(span=length, adjust=False).mean()
    
    return zlema_values.values


def calculate_rsi(series, period=14):
    """
    Relative Strength Index.
    Works on smoothed data for cleaner signals.
    """
    series = pd.Series(series)
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values


def laguerre_rsi(price, gamma=0.7):
    """
    LaGuerre RSI by John Ehlers.
    gamma controls smoothness (0.5–0.9 typical)
    """
    price = np.array(price, dtype=float)
    n = len(price)

    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    RSI = np.zeros(n)

    for i in range(1, n):
        L0[i] = (1 - gamma) * price[i] + gamma * L0[i-1]
        L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
        L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
        L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]

        up = (L0[i] - L1[i]) + (L2[i] - L3[i])
        down = abs(L0[i] - L1[i]) + abs(L1[i] - L2[i]) + abs(L2[i] - L3[i])

        RSI[i] = up / down if down != 0 else 0

    return RSI


# ================================================================
# ⭐ ADDITIONAL FILTER FUNCTIONS (EHLERS ROOFING, SUPER30, T3 FAST/SLOW)
# ================================================================

def ehlers_roofing(series, hp_period=40, lp_period=20):
    """
    Ehlers Roofing Filter: HP + LP smoothing.
    The strictest low-lag noise filter available.
    """
    series = pd.Series(series)
    n = len(series)

    # High-pass filter
    hp_out = np.full(n, np.nan)
    alpha_hp = (np.cos(2*np.pi/hp_period) + np.sin(2*np.pi/hp_period) - 1) / np.cos(2*np.pi/hp_period)
    
    hp_out[0] = 0
    hp_out[1] = 0
    for i in range(2, n):
        hp_out[i] = 0.5*(1+alpha_hp)*(series.iloc[i] - series.iloc[i-1]) + alpha_hp * hp_out[i-1]

    # Low-pass smoothing
    lp_out = np.full(n, np.nan)
    alpha_lp = np.exp(-np.sqrt(2)*np.pi/lp_period)

    lp_out[0] = hp_out[0]
    lp_out[1] = hp_out[1]
    for i in range(2, n):
        lp_out[i] = (1 - alpha_lp)**2 * hp_out[i] + \
                    2*alpha_lp*(1-alpha_lp)*lp_out[i-1] + \
                    alpha_lp**2 * lp_out[i-2]

    return lp_out


def supersmoother_30(series):
    return supersmoother(series, length=30)


def t3_single(series, length=5):
    """Simple wrapper for T3 single-period."""
    return t3(series, length, 0.7)


# ================================================================
# 1. Load Data
# ================================================================
day_to_plot = 14
success, price_pairs, df_day = get_pairs_for_day(day_to_plot)
if not success or df_day is None:
    raise FileNotFoundError(f"Failed to load data for day {day_to_plot}")

print(f"✓ Loaded Day {day_to_plot}: {len(df_day)} rows")

# ================================================================
# 2. Prepare Data
# ================================================================
price_col = CONFIG["PRICE_COLUMN"]

df_day = df_day.dropna(subset=[price_col]).reset_index(drop=True)
df_day = df_day.sort_index()
df_day["Index"] = np.arange(len(df_day))

# ================================================================
# 2. Prepare Data
# ================================================================
prices = df_day[price_col].astype(float).values

print(f"✓ Price range: {prices.min():.4f} to {prices.max():.4f}")

# ================================================================
# ⭐ FINAL Sigmoid Z-Score (Raw Price)
# ================================================================
print("\n🎯 Computing FINAL Sigmoid Z-Score (Raw Price, 1800 window)...")

Z_LOOKBACK = 3600
EPS = 1e-10

z_raw, z_mean, z_std = zscore_mmap_style(prices, Z_LOOKBACK, EPS)
z_sigmoid = 1 / (1 + np.exp(-z_raw))
zscore_final = 2 * z_sigmoid - 1

df_day["ZScore_Final"] = zscore_final
print("✓ ZScore_Final created (range: −1 to +1)")



# ================================================================
# 3. LAYER 1 - Apply Heavy Noise Filters
# ================================================================

print("\n🔧 LAYER 1: Applying Heavy Noise Filters...")

# Kalman - Very smooth (increased R for more smoothing)
print("  → Kalman Heavy Smooth (Q=0.0001, R=0.1)...")
df_day["Base_Kalman"] = kalman_smooth(prices, Q=0.0001, R=0.1)

# ================================================================
# ⭐ PAIRS TRADING Z-SCORE (Kalman β, rolling window = 1800)
# ================================================================
print("🎯 Computing Pairs Trading Z-Score (Kalman β)...")

priceA = df_day["Price"].values
priceB = df_day["Base_Kalman"].values   # synthetic second leg

pairs_z, beta_series, spread_series = compute_pairs_zscore(priceA, priceB, window=1800)

df_day["Pairs_ZScore"] = pairs_z
df_day["Pairs_Beta"] = beta_series
df_day["Pairs_Spread"] = spread_series

print("✓ Pairs Z-Score added (range: −1 to +1)")
# ----------------------------------------------------

# Supersmoother - Ehlers' method
print("  → Supersmoother (20)...")
df_day["Base_Supersmoother"] = supersmoother(prices, length=20)

# Adaptive EMA - adjusts to market conditions
print("  → Adaptive EMA (2/30)...")
df_day["Base_AEMA"] = adaptive_ema(prices, fast_period=2, slow_period=30)

# Ehlers Filter - specifically designed for noise
print("  → Ehlers Filter (25)...")
df_day["Base_Ehlers"] = ehlers_filter(prices, length=25)

# Double T3 - applying T3 twice for extreme smoothing
print("  → Double T3 (first pass: 15)...")
t3_first = t3(prices, length=15, volume_factor=0.7)
print("  → Double T3 (second pass: 10)...")
df_day["Base_DoubleT3"] = t3(t3_first, length=10, volume_factor=0.7)

print("\n✅ Noise filters applied!")

print("\n⭐ Adding Additional Filter Features...")

# Roofing Filters
print("  → Roofing Filter (HP40-LP20)")
df_day["Base_Roofing40_20"] = ehlers_roofing(prices, hp_period=40, lp_period=20)

print("  → Roofing Filter (HP20-LP10)")
df_day["Base_Roofing20_10"] = ehlers_roofing(prices, hp_period=20, lp_period=10)

# T3 Fast & Slow
print("  → T3(5)")
df_day["Base_T3_5"] = t3_single(prices, length=5)

print("  → T3(15)")
df_day["Base_T3_15"] = t3_single(prices, length=15)

# Supersmoother(30)
print("  → Supersmoother(30)")
df_day["Base_Super30"] = supersmoother_30(prices)


# ================================================================
# 4. ⭐ NEW: SMA AND EMA FOR MULTIPLE PERIODS
# ================================================================

print("\n📊 Adding SMA and EMA for multiple periods...")

SMA_PERIODS = [30, 60, 90, 120, 150, 180, 300, 600, 240]
EMA_PERIODS = [30, 60, 90, 120, 150, 180, 300, 600, 130]

for period in SMA_PERIODS:
    print(f"  → SMA({period})...")
    df_day[f"SMA_{period}"] = sma(prices, period)

for period in EMA_PERIODS:
    print(f"  → EMA({period})...")
    df_day[f"EMA_{period}"] = ema(prices, period)

print("\n✅ All SMAs and EMAs calculated!")



print("\n🔷 Adding Fourier Transform Features...")

windows = [60, 120, 180]

for w in windows:
    print(f"  → FFT Reconstruction Window {w} (Top 3 components)")
    df_day[f"FFT_Recon_{w}"] = rolling_fft_reconstruction(prices, window=w, top_k=3)

    print(f"  → FFT Spectrum Window {w}")
    df_day[f"FFT_DomFreq_{w}"], df_day[f"FFT_DomAmp_{w}"], df_day[f"FFT_SNR_{w}"] = \
        rolling_fft_spectrum(prices, window=w)

print("✅ Fourier features added!\n")


# ================================================================
# 6. LAYER 2 - Apply Crossovers to Smoothed Data
# ================================================================

print("\n📊 LAYER 2: Applying Crossovers to Smoothed Data...")

# Choose the best base signal (you can test each)
BASE_OPTIONS = {
    "Kalman": "Base_Kalman",
    "Supersmoother": "Base_Supersmoother",
    "Adaptive EMA": "Base_AEMA",
    "Ehlers": "Base_Ehlers",
    "Double T3": "Base_DoubleT3"
}

for name, col in BASE_OPTIONS.items():
    base_signal = df_day[col].values
    
    # Apply crossovers on the SMOOTHED data
    df_day[f"{name}_Fast"] = ema(base_signal, 8)
    df_day[f"{name}_Slow"] = ema(base_signal, 21)
    df_day[f"{name}_Crossover"] = df_day[f"{name}_Fast"] - df_day[f"{name}_Slow"]
    
    print(f"  → {name}: Fast(8) & Slow(21) crossovers calculated")

print("\n✅ All crossover systems ready!")


# ================================================================
# 7. ⭐ APPLY HMA, ZLEMA, RSI ON KALMAN SMOOTHED DATA
# ================================================================

print("\n🎯 LAYER 3: Applying HMA, ZLEMA, RSI on Kalman-Smoothed Data...")

# Use Kalman as the base (best noise reduction)
kalman_smoothed = df_day["Base_Kalman"].values

# HMA Crossover (Primary Signal)
print("  → HMA Fast (9) & Slow (21)...")
df_day["HMA_Fast_9"] = hma(kalman_smoothed, 9)
df_day["HMA_Slow_21"] = hma(kalman_smoothed, 21)
df_day["HMA_Crossover"] = df_day["HMA_Fast_9"] - df_day["HMA_Slow_21"]

# ZLEMA Crossover (Confirmation)
print("  → ZLEMA Fast (8) & Slow (18)...")
df_day["ZLEMA_Fast_8"] = zlema(kalman_smoothed, 8)
df_day["ZLEMA_Slow_18"] = zlema(kalman_smoothed, 18)
df_day["ZLEMA_Crossover"] = df_day["ZLEMA_Fast_8"] - df_day["ZLEMA_Slow_18"]

# RSI on Kalman smoothed (Range Filter)
print("  → RSI (14) on Kalman smoothed...")
df_day["RSI_14"] = calculate_rsi(kalman_smoothed, 14)

print("  → LaGuerre RSI (γ=0.7)...")
df_day["LaGuerre_RSI"] = laguerre_rsi(kalman_smoothed, gamma=0.7)

# Now we can normalize them
df_day["RSI_14_norm"] = 95 + (df_day["RSI_14"] / 100.0) * 10.0
df_day["LaGuerre_RSI_norm"] = 95 + (df_day["LaGuerre_RSI"] * 10.0)

print("\n✅ HMA, ZLEMA, RSI calculated on Kalman-smoothed data!")

# ================================================================
# ⭐ IMPORTED INDICATORS FROM FEATURE ANALYZER (your other script)
# ================================================================

print("\n🔵 Adding Advanced Indicators from Secondary Script...")

prices = df_day["Price"].values
n = len(prices)

# ---------- 1) Ehlers SuperSmoother Slow ----------
period = 1800
pi = math.pi
a1 = math.exp(-1.414 * pi / period)
b1 = 2 * a1 * math.cos(1.414 * pi / period)
c1 = 1 - b1 + a1*a1

ss_slow = np.zeros(n)
ss_slow[0] = prices[0]
ss_slow[1] = prices[1]

c2 = b1
c3 = -a1*a1

for i in range(2, n):
    ss_slow[i] = c1*prices[i] + c2*ss_slow[i-1] + c3*ss_slow[i-2]

df_day["EhlersSuperSmoother_Slow"] = ss_slow

# ---------- 2) Ehlers SuperSmoother Fast ----------
period = 30
a1 = math.exp(-1.414 * pi / period)
b1 = 2 * a1 * math.cos(1.414 * pi / period)
c1 = 1 - b1 + a1*a1

ss = np.zeros(n)
ss[0] = prices[0]
ss[1] = prices[1]

c2 = b1
c3 = -a1*a1

for i in range(2, n):
    ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]

df_day["EhlersSuperSmoother"] = ss

# ---------- 3) EAMA ----------
eama_period = 10
eama_fast = 15
eama_slow = 40

direction_ss = pd.Series(ss).diff(eama_period).abs()
volatility_ss = pd.Series(ss).diff(1).abs().rolling(eama_period).sum()
er_ss = (direction_ss / volatility_ss).fillna(0).values

fast_sc = 2/(eama_fast+1)
slow_sc = 2/(eama_slow+1)
sc_ss = ((er_ss*(fast_sc - slow_sc)) + slow_sc)**2

eama = np.zeros(n)
eama[0] = ss[0]
for i in range(1, n):
    eama[i] = eama[i-1] + sc_ss[i] * (ss[i] - eama[i-1])

df_day["EAMA"] = eama
df_day["EAMA_Slope"] = pd.Series(eama).diff().fillna(0)
df_day["EAMA_Slope_MA"] = pd.Series(eama).rolling(5).mean().fillna(0)

# ---------- 5) HAMA ----------
close_vals = prices
open_vals = np.roll(close_vals, 1)
open_vals[0] = close_vals[0]

high_vals = df_day[price_col].rolling(5).max().values
low_vals = df_day[price_col].rolling(5).min().values

ha_open = np.zeros(n)
ha_close = np.zeros(n)
ha_high = np.zeros(n)
ha_low = np.zeros(n)

ha_open[0] = open_vals[0]
ha_close[0] = (open_vals[0]+high_vals[0]+low_vals[0]+close_vals[0]) / 4
ha_high[0] = high_vals[0]
ha_low[0] = low_vals[0]

for i in range(1, n):
    ha_close[i] = (open_vals[i]+high_vals[i]+low_vals[i]+close_vals[i]) / 4
    ha_open[i]  = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high[i]  = max(high_vals[i], ha_open[i], ha_close[i])
    ha_low[i]   = min(low_vals[i], ha_open[i], ha_close[i])

df_day["HAMA"] = pd.Series(ha_close).rolling(180).mean().bfill().values

# ---------- 6) EKFTrend ----------
ekf_vals = []
ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
ekf.x = np.array([[prices[0]]])
ekf.F = np.array([[1.0]])
ekf.Q = np.array([[0.05]])
ekf.R = np.array([[0.2]])

def h(x): return np.log(x)
def H_jac(x): return np.array([[1.0/x[0][0]]])

for p in prices:
    ekf.predict()
    ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
    ekf_vals.append(ekf.x.item())

df_day["EKFTrend"] = pd.Series(ekf_vals).shift(1).fillna(method='bfill')

# ---------- 7) MACD ----------
ema_12 = df_day["Price"].ewm(span=12*60).mean()
ema_26 = df_day["Price"].ewm(span=26*60).mean()

df_day["MACD_Line"] = ema_12 - ema_26
df_day["MACD_Signal"] = df_day["MACD_Line"].ewm(span=9*60).mean()
df_day["MACD_Hist"] = df_day["MACD_Line"] - df_day["MACD_Signal"]

# ---------- 8) ATR ----------
df_day["ATR"] = df_day["Price"].diff().abs().ewm(60).mean()

# ---------- 9) Slow/Fast MA ----------
df_day["Slow_MA"] = df_day["Price"].rolling(1800).mean()
df_day["Fast_MA"] = df_day["Price"].rolling(180).mean()

# ---------- 10) EMA_Fast / EMA_Slow ----------
df_day["EMA_Fast"] = df_day["Price"].ewm(span=25).mean()
df_day["EMA_Slow"] = df_day["Price"].ewm(span=1800).mean()

# ---------- 11) ER ----------
w = 30
er = np.full(n, np.nan)
for i in range(w, n):
    win = prices[i-w:i+1]
    denom = np.sum(np.abs(np.diff(win)))
    er[i] = abs(win[-1]-win[0]) / denom if denom != 0 else 0

df_day["ER"] = er

# ---------- 12) BC ----------
period_bc = 600
sk = df_day["Price"].rolling(period_bc).skew()
ku = df_day["Price"].rolling(period_bc).kurt() + 3
corr = (3*(period_bc-1)**2) / ((period_bc-2)*(period_bc-3))
df_day["BC"] = (sk**2 + 1) / (ku + corr)

# ---------- 13) FFT ----------
df_day["FFT"] = rolling_fft_reconstruction(df_day["Price"], 120, 3)

# ---------- 14) KAMA ----------
P2_KAMA_PERIOD = 15
P2_KAMA_FAST = 30
P2_KAMA_SLOW = 180

direction = df_day["Price"].diff(P2_KAMA_PERIOD).abs()
vol = df_day["Price"].diff().abs().rolling(P2_KAMA_PERIOD).sum()
sc = (((direction/vol).fillna(0)) * (2/(P2_KAMA_FAST+1) - 2/(P2_KAMA_SLOW+1)) + (2/(P2_KAMA_SLOW+1)))**2

kama = np.zeros(n)
kama[0] = prices[0]
sc_vals = sc.values

for i in range(1, n):
    kama[i] = kama[i-1] + sc_vals[i] * (prices[i] - kama[i-1])

df_day["Kama"] = kama

# ---------- 15) HAM ----------
er_vals = df_day["ER"].fillna(0).values
er_smooth = pd.Series(er_vals).ewm(span=50).mean().values
er_norm = np.clip(er_smooth, 0.0, 1.0)

ss_fast = df_day["EhlersSuperSmoother"].values
kama_slow = df_day["Kama"].values

ham = er_norm*ss_fast + (1-er_norm)*kama_slow

df_day["HAM"] = ham
df_day["HAM_MA"] = pd.Series(ham).rolling(30).mean().fillna(0)
df_day["HAM_EMA"] = pd.Series(ham).ewm(span=30).mean().fillna(0)
df_day["HAM_Slope"] = pd.Series(ham).diff().fillna(0)

# ---------- 16) HAMF ----------
fft_slow = df_day["FFT"].values
hamf = er_norm*ss_fast + (1-er_norm)*fft_slow

df_day["HAMF"] = hamf
df_day["HAMF_MA"] = pd.Series(hamf).rolling(30).mean().fillna(0)
df_day["HAMF_EMA"] = pd.Series(hamf).ewm(span=30).mean().fillna(0)
df_day["HAMF_Slope"] = pd.Series(hamf).diff().fillna(0)

# ---------- 17) UCM ----------
alpha_ucm, beta_ucm = 0.005, 0.0001
mu = np.zeros(n)
beta_slope = np.zeros(n)
filt = np.zeros(n)

mu[0] = prices[0]

for i in range(1, n):
    mu_pred = mu[i-1] + beta_slope[i-1]
    mu[i] = mu_pred + alpha_ucm*(prices[i] - mu_pred)
    beta_slope[i] = beta_slope[i-1] + beta_ucm*(prices[i] - mu_pred)
    filt[i] = mu[i] + beta_slope[i]

df_day["UCM"] = filt

print("✅ All Advanced Indicators Added!")


# ================================================================
# COLORS FOR INDICATORS (needed for single-panel plot)
# ================================================================

color_map = {
    "Base_Kalman": "darkblue",
    "Base_Supersmoother": "red",
    "Base_AEMA": "green",
    "Base_Ehlers": "orange",
    "Base_DoubleT3": "purple",
    "Base_Roofing40_20": "brown",
    "Base_Roofing20_10": "saddlebrown",
    "Base_T3_5": "magenta",
    "Base_T3_15": "gray",
    "Base_Super30": "teal"
}

sma_colors = [
    'blue', 'cyan', 'navy', 'skyblue', 
    'steelblue', 'dodgerblue', 'royalblue', 'midnightblue'
]

ema_colors = [
    'red', 'orange', 'coral', 'tomato', 
    'orangered', 'crimson', 'darkred', 'firebrick'
]

fft_colors = {
    60: "gold",
    120: "darkorange",
    180: "salmon"
}



# ================================================================
# ⭐ NEW: SINGLE PLOT — PRICE PANEL WITH ALL INDICATORS
# ================================================================

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.70, 0.30],
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=(
        f"Day {day_to_plot} — Price + All Filters & Indicators",
        "RSI Panel"
    ),
    specs=[
        [{"secondary_y": True}],
        [{"secondary_y": False}]
    ]
)



# ================================================================
# 1️⃣ RAW PRICE — ALWAYS VISIBLE
# ================================================================
fig.add_trace(go.Scatter(
    x=df_day["Index"], y=prices,
    name="Price (Raw)",
    line=dict(color="black", width=2.5)
), row=1, col=1)

# ================================================================
# 📌 ADD HMA-9 PLOT (Always Visible)
# ================================================================
fig.add_trace(go.Scatter(
    x=df_day["Index"],
    y=df_day["HMA_Fast_9"],
    name="HMA-9",
    line=dict(color="purple", width=2.5),
    visible=True
), row=1, col=1)





# ================================================================
# 2️⃣ EXISTING FILTERS (Kalman, HMA, ZLEMA…) 
#     — KEEP SAME COLOR / BEHAVIOR
# ================================================================

# Base smoothing filters (legend-only)
for col, color in color_map.items():
    if col in df_day:
        fig.add_trace(go.Scatter(
            x=df_day["Index"], y=df_day[col],
            name=col,
            line=dict(color=color, width=2, dash="dot"),
            visible="legendonly"
        ), row=1, col=1)

# SMA (hidden by default)
for i, period in enumerate(SMA_PERIODS):
    fig.add_trace(go.Scatter(
        x=df_day["Index"],
        y=df_day[f"SMA_{period}"],
        name=f"SMA({period})",
        line=dict(color=sma_colors[i % len(sma_colors)], width=1.5, dash="dash"),
        visible="legendonly"
    ), row=1, col=1)



# EMA (hidden by default)
for i, period in enumerate(EMA_PERIODS):
    fig.add_trace(go.Scatter(
        x=df_day["Index"],
        y=df_day[f"EMA_{period}"],
        name=f"EMA({period})",
        line=dict(color=ema_colors[i % len(ema_colors)], width=1.5),
        visible="legendonly"
    ), row=1, col=1)

# ================================================================
# 3️⃣ ADD FFT Reconstruction (legend-only)
# ================================================================
for w, colr in fft_colors.items():
    fig.add_trace(go.Scatter(
        x=df_day["Index"], y=df_day[f"FFT_Recon_{w}"],
        name=f"FFT_Recon_{w}",
        line=dict(color=colr, width=2),
        visible="legendonly"
    ), row=1, col=1)

# ================================================================
# 4️⃣ ALWAYS VISIBLE CROSSOVERS (HMA, ZLEMA)
# ================================================================
fig.add_trace(go.Scatter(
    x=df_day["Index"], 
    y=df_day["HMA_Fast_9"],
    name="HMA Fast (9)",
    line=dict(color="lime", width=2.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_day["Index"], 
    y=df_day["HMA_Slow_21"],
    name="HMA Slow (21)",
    line=dict(color="darkgreen", width=2.5, dash="dash")
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_day["Index"], 
    y=df_day["ZLEMA_Fast_8"],
    name="ZLEMA Fast (8)",
    line=dict(color="red", width=2.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_day["Index"], 
    y=df_day["ZLEMA_Slow_18"],
    name="ZLEMA Slow (18)",
    line=dict(color="darkred", width=2.5, dash="dash")
), row=1, col=1)


# ================================================================
# ⭐ ADD Z-SCORE (RIGHT AXIS)
# ================================================================
fig.add_trace(
    go.Scatter(
        x=df_day["Index"],
        y=df_day["ZScore_Final"],
        name="ZScore (Sigmoid, 1800)",
        line=dict(color="red", width=2)
    ),
    row=1, col=1,
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=df_day["Index"],
        y=df_day["Pairs_ZScore"],
        name="Pairs ZScore (Kalman β Spread)",
        line=dict(color="blue", width=2)
    ),
    row=1, col=1,
    secondary_y=True
)




# ================================================================
# 5️⃣ ⭐ ADD ALL SCRIPT-1 INDICATORS (23 FEATURES) 
#     → ADDED AS legend-only traces
# ================================================================
features_to_add = [
    "ATR", "Kama", "ER", "BC", "EMA_Fast", "EMA_Slow", "ZLEMA_Fast_8",
    "HAM", "HAM_Slope", "Slow_MA", "Fast_MA", "UCM", "EAMA", 
    "EAMA_Slope", "EAMA_Slope_MA", "FFT", "EKFTrend",
    "EhlersSuperSmoother", "HAMA", "MACD_Hist",
    "EhlersSuperSmoother_Slow",
    "LaGuerre_RSI", "RSI_14_norm", "LaGuerre_RSI_norm"
]


extra_colors = [
    # original colors
    "blue","orange","green","red","purple","brown","pink","gray","olive","cyan",
    "gold","magenta","teal","indigo","maroon","darkblue","darkcyan","darkorange",
    "navy","darkgreen","lightcoral","darkgrey","slateblue",

    # extra colors to avoid IndexError
    "forestgreen","deeppink","chocolate","darkviolet","mediumvioletred",
    "turquoise","steelblue","limegreen","darkgoldenrod","firebrick"
]


for i, col in enumerate(features_to_add):
    if col in df_day.columns:
        fig.add_trace(go.Scatter(
            x=df_day["Index"],
            y=df_day[col],
            name=col,
            line=dict(color=extra_colors[i], width=1.8),
            visible="legendonly"
        ), row=1, col=1)

# ================================================================
#  📉  ADD RSI SUBPLOT (ROW 2)
# ================================================================

# Clip raw RSI values to proper ranges before normalization (safety)
df_day["RSI_14"] = df_day["RSI_14"].clip(0, 100)
df_day["LaGuerre_RSI"] = df_day["LaGuerre_RSI"].clip(0, 1)

# Add normalized RSI(14)
fig.add_trace(go.Scatter(
    x=df_day["Index"],
    y=df_day["RSI_14_norm"],
    name="RSI14 (Norm 95–105)",
    line=dict(color="blue", width=2)
), row=2, col=1)

# Add normalized LaGuerre RSI
fig.add_trace(go.Scatter(
    x=df_day["Index"],
    y=df_day["LaGuerre_RSI_norm"],
    name="LaGuerre RSI (Norm 95–105)",
    line=dict(color="red", width=2)
), row=2, col=1)

# Raw RSI (hidden by default)
fig.add_trace(go.Scatter(
    x=df_day["Index"],
    y=df_day["RSI_14"],
    name="RSI14 Raw",
    line=dict(color="cyan", width=1, dash="dot"),
    visible="legendonly"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=df_day["Index"],
    y=df_day["LaGuerre_RSI"],
    name="LaGuerre RSI Raw",
    line=dict(color="orange", width=1, dash="dot"),
    visible="legendonly"
), row=2, col=1)



# ================================================================
# 6️⃣ Layout — SINGLE PANEL ONLY
# ================================================================
fig.update_xaxes(showgrid=True, gridcolor="lightgray", title_text="Time Index")
fig.update_yaxes(showgrid=True, gridcolor="lightgray", title_text="Price")

fig.update_layout(
    title={
        'text': f"📈 Unified Price Panel — All Indicators | Day {day_to_plot}",
        'x': 0.5,
        'xanchor': 'center'
    },
    width=2000,
    height=1000,
    plot_bgcolor="white",
    hovermode="x unified",
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.01,
        bgcolor="rgba(255,255,255,0.9)"
    )
)

fig.update_yaxes(
    title_text="Price", 
    row=1, col=1, 
    secondary_y=False
)

fig.update_yaxes(
    title_text="Z-Score (−1 to +1)",
    row=1, col=1,
    secondary_y=True,
    range=[-1, 1]
)


# # Add RIGHT AXIS after main layout
# fig.update_layout(
#     yaxis2=dict(
#         title="Z-Score (−1 to +1)",
#         overlaying="y",
#         side="right",
#         range=[-1, 1]
#     )
# )


output_file = f"PRICE_ONLY_ALL_INDICATORS_DAY_{day_to_plot}.html"
fig.write_html(output_file)
fig.show()
print(f"Saved to {output_file}")
