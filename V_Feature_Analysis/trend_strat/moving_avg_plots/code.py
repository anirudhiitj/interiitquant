import os
import glob
import re
import math
import sys
import warnings
import numpy as np
import pandas as pd
import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from filterpy.kalman import ExtendedKalmanFilter

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def extract_day_num(filepath):
    """Extracts day number from filename like 'day103.parquet'"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

# ---------------------------------------------------------
# Core Analysis & Plotting Logic
# ---------------------------------------------------------

def analyze_and_save_day(data_df, day_num, output_dir, selected_features=None):
    
    # 1. Data Conversion (GPU -> CPU)
    if isinstance(data_df, dask_cudf.DataFrame):
        data_df = data_df.compute().to_pandas()
    elif isinstance(data_df, cudf.DataFrame):
        data_df = data_df.to_pandas()

    if data_df.empty:
        print(f"Day {day_num}: DataFrame is empty.")
        return

    # 2. Preprocessing
    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    
    # Create artificial 'Day' column if not present
    if "Day" not in data_df.columns:
        data_df["Day"] = day_num

    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    price_vals = data_df["Price"].values
    n = len(price_vals)

    print(f"  > Calculating Indicators for Day {day_num} ({n} rows)...")

    # 3. Indicator Calculation
    
    # --- Ehlers Super Smoother ---
    period = 120
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    ss_slow = np.zeros(n)
    if n > 0: ss_slow[0] = price_vals[0]
    if n > 1: ss_slow[1] = price_vals[1]
    
    c2 = b1
    c3 = -a1*a1
    for i in range(2, n):
        ss_slow[i] = c1*price_vals[i] + c2*ss_slow[i-1] + c3*ss_slow[i-2]
    data_df["EhlersSuperSmoother_Slow"] = ss_slow
    
    # Standard Super Smoother
    period = 25
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    ss = np.zeros(n)
    if n > 0: ss[0] = price_vals[0]
    if n > 1: ss[1] = price_vals[1]
    c2 = b1
    c3 = -a1*a1
    for i in range(2, n):
        ss[i] = c1*price_vals[i] + c2*ss[i-1] + c3*ss[i-2]
    data_df["EhlersSuperSmoother"] = ss

    # --- EAMA ---
    eama_period = 15
    eama_fast = 30
    eama_slow = 120
    ss_series = pd.Series(ss)
    direction_ss = ss_series.diff(eama_period).abs()
    volatility_ss = ss_series.diff(1).abs().rolling(eama_period).sum()
    er_ss = (direction_ss / volatility_ss).fillna(0).values

    fast_sc = 2 / (eama_fast + 1)
    slow_sc = 2 / (eama_slow + 1)
    sc_ss = ((er_ss * (fast_sc - slow_sc)) + slow_sc) ** 2

    eama = np.zeros(n)
    if n > 0: eama[0] = ss[0]
    for i in range(1, n):
        c = sc_ss[i] if not np.isnan(sc_ss[i]) else 0
        eama[i] = eama[i-1] + c * (ss[i] - eama[i-1])

    data_df["EAMA"] = eama
    data_df["EAMA_Slope"] = pd.Series(eama).diff().fillna(0)
    data_df["EAMA_Slope_MA"] = data_df["EAMA_Slope"].rolling(window=5, min_periods=1).mean()

    # --- JMA ---
    length = 180
    alpha = 2.0 / (length + 1.0)
    prices = data_df["Price"].values
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(window=length, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(window=length, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    data_df["JMA"] = out

    # --- HAMA ---
    close_vals = data_df["Price"].values
    open_vals = np.roll(close_vals, 1)
    open_vals[0] = close_vals[0]
    
    if "High" in data_df.columns and "Low" in data_df.columns:
        high_vals = data_df["High"].values
        low_vals = data_df["Low"].values
    else:
        high_vals = data_df["Price"].rolling(5, min_periods=1).max().values
        low_vals = data_df["Price"].rolling(5, min_periods=1).min().values

    ha_open = np.zeros(n)
    ha_close = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    ha_open[0] = open_vals[0]
    ha_close[0] = (open_vals[0] + high_vals[0] + low_vals[0] + close_vals[0]) / 4
    ha_high[0] = high_vals[0]
    ha_low[0] = low_vals[0]

    for i in range(1, n):
        ha_close[i] = (open_vals[i] + high_vals[i] + low_vals[i] + close_vals[i]) / 4
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(high_vals[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_vals[i], ha_open[i], ha_close[i])

    data_df["HAMA"] = pd.Series(ha_close).rolling(180).mean().bfill().values

    # --- EKF ---
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[data_df["Price"].iloc[0]]])
    ekf.F = np.array([[1.0]])
    ekf.Q = np.array([[0.05]])
    ekf.R = np.array([[0.2]])
    def h(x): return np.log(x)
    def H_jac(x): return np.array([[1.0/x[0][0]]])
    
    vals = []
    for p in data_df["Price"].dropna().values:
        ekf.predict()
        ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
        vals.append(ekf.x.item())
    data_df["EKFTrend"] = pd.Series(vals, index=data_df["Price"].dropna().index).ffill().shift(1)

    # --- MACD ---
    ema_12 = data_df['Price'].ewm(span=12*60, adjust=False).mean()
    ema_26 = data_df['Price'].ewm(span=26*60, adjust=False).mean()
    data_df['MACD_Line'] = ema_12 - ema_26
    data_df['MACD_Signal'] = data_df['MACD_Line'].ewm(span=9*60, adjust=False).mean()
    data_df['MACD_Hist'] = data_df['MACD_Line'] - data_df['MACD_Signal']

    # --- Other Indicators ---
    data_df['ATR'] = data_df['Price'].diff().abs().ewm(60).mean()
    data_df['Slowest_MA'] = data_df['Price'].rolling(1800).mean()
    data_df['Slow_MA'] = data_df['Price'].rolling(1200).mean()
    data_df['Fast_MA'] = data_df['Price'].rolling(180).mean()
    data_df['EMA_Slow'] = data_df['Price'].ewm(span=180, adjust=False).mean()
    data_df['EMA_Fast'] = data_df['Price'].ewm(span=25, adjust=False).mean()

    zlema_period = 25
    zlema_lag = int((zlema_period - 1) / 2)
    zlema_data = 2 * data_df['Price'] - data_df['Price'].shift(zlema_lag)
    data_df['ZLEMA'] = zlema_data.ewm(span=zlema_period, adjust=False).mean()
    
    # Efficiency Ratio (ER)
    w = 15
    er = np.full(n, np.nan)
    for i in range(w, n):
        win = price_vals[i-w:i+1]
        denom = np.sum(np.abs(np.diff(win)))
        er[i] = abs(win[-1] - win[0]) / denom if denom != 0 else 0
    data_df["ER"] = er

    # Bias Criterion (BC)
    period_bc = 600
    sk = data_df["Price"].rolling(period_bc).skew()
    ku = data_df["Price"].rolling(period_bc).kurt() + 3
    corr = (3*(period_bc-1)**2) / ((period_bc-2)*(period_bc-3))
    data_df["BC"] = (sk**2 + 1) / (ku + corr)

    # --- KAMA & Regime Detection ---
    P2_KAMA_PERIOD = 15
    P2_KAMA_FAST = 25
    P2_KAMA_SLOW = 100
    
    direction = data_df['Price'].diff(P2_KAMA_PERIOD).abs()
    volatility = data_df['Price'].diff(1).abs().rolling(window=P2_KAMA_PERIOD).sum()
    sc = (((direction/volatility).fillna(0) * (2/(P2_KAMA_FAST+1) - 2/(P2_KAMA_SLOW+1))) + (2/(P2_KAMA_SLOW+1)))**2
    
    sc_vals = sc.values
    kama_vals = np.zeros_like(price_vals)
    kama_vals[0] = price_vals[0]
    for i in range(1, len(price_vals)):
        sc_i = sc_vals[i] if not np.isnan(sc_vals[i]) else 0
        kama_vals[i] = kama_vals[i-1] + sc_i * (price_vals[i] - kama_vals[i-1])
    data_df['Kama'] = kama_vals

    sc_series = pd.Series(sc_vals).fillna(0)
    sc_threshold_rolling = sc_series.rolling(window=1800, min_periods=1).median()
    data_df["KAMA_Mode"] = (sc_series > sc_threshold_rolling).astype(int)
    data_df["KAMA_Switch"] = data_df["KAMA_Mode"].diff().abs().fillna(0)

    # --- UCM ---
    alpha_ucm, beta_ucm = 0.005, 0.0001
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    fvi = data_df['Price'].first_valid_index()
    if fvi is not None:
        mu[fvi] = data_df['Price'].loc[fvi]
        filtered[fvi] = mu[fvi]
        for t in range(fvi+1, n):
            if np.isnan(data_df['Price'].iloc[t]):
                mu[t], beta_slope[t], filtered[t] = mu[t-1], beta_slope[t-1], filtered[t-1]
                continue
            mu_pred = mu[t-1] + beta_slope[t-1]
            mu[t] = mu_pred + alpha_ucm * (data_df['Price'].iloc[t] - mu_pred)
            beta_slope[t] = beta_slope[t-1] + beta_ucm * (data_df['Price'].iloc[t] - mu_pred)
            filtered[t] = mu[t] + beta_slope[t]
    data_df["UCM"] = filtered

    # 4. Plotting
    print(f"  > Generating Plot for Day {day_num}...")
    
    feature_columns = [c for c in data_df.columns if c not in ['Time','Price','Day','PlotIndex','KAMA_Mode','KAMA_Switch']]
    if selected_features:
        feature_columns = [c for c in feature_columns if c in selected_features]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Price (Primary Y)
    ts = data_df["Time"].apply(lambda x: str(x).split()[-1]).tolist()
    fig.add_trace(go.Scatter(
        x=data_df["PlotIndex"], y=data_df["Price"], mode='lines',
        name=f"Price Day {day_num}",
        line=dict(width=2, color='rgba(0,0,0,0.7)'),
        hovertemplate="<b>Day</b>: %{customdata[0]}<br><b>Time</b>: %{customdata[1]}<br><b>Price</b>: %{y:.4f}<extra></extra>",
        customdata=[[day_num, t] for t in ts],
        showlegend=False
    ), secondary_y=False)

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    
    # Plot Indicators (Secondary Y)
    for i, col in enumerate(feature_columns):
        norm_col = f"{col}_viz"
        data_df[norm_col] = data_df[col] 
        
        mask = data_df[norm_col].notna()
        if mask.sum() == 0: continue
        
        if 'Hist' in col:
            fig.add_trace(go.Bar(
                x=data_df.loc[mask, "PlotIndex"], 
                y=data_df.loc[mask, norm_col],
                name=col, 
                legendgroup=col, 
                visible='legendonly',
                marker_color=colors[i % len(colors)],
                opacity=0.7
            ), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(
                x=data_df.loc[mask, "PlotIndex"], 
                y=data_df.loc[mask, norm_col],
                mode='lines', 
                name=col, 
                legendgroup=col, 
                visible='legendonly',
                line=dict(color=colors[i % len(colors)], width=1.5),
            ), secondary_y=True)

    # Plot Regime Switches (KAMA)
    switches = data_df[data_df["KAMA_Switch"] == 1]
    if not switches.empty:
        fig.add_trace(go.Scatter(
            x=switches["PlotIndex"], y=switches["Price"], mode="markers",
            marker=dict(size=18, color="purple", symbol="star", line=dict(color="black", width=1)),
            name="Regime Switch"
        ), secondary_y=False)

    # Regime Background Shading
    df_plot = data_df[["PlotIndex", "KAMA_Mode"]].copy()
    df_plot['change'] = df_plot['KAMA_Mode'].diff().fillna(0)
    starts = df_plot[(df_plot['change'] == 1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == 0))]['PlotIndex'].values
    ends = df_plot[(df_plot['change'] == -1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == len(df_plot)-1))]['PlotIndex'].values

    if len(starts) > 0 and len(ends) > 0:
        if starts[0] > ends[0]: starts = np.insert(starts, 0, df_plot['PlotIndex'].iloc[0])
        if len(starts) > len(ends): ends = np.append(ends, df_plot['PlotIndex'].iloc[-1])

    shapes = [dict(type="rect", xref="x", yref="paper", x0=s, y0=0, x1=e, y1=1, fillcolor="rgba(0,255,0,0.1)", line_width=0, layer="below") for s, e in zip(starts, ends)]
    
    # Layout and Save
    fig.update_layout(
        title=f'Analysis: Day {day_num}', 
        hovermode='x unified', 
        height=700, 
        width=1600, 
        plot_bgcolor='white',
        shapes=shapes
    )
    
    save_path = os.path.join(output_dir, f"day_{day_num}_analysis.html")
    print(f"  > Saving HTML to {save_path}...")
    fig.write_html(save_path)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'DATA_DIR': '/data/quant14/EBX/',
        'OUTPUT_DIR': '/home/raid/Quant14/V_Feature_Analysis/trend_strat/moving_avg_plots', # <--- Output Directory
        'TIME_COLUMN': 'Time',
        'PRICE_COLUMN': 'Price'
    }

    # Features to plot (Removed Z-Score, Z-Sigmoid, ZurichMA)
    FEATURES = [
        'ATR', 'Kama', 'ER', 'BC', 'EMA_Fast', 'EMA_Slow', 'ZLEMA',
        'Slow_MA', 'Fast_MA', 'UCM', 'EAMA', 'EAMA_Slope', 'EAMA_Slope_MA',
        'EKFTrend', 'EhlersSuperSmoother', 'JMA', 'HAMA', 'MACD_Hist', 'EhlersSuperSmoother_Slow'
    ]

    # Create output directory
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # Find files
    files_pattern = os.path.join(CONFIG['DATA_DIR'], "day*.parquet")
    all_files = glob.glob(files_pattern)
    
    # Map day number to file path
    day_files = {extract_day_num(f): f for f in all_files if extract_day_num(f) != -1}
    
    if not day_files:
        print(f"No parquet files found in {CONFIG['DATA_DIR']}")
        sys.exit(1)

    # Define which days to process
    TARGET_DAYS = [49, 170, 382, 118, 265, 483] 

    if not TARGET_DAYS:
        TARGET_DAYS = sorted(day_files.keys())

    print(f"Found {len(day_files)} total files. Processing {len(TARGET_DAYS)} selected days.")
    print(f"Outputs will be saved to: {CONFIG['OUTPUT_DIR']}")

    for day_num in TARGET_DAYS:
        if day_num not in day_files:
            print(f"Warning: File for Day {day_num} not found. Skipping.")
            continue
            
        file_path = day_files[day_num]
        print(f"\n[{day_num}] Reading {file_path}...")
        
        try:
            # Load Data using Dask/CuDF
            required_cols = [CONFIG['TIME_COLUMN'], CONFIG['PRICE_COLUMN']]
            ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
            gdf = ddf.compute()
            
            # Run Analysis
            analyze_and_save_day(
                gdf, 
                day_num=day_num, 
                output_dir=CONFIG['OUTPUT_DIR'],
                selected_features=FEATURES
            )
            
        except Exception as e:
            print(f"Error processing day {day_num}: {e}")
            import traceback
            traceback.print_exc()