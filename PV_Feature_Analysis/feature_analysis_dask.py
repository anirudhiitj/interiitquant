import dask_cudf 
import cudf 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots 
import warnings 
import pandas as pd 
import numpy as np 
from filterpy.kalman import ExtendedKalmanFilter 
from filter import get_pairs_for_day 
import math 
import sys 

# Move warnings to global scope or keep inside if preferred 
warnings.filterwarnings('ignore') 

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

def analyze_feature(data_df, price_pairs=None, selected_features=None): 
     
    # 1. Data Conversion 
    if isinstance(data_df, dask_cudf.DataFrame): 
        data_df = data_df.compute().to_pandas() 
    elif isinstance(data_df, cudf.DataFrame): 
        data_df = data_df.to_pandas() 

    # 2. Preprocessing 
    data_df["Time"] = pd.to_timedelta(data_df["Time"]) 
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum() 
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True) 
    data_df["PlotIndex"] = range(len(data_df)) 

    price_vals = data_df["Price"].values 
    n = len(price_vals) 

    # 3. Indicator Calculation 
    
    # --- Ehlers Super Smoother --- 
    period = 1800 
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

    period = 30 
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

    # --- EAMA (Ehlers Adaptive Moving Average) --- 
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

    # --- FTAMA (Fourier Transform Adaptive Moving Average) ---
    # 1. Generate FFT Series (The "Smoothed" Input)
    fft_window = 120
    fft_vals = rolling_fft_reconstruction(data_df["Price"], window=fft_window, top_k=3)
    data_df["FFT"] = fft_vals # Saving raw FFT for reference

    # 2. Calculate Efficiency Ratio (ER) on the FFT Series
    ftama_period = 15
    ftama_fast = 30
    ftama_slow = 120
    
    fft_s = pd.Series(fft_vals)
    direction_ft = fft_s.diff(ftama_period).abs()
    volatility_ft = fft_s.diff(1).abs().rolling(ftama_period).sum()
    er_ft = (direction_ft / volatility_ft).fillna(0).values

    # 3. Calculate Smoothing Constant (SC)
    fast_sc_ft = 2 / (ftama_fast + 1)
    slow_sc_ft = 2 / (ftama_slow + 1)
    sc_ft = ((er_ft * (fast_sc_ft - slow_sc_ft)) + slow_sc_ft) ** 2

    # 4. Apply KAMA Recursive Formula to the FFT Series
    ftama = np.full(n, np.nan)
    
    # Initialization: We need valid FFT data to start. 
    # FFT data starts at index `fft_window`.
    start_idx = fft_window
    
    if n > start_idx:
        ftama[start_idx] = fft_vals[start_idx] # Initialize with first valid FFT value
        for i in range(start_idx + 1, n):
            c = sc_ft[i] if not np.isnan(sc_ft[i]) else 0
            # Target is the current FFT value
            target = fft_vals[i]
            if not np.isnan(target):
                ftama[i] = ftama[i-1] + c * (target - ftama[i-1])
            else:
                ftama[i] = ftama[i-1]
                
    data_df["FTAMA"] = ftama

    # --- JMA --- 
    length = 1800 
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

    data_df["HA_Open"] = ha_open 
    data_df["HA_Close"] = ha_close 
    data_df["HA_High"] = ha_high 
    data_df["HA_Low"] = ha_low 
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
    period = 45 
    tr = data_df['Price'].diff().abs() 
    atr = tr.ewm(span=period).mean() 
    data_df['ATR'] = data_df['Price'].diff().abs().ewm(60).mean() 
    data_df['Slowest_MA'] = data_df['Price'].rolling(1800).mean() 
    data_df['Slow_MA'] = data_df['Price'].rolling(1800).mean() 
    data_df['Fast_MA'] = data_df['Price'].rolling(180).mean() 
    data_df['STD'] = data_df['Price'].rolling(20).std() 
    data_df['EMA_Slow'] = data_df['Price'].ewm(span=1800, adjust=False).mean() 
    data_df['EMA_Fast'] = data_df['Price'].ewm(span=25, adjust=False).mean() 
    # data_df["FFT"] already computed in FTAMA section above

    zlema_period = 1800 
    zlema_lag = int((zlema_period - 1) / 2) 
    zlema_data = 2 * data_df['Price'] - data_df['Price'].shift(zlema_lag) 
    data_df['ZLEMA'] = zlema_data.ewm(span=zlema_period, adjust=False).mean() 
     
    w = 30 
    er = np.full(n, np.nan) 
    for i in range(w, n): 
        win = price_vals[i-w:i+1] 
        denom = np.sum(np.abs(np.diff(win))) 
        er[i] = abs(win[-1] - win[0]) / denom if denom != 0 else 0 
    data_df["ER"] = er 

    period_bc = 600 
    sk = data_df["Price"].rolling(period_bc).skew() 
    ku = data_df["Price"].rolling(period_bc).kurt() + 3 
    corr = (3*(period_bc-1)**2) / ((period_bc-2)*(period_bc-3)) 
    data_df["BC"] = (sk**2 + 1) / (ku + corr) 

    period = 900 
    rets = data_df["Price"].pct_change().values 
    tsmom_lin = np.full(len(rets), np.nan) 
    w_vec = np.arange(1, period+1) 
    wsum = w_vec.sum() 
    for i in range(period, len(rets)): 
        r = rets[i-period+1:i+1] 
        tsmom_lin[i] = np.sum(w_vec * r) / wsum 
    data_df["TSM_Lin"] = tsmom_lin     

    # --- KAMA & Regime Detection --- 
    P2_KAMA_PERIOD = 15 
    P2_KAMA_FAST = 30 
    P2_KAMA_SLOW = 180 
     
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

    # --- HAM --- 
    er_vals = data_df["ER"].fillna(0).values 
    er_smooth = pd.Series(er_vals).ewm(span=50, adjust=False).mean().values 
    er_norm = np.clip(er_smooth, 0.0, 1.0) 
    ss_fast = data_df["EhlersSuperSmoother"].values 
    kama_slow = data_df["Kama"].values 
    ham_vals = er_norm * ss_fast + (1.0 - er_norm) * kama_slow 
    data_df["HAM"] = ham_vals 
    data_df["HAM_MA"] = pd.Series(ham_vals).rolling(window=30, min_periods=1).mean().values
    data_df["HAM_EMA"] = pd.Series(ham_vals).ewm(span=30, adjust=False).mean().values
    data_df["HAM_Slope"] = pd.Series(ham_vals).diff().fillna(0) 

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

    # 4. Plotting Setup with SUBPLOTS 
    feature_columns = [c for c in data_df.columns if c not in ['Time','Price','Day','PlotIndex','KAMA_Mode','KAMA_Switch']] 
    if selected_features: 
        feature_columns = [c for c in feature_columns if c in selected_features] 

    # Create Subplots: Row 1 for Price/Lines, Row 2 for Histograms 
    fig = make_subplots( 
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3], 
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]] 
    ) 

    # Plot Price (Row 1, Primary Y) 
    for day, df_day in data_df.groupby("Day"): 
        ts = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist() 
        fig.add_trace(go.Scatter( 
            x=df_day["PlotIndex"], y=df_day["Price"], mode='lines', 
            name=f"Price Day {day}", 
            line=dict(width=2, color='rgba(0,0,0,0.7)'), 

            hovertemplate="<b>Day</b>: %{customdata[0]}<br><b>Time</b>: %{customdata[1]}<br><b>Price</b>: %{y:.4f}<extra></extra>", 
            customdata=[[day, t] for t in ts], 
            showlegend=False 
        ), row=1, col=1, secondary_y=False) 

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] 
     
    # Plot Indicators 
    for i, col in enumerate(feature_columns): 
        norm_col = f"{col}_norm" 
        data_df[norm_col] = data_df[col] 
         
        for day, df_day in data_df.groupby("Day"): 
            mask = df_day[norm_col].notna() 
            if mask.sum() == 0: continue 
             
            # --- HISTOGRAMS (Row 2) --- 
            if 'Hist' in col: 
                fig.add_trace(go.Bar( 
                    x=df_day.loc[mask, "PlotIndex"],  
                    y=df_day.loc[mask, norm_col], 
                    name=col,  
                    legendgroup=col,  
                    visible='legendonly', 
                    marker_color=colors[i % len(colors)], 
                    marker_line_width=0, # Makes bars look solid, not wavy 
                    opacity=0.7 
                ), row=2, col=1) 
             
            # --- LINE PLOTS (Row 1, Secondary Y) --- 
            else: 
                fig.add_trace(go.Scatter( 
                    x=df_day.loc[mask, "PlotIndex"],  
                    y=df_day.loc[mask, norm_col], 
                    mode='lines',  
                    name=col,  
                    legendgroup=col,  
                    visible='legendonly', 
                    line=dict(color=colors[i % len(colors)], width=1.5), 
                ), row=1, col=1, secondary_y=True) 

    # Plot Regime Switches (Row 1) 
    switches = data_df[data_df["KAMA_Switch"] == 1] 
    if not switches.empty: 
        fig.add_trace(go.Scatter( 
            x=switches["PlotIndex"], y=switches["Price"], mode="markers", 
            marker=dict(size=18, color="purple", symbol="star", line=dict(color="black", width=1)), 
            name="Regime Switch" 
        ), row=1, col=1, secondary_y=False) 

    # Regime Background Shading (Applies to whole height) 
    df_plot = data_df[["PlotIndex", "KAMA_Mode"]].copy() 
    df_plot['change'] = df_plot['KAMA_Mode'].diff().fillna(0) 
    starts = df_plot[(df_plot['change'] == 1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == 0))]['PlotIndex'].values 

    ends = df_plot[(df_plot['change'] == -1) | ((df_plot['KAMA_Mode'] == 1) & (df_plot.index == len(df_plot)-1))]['PlotIndex'].values 

    if len(starts) > 0 and len(ends) > 0: 
        if starts[0] > ends[0]: starts = np.insert(starts, 0, df_plot['PlotIndex'].iloc[0]) 
        if len(starts) > len(ends): ends = np.append(ends, df_plot['PlotIndex'].iloc[-1]) 

    # yref="paper" ensures shading covers both subplots 

    shapes = [dict(type="rect", xref="x", yref="paper", x0=s, y0=0, x1=e, y1=1, fillcolor="rgba(0,255,0,0.15)", line_width=0, layer="below") for s, e in zip(starts, ends)] 
    fig.update_layout(shapes=shapes) 


    tick_vals = [data_df[data_df["Day"]==d]["PlotIndex"].iloc[0] for d in range(int(data_df["Day"].max())+1) if not data_df[data_df["Day"]==d].empty] 
    tick_txt = [f"Day {d}" for d in range(len(tick_vals))] 
     
    fig.update_layout(title='Multi-Day Analysis', hovermode='x unified', height=800, width=1600, plot_bgcolor='white') 
     
    # Ensure Ticks are on the bottom-most axis (Row 2) 
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_txt, row=2, col=1) 
     
    # Optional: Light grid for easier reading 
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)') 
     
    fig.show() 

if __name__=="__main__": 
    import sys 
    sys.path.append('.') 
    days = [103, 110, 31] 

    for d in days: 
        print(f"Processing Day {d}...") 
        success, pairs, df = get_pairs_for_day(d) 
        if not success: continue 

        feats = [c for c in df.columns if c.startswith(('BB4_T9','BB6_T11'))] 
         
        # Added FTAMA to the list of features to visualize
        feats.extend(['ATR', 'Kama', 'ER', 'BC', 'EMA_Fast', 'EMA_Slow', 'ZLEMA', 'HAM', 'HAM_Slope', 'HAM_MA', 'HAM_EMA',
                        'Slow_MA', 'Fast_MA', 'UCM', 'EAMA', 'EAMA_Slope', 'EAMA_Slope_MA', 'FFT', 'FTAMA',
                        'EKFTrend', 'EhlersSuperSmoother', 'JMA', 'HAMA', 'MACD_Hist', 'EhlersSuperSmoother_Slow']) 
         
        analyze_feature(cudf.from_pandas(df), price_pairs=pairs, selected_features=feats)
