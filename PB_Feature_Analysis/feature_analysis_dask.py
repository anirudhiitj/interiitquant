import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from filterpy.kalman import ExtendedKalmanFilter
from pykalman import KalmanFilter
# from filter import get_pairs_for_day # Assuming this exists in your environment
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.nonparametric.smoothers_lowess import lowess
from numba import jit

def strided_windows(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def analyze_feature(data_df, price_pairs=None, selected_features=None):
    warnings.filterwarnings('ignore')

    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price

    # --- UCM Calculation ---
    series = data_df["PB9_T1"]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError("PB9_T1 has no valid values")

    alpha = 0.8    
    beta = 0.08
    n = len(series)

    mu = np.zeros(n)         # level
    beta_slope = np.zeros(n)  # trend
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

    data_df["UCM"] = filtered

    # --- EKF Calculation ---
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

    # --- Kalman Filter Fast ---
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.24
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = data_df["Price"].iloc[0]
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(data_df["PB9_T1"].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=data_df["PB9_T1"].dropna().index)
    data_df["KFTrendFast"] = smoothed_prices.shift(1)

    # --- KAMA Calculation ---
    price = data_df["PB9_T1"].to_numpy(dtype=float)
    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) > 0:
        start = valid_start[0]
        kama[start] = price[start]
        for i in range(start + 1, len(price)):
            if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    data_df["KAMA"] = kama
    data_df["KAMA_Slope"] = data_df["KAMA"].diff(2)
    data_df["KAMA_Slope_abs"] = data_df["KAMA_Slope"].abs()

    # --- SMA KAMA ---
    n = len(price)
    kama_sma = np.full(n, np.nan)
    signal = np.abs(pd.Series(price).diff(window))
    noise = pd.Series(np.abs(pd.Series(price).diff())).ewm(span=window, adjust=False).mean()
    er = (signal / noise.replace(0, np.nan)).fillna(0).to_numpy()
    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    adaptive_window = (1 / sc).clip(3, 300).astype(int)

    for t in range(n):
        if np.isnan(price[t]):
            kama_sma[t] = np.nan
            continue
        w = adaptive_window[t]
        start = max(0, t - w + 1)
        kama_sma[t] = np.nanmean(price[start:t+1])

    data_df["SMA_KAMA"] = kama_sma
    data_df["SMA_KAMA_Slope"] = data_df["SMA_KAMA"].diff(2)
    data_df["SMA_KAMA_Slope_abs"] = data_df["SMA_KAMA_Slope"].abs()


    # --- SMAs ---
    data_df["SMA_90"]=data_df["Price"].rolling(window=90, min_periods=1).mean()
    data_df["SMA_15"]=data_df["Price"].rolling(window=15, min_periods=1).mean()
    data_df["SMA_30"]=data_df["Price"].rolling(window=30, min_periods=1).mean()
    data_df["SMA_20"]=data_df["Price"].rolling(window=20, min_periods=1).mean()
    data_df["SMA_25"]=data_df["Price"].rolling(window=25, min_periods=1).mean()
    data_df["SMA_10"]=data_df["Price"].rolling(window=10, min_periods=1).mean()
    data_df["SMA_45"]=data_df["Price"].rolling(window=45, min_periods=1).mean()
    
    data_df['Diff_STD'] = data_df['Price'].diff().rolling(120).std()
    data_df['Diff_STD_STD'] = data_df['Diff_STD'].rolling(120).std()
    times = data_df["Time"]
    start_time = times.iloc[0]
    # Define first hour window
    cutoff = start_time + pd.Timedelta(minutes=60)
    
    # Filter for first hour
    mask = times < cutoff
    first_hour_data = data_df.loc[mask, "Diff_STD_STD"].dropna()
    
    if first_hour_data.empty:
        data_df["STD_Threshold"] = np.nan
    else:
        p99 = np.percentile(first_hour_data, 99)
        threshold = p99 / 2.5
        data_df["STD_Threshold"] = threshold
    price_vals = data_df["Price"].values
    n = len(price_vals)

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
    data_df["HAMA_Slope"] = data_df["HAMA"].diff().fillna(0)

    # Ehlers Super Smoother
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
    data_df["EhlersSuperSmoother_Slope"] = data_df["EhlersSuperSmoother"].diff()
    
    ss_series = data_df["EhlersSuperSmoother"]
    
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

    eama_period = 15
    eama_fast = 60
    eama_slow = 180 
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

    data_df["EAMA_Slow"] = eama
    data_df["EAMA_Slow_Slope"] = pd.Series(eama).diff().fillna(0)

    data_df["Mean"] = data_df["Price"].expanding().mean()
    data_df["StdDev"] = data_df["Price"].expanding().std().fillna(0)
    data_df["Upper"] = data_df["Mean"] + 2 * data_df["StdDev"]
    data_df["Lower"] = data_df["Mean"] - 2 * data_df["StdDev"]

    price = data_df["PB9_T1"].to_numpy(dtype=float)

    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
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

    data_df["KAMA"] = kama
    data_df["KAMA_Slope"] = data_df["KAMA"].diff(2)
    data_df["KAMA_Slope_abs"] = data_df["KAMA_Slope"].abs()

    window = 300
    raw_signal = np.where(data_df["Diff_STD_STD"] > data_df["STD_Threshold"], 1, 0)
    
    high_vol_count = pd.Series(raw_signal).rolling(window=window).sum().fillna(0)
    mix_weight = high_vol_count / window
    
    data_df["EDAMA"] = (mix_weight * data_df["EAMA"]) + ((1 - mix_weight) * data_df["EAMA_Slow"])
    start_time = data_df['Time'].iloc[0]
    cutoff = start_time + pd.Timedelta(minutes=60)
    
    mask_first_hour = data_df['Time'] < cutoff
    
    data_df.loc[mask_first_hour, "EDAMA"] = data_df.loc[mask_first_hour, "EAMA_Slow"]

    length = 1500
    prices = data_df["Price"].values
    
    vol = np.abs(np.diff(prices, prepend=prices[0]))
    vol_smooth = pd.Series(vol).rolling(window=length, min_periods=1).mean().fillna(0).values
    vol_mean = pd.Series(vol_smooth).rolling(window=length, min_periods=1).mean().replace(0, 1.0).values
    vol_norm = np.clip(vol_smooth / vol_mean, 0.1, 10.0)
    
    alpha = 2.0 / (length + 1.0)
    alpha_adapt = np.clip(alpha * (1 + 0.5 * (vol_norm - 1)), 0.001, 0.999)
    
    out = np.empty(len(prices))
    out[0] = prices[0]
    for i in range(1, len(prices)):
        out[i] = alpha_adapt[i] * prices[i] + (1.0 - alpha_adapt[i]) * out[i-1]
    
    data_df["JMA"] = out

    alpha = 1.2
    beta = 1.8
    threshold = 0.01
    time_delta = 1
    decay = np.exp(-beta * time_delta)
    H_buy = np.zeros(n)
    H_sell = np.zeros(n)
    H_sig = np.zeros(n)
    for i in range(1, n):
        H_buy[i] = H_buy[i-1] * decay
        H_sell[i] = H_sell[i-1] * decay

        if data_df["EhlersSuperSmoother_Slope"][i] > threshold:
            H_buy[i] += alpha
            H_sig[i] += alpha
        if data_df["EhlersSuperSmoother_Slope"][i] < -threshold:
            H_sell[i] += alpha
            H_sig[i] +=alpha

    data_df["H_Buy"] = H_buy
    data_df["H_Sell"] = H_sell
    data_df["H_Sig"] = H_sig

    # --- Z-Score ---
    roll_mean = data_df["Price"].rolling(window=1800).mean()
    roll_std  = data_df["Price"].rolling(window=1800).std(ddof=1)
    data_df["Z-Score"] = (data_df["Price"] - roll_mean) / roll_std
    data_df["Z-Score"].iloc[:1800] = np.nan
    sig = 1 / (1 + np.exp(-data_df["Z-Score"]))
    data_df["Z-Sigmoid"] = (sig - 0.5) * 2
    data_df["Z-Sigmoid"] = data_df["Z-Sigmoid"].rolling(window=10, min_periods=1).mean()
    data_df["Z-Sigmoid_MA"] = data_df["Z-Sigmoid"].rolling(window=600, min_periods=1).mean()
    data_df["Z-Sigmoid_Slope"] = data_df["Z-Sigmoid"].diff(5)

    data_df["ATR"] = data_df["Price"].diff().abs().rolling(window=30, min_periods=1).mean()

    # --- Feature Selection ---
    feature_columns = [c for c in data_df.columns if c not in ['Time', 'Price', 'Day', 'PlotIndex']]
    all_available_features = list(feature_columns)
    
    # Updated Defaults to include Hawkes
    default_primary = ["UCM", "KFTrendFast", "SMA_10", "SMA_30", "SMA_45","EDAMA" ,"EAMA","EAMA_Slow" , "HAMA","Upper","Lower","EhlersSuperSmoother", "EKFTrend"]
    default_secondary = ["KAMA_Slope_abs", "KAMA", "SMA_KAMA", "SMA_KAMA_Slope_abs","EAMA_Slow_Slope", "EAMA_Slope", "HAMA_Slope","EhlersSuperSmoother_Slope", "H_Buy", "H_Sell","H_Long","H_Short","ATR","Diff_STD_STD"]

    if selected_features is not None:
        if isinstance(selected_features, dict):
            sel_names = list(selected_features.get('primary', [])) + list(selected_features.get('secondary', []))
        else:
            try:
                sel_list = list(selected_features)
            except Exception:
                sel_list = []
            sel_names = sel_list + [c for c in default_primary if c in all_available_features and c not in sel_list]

        feature_columns = [c for c in all_available_features if c in sel_names]
        print(f"  Selected {len(feature_columns)} features out of {len(all_available_features)} available")
    else:
        print(f"  Using all {len(feature_columns)} features")

    available_feature_set = set(feature_columns)
    
    feature_groups = {
        'primary': [c for c in default_primary if c in available_feature_set],
        'secondary': [c for c in default_secondary if c in available_feature_set]
    }

    if selected_features is None:
        remaining = [c for c in sorted(feature_columns) if c not in feature_groups['primary']]
        feature_groups['secondary'] = [c for c in feature_groups['secondary']] + [c for c in remaining if c not in feature_groups['secondary']]
        print(f"  No selected_features provided. Using {len(feature_groups['primary'])} primary and {len(feature_groups['secondary'])} secondary features.")
    elif isinstance(selected_features, dict):
        for grp in ('primary', 'secondary'):
            if grp in selected_features and selected_features[grp]:
                feature_groups[grp] = [c for c in selected_features[grp] if c in available_feature_set]
            else:
                feature_groups[grp] = [c for c in feature_groups.get(grp, []) if c in available_feature_set]
        print(f"  Using {len(feature_groups['primary'])} primary and {len(feature_groups['secondary'])} secondary features (from dict).")
    else:
        try:
            selected_list = list(selected_features)
        except Exception:
            selected_list = []
        feature_groups['secondary'] = [c for c in selected_list if c in available_feature_set and c not in feature_groups['primary']]
        print(f"  Using {len(feature_groups['primary'])} primary (defaults) and {len(feature_groups['secondary'])} selected secondary features.")

    normalized_feature_columns = {}
    for grp in ('primary', 'secondary'):
        for col in feature_groups[grp]:
            if col in normalized_feature_columns:
                continue
            norm_col_name = f"{col}_norm"
            normalized_feature_columns[col] = norm_col_name
            data_df[norm_col_name] = data_df[col]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Price (Primary Y-axis) - Day by Day (disconnected)
    for day, df_day in data_df.groupby("Day"):
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"],
            y=df_day["Price"],
            mode='lines',
            name=f"Price Day {day}", 
            line=dict(width=2, color='rgba(0,0,0,0.7)'), 
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Price</b>: %{y:.8f}<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False,
            showlegend=False 
        ), secondary_y=False)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    plotted_legend_items = set() 

    all_feature_list = feature_groups['primary'] + feature_groups['secondary']
    for idx, original_feature_name in enumerate(all_feature_list):
        norm_feature = normalized_feature_columns.get(original_feature_name)
        if norm_feature is None:
            continue
        group = 'secondary' if original_feature_name in feature_groups['secondary'] else 'primary'
        color = colors[idx % len(colors)]

        for day, df_day in data_df.groupby("Day"):
            feature_mask = df_day[norm_feature].notna()
            feature_data = df_day.loc[feature_mask, norm_feature]
            if len(feature_data) == 0:
                continue
            corresponding_times = df_day.loc[feature_mask, "Time"].apply(lambda x: str(x).split()[-1]).tolist()
            corresponding_plot_indices = df_day.loc[feature_mask, "PlotIndex"].tolist()

            show_legend_for_this_trace = (original_feature_name not in plotted_legend_items)
            if show_legend_for_this_trace:
                plotted_legend_items.add(original_feature_name)

            fig.add_trace(
                go.Scatter(
                    x=corresponding_plot_indices,
                    y=feature_data.values.tolist(),
                    mode='lines',
                    name=original_feature_name,
                    legendgroup=original_feature_name,
                    line=dict(color=color, width=1.5),
                    visible='legendonly',
                    hovertemplate=(
                        "<b>Day</b>: %{customdata[0]}<br>"
                        "<b>Time</b>: %{customdata[1]}<br>"
                        f"<b>{original_feature_name} (Norm)</b>: %{{y:.8f}}<br>"
                        f"<b>{original_feature_name} (Orig)</b>: %{{customdata[2]:.8f}}<br>"
                        "<extra></extra>"
                    ),
                    customdata=[[day, time_str, orig_val] for day, time_str, orig_val in zip(
                        [day]*len(corresponding_times),
                        corresponding_times,
                        df_day.loc[feature_mask, original_feature_name].tolist()
                    )],
                    connectgaps=False,
                    showlegend=show_legend_for_this_trace
                ),
                secondary_y=(group == 'secondary')
            )

    if price_pairs is not None and len(price_pairs) > 0:
        print(f"  Plotting {len(price_pairs)} price jump pairs...")
        for pair_idx, (start_idx, end_idx) in enumerate(price_pairs):
            if start_idx >= len(data_df) or end_idx >= len(data_df):
                continue
            
            try:
                start_plot_idx = data_df.iloc[start_idx]['PlotIndex']
                end_plot_idx = data_df.iloc[end_idx]['PlotIndex']
                start_price = data_df.iloc[start_idx]['Price']
                end_price = data_df.iloc[end_idx]['Price']
                start_time = str(data_df.iloc[start_idx]['Time']).split()[-1]
                end_time = str(data_df.iloc[end_idx]['Time']).split()[-1]
                
                fig.add_trace(go.Scatter(
                    x=[start_plot_idx],
                    y=[start_price],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='circle', 
                               line=dict(color='darkgreen', width=2)),
                    name='Jump Start' if pair_idx == 0 else None,
                    legendgroup='jump_start',
                    showlegend=(pair_idx == 0),
                    hovertemplate=f"<b>Jump Start</b><br>Time: {start_time}<br>Price: {start_price:.8f}<extra></extra>"
                ), secondary_y=False)
                
                fig.add_trace(go.Scatter(
                    x=[end_plot_idx],
                    y=[end_price],
                    mode='markers',
                    marker=dict(size=10, color='orange', symbol='circle',
                               line=dict(color='darkorange', width=2)),
                    name='Jump End' if pair_idx == 0 else None,
                    legendgroup='jump_end',
                    showlegend=(pair_idx == 0),
                    hovertemplate=f"<b>Jump End</b><br>Time: {end_time}<br>Price: {end_price:.8f}<extra></extra>"
                ), secondary_y=False)
                
                fig.add_trace(go.Scatter(
                    x=[start_plot_idx, end_plot_idx],
                    y=[start_price, end_price],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ), secondary_y=False)
            except Exception as e:
                print(f"  Error plotting pair {pair_idx}: {e}")

    num_days = int(data_df["Day"].max()) + 1
    tick_positions = []
    tick_labels = []
    for day in range(num_days):
        day_data = data_df[data_df["Day"] == day]
        if len(day_data) > 0:
            tick_positions.append(day_data["PlotIndex"].iloc[0])
            tick_labels.append(f"Day {day}")
    
    title_text = 'Combined Multi-Day Price Analysis with Hawkes Intensity'
    if price_pairs is not None and len(price_pairs) > 0:
        title_text += f' ({len(price_pairs)} Price Jump Pairs)'
    
    fig.update_layout(
        title={'text': title_text, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': 'black'}},
        hovermode='x unified',
        height=600,
        width=1600,
        showlegend=True,
        plot_bgcolor='#FDFFF7',
        paper_bgcolor='#FDFFF7',
        font=dict(size=12)
    )
    
    fig.update_xaxes(title_text="Time (HH:MM:SS)", tickvals=tick_positions, ticktext=tick_labels, gridcolor='lightgray')
    fig.update_yaxes(title_text="Price / Primary Features", tickformat='.8f', gridcolor='lightgray', secondary_y=False)
    fig.update_yaxes(title_text="Hawkes Intensity / Secondary Features", tickformat='.2f', secondary_y=True)
    
    y_padding = (max_price - min_price) * 0.1
    fig.update_yaxes(range=[min_price - y_padding, max_price + y_padding], secondary_y=False)
    
    stats_text = (
        f'Number of Days: {num_days}<br>'
        f'Initial Price: ${data_df["Price"].iloc[0]:.2f}<br>'
        f'Final Price: ${data_df["Price"].iloc[-1]:.2f}<br>'
    )
    
    fig.add_annotation(
        text="annotations",
        hovertext=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="black",
        borderwidth=2,
        borderpad=10,
        align="left",
        font=dict(size=11, family="monospace")
    )
    
    fig.write_html("EBX_Hawkes_Analysis.html")
    fig.show()

if __name__ == "__main__":
    import sys
    sys.path.append('.') 
    
    # Import from your analysis script
    try:
        from filter import get_pairs_for_day
        day_to_plot = 421
        success, pairs, df_pandas = get_pairs_for_day(day_to_plot)
        
        if success:
            print(f"Successfully loaded day {day_to_plot}")
            
            # --- ADDED: Hawkes to selected features list ---
            available_cols = df_pandas.columns.tolist()
            selected_features = [c for c in available_cols if c=="PB9_T1"]
            selected_features.extend([
                "KFTrendFast", "UCM","KAMA_Slope_abs", "EAMA","EDAMA","EAMA_Slope","EAMA_Slow","EAMA_Slow_Slope","HAMA","HAMA_Slope","EhlersSuperSmoother","EhlersSuperSmoother_Slope", 
                "H_Buy", "H_Sell", "H_Long", "H_Short","ATR","Diff_STD_STD"
            ])
            
            analyze_feature(df_pandas, price_pairs=pairs, selected_features=selected_features)
        else:
            print(f"Failed to load day {day_to_plot}")
            
    except ImportError:
        print("Could not import 'filter'. Please ensure 'filter.py' is in the working directory.")