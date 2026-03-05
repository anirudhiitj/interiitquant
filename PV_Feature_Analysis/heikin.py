import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cudf
import dask_cudf
from filter import get_pairs_for_day
from filterpy.kalman import ExtendedKalmanFilter
import sys

# -------------------------------------------------------------------------
# Indicator Calculations
# -------------------------------------------------------------------------

def calculate_all_indicators(data_df):
    """
    Calculate indicators that require TICK data (EKF, UCM, KAMA, MAs).
    """
    df = data_df.copy()
    
    # --- Extended Kalman Filter (Trend) ---
    ekf = ExtendedKalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([[df["Price"].iloc[0]]])
    ekf.F = np.array([[1.0]])
    ekf.Q = np.array([[0.05]])
    ekf.R = np.array([[0.2]])
    
    def h(x): return np.log(x)
    def H_jac(x): return np.array([[1.0/x[0][0]]])
    
    prices = df["Price"].dropna().values
    vals = []
    
    for p in prices:
        ekf.predict()
        ekf.update(np.array([[np.log(p)]]), HJacobian=H_jac, Hx=h)
        vals.append(ekf.x.item())
    
    df["EKFTrend"] = pd.Series(vals, index=df["Price"].dropna().index).ffill().shift(1)
    
    # --- UCM (Unobserved Components Model) ---
    alpha = 0.005
    beta = 0.0001
    series = df['Price']
    n = len(series)
    
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None:
        mu[first_valid_idx] = series.loc[first_valid_idx]
        beta_slope[first_valid_idx] = 0
        filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
        
        for t in range(first_valid_idx+1, n):
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
    
    df["UCM"] = filtered
    
    # --- KAMA ---
    P2_KAMA_PERIOD = 15
    P2_KAMA_FAST = 25
    P2_KAMA_SLOW = 100
    
    direction = df['Price'].diff(P2_KAMA_PERIOD).abs()
    volatility = df['Price'].diff(1).abs().rolling(window=P2_KAMA_PERIOD).sum()
    sc = (((direction/volatility).fillna(0) *
          (2/(P2_KAMA_FAST+1) - 2/(P2_KAMA_SLOW+1))) +
          (2/(P2_KAMA_SLOW+1)))**2
    
    price_vals = df['Price'].values
    sc_vals = sc.values
    kama_vals = np.zeros_like(price_vals)
    if len(price_vals) > 0:
        kama_vals[0] = price_vals[0]
        for i in range(1, len(price_vals)):
            sc_i = sc_vals[i] if not np.isnan(sc_vals[i]) else 0
            kama_vals[i] = kama_vals[i-1] + sc_i * (price_vals[i] - kama_vals[i-1])
    
    df['Kama'] = kama_vals
    
    # --- Basic MAs (Tick Based) ---
    df['Slowest_MA'] = df['Price'].rolling(1800).mean()
    df['Slow_MA'] = df['Price'].rolling(1300).mean()
    df['Fast_MA'] = df['Price'].rolling(420).mean()

    return df

def calculate_ha_indicators(ha_df):
    """
    Calculate indicators specifically on Heikin Ashi data.
    """
    df = ha_df.copy()
    
    # --- Choppiness Index ---
    period = 14
    prev_close = df['HA_Close'].shift(1)
    
    tr1 = df['HA_High'] - df['HA_Low']
    tr2 = (df['HA_High'] - prev_close).abs()
    tr3 = (df['HA_Low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period).sum()
    
    highest_high = df['HA_High'].rolling(window=period).max()
    lowest_low = df['HA_Low'].rolling(window=period).min()
    range_hl = highest_high - lowest_low
    
    epsilon = 1e-10
    # Formula: 100 * LOG10( SUM(TR, n) / ( MaxHi(n) - MinLo(n) ) ) / LOG10(n)
    ci = 100 * np.log10(atr_sum / (range_hl + epsilon)) / np.log10(period)
    
    df['Choppiness'] = ci
    
    # --- HAMA (Heikin Ashi Moving Averages) ---
    df['HAMA_Fast'] = df['HA_Close'].rolling(window=21).mean()
    df['HAMA_Slow'] = df['HA_Close'].rolling(window=55).mean()
    
    return df

# -------------------------------------------------------------------------
# Resampling Logic
# -------------------------------------------------------------------------

def resample_to_ohlc(df, tick_window=5):
    """Resample tick data into OHLC bars every tick_window ticks."""
    n_complete_bars = len(df) // tick_window
    df_truncated = df.iloc[:n_complete_bars * tick_window].copy()
    df_truncated['BarGroup'] = np.arange(len(df_truncated)) // tick_window
    
    agg_dict = {'Price': ['first', 'max', 'min', 'last', lambda x: list(x)], 'Time': ['first', 'last']}
    ohlc = df_truncated.groupby('BarGroup').agg(agg_dict).reset_index()
    ohlc.columns = ['BarGroup', 'Open', 'High', 'Low', 'Close', 'Tick_Prices', 'Time_Start', 'Time_End']
    return ohlc

def calculate_heikin_ashi(ohlc_df):
    """Calculate HA columns."""
    ha_df = ohlc_df.copy()
    
    ha_close = np.zeros(len(ohlc_df))
    ha_open = np.zeros(len(ohlc_df))
    ha_high = np.zeros(len(ohlc_df))
    ha_low = np.zeros(len(ohlc_df))
    
    for i in range(len(ohlc_df)):
        ha_close[i] = (ohlc_df['Open'].iloc[i] + ohlc_df['High'].iloc[i] + ohlc_df['Low'].iloc[i] + ohlc_df['Close'].iloc[i]) / 4
        if i == 0:
            ha_open[i] = (ohlc_df['Open'].iloc[i] + ohlc_df['Close'].iloc[i]) / 2
        else:
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(ohlc_df['High'].iloc[i], ha_open[i], ha_close[i])
        ha_low[i] = min(ohlc_df['Low'].iloc[i], ha_open[i], ha_close[i])
    
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_high
    ha_df['HA_Low'] = ha_low
    ha_df['HA_Close'] = ha_close
    
    return ha_df

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_single_heikin_ashi(ha_df, tick_data_with_indicators, day_number, tick_window):
    """
    Plots exactly 1 row:
    1-Min Heikin Ashi (Price on Left, Choppiness on Right)
    """
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        subplot_titles=(f"1-Min Heikin Ashi ({tick_window}-Tick) Analysis",),
        specs=[[{"secondary_y": True}]]
    )
    
    # --- Map ticks to bars for indicator alignment ---
    bar_to_tick_map = {}
    for i in range(len(ha_df)):
        tick_end_idx = min((i + 1) * tick_window - 1, len(tick_data_with_indicators) - 1)
        bar_to_tick_map[i] = tick_end_idx

    # --- PRIMARY AXIS (Left): Price & HAMA ---
    
    # 1. Candlesticks (HA)
    hover_texts = []
    for i in range(len(ha_df)):
        time_start = str(ha_df['Time_Start'].iloc[i]).split()[-1]
        chop_val = ha_df['Choppiness'].iloc[i] if 'Choppiness' in ha_df else np.nan
        raw_close = ha_df['Close'].iloc[i]
        
        hover_text = (f"<b>Time:</b> {time_start}<br>"
                     f"<b>Raw:</b> {raw_close:.2f} | <b>HA:</b> {ha_df['HA_Close'].iloc[i]:.2f}<br>"
                     f"<b>Chop:</b> {chop_val:.2f}<extra></extra>")
        hover_texts.append(hover_text)

    fig.add_trace(go.Candlestick(
        x=ha_df['BarGroup'],
        open=ha_df['HA_Open'], high=ha_df['HA_High'],
        low=ha_df['HA_Low'], close=ha_df['HA_Close'],
        increasing_line_color='#089981', decreasing_line_color='#f23645',
        name=f'HA Candles', hovertext=hover_texts, hoverinfo='text',
        showlegend=False
    ), secondary_y=False)

    # 2. Raw Price (Dotted Line)
    fig.add_trace(go.Scatter(
        x=ha_df['BarGroup'], y=ha_df['Close'],
        mode='lines', line=dict(color='gray', width=1, dash='dot'),
        name='Raw Price', opacity=0.8
    ), secondary_y=False)

    # 3. HAMA (Overlay)
    if 'HAMA_Fast' in ha_df.columns:
        fig.add_trace(go.Scatter(
            x=ha_df['BarGroup'], y=ha_df['HAMA_Fast'],
            mode='lines', line=dict(color='cyan', width=1.5),
            name='HAMA Fast (21)'
        ), secondary_y=False)
        
    if 'HAMA_Slow' in ha_df.columns:
        fig.add_trace(go.Scatter(
            x=ha_df['BarGroup'], y=ha_df['HAMA_Slow'],
            mode='lines', line=dict(color='blue', width=1.5),
            name='HAMA Slow (55)'
        ), secondary_y=False)

    # --- SECONDARY AXIS (Right): Choppiness ---
    if 'Choppiness' in ha_df.columns:
        # Threshold lines
        fig.add_hline(y=61.8, line_dash="dot", line_color="gray", opacity=0.3, secondary_y=True)
        fig.add_hline(y=38.2, line_dash="dot", line_color="gray", opacity=0.3, secondary_y=True)
        
        # Choppiness Plot
        fig.add_trace(go.Scatter(
            x=ha_df['BarGroup'], y=ha_df['Choppiness'],
            mode='lines', line=dict(color='black', width=1, dash='solid'),
            name='Choppiness', opacity=0.6
        ), secondary_y=True)

    # --- Tick Indicators (Optional, Hidden by default) ---
    tick_indicators = ['EKFTrend', 'UCM', 'Kama']
    colors = {'EKFTrend': '#ff7f0e', 'UCM': '#2ca02c', 'Kama': '#9467bd'}
    
    for col in tick_indicators:
        if col in tick_data_with_indicators.columns:
            bar_x = []
            ind_y = []
            step = 1 if len(ha_df) < 1000 else 2 
            for bar_i in range(0, len(ha_df), step):
                if bar_i in bar_to_tick_map:
                    tick_idx = bar_to_tick_map[bar_i]
                    val = tick_data_with_indicators[col].iloc[tick_idx]
                    if pd.notna(val):
                        bar_x.append(bar_i)
                        ind_y.append(val)
            
            if ind_y:
                fig.add_trace(go.Scatter(
                    x=bar_x, y=ind_y,
                    mode='lines', name=col,
                    line=dict(color=colors.get(col, 'gray'), width=1.5),
                    visible='legendonly'
                ), secondary_y=False)

    # --- Layout ---
    fig.update_layout(
        title=f'Market Analysis Day {day_number}: Price, HAMA & Choppiness',
        height=700, # Reduced height for single plot
        width=1600,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Axes config
    fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True, gridcolor='#f2f2f2')
    fig.update_yaxes(title_text="Chop", range=[0, 100], secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Bar Number", showgrid=True, gridcolor='#f2f2f2')

    fig.show()

# -------------------------------------------------------------------------
# Execution / Driver
# -------------------------------------------------------------------------

def process_and_plot_day(day_number, tick_window=5):
    print(f"\nProcessing Day {day_number}...")
    success, pairs, df_pandas = get_pairs_for_day(day_number)
    
    if not success:
        print(f"Failed to load day {day_number}")
        return False
    
    if isinstance(df_pandas, dask_cudf.DataFrame):
        df_pandas = df_pandas.compute().to_pandas()
    elif isinstance(df_pandas, cudf.DataFrame):
        df_pandas = df_pandas.to_pandas()
    
    df_pandas["Time"] = pd.to_timedelta(df_pandas["Time"])
    df_pandas = df_pandas.dropna(subset=["Price"]).reset_index(drop=True)
    
    # Calculate Tick Indicators
    tick_data_with_indicators = calculate_all_indicators(df_pandas)
    
    # 1-Min Data (tick_window defines the aggregation)
    ohlc_df = resample_to_ohlc(tick_data_with_indicators, tick_window=tick_window)
    if len(ohlc_df) == 0: return False
    
    ha_df = calculate_heikin_ashi(ohlc_df)
    ha_df = calculate_ha_indicators(ha_df)
    
    # Plot Only 1-Min
    plot_single_heikin_ashi(ha_df, tick_data_with_indicators, day_number, tick_window)
    return True

if __name__ == "__main__":
    sys.path.append('.')
    
    days_to_plot = [277, 31] 
    
    for day in days_to_plot:
        # Assuming tick_window=60 creates approx 1-min bars if data is 1 tick/sec, 
        # or adjust tick_window to match your data density for 1-minute bars.
        process_and_plot_day(day, tick_window=60)