# rolling_hurst_pipeline.py
# Requirements: pip install pandas numpy plotly tqdm

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from math import ceil

# ---------------------------
# Parameters (tweak these)
# ---------------------------
PARQUET_FOLDER = "/data/quant14/EBY/"   # set this
PARQUET_GLOB = "*.parquet"
PRICE_COL = "Price"         # price column name
TS_COL = "Time"        # timestamp col if present; else index order used
WINDOW_SECONDS = 900        # rolling window size (seconds) -- default 15 minutes
STRIDE_SECONDS = 30         # stride (seconds) between successive windows
MIN_POINTS = int(0.85 * WINDOW_SECONDS)            # minimum points in window to compute H
MAX_LAG = 200               # max lag for variogram estimator
MIN_LAGS = 10               # minimal number of lags required in estimator
SAVE_H_DF = "rolling_hurst_per_day.parquet"


# ---------------------------

def hurst_variogram(ts, max_lag=MAX_LAG, min_lags=MIN_LAGS):
    ts = np.asarray(ts, dtype=float)
    N = ts.size
    if N < 10:
        return np.nan
    max_lag = int(min(max_lag, max(2, N//4)))
    lags = np.arange(2, max_lag + 1)
    if lags.size < min_lags:
        return np.nan
    taus = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        if diffs.size == 0:
            taus.append(np.nan)
        else:
            taus.append(np.sqrt(np.mean(diffs * diffs)))
    taus = np.array(taus, dtype=float)
    mask = (taus > 0) & np.isfinite(taus)
    if mask.sum() < min_lags:
        return np.nan
    lags = lags[mask]; taus = taus[mask]
    # linear fit on log-log
    A = np.vstack([np.log(lags), np.ones_like(lags)]).T
    slope, intercept = np.linalg.lstsq(A, np.log(taus), rcond=None)[0]
    return float(slope)

def rolling_hurst_for_series(ts_values, timestamps=None,
                             window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS,
                             min_points=MIN_POINTS):
    """
    Compute leak-free rolling Hurst for a single day's series.
    ts_values: 1D array-like of price (prefer log-price or price>0)
    timestamps: same length array-like of POSIX seconds (or pandas.Timestamp). If None, indices are used and
                window/stride are in number-of-points units (not recommended).
    Returns: DataFrame with columns ['t_end', 't_start', 'n_points', 'hurst']
    where t_end is the timestamp of the last point included (i.e. H(t) uses data <= t_end)
    """
    # convert inputs
    x = np.asarray(ts_values, dtype=float)
    N = len(x)
    if N == 0:
        return pd.DataFrame(columns=['t_end','t_start','n_points','hurst'])

    # Prefer log-price input (caller should pass np.log(price))
    if timestamps is None:
        # treat window_seconds/stride_seconds as counts (not ideal)
        idx = np.arange(N)
        times = idx
        window_pts = window_seconds  # caller aware
        stride_pts = stride_seconds
        use_time_index = False
    else:
        # convert timestamps to numeric seconds if pandas.Timestamp
        times = np.array([t.timestamp() if hasattr(t, "timestamp") else float(t) for t in timestamps], dtype=float)
        use_time_index = True
        # We'll compute windows by time: include points with times >= t_end - window_seconds
        window_pts = window_seconds
        stride_pts = stride_seconds

    results = []
    # starting time is the first timestamp + window_seconds (we compute H at t_end values)
    t0 = times[0]
    t_last = times[-1]
    # We'll compute t_end sequence: from t0 + window to t_last, stepping by stride
    t_end = t0 + window_pts
    while t_end <= t_last + 1e-9:
        # include all points with time <= t_end and >= t_end - window
        left = t_end - window_pts
        if use_time_index:
            # boolean mask of indices in window
            mask = (times > left) & (times <= t_end)
            idxs = np.nonzero(mask)[0]
        else:
            # use integer indexing approximate
            idxs = np.where((times > left) & (times <= t_end))[0]

        n_pts = idxs.size
        if n_pts >= min_points:
            segment = x[idxs]
            h = hurst_variogram(segment, max_lag=MAX_LAG, min_lags=MIN_LAGS)
        else:
            h = np.nan
        results.append({'t_start': left, 't_end': t_end, 'n_points': int(n_pts), 'hurst': h})
        t_end += stride_pts

    return pd.DataFrame(results)

# -------------------------
# Top-level orchestration
# -------------------------
def compute_rolling_hurst_all_days(folder=PARQUET_FOLDER, pattern=PARQUET_GLOB,
                                   price_col=PRICE_COL, ts_col=TS_COL,
                                   window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS,
                                   min_points=MIN_POINTS):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {folder} with pattern {pattern}")

    all_days = []
    for fp in tqdm(files, desc="Days"):
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"skip {fp} read error: {e}")
            continue

        # infer day id
        day_id = os.path.splitext(os.path.basename(fp))[0]

        # sort by timestamp if exists
        if ts_col in df.columns:
            df = df.sort_values(ts_col)
            # Parse datetime efficiently - use format='mixed' for pandas 2.0+ (handles mixed formats)
            # or fallback to standard parsing with errors='coerce' to avoid warnings
            try:
                times = pd.to_datetime(df[ts_col], format='mixed', errors='coerce').tolist()
            except (ValueError, TypeError):
                # Fallback for older pandas versions
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    times = pd.to_datetime(df[ts_col], errors='coerce').tolist()
        else:
            times = None

        # price series; use log-price if possible
        if price_col in df.columns:
            price = df[price_col].astype(float).values
            if np.all(price > 0):
                series = np.log(price)
            else:
                # fallback to cumulative returns if prices not positive
                ret = np.zeros_like(price)
                ret[1:] = price[1:] / price[:-1] - 1.0
                series = np.cumsum(ret)
        else:
            # fallback to first numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                print(f"No numeric columns for {fp}, skipping")
                continue
            series = df[num_cols[0]].astype(float).values

        # compute rolling H for this day
        roll_df = rolling_hurst_for_series(series, timestamps=times,
                                           window_seconds=window_seconds,
                                           stride_seconds=stride_seconds,
                                           min_points=min_points)
        if roll_df.empty:
            continue
        roll_df['day_id'] = day_id
        all_days.append(roll_df)

    if not all_days:
        return pd.DataFrame()
    res = pd.concat(all_days, ignore_index=True)
    # optionally persist
    res.to_parquet(SAVE_H_DF, index=False)
    return res

# -------------------------
# Plotting helpers
# -------------------------
def plot_single_day_hurst(roll_df, day_id, show_n_points=False):
    df_day = roll_df[roll_df['day_id'] == day_id].copy()
    if df_day.empty:
        print("No data for", day_id); return
    # convert numeric t_end to pandas datetime if needed
    if np.issubdtype(type(df_day['t_end'].iloc[0]), np.floating):
        # treat as epoch seconds
        df_day['t_end_dt'] = pd.to_datetime(df_day['t_end'], unit='s')
    else:
        df_day['t_end_dt'] = pd.to_datetime(df_day['t_end'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_day['t_end_dt'], y=df_day['hurst'], mode='lines+markers', name='H(t)'))
    fig.add_hline(y=0.5, line_dash='dash', line_color='black', annotation_text='H=0.5', annotation_position='top left')
    fig.update_layout(title=f"Rolling Hurst for {day_id}", xaxis_title='time', yaxis_title='Hurst')
    if show_n_points:
        fig.add_trace(go.Bar(x=df_day['t_end_dt'], y=df_day['n_points'], name='n_points', yaxis='y2', opacity=0.3))
    fig.show()

def plot_heatmap_days(roll_df, sample_days=None, vmax=None):
    """
    Create a heatmap: rows = days, columns = time bins (relative), values = H
    We'll align on window index (0..T-1) per day; missing fill NaN.
    """
    # pick days in order
    days = sorted(roll_df['day_id'].unique())
    if sample_days is not None:
        days = [d for d in days if d in sample_days]
    # pivot by day and t_end
    pivot = roll_df.pivot(index='day_id', columns='t_end', values='hurst')
    # order rows by day list
    pivot = pivot.reindex(days)
    # for plotting, convert column names to relative minutes or indices
    z = pivot.values
    fig = go.Figure(data=go.Heatmap(z=z, x=pivot.columns.astype(str), y=pivot.index, colorscale='RdBu', zmid=0.5, zmin=0.35 if vmax is None else None, zmax=0.65 if vmax is None else None))
    fig.update_layout(title='Heatmap of rolling Hurst (rows=days, cols=time bins)', xaxis_title='t_end', yaxis_title='day')
    fig.show()
    return pivot

# -------------------------
# Example run (call in notebook)
# -------------------------
if __name__ == "__main__":
    # compute all days (this can take time depending on number of days and window settings)
    # rolling_df = compute_rolling_hurst_all_days(folder=PARQUET_FOLDER, pattern=PARQUET_GLOB,
    #                                            price_col=PRICE_COL, ts_col=TS_COL,
    #                                            window_seconds=WINDOW_SECONDS, stride_seconds=STRIDE_SECONDS,
    #                                            min_points=MIN_POINTS)
    # print("Saved rolling Hurst to", SAVE_H_DF)
    # print("Example entries:", rolling_df.head())

    rolling_df = pd.read_parquet(SAVE_H_DF)

    # 1. Smooth H with EMA (optional)
    rolling_df['H_smooth'] = rolling_df.groupby('day_id')['hurst'].transform(lambda x: x.ewm(span=5, adjust=False).mean())

    # 2. Regime mark (adjust thresholds if desired)
    def regime(h):
        if np.isnan(h): return 'nan'
        if h > 0.535: return 'trend'
        if h < 0.465: return 'revert'
        return 'neutral'
    rolling_df['regime'] = rolling_df['H_smooth'].apply(regime)

    # 3. Count regime durations per day
    regime_counts = rolling_df.groupby(['day_id','regime']).size().unstack(fill_value=0)

    # 4. Average H per day & fraction of time in each regime
    daily_summary = rolling_df.groupby('day_id').agg(
        H_mean=('hurst','mean'),
        H_median=('hurst','median'),
        H_std=('hurst','std'),
        n_windows=('hurst','count')
    ).join(regime_counts)
    print(daily_summary.head())

    # 5. Autocorrelation of H (to measure persistence)
    def acf1(series):
        s = series.dropna()
        if len(s) < 10: return np.nan
        return s.autocorr(lag=1)
    acf_by_day = rolling_df.groupby('day_id')['H_smooth'].apply(acf1)
    print("Mean lag-1 autocorr of H:", acf_by_day.mean())

    # 6. Visual checks
    # - single day plot (use plot_single_day_hurst(rolling_df, some_day))
    # - heatmap (plot_heatmap_days) but you'll see fewer columns now



    # quick plot for one example day
    if not rolling_df.empty:
        some_day = rolling_df['day_id'].unique()[0]
        plot_single_day_hurst(rolling_df, some_day)
        # heatmap for a subset of days (first 30)
        sample = list(rolling_df['day_id'].unique()[:30])
        pivot = plot_heatmap_days(rolling_df, sample_days=sample)
