# hurst_by_day_full.py
# Requirements:
# pip install pandas numpy pyarrow plotly tqdm scipy

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# -------------------------
# Configurable parameters
# -------------------------
FOLDER = "/data/quant14/EBY/"   # <-- set this to your folder with per-day parquet files
PARQUET_GLOB = "*.parquet"         # or "*.pqt"
TS_COL = "Time"               # or your timestamp column name if any
PRICE_COL = "Price"                # price column to use; fallback to numeric column if missing
MIN_POINTS = 1000                   # skip days with fewer points (adjust if needed)
MAX_LAG = 200                      # max lag for variogram method (cap)
MIN_LAGS = 20                      # minimal number of lags required
BOOTSTRAP_N = 200                  # bootstrap replications for CI (200 is a reasonable start)
BOOTSTRAP_BLOCK_PCT = 0.05         # block size as fraction of N for block bootstrap (5%)
SAVE_CSV = "hurst_by_day_results.csv"
PLOT_HTML = "hurst_distribution.html"
# -------------------------

def hurst_variogram(ts, max_lag=None, min_lags=MIN_LAGS):
    """
    Variogram-based Hurst estimator.
    ts: 1D numpy array (preferably log(price) or integrated returns)
    returns H (float) or np.nan
    """
    ts = np.asarray(ts, dtype=float)
    N = ts.size
    if N < 10:
        return np.nan

    if max_lag is None:
        max_lag = min(MAX_LAG, max(2, N // 4))
    max_lag = int(max_lag)
    lags = np.arange(2, max_lag + 1)
    if lags.size < min_lags:
        return np.nan

    # compute tau(lag) = sqrt(mean((x[t+lag]-x[t])^2))
    taus = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        # if diffs empty or invalid, skip
        if diffs.size == 0:
            taus.append(np.nan)
        else:
            taus.append(np.sqrt(np.mean(diffs * diffs)))
    taus = np.array(taus, dtype=float)

    mask = (taus > 0) & np.isfinite(taus)
    if mask.sum() < min_lags:
        return np.nan

    lags = lags[mask]
    taus = taus[mask]

    # linear regression on log-log: log(tau) = H * log(lag) + const
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(lags), np.log(taus))
    H = float(slope)
    return H

def hurst_dfa(ts, min_win=4, max_win=None, n_windows=20):
    """
    Simple Detrended Fluctuation Analysis (DFA) estimator.
    Returns H or np.nan. More robust to polynomial trends.
    """
    ts = np.asarray(ts, dtype=float)
    N = ts.size
    if N < 20:
        return np.nan
    if max_win is None:
        max_win = max(8, N // 4)
    # construct scales (window sizes) log-spaced
    scales = np.unique(np.floor(np.logspace(np.log10(min_win), np.log10(max_win), n_windows)).astype(int))
    # center the series and integrate (profile)
    x = ts - np.mean(ts)
    y = np.cumsum(x)
    F = []
    scales_used = []
    for s in scales:
        if s < 4:
            continue
        nseg = N // s
        if nseg < 2:
            continue
        # compute RMS of detrended segments
        rms = []
        for v in range(nseg):
            seg = y[v*s:(v+1)*s]
            t = np.arange(s)
            # linear detrend (1st order)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        if len(rms) > 0:
            F.append(np.mean(rms))
            scales_used.append(s)
    if len(F) < 5:
        return np.nan
    F = np.array(F)
    scales_used = np.array(scales_used)
    mask = (F > 0) & np.isfinite(F)
    if mask.sum() < 5:
        return np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(scales_used[mask]), np.log(F[mask]))
    H = float(slope)
    return H

def block_bootstrap(series, estimator, n_boot=BOOTSTRAP_N, block_size=None, rng_seed=None):
    """
    Block bootstrap for dependent time series. Returns percentiles and stats.
    block_size: if None, set to max(10, int(BOOTSTRAP_BLOCK_PCT * N))
    """
    rng = np.random.RandomState(rng_seed)
    N = len(series)
    if N < 2:
        return None
    if block_size is None:
        block_size = max(10, int(max(1, BOOTSTRAP_BLOCK_PCT * N)))
    n_blocks = int(np.ceil(N / block_size))
    Hs = []
    for _ in range(n_boot):
        sampled = []
        for _b in range(n_blocks):
            start = rng.randint(0, max(1, N - block_size + 1))
            sampled.extend(series[start:start+block_size])
        sampled = np.array(sampled[:N])
        h = estimator(sampled)
        Hs.append(h)
    Hs = np.array(Hs, dtype=float)
    Hs = Hs[np.isfinite(Hs)]
    if Hs.size == 0:
        return None
    summary = {
        'boot_n': Hs.size,
        'mean': float(np.mean(Hs)),
        'median': float(np.median(Hs)),
        'std': float(np.std(Hs)),
        'ci_lower': float(np.percentile(Hs, 2.5)),
        'ci_upper': float(np.percentile(Hs, 97.5)),
        'all': Hs  # keep for debugging if needed
    }
    return summary

def compute_hurst_for_folder(folder_path,
                             pattern=PARQUET_GLOB,
                             ts_col=TS_COL,
                             price_col=PRICE_COL,
                             min_points=MIN_POINTS):
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in {folder_path} with pattern {pattern}")

    rows = []
    for fp in tqdm(files, desc="Processing files"):
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
            continue

        # infer day id
        day_id = None
        if 'date_id' in df.columns:
            day_id = df['date_id'].iloc[0]
        elif 'date' in df.columns:
            day_id = df['date'].iloc[0]
        else:
            # fallback: filename (without extension)
            day_id = os.path.splitext(os.path.basename(fp))[0]

        # sort by timestamp if exists
        if ts_col in df.columns:
            try:
                df = df.sort_values(ts_col)
            except Exception:
                df = df.reset_index(drop=True)

        # pick series: prefer log-price
        if price_col in df.columns:
            price = df[price_col].astype(float).values
            if np.all(price > 0):
                series = np.log(price)
            else:
                # fallback: cumulative returns (integrated returns)
                ret = np.zeros_like(price)
                ret[1:] = price[1:] / price[:-1] - 1.0
                series = np.cumsum(ret)
        else:
            # fallback: take first numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) == 0:
                print(f"No numeric columns found in {fp}, skipping")
                continue
            col = num_cols[0]
            series = df[col].astype(float).values

        n = len(series)
        if n < min_points:
            # skip / record as NaN
            rows.append({
                'file': os.path.basename(fp),
                'day_id': day_id,
                'n_points': n,
                'var_h': np.nan,
                'dfa_h': np.nan,
                'var_h_bs_median': np.nan,
                'var_h_bs_ci_lower': np.nan,
                'var_h_bs_ci_upper': np.nan,
            })
            continue

        # compute estimators
        var_h = hurst_variogram(series, max_lag=MAX_LAG, min_lags=MIN_LAGS)
        dfa_h = hurst_dfa(series)

        # bootstrap variogram for CI
        bs = block_bootstrap(series, lambda s: hurst_variogram(s, max_lag=MAX_LAG, min_lags=MIN_LAGS),
                             n_boot=BOOTSTRAP_N, block_size=max(10, int(BOOTSTRAP_BLOCK_PCT * n)), rng_seed=42)
        if bs is None:
            bs_median = np.nan
            bs_lower = np.nan
            bs_upper = np.nan
        else:
            bs_median = bs['median']
            bs_lower = bs['ci_lower']
            bs_upper = bs['ci_upper']

        rows.append({
            'file': os.path.basename(fp),
            'day_id': day_id,
            'n_points': n,
            'var_h': var_h,
            'dfa_h': dfa_h,
            'var_h_bs_median': bs_median,
            'var_h_bs_ci_lower': bs_lower,
            'var_h_bs_ci_upper': bs_upper,
        })

    res = pd.DataFrame(rows)
    # Save CSV
    res.to_csv(SAVE_CSV, index=False)
    print(f"Saved results to {SAVE_CSV}")
    return res

# -------------------------
# Visualization
# -------------------------
def plot_hurst_results(df, hurst_col='var_h', bs_lower='var_h_bs_ci_lower', bs_upper='var_h_bs_ci_upper'):
    """
    Plot histogram + box + scatter H vs n_points and show bootstrap CI summary.
    """
    df_plot = df.copy()
    # drop NaNs for plotting
    df_plot = df_plot[df_plot[hurst_col].notna()].reset_index(drop=True)
    if df_plot.empty:
        print("No valid hurst values to plot.")
        return

    mean_h = df_plot[hurst_col].mean()
    median_h = df_plot[hurst_col].median()

    # Histogram with box marginal
    fig_hist = px.histogram(df_plot, x=hurst_col, nbins=60, marginal="box",
                            title=f"Hurst exponent distribution (estimator={hurst_col})")
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="H=0.5", annotation_position="top left")
    fig_hist.add_vline(x=mean_h, line_dash="dot", line_color="green", annotation_text=f"mean={mean_h:.3f}", annotation_position="top right")
    fig_hist.add_vline(x=median_h, line_dash="dot", line_color="blue", annotation_text=f"median={median_h:.3f}", annotation_position="top right")

    # Scatter H vs n_points with CI bars
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_plot['n_points'], y=df_plot[hurst_col],
        mode='markers',
        name='Hurst',
        marker=dict(size=8, opacity=0.8)
    ))
    # add CI error bars if available
    if bs_lower in df_plot.columns and bs_upper in df_plot.columns:
        fig_scatter.add_trace(go.Scatter(
            x=df_plot['n_points'],
            y=df_plot[hurst_col],
            mode='markers',
            marker=dict(opacity=0),
            error_y=dict(type='data',
                         array=(df_plot[hurst_col] - df_plot[bs_lower]).fillna(0).values,
                         arrayminus=(df_plot[bs_upper] - df_plot[hurst_col]).fillna(0).values,
                         thickness=1.5, width=3),
            showlegend=False
        ))
    fig_scatter.update_layout(title="Hurst vs number of points (per day)", xaxis_title="n_points", yaxis_title="Hurst")

    # Boxplot / violin of H per estimator if multiple estimators exist
    fig_box = go.Figure()
    if 'var_h' in df.columns:
        fig_box.add_trace(go.Box(y=df['var_h'], name='variogram', boxmean=True))
    if 'dfa_h' in df.columns:
        fig_box.add_trace(go.Box(y=df['dfa_h'], name='DFA', boxmean=True))

    fig_box.update_layout(title="Hurst estimators distribution (boxplots)", yaxis_title="Hurst")

    # Combine into a single HTML
    with open(PLOT_HTML, "w") as f:
        f.write("<html><body>\n")
        f.write("<h2>Hurst per-day analysis</h2>\n")
        f.write("<div>\n")
        f.write(fig_hist.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</div>\n<hr>\n")
        f.write(fig_scatter.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>\n")
        f.write(fig_box.to_html(full_html=False, include_plotlyjs=False))
        f.write("</div>\n</body></html>")

    print(f"Saved interactive plots to {PLOT_HTML}")
    # show summary (console)
    print("\nSummary statistics for H (variogram):")
    print(df['var_h'].describe())
    print("\nDays with low confidence (CI contains 0.5):")
    if 'var_h_bs_ci_lower' in df.columns and 'var_h_bs_ci_upper' in df.columns:
        cond = (df['var_h_bs_ci_lower'] <= 0.5) & (df['var_h_bs_ci_upper'] >= 0.5)
        low_conf = df[cond]
        if low_conf.empty:
            print("None (all have CI clearly away from 0.5) or no bootstrap info.")
        else:
            print(low_conf[['file','day_id','n_points','var_h','var_h_bs_ci_lower','var_h_bs_ci_upper']].head(10))
    return fig_hist, fig_scatter, fig_box

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    print("Starting Hurst per-day computation...")
    results_df = compute_hurst_for_folder(FOLDER, pattern=PARQUET_GLOB, ts_col=TS_COL, price_col=PRICE_COL, min_points=MIN_POINTS)
    print("Computed Hurst for files:", len(results_df))
    # Quick inspection
    print(results_df.head())
    # Visualization
    fig_hist, fig_scatter, fig_box = plot_hurst_results(results_df, hurst_col='var_h')
    print("Done.")
