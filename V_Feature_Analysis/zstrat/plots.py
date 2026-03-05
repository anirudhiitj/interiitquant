import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import random
from scipy.signal import savgol_filter


# =======================================================
# CONFIG
# =======================================================
DATA_DIR = Path("/data/quant14/EBX")    # INPUT PARQUET DIRECTORY
OUTPUT_DIR = Path("/home/raid/Quant14/V_Feature_Analysis/zstrat/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_COL = "Price"

# 👉 CHOOSE A SINGLE DAY
DAY = 25

# 👉 OPTIONAL RANDOM DAYS (keep commented)
# NUM_DAYS = 7

# WINDOWS IN SECONDS (1 bar = 1 sec)
ROLLING_WINDOW = 30 * 60     # 30min rolling z-window
CALIB_WINDOW = 30 * 60       # 30min baseline window

# SMOOTHING PARAMETERS
SMOOTH_WINDOW = 10
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

# =======================================================
# HELPERS
# =======================================================
def prep_df(df):
    df = df.copy()

    if "Time" not in df or PRICE_COL not in df:
        raise KeyError("Missing Time or Price columns.")

    df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
    df.dropna(subset=["Time"], inplace=True)

    df["TimeSeconds"] = df["Time"].dt.total_seconds()

    df["TimeLabel"] = df["Time"].apply(
        lambda x: f"{int(x.total_seconds()//3600):02d}:{int((x.total_seconds()%3600)//60):02d}:{int(x.total_seconds()%60):02d}"
    )

    df.sort_values("TimeSeconds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_returns(df):
    df = df.copy()
    df["ret"] = df[PRICE_COL].pct_change()
    df["log_ret"] = np.log(df[PRICE_COL] / df[PRICE_COL].shift(1))
    return df


def add_smoothing(df, zcol):
    df[f"{zcol}_ema"] = df[zcol].ewm(span=SMOOTH_WINDOW, adjust=False).mean()
    df[f"{zcol}_sma"] = df[zcol].rolling(window=SMOOTH_WINDOW).mean()

    if df[zcol].notna().sum() > SAVGOL_WINDOW:
        try:
            temp_series = df[zcol].fillna(method='ffill').fillna(method='bfill')
            smoothed = savgol_filter(temp_series, SAVGOL_WINDOW, SAVGOL_POLY)
            df[f"{zcol}_savgol"] = np.where(df[zcol].isna(), np.nan, smoothed)
        except:
            df[f"{zcol}_savgol"] = np.nan
    else:
        df[f"{zcol}_savgol"] = np.nan

    return df


def add_rolling_z(df, col, window=ROLLING_WINDOW):
    mu = df[col].rolling(window).mean()
    sigma = df[col].rolling(window).std(ddof=0)

    df[f"{col}_z_roll"] = np.tanh((df[col] - mu) / sigma)
    return add_smoothing(df, f"{col}_z_roll")


def add_fixed_z(df, col, calib=CALIB_WINDOW):
    base = df.iloc[:calib][col].dropna()
    mu = base.mean()
    sigma = base.std(ddof=0)

    df[f"{col}_z_fix"] = np.nan
    if sigma > 0:
        df.loc[calib:, f"{col}_z_fix"] = np.tanh((df.loc[calib:, col] - mu) / sigma)

    return add_smoothing(df, f"{col}_z_fix")


# =======================================================
# PLOTTING
# =======================================================

from plotly.subplots import make_subplots
def plot_price_and_z_subplots(df, zcol, title, outname):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Price", "Z-Score (Smoothed)"),
        row_heights=[0.55, 0.45]
    )

    fig.add_trace(go.Scatter(
        x=df["TimeLabel"], y=df[PRICE_COL],
        name="Price", line=dict(color="black", width=1.3)
    ), row=1, col=1)

    for sub in ["ema", "sma", "savgol"]:
        col = f"{zcol}_{sub}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["TimeLabel"], y=df[col],
                name=col, line=dict(width=1.7)
            ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dot", row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        width=1500, height=900,
        showlegend=True
    )
    fig.write_html(OUTPUT_DIR / outname)
    print(f"💾 Saved {outname}")


# =======================================================
# LOAD SINGLE DAY
# =======================================================
day_file = DATA_DIR / f"day{DAY}.parquet"
print(f"📁 Loading → {day_file}")
df = pd.read_parquet(day_file)

df = prep_df(df)
df = add_returns(df)

# Z
df = add_rolling_z(df, "ret")
df = add_fixed_z(df, "ret")

# =======================================================
# OUTPUT
# =======================================================

plot_price_and_z_subplots(
    df, "ret_z_roll",
    f"Day{DAY} — Price + Rolling Z(ret)",
    f"day{DAY}_retZ_roll.html"
)

plot_price_and_z_subplots(
    df, "ret_z_fix",
    f"Day{DAY} — Price + Fixed Z(ret,30m)",
    f"day{DAY}_retZ_fix.html"
)

print("\n✨ DONE.")
