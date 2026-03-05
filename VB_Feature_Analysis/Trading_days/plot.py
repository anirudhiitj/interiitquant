import cudf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBY")
DAY_TO_PLOT = 104
PRICE_COLUMN = "Price"
VOL_COLUMN = "PB9_T1"
TIME_COLUMN = "Time"

ATR_WINDOW = 600
ATR_THRESHOLD = 0.01
# ==========================================================

# ----------------------------------------------------------
# Load data using cuDF (GPU)
# ----------------------------------------------------------
file_path = DATA_DIR / f"day{DAY_TO_PLOT}.parquet"
print(f"\nLoading {file_path} ...")
df = cudf.read_parquet(file_path)
print(f"Loaded {len(df):,} rows from Day {DAY_TO_PLOT}")

# ----------------------------------------------------------
# Compute ATR proxy on GPU
# ----------------------------------------------------------
df = df[[TIME_COLUMN, PRICE_COLUMN, VOL_COLUMN]].dropna()

# Compute ATR (rolling mean of absolute diff of PB9_T1)
df["ATR"] = (
    df[VOL_COLUMN].diff().abs().rolling(window=ATR_WINDOW, min_periods=1).mean().fillna(0)
)

# Convert to pandas for Plotly
df_pd = df.to_pandas()

# Ensure Time is in seconds
if not np.issubdtype(df_pd[TIME_COLUMN].dtype, np.number):
    df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN]).dt.total_seconds()

# ----------------------------------------------------------
# Detect ATR threshold crossing
# ----------------------------------------------------------
atr_cross_indices = np.where(df_pd["ATR"].values >= ATR_THRESHOLD)[0]
if len(atr_cross_indices) > 0:
    atr_cross_time = df_pd[TIME_COLUMN].iloc[atr_cross_indices[0]]
    atr_cross_value = df_pd["ATR"].iloc[atr_cross_indices[0]]
    print(f"ATR first crosses {ATR_THRESHOLD} at {atr_cross_time:.2f}s (ATR={atr_cross_value:.4f})")
else:
    atr_cross_time = None
    print("ATR never crosses the threshold.")

# ----------------------------------------------------------
# Plotly Visualization (Dual Y-Axis)
# ----------------------------------------------------------
fig = go.Figure()

# --- PRICE LINE (Left Axis) ---
fig.add_trace(go.Scatter(
    x=df_pd[TIME_COLUMN],
    y=df_pd[PRICE_COLUMN],
    mode="lines",
    name="Price",
    line=dict(color="black", width=2),
    yaxis="y1"
))

# --- ATR LINE (Right Axis) ---
fig.add_trace(go.Scatter(
    x=df_pd[TIME_COLUMN],
    y=df_pd["ATR"],
    mode="lines",
    name=f"ATR (window={ATR_WINDOW})",
    line=dict(color="red", width=2),
    yaxis="y2"
))

# --- HORIZONTAL ATR THRESHOLD LINE (on right axis) ---
fig.add_hline(
    y=ATR_THRESHOLD,
    line=dict(color="red", dash="dot", width=2),
    annotation_text=f"ATR Threshold = {ATR_THRESHOLD}",
    annotation_position="top right",
    yref="y2"
)

# --- VERTICAL LINE AT FIRST CROSSING ---
if atr_cross_time is not None:
    fig.add_vline(
        x=atr_cross_time,
        line=dict(color="green", dash="dot", width=2),
        annotation_text=f"ATR crosses @ {atr_cross_time:.2f}s",
        annotation_position="top left"
    )

# ----------------------------------------------------------
# Layout Settings
# ----------------------------------------------------------
fig.update_layout(
    title=f"<b>Day {DAY_TO_PLOT} — Price (Left Axis) vs ATR (Right Axis) with Threshold</b>",
    xaxis=dict(title="Time (seconds)"),
    yaxis=dict(
        title="Price ($)",
        side="left",
        showgrid=True
    ),
    yaxis2=dict(
        title="ATR Value",
        side="right",
        overlaying="y",
        showgrid=False
    ),
    legend=dict(x=0, y=1.1, orientation="h"),
    template="plotly_white",
    height=750,
    hovermode="x unified"
)

fig.show()
