import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# =======================================================
# CONFIG
# =======================================================
DATA_DIR = Path("/data/quant14/EBX")
OUTPUT_DIR = Path("/home/raid/Quant14/V_Feature_Analysis/zstrat/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAY = 1
PRICE_COL = "Price"

# SMA windows (1s bars)
SMA_5M = 5 * 60     # 300 bars
SMA_15M = 22 * 60   # 900 bars
SMA_60M = 60 * 60   # 3600 bars


# =======================================================
# LOAD + PREP
# =======================================================
df = pd.read_parquet(DATA_DIR / f"day{DAY}.parquet")

# Validate input
if "Time" not in df or PRICE_COL not in df:
    raise KeyError("Dataset missing 'Time' or 'Price' column.")

# Parse time column
df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
df = df.dropna(subset=["Time"])

df["TimeSeconds"] = df["Time"].dt.total_seconds()

# Human readable HH:MM:SS
df["TimeLabel"] = df["Time"].apply(
    lambda x: f"{int(x.total_seconds()//3600):02d}:{int((x.total_seconds()%3600)//60):02d}:{int(x.total_seconds()%60):02d}"
)

df = df.sort_values("TimeSeconds").reset_index(drop=True)


# =======================================================
# SIMPLE MOVING AVERAGES
# =======================================================
df["SMA_5m"] = df[PRICE_COL].rolling(SMA_5M).mean()
df["SMA_15m"] = df[PRICE_COL].rolling(SMA_15M).mean()
df["SMA_60m"] = df[PRICE_COL].rolling(SMA_60M).mean()


# =======================================================
# PLOT
# =======================================================
fig = go.Figure()

# Price
fig.add_trace(go.Scatter(
    x=df["TimeLabel"],
    y=df[PRICE_COL],
    name="Price",
    line=dict(color="black", width=2),
    hovertemplate="Time: %{x}<br>Price: %{y:.2f}<extra></extra>"
))

# 5m SMA
fig.add_trace(go.Scatter(
    x=df["TimeLabel"],
    y=df["SMA_5m"],
    name="SMA 5m",
    line=dict(color="#0a7ff9", width=1.5),
    hovertemplate="Time: %{x}<br>SMA5: %{y:.2f}<extra></extra>"
))

# 15m SMA
fig.add_trace(go.Scatter(
    x=df["TimeLabel"],
    y=df["SMA_15m"],
    name="SMA 15m",
    line=dict(color="#23C552", width=1.5),
    hovertemplate="Time: %{x}<br>SMA15: %{y:.2f}<extra></extra>"
))

# 60m SMA
fig.add_trace(go.Scatter(
    x=df["TimeLabel"],
    y=df["SMA_60m"],
    name="SMA 60m",
    line=dict(color="#E83A14", width=1.5),
    hovertemplate="Time: %{x}<br>SMA60: %{y:.2f}<extra></extra>"
))


fig.update_layout(
    title=f"Day {DAY} — Price + SMA(5m,15m,60m)",
    template="plotly_white",
    hovermode="x unified",
    width=1500,
    height=750,
    xaxis=dict(
        title="Time (HH:MM:SS)",
        tickangle=-45,
        nticks=20
    ),
    yaxis=dict(title="Price"),
)

out_path = OUTPUT_DIR / f"day{DAY}_price_sma.html"
fig.write_html(out_path)
print(f"💾 Saved → {out_path}")
