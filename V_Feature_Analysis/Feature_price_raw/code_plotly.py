import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "/data/quant14/EBX"
OUTPUT_DIR = "/home/raid/Quant14/V_Feature_Analysis/Feature_price_raw/plots_interactive/"
DAYS = [0, 10, 20, 50, 150, 180, 100, 200, 252]  # same as before
PRICE_COL = "Price"

# =============================================================================
# SELECTED FEATURES
# =============================================================================
top_features = [
    "V1_T4",
    "V2_T4", "V2_T8", "V2_T10",
    "V3_T7", "V3_T12",
    "V4_T2", "V5",
    "V8_T1_T12", "V8_T3_T4", "V8_T4_T5", "V8_T9_T12", "V8_T7_T11"
]

# =============================================================================
# SETUP
# =============================================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"📊 Feature vs Price plotting (RAW DATA, INTERACTIVE) — saving HTMLs to {OUTPUT_DIR}\n")

# =============================================================================
# MAIN LOOP
# =============================================================================
for DAY in DAYS:
    print(f"📅 Processing day {DAY}...")
    day_path = Path(DATA_DIR) / f"day{DAY}.csv"
    if not day_path.exists():
        print(f"⚠️ Missing file: {day_path}")
        continue

    df = pd.read_csv(day_path)
    if PRICE_COL not in df.columns:
        print(f"⚠️ No '{PRICE_COL}' column in day{DAY}.csv")
        continue

    # Handle time column
    if np.issubdtype(df["Time"].dtype, np.number):
        df["Time"] = pd.to_timedelta(df["Time"], unit="s")
    else:
        df["Time"] = pd.to_timedelta(df["Time"])

    # Convert Timedelta to datetime for readable x-axis
    df["Time_dt"] = pd.to_datetime(df["Time"].dt.total_seconds(), unit="s")

    day_out_dir = Path(OUTPUT_DIR) / f"day{DAY}"
    day_out_dir.mkdir(exist_ok=True, parents=True)

    for feature in top_features:
        if feature not in df.columns:
            print(f"  ⚠️ Skipping {feature} — not in file.")
            continue

        f_values = df[feature]
        p_values = df[PRICE_COL]

        # Create Plotly figure
        fig = go.Figure()

        # Feature trace (left axis)
        fig.add_trace(go.Scatter(
            x=df["Time_dt"],
            y=f_values,
            name=f"{feature} (raw)",
            line=dict(color="royalblue", width=1.8),
            yaxis="y1",
            hovertemplate="Time=%{x}<br>" + f"{feature}=%{{y:.4f}}<extra></extra>",
        ))

        # Price trace (right axis)
        fig.add_trace(go.Scatter(
            x=df["Time_dt"],
            y=p_values,
            name="Price",
            line=dict(color="darkorange", width=1.2, dash="dash"),
            yaxis="y2",
            hovertemplate="Time=%{x}<br>Price=%{y:.4f}<extra></extra>",
        ))

        # Layout
        fig.update_layout(
            title=f"{feature} vs Price (RAW DATA, INTERACTIVE) — Day {DAY}",
            xaxis=dict(title="Time", showgrid=False),
            yaxis=dict(title=f"{feature} (raw)", side="left", showgrid=True),
            yaxis2=dict(title="Price", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.02, y=0.98, bordercolor="gray", borderwidth=1),
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=60, r=60, t=60, b=40)
        )

        # Save HTML file
        out_path = day_out_dir / f"{feature}_vs_price_day{DAY}_raw.html"
        fig.write_html(str(out_path))
        print(f"  ✅ Saved {out_path.name}")

print("\n✨ All done! Interactive HTML plots saved successfully.")
