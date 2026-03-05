import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "/data/quant14/EBX"
OUTPUT_DIR = "/home/raid/Quant14/V_Feature_Analysis/Feature_price_raw/plots/"
FIGURE_DPI = 300
DAYS = [0, 10, 20, 50, 150, 180, 100, 200, 252]  # Change this list as needed
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
plt.style.use("seaborn-v0_8-whitegrid")
print(f"📊 Feature vs Price plotting (RAW DATA) — saving PNGs to {OUTPUT_DIR}\n")

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

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        # Plot feature (raw)
        ax1.plot(df["Time_dt"], f_values, color="royalblue", linewidth=1.8, label=f"{feature}")
        ax1.set_ylabel(f"{feature} (raw)", color="royalblue", fontweight="bold")

        # Plot price
        ax2.plot(df["Time_dt"], p_values, color="darkorange", linewidth=1.2, linestyle="--", label="Price")
        ax2.set_ylabel("Price", color="darkorange", fontweight="bold")

        # Format x-axis for time
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate(rotation=45)

        # Title and layout
        plt.title(f"{feature} vs Price (RAW DATA) — Day {DAY}", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Time")
        fig.tight_layout()

        # Save PNG
        out_path = day_out_dir / f"{feature}_vs_price_day{DAY}_raw.png"
        plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✅ Saved {out_path.name}")

print("\n✨ All done! RAW DATA PNG plots saved successfully.")
