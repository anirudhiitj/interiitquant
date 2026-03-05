import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "/data/quant14/EBX/day10.csv"
FEATURE = "V8_T1_T12"
PRICE_COL = "Price"
OUTPUT_DIR = "/home/raid/Quant14/V_Feature_Analysis/Feature_price_raw/feature_behavior/V8_T1_T12/"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
df = pd.read_csv(DATA_PATH)
# Handle both numeric and string time columns safely
if np.issubdtype(df["Time"].dtype, np.number):
    df["Time_dt"] = pd.to_timedelta(df["Time"], unit="s")
else:
    df["Time_dt"] = pd.to_timedelta(df["Time"])

df["Time_dt"] = pd.to_datetime(df["Time_dt"].dt.total_seconds(), unit="s")



# =============================================================================
# SPIKE DETECTION
# =============================================================================
mu, sigma = df[FEATURE].mean(), df[FEATURE].std()
threshold = mu + 3*sigma

spikes = df[df[FEATURE] > threshold].copy()
print(f"Detected {len(spikes)} spikes in {FEATURE}")
print(f"Threshold value = {threshold:.2f}")

# =============================================================================
# BASIC VISUALIZATION
# =============================================================================
plt.figure(figsize=(12,5))
plt.plot(df["Time_dt"], df[FEATURE], color='royalblue', linewidth=1.2, label=FEATURE)
plt.axhline(threshold, color='red', linestyle='--', label='Spike Threshold (μ + 3σ)')
plt.scatter(spikes["Time_dt"], spikes[FEATURE], color='darkred', label='Detected Spikes', zorder=5)
plt.title(f"{FEATURE} — Spike Detection (Day 10)", fontsize=14, fontweight="bold")
plt.xlabel("Time")
plt.ylabel("Feature Value")
plt.legend()
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f"{FEATURE}_spikes_day10.png", dpi=300)
plt.close()

# =============================================================================
# HISTOGRAM OF SPIKES
# =============================================================================
plt.figure(figsize=(7,4))
plt.hist(df[FEATURE], bins=100, color='skyblue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', linewidth=1.2)
plt.title(f"{FEATURE} Value Distribution", fontsize=13)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f"{FEATURE}_distribution_day10.png", dpi=300)
plt.close()

# =============================================================================
# PRICE AROUND SPIKES (simple correlation check)
# =============================================================================
df["Price_Return"] = df[PRICE_COL].pct_change()
spike_corr = df[FEATURE].corr(df["Price_Return"])
print(f"Correlation between {FEATURE} and price returns: {spike_corr:.4f}")

# Optional: Price during spikes
plt.figure(figsize=(12,5))
plt.plot(df["Time_dt"], df[PRICE_COL], color='orange', label="Price", linewidth=1)
plt.scatter(spikes["Time_dt"], spikes[PRICE_COL], color='red', marker='x', label="Price at Spikes")
plt.title(f"Price During {FEATURE} Spikes (Day 10)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f"{FEATURE}_price_during_spikes_day10.png", dpi=300)
plt.close()
