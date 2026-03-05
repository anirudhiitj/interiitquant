import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import warnings
import time
import dask_cudf
import gc
from multiprocessing import Pool
import re
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "/data/quant14/EBX"
OUTPUT_DIR = "/home/raid/Quant14/V_Feature_Analysis/Feature_price_raw/plots_interactive/"
FEATURE = "V5"
PRICE_COL = "Price"
TIME_COL = "Time"
FIRST_20_MINS = 30 * 60  # 1200 seconds
EXCLUDED_DAYS = list(range(60, 80))  # Days 60-79
TARGET_DAY_RANGE = list(range(107, 147))  # Days 107-146
PLOT_DAYS = [107, 108, 137, 124, 14, 183, 311]
ALL_DAYS = list(range(510))
MAX_WORKERS = 32

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.(parquet|csv)', str(filepath))
    return int(match.group(1)) if match else -1

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================
def load_day_data(file_path: str, day_num: int, required_cols: list) -> pd.DataFrame:
    """Load a single day file using GPU acceleration"""
    try:
        # Try parquet first
        if Path(file_path).suffix == '.parquet':
            ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
            gdf = ddf.compute()
            df = gdf.to_pandas()
            del ddf, gdf
            gc.collect()
        else:
            # Fallback to CSV
            df = pd.read_csv(file_path, usecols=required_cols)
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        return df
    
    except Exception as e:
        print(f"  ⚠️ Error loading day {day_num}: {e}")
        return None

def process_day_for_stats(args):
    """Process a single day to extract V5 statistics from first 20 minutes"""
    file_path, day_num, excluded_days, target_range = args
    
    if day_num in excluded_days:
        return None, None
    
    required_cols = [TIME_COL, FEATURE]
    df = load_day_data(file_path, day_num, required_cols)
    
    if df is None or FEATURE not in df.columns:
        return None, None
    
    try:
        # Handle time column
        if np.issubdtype(df[TIME_COL].dtype, np.number):
            df[TIME_COL] = pd.to_timedelta(df[TIME_COL], unit="s")
        else:
            df[TIME_COL] = pd.to_timedelta(df[TIME_COL])
        
        # Get first 20 mins data
        first_20_mask = df[TIME_COL].dt.total_seconds() <= FIRST_20_MINS
        v5_first_20 = df.loc[first_20_mask, FEATURE].dropna().values
        
        if len(v5_first_20) == 0:
            return None, None
        
        # Return data for overall and target range (if applicable)
        overall_data = v5_first_20
        target_data = v5_first_20 if day_num in target_range else None
        
        return overall_data, target_data
    
    except Exception as e:
        print(f"  ⚠️ Error processing day {day_num}: {e}")
        return None, None

# =============================================================================
# SETUP
# =============================================================================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"📊 V5 Feature Analysis (RAW DATA, INTERACTIVE, GPU-ACCELERATED)\n")
print(f"🎯 Target day range for stats: {TARGET_DAY_RANGE[0]}-{TARGET_DAY_RANGE[-1]}")
print(f"📈 Days to plot: {PLOT_DAYS}\n")

start_time = time.time()

# =============================================================================
# COLLECT ALL FILES
# =============================================================================
print("📂 Scanning for data files...")
all_files = []
for day_num in ALL_DAYS:
    # Try parquet first, then CSV
    parquet_path = Path(DATA_DIR) / f"day{day_num}.parquet"
    csv_path = Path(DATA_DIR) / f"day{day_num}.csv"
    
    if parquet_path.exists():
        all_files.append((str(parquet_path), day_num))
    elif csv_path.exists():
        all_files.append((str(csv_path), day_num))

print(f"Found {len(all_files)} data files")

# =============================================================================
# PARALLEL PROCESSING FOR STATISTICS
# =============================================================================
print(f"\n📐 Calculating statistics using {MAX_WORKERS} workers...")

pool_args = [
    (file_path, day_num, EXCLUDED_DAYS, TARGET_DAY_RANGE)
    for file_path, day_num in all_files
]

overall_v5_values = []
target_range_v5_values = []

with Pool(processes=MAX_WORKERS) as pool:
    results = pool.map(process_day_for_stats, pool_args)

# Collect results
for overall_data, target_data in results:
    if overall_data is not None:
        overall_v5_values.extend(overall_data)
    if target_data is not None:
        target_range_v5_values.extend(target_data)

# Convert to numpy arrays
overall_v5_values = np.array(overall_v5_values)
target_range_v5_values = np.array(target_range_v5_values)

# =============================================================================
# CALCULATE AND DISPLAY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("📊 OVERALL STATISTICS (All 510 days, first 20 mins, excluding 60-79)")
print("="*70)
print(f"Average V5:       {np.mean(overall_v5_values):.6f}")
print(f"25th Percentile:  {np.percentile(overall_v5_values, 25):.6f}")
print(f"50th Percentile:  {np.percentile(overall_v5_values, 50):.6f}")
print(f"75th Percentile:  {np.percentile(overall_v5_values, 75):.6f}")
print(f"90th Percentile:  {np.percentile(overall_v5_values, 90):.6f}")
print(f"Total samples:    {len(overall_v5_values):,}")

print("\n" + "="*70)
print(f"📊 TARGET RANGE STATISTICS (Days {TARGET_DAY_RANGE[0]}-{TARGET_DAY_RANGE[-1]}, first 20 mins)")
print("="*70)
print(f"Average V5:       {np.mean(target_range_v5_values):.6f}")
print(f"25th Percentile:  {np.percentile(target_range_v5_values, 25):.6f}")
print(f"50th Percentile:  {np.percentile(target_range_v5_values, 50):.6f}")
print(f"75th Percentile:  {np.percentile(target_range_v5_values, 75):.6f}")
print(f"90th Percentile:  {np.percentile(target_range_v5_values, 90):.6f}")
print(f"Total samples:    {len(target_range_v5_values):,}")
print("="*70 + "\n")

# Store percentiles for plotting
overall_percentiles = {
    'p25': np.percentile(overall_v5_values, 25),
    'p50': np.percentile(overall_v5_values, 50),
    'p75': np.percentile(overall_v5_values, 75),
    'p90': np.percentile(overall_v5_values, 90),
}

target_percentiles = {
    'p25': np.percentile(target_range_v5_values, 25),
    'p50': np.percentile(target_range_v5_values, 50),
    'p75': np.percentile(target_range_v5_values, 75),
    'p90': np.percentile(target_range_v5_values, 90),
}

# =============================================================================
# PLOT SELECTED DAYS
# =============================================================================
print("📈 Creating interactive plots...")

v5_out_dir = Path(OUTPUT_DIR) / "V5_analysis"
v5_out_dir.mkdir(exist_ok=True, parents=True)

for DAY in PLOT_DAYS:
    print(f"\n📅 Processing day {DAY}...")
    
    # Find the file for this day
    file_path = None
    for fp, day_num in all_files:
        if day_num == DAY:
            file_path = fp
            break
    
    if file_path is None:
        print(f"  ⚠️ Missing file for day {DAY}")
        continue

    try:
        required_cols = [TIME_COL, PRICE_COL, FEATURE]
        df = load_day_data(file_path, DAY, required_cols)
        
        if df is None or PRICE_COL not in df.columns or FEATURE not in df.columns:
            print(f"  ⚠️ Missing required columns in day{DAY}")
            continue

        # Handle time column
        if np.issubdtype(df[TIME_COL].dtype, np.number):
            df[TIME_COL] = pd.to_timedelta(df[TIME_COL], unit="s")
        else:
            df[TIME_COL] = pd.to_timedelta(df[TIME_COL])

        # Convert Timedelta to datetime for readable x-axis
        df["Time_dt"] = pd.to_datetime(df[TIME_COL].dt.total_seconds(), unit="s")

        f_values = df[FEATURE]
        p_values = df[PRICE_COL]

        # Create Plotly figure
        fig = go.Figure()

        # V5 trace (left axis)
        fig.add_trace(go.Scatter(
            x=df["Time_dt"],
            y=f_values,
            name="V5 (raw)",
            line=dict(color="royalblue", width=2),
            yaxis="y1",
            hovertemplate="Time=%{x}<br>V5=%{y:.6f}<extra></extra>",
        ))

        # Price trace (right axis)
        fig.add_trace(go.Scatter(
            x=df["Time_dt"],
            y=p_values,
            name="Price",
            line=dict(color="darkorange", width=1.5, dash="dash"),
            yaxis="y2",
            hovertemplate="Time=%{x}<br>Price=%{y:.4f}<extra></extra>",
        ))

        # Add percentile lines for overall (solid lines)
        colors_overall = {'p25': 'lightgreen', 'p50': 'green', 'p75': 'darkgreen', 'p90': 'purple'}
        for pct_name, pct_val in overall_percentiles.items():
            fig.add_hline(
                y=pct_val, 
                line_dash="solid", 
                line_color=colors_overall[pct_name],
                line_width=1.5,
                annotation_text=f"Overall {pct_name.upper()}",
                annotation_position="left",
                yref="y1"
            )

        # Add percentile lines for target range (dotted lines)
        colors_target = {'p25': 'lightcoral', 'p50': 'red', 'p75': 'darkred', 'p90': 'maroon'}
        for pct_name, pct_val in target_percentiles.items():
            fig.add_hline(
                y=pct_val, 
                line_dash="dot", 
                line_color=colors_target[pct_name],
                line_width=1.5,
                annotation_text=f"Range {pct_name.upper()}",
                annotation_position="right",
                yref="y1"
            )

        # Layout
        fig.update_layout(
            title=f"V5 vs Price (Day {DAY}) — Raw Data with Percentiles<br>" +
                  f"<sub>Green: Overall (All Days) | Red: Target Range (Days {TARGET_DAY_RANGE[0]}-{TARGET_DAY_RANGE[-1]})</sub>",
            xaxis=dict(title="Time", showgrid=False),
            yaxis=dict(title="V5 (raw)", side="left", showgrid=True),
            yaxis2=dict(title="Price", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.02, y=0.98, bordercolor="gray", borderwidth=1),
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=80, r=80, t=100, b=60),
            height=700
        )

        # Save HTML file
        out_path = v5_out_dir / f"V5_vs_price_day{DAY}_raw.html"
        fig.write_html(str(out_path))
        print(f"  ✅ Saved {out_path.name}")
        
    except Exception as e:
        print(f"  ❌ Error processing day {DAY}: {e}")
        continue

# =============================================================================
# SUMMARY
# =============================================================================
elapsed = time.time() - start_time
print("\n" + "="*70)
print("✨ All done! Interactive HTML plots and statistics calculated successfully.")
print(f"📁 Files saved to: {v5_out_dir}")
print(f"⏱️  Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
print("="*70)