import cudf
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = Path("/data/quant14/EBY")
# CHANGE: Use a list of days instead of a single integer
DAYS_TO_PLOT = [25,22,50,61,31,77,92,116,107,128,147,153,175,177,176,211,244] 
PRICE_COLUMN = "Price"
TIME_COLUMN = "Time"
# ==========================================================

# Initialize the figure *before* the loop begins
fig = go.Figure()

print(f"Starting multiple day plot comparison for: {DAYS_TO_PLOT}\n")

# Loop through each day defined in the configuration list
for day_num in DAYS_TO_PLOT:
    file_path = DATA_DIR / f"day{day_num}.parquet"

    # 1. Sanity check: ensure file exists before trying to load with cuDF
    if not file_path.exists():
        print(f"[Warning] File missing for Day {day_num}. Skipping: {file_path.name}")
        continue

    try:
        print(f"Processing Day {day_num}...")
        
        # 2. Load data using GPU (cuDF)
        # Only reading necessary columns to save memory
        df = cudf.read_parquet(file_path, columns=[TIME_COLUMN, PRICE_COLUMN])
        
        # 3. Convert to pandas for plotting
        df_pd = df.to_pandas()
        df_pd[TIME_COLUMN] = pd.to_timedelta(df_pd[TIME_COLUMN])

        print(f"  -> Loaded {len(df_pd):,} rows. Range: ${df_pd[PRICE_COLUMN].min():.2f} - ${df_pd[PRICE_COLUMN].max():.2f}")

        # 4. Add trace to the figure
        fig.add_trace(
            go.Scatter(
                x=df_pd[TIME_COLUMN],
                y=df_pd[PRICE_COLUMN],
                mode="lines",
                # Removed line=dict(color="black") so Plotly cycles colors automatically
                line=dict(width=1.5), 
                name=f"Day {day_num}",  # Dynamic legend name
                # Updated hovertemplate to include the Day number
                hovertemplate=f"<b>Day: {day_num}</b><br><b>Time:</b> %{{x}}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>"
            )
        )
        
        # Optional: manually free up GPU memory memory before next iteration
        del df
        del df_pd

    except Exception as e:
        print(f"[Error] Could not process Day {day_num}: {e}")


# ==========================================================
# Final Layout Adjustments (happens after loop finishes)
# ==========================================================
day_list_str = ", ".join(map(str, DAYS_TO_PLOT))

fig.update_layout(
    title=f"Price Comparison — Days: {day_list_str}",
    xaxis_title="Time (from start of day)",
    yaxis_title="Price ($)",
    template="plotly_white",
    hovermode="x unified", # Great for comparing values at the same time across days
    height=700,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

print("\nRendering plot...")
fig.show()