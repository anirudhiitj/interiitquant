import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
DATA_FOLDER = r'/data/quant14/EBY/'   # Path to your daily parquet files
PRICE_COL = 'Price'          # The column representing the Asset Price
MAX_LAG = 300                      # Lookback window (300 seconds = 5 mins)
MIN_ROWS = 1000                    # Skip days with insufficient data

# ==========================================
# 📐 CUSTOM HURST CALCULATION (Vectorized)
# ==========================================
def calculate_hurst_custom(time_series, max_lag=100):
    """
    Calculates H using the Diffusion Volatility method.
    Math: StdDev(t) ~ t^H
    Slope of log(StdDev) vs log(Lag) = H
    """
    # Ensure numpy array for speed
    ts = np.array(time_series)
    
    # Create the range of lags (e.g., 2 seconds to 300 seconds)
    lags = np.arange(2, max_lag)
    
    # Vectorized calculation of Standard Deviation at each lag
    tau = []
    for lag in lags:
        # Price difference: Price(t + lag) - Price(t)
        diffs = ts[lag:] - ts[:-lag]
        tau.append(np.std(diffs))
    
    # Fit linear regression on Log-Log scale
    # log(std) = H * log(lag) + c
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # The slope (poly[0]) is the Hurst Exponent
    return poly[0]

# ==========================================
# 🚀 MAIN EXECUTION LOOP
# ==========================================
def main():
    # 1. Get list of files
    files = glob.glob(os.path.join(DATA_FOLDER, "*.parquet"))
    
    if not files:
        print(f"❌ No Parquet files found in {DATA_FOLDER}")
        return

    print(f"📂 Found {len(files)} files. Calculating daily Hurst exponents...")
    
    daily_hurst_values = []
    processed_count = 0

    # 2. Iterate through files with Progress Bar
    for file_path in tqdm(files, unit="day"):
        try:
            # OPTIMIZATION: Read only the necessary column
            df = pd.read_parquet(file_path, columns=[PRICE_COL])
            
            # Basic validation
            if len(df) < MIN_ROWS:
                continue
                
            # Drop NaNs just in case
            prices = df[PRICE_COL].dropna().values
            
            # Calculate H
            h_val = calculate_hurst_custom(prices, max_lag=MAX_LAG)
            
            # Store result (Filter out mathematical errors)
            if not np.isnan(h_val) and 0 < h_val < 1.5:
                daily_hurst_values.append(h_val)
                processed_count += 1
                
        except Exception as e:
            # Print error but keep going
            print(f"\n⚠️ Error processing {os.path.basename(file_path)}: {e}")
            continue

    # ==========================================
    # 📊 INTERACTIVE VISUALIZATION (PLOTLY)
    # ==========================================
    if not daily_hurst_values:
        print("❌ No valid Hurst values calculated.")
        return

    # Statistics
    avg_hurst = np.mean(daily_hurst_values)
    median_hurst = np.median(daily_hurst_values)
    std_hurst = np.std(daily_hurst_values)

    print("\n" + "="*40)
    print(f"📈 ANALYSIS COMPLETE ({processed_count} Days)")
    print("="*40)
    print(f"Mean H:   {avg_hurst:.4f}")
    print(f"Median H: {median_hurst:.4f}")
    print(f"Std Dev:  {std_hurst:.4f}")
    
    # Define Regime
    if avg_hurst < 0.45:
        regime = "MEAN REVERTING (Use Bollinger Bands)"
        zone_color = "green"
    elif avg_hurst > 0.55:
        regime = "TRENDING (Use Momentum/Breakout)"
        zone_color = "red"
    else:
        regime = "RANDOM/MIXED (High Risk - Filter carefully)"
        zone_color = "orange"
        
    print(f"Regime:   {regime}")

    # Create Plotly Figure
    fig = px.histogram(
        daily_hurst_values, 
        nbins=40,
        title=f'<b>Distribution of Daily Hurst Exponents</b><br><sub>Mean H: {avg_hurst:.3f} | Regime: {regime}</sub>',
        labels={'value': 'Hurst Exponent (H)', 'count': 'Frequency (Days)'},
        color_discrete_sequence=['#3366CC'],
        opacity=0.75
    )

    # Add Vertical Line for Random Walk (0.5)
    fig.add_vline(
        x=0.5, 
        line_width=3, 
        line_dash="dash", 
        line_color="#EF553B",
        annotation_text="Random Walk (0.5)", 
        annotation_position="top right"
    )

    # Add Vertical Line for Mean H
    fig.add_vline(
        x=avg_hurst, 
        line_width=4, 
        line_color="#00CC96",
        annotation_text=f"Mean ({avg_hurst:.2f})", 
        annotation_position="top left"
    )

    # Add Zone Annotations (Background Text)
    fig.add_annotation(
        x=0.35, y=0.95, 
        xref="x", yref="paper",
        text="Mean Reversion Zone", 
        showarrow=False, 
        font=dict(color="green", size=16, family="Arial Black")
    )
    
    fig.add_annotation(
        x=0.65, y=0.95, 
        xref="x", yref="paper",
        text="Trending Zone", 
        showarrow=False, 
        font=dict(color="red", size=16, family="Arial Black")
    )

    # Layout Polish
    fig.update_layout(
        xaxis_title="Hurst Exponent (H)",
        yaxis_title="Count of Days",
        showlegend=False,
        hovermode="x",
        bargap=0.05,
        template="plotly_white"
    )

    fig.show()

    # Save the figure with EBX in the filename
    fig.write_html("hurst_histogram_EBY.html")
    print("Figure saved as hurst_histogram_EBY.html")

if __name__ == "__main__":
    main()