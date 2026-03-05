import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FOLDER = r'/data/quant14/EBY/'
PRICE_COL = 'Price'
# You need to identify a Volatility Feature. 
# Look for a column starting with 'VB' and a short time horizon 'T1' or 'T2'
# If you don't know, pick the first 'VB' column you see in df.columns
VOLATILITY_COL = 'VB1_T1' 

MIN_ROWS = 1000
MAX_LAG = 300

# ==========================================
# HURST FUNCTION
# ==========================================
def calculate_hurst_custom(time_series, max_lag=100):
    ts = np.array(time_series)
    lags = np.arange(2, max_lag)
    # Variance/Diffusion method
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

# ==========================================
# MAIN ANALYSIS
# ==========================================
def main():
    files = glob.glob(os.path.join(DATA_FOLDER, "*.parquet"))
    
    results = []

    print(f"Analyzing relationship between Hurst and Volatility...")

    # Analyze a subset (first 100) or all files. Increase limit if needed.
    for file_path in tqdm(files[:200], unit="file"): 
        try:
            # Load Price AND Volatility
            # We read the whole file or specific cols if known to be safe. 
            # Reading just cols is faster if we are sure of names.
            # To be safe with auto-detection, we read parquet then filter.
            # Ideally, read specific columns if VOLATILITY_COL is confirmed.
            df = pd.read_parquet(file_path)
            
            # Find the actual Volatility column if configured one doesn't exist
            if VOLATILITY_COL in df.columns:
                current_vol_col = VOLATILITY_COL
            else:
                # Auto-detect first VB column
                vol_cols = [c for c in df.columns if 'VB' in c]
                if not vol_cols:
                    continue
                current_vol_col = vol_cols[0]
            
            if len(df) < MIN_ROWS: continue
            
            # 1. Calculate Day's Hurst
            h_val = calculate_hurst_custom(df[PRICE_COL].values, max_lag=MAX_LAG)
            
            # 2. Calculate Day's Average Volatility
            avg_vol = df[current_vol_col].mean()
            
            # Record Data
            if not np.isnan(h_val) and 0.1 < h_val < 0.9: # Filter absolute errors
                results.append({
                    'File': os.path.basename(file_path),
                    'Hurst': h_val, 
                    'Volatility': avg_vol
                })
                
        except Exception as e:
            # print(f"Skipping {file_path}: {e}")
            continue

    if not results:
        print("No valid data processed. Check column names/paths.")
        return

    res_df = pd.DataFrame(results)
    
    # Calculate Correlation
    corr = res_df['Volatility'].corr(res_df['Hurst'])

    # ==========================================
    # PLOTLY INTERACTIVE REGIME MAP
    # ==========================================
    
    # 1. Calculate Linear Regression Trendline (Manually to avoid statsmodels dependency)
    slope, intercept = np.polyfit(res_df['Volatility'], res_df['Hurst'], 1)
    res_df['Trendline'] = slope * res_df['Volatility'] + intercept
    
    # 2. Setup Figure
    fig = go.Figure()

    # Scatter Points
    fig.add_trace(go.Scatter(
        x=res_df['Volatility'],
        y=res_df['Hurst'],
        mode='markers',
        name='Daily Data',
        text=res_df['File'], # Hover text
        marker=dict(
            size=8,
            color=res_df['Hurst'], # Color by Hurst value
            colorscale='RdYlGn_r', # Red (High H) to Green (Low H)
            showscale=True,
            colorbar=dict(title="Hurst")
        )
    ))

    # Trendline
    fig.add_trace(go.Scatter(
        x=res_df['Volatility'],
        y=res_df['Trendline'],
        mode='lines',
        name=f'Trend (Corr: {corr:.2f})',
        line=dict(color='red', width=3, dash='solid')
    ))

    # Reference Lines & Zones
    # Random Walk Line
    fig.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="gray", annotation_text="Random Walk (0.5)")
    
    # Zone Backgrounds (Approximate visual guides)
    # We use annotations instead of shapes to keep it clean on zoom
    
    # Layout
    fig.update_layout(
        title=f'<b>Market Physics: Volatility vs Regime</b><br><sub>Correlation: {corr:.2f} (Target: Negative Correlation)</sub>',
        xaxis_title=f'Average Daily Volatility ({current_vol_col})',
        yaxis_title='Hurst Exponent (Regime)',
        template='plotly_white',
        hovermode='closest',
        height=600
    )

    # Add Text Annotations for Strategy Zones
    min_vol = res_df['Volatility'].min()
    max_vol = res_df['Volatility'].max()
    
    fig.add_annotation(
        x=min_vol, y=0.40,
        text="<b>Mean Reversion Zone</b><br>(High Volatility -> Low H)",
        showarrow=False,
        font=dict(color="green", size=14),
        xanchor="left"
    )
    
    fig.add_annotation(
        x=min_vol, y=0.55,
        text="<b>Random/Trend Zone</b><br>(Low Volatility -> H ~0.5)",
        showarrow=False,
        font=dict(color="red", size=14),
        xanchor="left"
    )

    fig.show()
    
    print("\n=== INTERPRETATION ===")
    print(f"Correlation: {corr:.4f}")
    if corr < -0.3:
        print("✅ SUCCESS: Strong Negative Correlation.")
        print("   Logic: As Volatility increases, Market becomes MORE Mean Reverting.")
        print("   Action: Enable your Mean Reversion strategy only when Volatility is HIGH.")
    elif -0.3 <= corr <= 0.1:
        print("⚠️ WARNING: No Correlation.")
        print("   Logic: Volatility does not predict regime. You need a different filter (e.g., Volume).")
    else:
        print("❌ DANGER: Positive Correlation.")
        print("   Logic: High Volatility makes the market Trend/Random. Do NOT fade spikes.")

if __name__ == "__main__":
    main()