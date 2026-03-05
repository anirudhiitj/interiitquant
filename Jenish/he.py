import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hurst import compute_Hc  # Library for Hurst Calculation
from tqdm import tqdm

# =================CONFIGURATION=================
# Path to your daily parquet files
DATA_FOLDER_PATH = '/data/quant14/EBY/' 
# The column name representing the "Core Time Series" (Price/Index)
# Check your feature dictionary. It might be 'close', 'price', or 'core'.
PRICE_COL = 'Price' 
# ===============================================

def calculate_daily_hurst(folder_path, price_col):
    results = []
    
    # Get list of all parquet files
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    
    if not files:
        print("No Parquet files found. Check your path.")
        return None

    print(f"Found {len(files)} files. Starting calculation...")

    for file_path in tqdm(files):
        try:
            # Load the day's data
            df = pd.read_parquet(file_path)
            
            # Ensure the price column exists
            if price_col not in df.columns:
                print(f"Warning: {price_col} not found in {os.path.basename(file_path)}")
                continue
            
            # Extract price series (drop NaNs)
            series = df[price_col].dropna().values
            
            # Hurst calculation requires a minimum length (e.g., 100 data points)
            if len(series) < 100:
                print(f"Skipping {os.path.basename(file_path)}: Not enough data points.")
                continue

            # Calculate Hurst Exponent
            # kind='price' means the input is the price series, not returns
            H, c, data = compute_Hc(series, kind='price', simplified=True)
            
            # Extract date from filename if possible (assuming 'date.parquet' format)
            file_name = os.path.basename(file_path)
            
            results.append({
                'filename': file_name,
                'hurst_exponent': H,
                'data_points': len(series)
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(results)

# ================= EXECUTION =================
hurst_df = calculate_daily_hurst(DATA_FOLDER_PATH, PRICE_COL)

if hurst_df is not None and not hurst_df.empty:
    # 1. Basic Statistics
    avg_hurst = hurst_df['hurst_exponent'].mean()
    print("\n=== Analysis Results ===")
    print(f"Mean Hurst Exponent: {avg_hurst:.4f}")
    print(f"Min Hurst: {hurst_df['hurst_exponent'].min():.4f}")
    print(f"Max Hurst: {hurst_df['hurst_exponent'].max():.4f}")

    # Interpretation
    if avg_hurst < 0.45:
        print("CONCLUSION: The asset is strongly MEAN REVERTING.")
    elif avg_hurst > 0.55:
        print("CONCLUSION: The asset is strongly TRENDING.")
    else:
        print("CONCLUSION: The asset behaves like a Random Walk (Mixed regimes).")

    # 2. Plotting the Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(hurst_df['hurst_exponent'], kde=True, bins=20, color='teal')
    
    # Add reference line at 0.5
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random Walk (0.5)')
    plt.axvline(avg_hurst, color='blue', linestyle='-', linewidth=2, label=f'Mean ({avg_hurst:.2f})')
    
    plt.title(f'Distribution of Daily Hurst Exponents\n(Feature: {PRICE_COL})', fontsize=14)
    plt.xlabel('Hurst Exponent (H)', fontsize=12)
    plt.ylabel('Frequency (Days)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. Save results to CSV for further analysis
    hurst_df.to_csv('hurst_analysis_results.csv', index=False)
    print("Results saved to hurst_analysis_results.csv")

else:
    print("No results generated.")