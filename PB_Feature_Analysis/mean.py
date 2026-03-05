import dask_cudf
import cudf
import glob
import numpy as np

# --- Configuration ---
ebx_path_pattern = "/data/quant14/EBX/day*.parquet"
range_low = 99.8
range_high = 100.2

# -------------------------------------------------------------------------
# 1. Existing Logic: Mean & Std (Global)
# -------------------------------------------------------------------------
print("Processing EBX Global Stats...")
ddfebx = dask_cudf.read_parquet(ebx_path_pattern, columns=['Price'])
meanebx = ddfebx['Price'].mean().compute()
stdebx = ddfebx['Price'].std().compute()

print(f"EBX Mean Price: {meanebx:.2f}, Standard Deviation: {stdebx:.2f}")
print("-" * 30)

# -------------------------------------------------------------------------
# 2. New Function: Last Value Analysis
# -------------------------------------------------------------------------
def get_days_in_range(file_pattern, low, high):
    """
    Reads the last 'Price' value from each file matching the pattern
    and calculates the percentage of days falling within [low, high].
    """
    files = glob.glob(file_pattern)
    
    if not files:
        print("No files found.")
        return 0.0

    last_prices = []

    # Iterate through files to ensure we get exactly one "Close" per file (Day)
    for f in files:
        try:
            # Read only the Price column to save memory/time
            df = cudf.read_parquet(f, columns=['Price'])
            if len(df) > 0:
                # Append the very last value (Closing Price)
                last_prices.append(df['Price'].iloc[-1])
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not last_prices:
        return 0.0

    # Convert to cuDF Series for GPU-accelerated counting
    s_closes = cudf.Series(last_prices)
    
    # Count days within range (inclusive)
    count_in_range = s_closes.between(low, high, inclusive="both").sum()
    total_days = len(s_closes)
    
    percentage = (count_in_range / total_days) * 100
    
    return percentage, total_days

# -------------------------------------------------------------------------
# 3. Execute Analysis
# -------------------------------------------------------------------------
print(f"Analyzing Closing Prices ({range_low} - {range_high})...")
pct, total = get_days_in_range(ebx_path_pattern, range_low, range_high)

print(f"Total Days Analyzed: {total}")
print(f"Days ending between {range_low}-{range_high}: {pct:.2f}%")