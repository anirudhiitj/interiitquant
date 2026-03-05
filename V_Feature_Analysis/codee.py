import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "/data/quant14/EBX/day0.csv"
OUTPUT_DIR = "/home/raid/Quant14/V_Feature_Analysis/EBX"
OUTPUT_FILE = "day0_V5_only.csv"

# =============================================================================
# EXTRACT V5 COLUMN
# =============================================================================
print(f"📂 Reading file: {INPUT_FILE}")

try:
    # Read the CSV file
    df = pd.read_csv(INPUT_FILE)
    
    print(f"✓ File loaded successfully")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Check if V5 exists
    if 'V5' not in df.columns:
        print(f"\n❌ ERROR: Column 'V5' not found in the file!")
        print(f"Please check the column names above.")
    else:
        # Extract V5 column
        v5_df = df[['V5']].copy()
        
        # Save to CSV
        output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
        v5_df.to_csv(output_path, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"📊 V5 Statistics:")
        print(f"  Mean:     {v5_df['V5'].mean():.6f}")
        print(f"  Std:      {v5_df['V5'].std():.6f}")
        print(f"  Min:      {v5_df['V5'].min():.6f}")
        print(f"  Max:      {v5_df['V5'].max():.6f}")
        print(f"  Nulls:    {v5_df['V5'].isna().sum()}")
        print(f"\n💾 Saved to: {output_path}")
        print(f"📏 File size: {len(v5_df):,} rows")

except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at {INPUT_FILE}")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()