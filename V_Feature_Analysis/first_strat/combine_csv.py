import os
import pandas as pd
import glob
from tqdm import tqdm

# ===============================================================
# CONFIGURATION
# ===============================================================
SIGNALS_DIR = '/home/raid/Quant14/V_Feature_Analysis/first_strat/daily_signals'
OUTPUT_FILE = os.path.join('/home/raid/Quant14/V_Feature_Analysis/first_strat', 'trading_signals.csv')

# ===============================================================
# TRY IMPORT DASK
# ===============================================================
try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    print("⚠️ Dask not found — falling back to Pandas mode.\nInstall with: pip install dask[complete]\n")

# ===============================================================
# FETCH FILES
# ===============================================================
csv_files = sorted(glob.glob(os.path.join(SIGNALS_DIR, 'trading_signals_day*.csv')))
if not csv_files:
    raise FileNotFoundError(f"No daily signal files found in {SIGNALS_DIR}")

print(f"Found {len(csv_files)} daily signal files.\n")

# ===============================================================
# USE DASK IF AVAILABLE & MANY FILES
# ===============================================================
if HAS_DASK and len(csv_files) > 100:
    print("⚡ Using Dask for high-speed parallel merge...\n")

    # Read all CSVs in parallel and include filename info
    ddf = dd.read_csv(
        csv_files,
        assume_missing=True,
        dtype={'Time': 'object', 'Price': 'float64', 'Signal': 'int64'},
        include_path_column='source_file'  # keeps original filename
    )

    # Extract day number directly from filename
    ddf['Day'] = ddf['source_file'].map_partitions(
        lambda s: s.apply(lambda x: int(os.path.basename(x).split('day')[1].split('.')[0])),
        meta=('Day', 'int64')
    )

    # Drop the filename helper column
    ddf = ddf.drop(columns='source_file')

    # Compute (execute the lazy graph)
    print("⏳ Combining and computing large dataset...")
    combined_df = ddf.compute()
    print("✅ Dask merge complete.")

else:
    # ===============================================================
    # PANDAS FALLBACK WITH PROGRESS BAR
    # ===============================================================
    print("🐍 Using Pandas merge with progress bar...\n")
    combined_list = []
    for idx, file in enumerate(tqdm(csv_files, desc="Combining CSVs", unit="file")):
        df = pd.read_csv(file, usecols=["Time", "Price", "Signal"])
        df["Day"] = idx
        df["Time"] = pd.to_timedelta(df["Time"], errors='coerce')
        df.dropna(subset=["Time", "Price", "Signal"], inplace=True)
        combined_list.append(df)

    combined_df = pd.concat(combined_list, ignore_index=True)
    print("\n✅ Pandas merge complete.\n")

# ===============================================================
# SORT, CLEAN, SAVE
# ===============================================================
combined_df = combined_df.sort_values(by=["Day", "Time"]).reset_index(drop=True)
combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"🎯 Combined CSV saved → {OUTPUT_FILE}")
print(f"📊 Total rows: {len(combined_df):,} | Total days: {combined_df['Day'].nunique()}")
