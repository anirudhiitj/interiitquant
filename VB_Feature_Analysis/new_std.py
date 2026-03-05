import numpy as np
import cupy as cp
import cudf
import dask_cudf
import pandas as pd
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_FILE = '/data/quant14/EBY/combined.csv'
BLOCKSIZE = '256MB'
OUTPUT_DIR = '.'

ROLLING_WINDOWS = {
    'T1': 5, 'T2': 10, 'T3': 30, 'T4': 60, 'T5': 90, 'T6': 120,
    'T7': 300, 'T8': 600, 'T9': 900, 'T10': 1500, 'T11': 1800, 'T12': 3600
}

VB_COUNT = 6
T_COUNT = 12
# =====================================================================


def gpu_pearson_corr(series1, series2):
    """GPU-accelerated Pearson correlation using CuPy."""
    try:
        x = cp.asarray(series1)
        y = cp.asarray(series2)
        mask = ~(cp.isnan(x) | cp.isnan(y))
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            return np.nan
        corr_matrix = cp.corrcoef(x, y)
        corr = float(corr_matrix[0, 1])
        return corr
    except Exception as e:
        print(f"Error in gpu_pearson_corr: {e}")
        return np.nan


def load_and_compute(file_path, blocksize='256MB'):
    print(f"\nLoading data from: {file_path}")
    print(f"Blocksize: {blocksize}")
    print("=" * 100)

    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' not found!")
        return None

    try:
        # Generate all VB feature names
        feature_cols = [f"VB{vb}_T{t}" for vb in range(1, VB_COUNT + 1) for t in range(1, T_COUNT + 1)]
        cols_to_load = feature_cols + ['Price']

        print("Reading small sample for column verification...")
        sample_df = pd.read_csv(file_path, nrows=10)
        all_columns = sample_df.columns.tolist()

        valid_features = [c for c in feature_cols if c in all_columns]
        if 'Price' not in all_columns:
            raise ValueError("Price column not found in CSV!")

        print(f"✓ Found {len(valid_features)} valid VB features")

        print("\nLoading full data (GPU accelerated with Dask-cuDF)...")
        ddf = dask_cudf.read_csv(file_path, blocksize=blocksize, usecols=valid_features + ['Price'])
        df = ddf.compute()
        print(f"✓ Loaded {len(df):,} rows total")

        # Convert to cuDF for GPU rolling computations
        gdf = cudf.DataFrame(df)
        del df

        # -----------------------------------------------------------------
        # Compute rolling std for each lookback
        # -----------------------------------------------------------------
        print("\n[1/3] Computing rolling standard deviations of Price...")
        for t_label, window in ROLLING_WINDOWS.items():
            gdf[f'Price_STD_{t_label}'] = gdf['Price'].rolling(window=window, min_periods=max(2, window // 2)).std()
            print(f"   Computed Price_STD_{t_label} (window={window}s)")
        print("✓ All rolling std columns computed.")

        # -----------------------------------------------------------------
        # Compute correlations
        # -----------------------------------------------------------------
        print("\n[2/3] Computing GPU Pearson correlations for VBx_Ty ↔ Price_STD_Ty ...")

        results = []

        for vb in range(1, VB_COUNT + 1):
            for t in range(1, T_COUNT + 1):
                feature_name = f'VB{vb}_T{t}'
                std_col = f'Price_STD_T{t}'
                if feature_name not in gdf.columns:
                    continue
                if std_col not in gdf.columns:
                    continue

                feature_data = gdf[feature_name].to_numpy()
                std_data = gdf[std_col].to_numpy()

                corr = gpu_pearson_corr(feature_data, std_data)
                results.append({
                    'Feature': feature_name,
                    'Family': f'VB{vb}',
                    'Lookback': f'T{t}',
                    'Rolling_Window(sec)': ROLLING_WINDOWS[f'T{t}'],
                    'Corr(VBx_Ty, Price_STD_Ty)': corr
                })
                print(f"   {feature_name} ↔ {std_col}: Corr = {corr:.5f}")

        # -----------------------------------------------------------------
        # Save results
        # -----------------------------------------------------------------
        results_df = pd.DataFrame(results)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_csv = os.path.join(OUTPUT_DIR, f'vb_price_std_correlations_{timestamp}.csv')
        results_df.to_csv(out_csv, index=False)

        print("\n" + "=" * 100)
        print("FINAL RESULTS (Top 20)")
        print("=" * 100)
        print(results_df.head(20).to_string(index=False))
        print(f"\n✓ Results saved to: {out_csv}")
        print("=" * 100)

        return results_df

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# =====================================================================
# RUN MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("GPU CORRELATION: VBx_Ty ↔ Rolling STD(Price)_Ty")
    print("=" * 100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 100 + "\n")

    load_and_compute(DATA_FILE, BLOCKSIZE)
