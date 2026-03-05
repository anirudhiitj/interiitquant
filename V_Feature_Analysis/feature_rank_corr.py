import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
CORRELATION_FILE = 'V_Feature_Analysis/EBX_combined.csv'
DATA_FILE = '/data/quant14/EBX/combined.csv'
OUTPUT_FILE = 'V_Feature_Analysis/feature_ranking_dask_top20.csv'
REDUNDANCY_THRESHOLD = 0.8  # max allowed correlation between selected features
MAX_FEATURES = 20

# =============================================================================
# STEP 1. LOAD CORRELATION SUMMARY
# =============================================================================
print("\nLoading correlation summary...")
corr_df = pd.read_csv(CORRELATION_FILE)

# Compute composite informativeness score (same weights as before)
corr_df['info_score'] = (
    0.3 * corr_df['Pearson'].abs() +
    0.2 * corr_df['Spearman'].abs() +
    0.2 * corr_df['Mutual_Info'] +
    0.1 * corr_df['Kendall'].abs() +
    0.2 * (corr_df['Granger→Price'].fillna('NO').eq('YES')).astype(float) +
    0.25 * (1 - (corr_df['Best_Lag'].abs() / corr_df['Best_Lag'].abs().max())) +
    0.1 * corr_df['Best_Lag_Corr'].abs()
)

# Rank features by informativeness
corr_df = corr_df.sort_values('info_score', ascending=False).reset_index(drop=True)
feature_list = corr_df['Feature'].tolist()
print(f"✓ Ranked {len(feature_list)} features by informativeness.\n")

# =============================================================================
# STEP 2. LOAD FEATURE DATA USING DASK
# =============================================================================
print("Loading large dataset with Dask...")
ddf = dd.read_csv(DATA_FILE, usecols=feature_list, blocksize="512MB")
print(f"✓ Loaded {len(ddf.columns)} columns across {ddf.npartitions} partitions")

# =============================================================================
# STEP 3. COMPUTE CORRELATION MATRIX IN DASK
# =============================================================================
print("\nComputing Dask correlation matrix (may take several minutes)...")
corr_matrix = ddf.corr().compute().abs()
print("✓ Correlation matrix computed on Dask cluster")

# =============================================================================
# STEP 4. REDUNDANCY FILTERING
# =============================================================================
print("\nFiltering redundant features...")
selected = []
for feat in corr_df['Feature']:
    if len(selected) >= MAX_FEATURES:
        break
    redundant = False
    for s in selected:
        if corr_matrix.loc[feat, s] > REDUNDANCY_THRESHOLD:
            redundant = True
            break
    if not redundant:
        selected.append(feat)

print(f"✓ Selected {len(selected)} non-redundant features")

# =============================================================================
# STEP 5. SAVE AND DISPLAY RESULTS
# =============================================================================
selected_df = corr_df[corr_df['Feature'].isin(selected)].copy()
selected_df = selected_df.sort_values('info_score', ascending=False)
selected_df.to_csv(OUTPUT_FILE, index=False)

print("\nTop 20 Features (Informative & Unique):")
print(selected_df[['Feature', 'info_score']])

print(f"\n✓ Saved ranked features → {OUTPUT_FILE}")
print("="*80)
