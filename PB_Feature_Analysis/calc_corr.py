import numpy as np
import cupy as cp
import dask_cudf
import cudf
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE
# ============================================================================
DATA_FILE = '/data/quant14/EBY/combined.csv'
BLOCKSIZE = '256MB'  # Size of each chunk for processing
TIME_COLUMN = 'Time'  # Name of the time column
PRICE_COLUMN = 'Price'  # Name of the price column
ANALYZE_RETURNS = True  # <--- NEW: Set to True to analyze returns, False for price
MAX_LAG = 10  # Maximum lag to test for lagged correlations
OUTPUT_DIR = 'PB_Feature_Analysis/'  # Directory to save results

# FEATURE SELECTION - Choose one of these options:
# Option 1: Analyze specific features (leave empty list to analyze all)
FEATURES_TO_ANALYZE = []

# Option 2: Use pattern matching (set to None to disable, or use patterns like 'BB*', 'RSI*', etc.)
FEATURE_PATTERN = 'PB*'  # Example: 'BB*' will match all features starting with BB

# Option 3: Analyze features by index range (set to None to disable)
FEATURE_INDEX_RANGE = None  # Example: (0, 50) will analyze first 50 features
# ============================================================================


def gpu_pearson_corr(series1, series2):
    """GPU-accelerated Pearson correlation using CuPy."""
    try:
        x = cp.asarray(series1)
        y = cp.asarray(series2)
        
        mask = ~(cp.isnan(x) | cp.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 2:
            return np.nan, 0
        
        corr_matrix = cp.corrcoef(x, y)
        corr = float(corr_matrix[0, 1])
        return corr, int(cp.sum(mask))
    except Exception as e:
        return np.nan, 0


def spearman_correlation(series1, series2):
    """Spearman's rank correlation."""
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        if len(s1) < 2:
            return np.nan
        corr, _ = spearmanr(s1, s2)
        return corr
    except:
        return np.nan


def kendall_correlation(series1, series2):
    """Kendall's tau correlation (sampled for large datasets)."""
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        if len(s1) < 2:
            return np.nan
        
        # Sample if too large (Kendall is O(n²))
        if len(s1) > 10000:
            sample_idx = np.random.choice(len(s1), 10000, replace=False)
            s1 = s1[sample_idx]
            s2 = s2[sample_idx]
        
        corr, _ = kendalltau(s1, s2)
        return corr
    except:
        return np.nan


def mutual_information(series1, series2):
    """Mutual Information (sampled for large datasets)."""
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        if len(s1) < 10:
            return np.nan
        
        # Sample if too large
        if len(s1) > 50000:
            sample_idx = np.random.choice(len(s1), 50000, replace=False)
            s1 = s1[sample_idx]
            s2 = s2[sample_idx]
        
        s1 = s1.reshape(-1, 1)
        mi = mutual_info_regression(s1, s2, random_state=42)[0]
        return mi
    except:
        return np.nan


def lagged_correlation(series1, series2, max_lag=10):
    """Find best lagged correlation using GPU."""
    try:
        s1 = cp.asarray(series1)
        s2 = cp.asarray(series2)
        
        mask = ~(cp.isnan(s1) | cp.isnan(s2))
        s1 = s1[mask]
        s2 = s2[mask]
        
        if len(s1) < 2:
            return 0, np.nan
        
        best_lag = 0
        best_corr = cp.nan
        
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag > 0:
                    if len(s1) - lag < 2:
                        continue
                    corr_matrix = cp.corrcoef(s1[:-lag], s2[lag:])
                    corr = corr_matrix[0, 1]
                elif lag < 0:
                    if len(s1) + lag < 2:
                        continue
                    corr_matrix = cp.corrcoef(s1[-lag:], s2[:lag])
                    corr = corr_matrix[0, 1]
                else:
                    corr_matrix = cp.corrcoef(s1, s2)
                    corr = corr_matrix[0, 1]
                
                if cp.isnan(best_corr) or abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            except:
                continue
        
        return int(best_lag), float(best_corr) if not cp.isnan(best_corr) else np.nan
    except:
        return 0, np.nan


def granger_causality(series1, series2, max_lag=5):
    """Test if series1 Granger-causes series2 (sampled for large datasets)."""
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        
        if len(s1) < 50:
            return False
        
        # Sample if too large (Granger test is computationally expensive)
        if len(s1) > 5000:
            sample_idx = np.random.choice(len(s1), 5000, replace=False)
            sample_idx = np.sort(sample_idx)  # Maintain time order
            s1 = s1[sample_idx]
            s2 = s2[sample_idx]
        
        data = pd.DataFrame({'series2': s2, 'series1': s1})
        
        gc_results = grangercausalitytests(data[['series2', 'series1']], 
                                          maxlag=min(max_lag, len(s1)//4), 
                                          verbose=False)
        for lag in range(1, min(max_lag, len(s1)//4) + 1):
            f_test = gc_results[lag][0]['ssr_ftest']
            if f_test[1] < 0.05:
                return True
        return False
    except:
        return False


def load_and_analyze_data(file_path, blocksize='256MB'):
    """Load and analyze data from combined CSV file."""
    
    print(f"Loading data from: {file_path}")
    print(f"Blocksize: {blocksize}")
    print("="*100)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' does not exist!")
        return None
    
    try:
        # First, read a small sample to get column names
        print("Reading CSV header to identify columns...")
        sample_df = pd.read_csv(file_path, nrows=10)
        
        columns = sample_df.columns.tolist()
        print(f"✓ Found {len(columns)} columns")
        
        # Identify price column (case insensitive)
        price_col = None
        for col in columns:
            if col.lower() == PRICE_COLUMN.lower():
                price_col = col
                break
        
        if price_col is None:
            print(f"ERROR: Price column '{PRICE_COLUMN}' not found!")
            print(f"Available columns: {columns[:10]}...")
            return None
        
        # Identify time column (optional)
        time_col = None
        for col in columns:
            if col.lower() == TIME_COLUMN.lower():
                time_col = col
                break
        
        # Get feature columns (all except price and time)
        all_feature_cols = [col for col in columns if col != price_col and col != time_col]
        
        # Filter features based on configuration
        if FEATURES_TO_ANALYZE:
            # Use specific feature list
            feature_cols = [f for f in FEATURES_TO_ANALYZE if f in all_feature_cols]
            missing_features = [f for f in FEATURES_TO_ANALYZE if f not in all_feature_cols]
            if missing_features:
                print(f"\n⚠ WARNING: These features were not found in the data:")
                for mf in missing_features:
                    print(f"  - {mf}")
        elif FEATURE_PATTERN:
            # Use pattern matching
            import fnmatch
            feature_cols = [f for f in all_feature_cols if fnmatch.fnmatch(f, FEATURE_PATTERN)]
            print(f"✓ Pattern '{FEATURE_PATTERN}' matched {len(feature_cols)} features")
        elif FEATURE_INDEX_RANGE:
            # Use index range
            start_idx, end_idx = FEATURE_INDEX_RANGE
            feature_cols = all_feature_cols[start_idx:end_idx]
            print(f"✓ Selected features {start_idx} to {end_idx-1} ({len(feature_cols)} features)")
        else:
            # Analyze all features
            feature_cols = all_feature_cols
        
        target_name = "Returns" if ANALYZE_RETURNS else "Price"
        
        print(f"✓ Price column (source): {price_col}")
        print(f"✓ Analysis Target: {target_name}")
        if time_col:
            print(f"✓ Time column: {time_col}")
        print(f"✓ Total available features: {len(all_feature_cols)}")
        print(f"✓ Features to analyze: {len(feature_cols)}")
        if len(feature_cols) <= 20:
            print(f"  Features: {feature_cols}")
        else:
            print(f"  First 10: {feature_cols[:10]}")
            print(f"  Last 10: {feature_cols[-10:]}")
        
        # Load only necessary columns using dask_cudf with smaller blocks
        print("\nLoading data with dask_cudf (GPU accelerated, chunked processing)...")
        cols_to_load = [price_col] + feature_cols
        
        # Use dask_cudf to read only selected columns
        ddf = dask_cudf.read_csv(file_path, blocksize=blocksize, usecols=cols_to_load)
        
        print(f"✓ Data will be processed in chunks of {blocksize}")
        
        # Process data chunk by chunk without loading all into GPU memory at once
        print("Processing chunks and extracting data...")
        
        target_data_list = []
        feature_data_dict = {col: [] for col in feature_cols}
        
        last_price = np.nan
        total_rows = 0
        n_partitions = ddf.npartitions
        
        for part_idx in range(n_partitions):
            # Process one partition at a time
            partition = ddf.get_partition(part_idx).compute()
            
            total_rows += len(partition)
            
            # --- MODIFIED: Calculate returns or get price ---
            if ANALYZE_RETURNS:
                price_series_gpu = partition[price_col]
                
                # Use cupy for GPU-native array operations
                price_cp = cp.asarray(price_series_gpu)
                
                # Use roll to get previous price, then fix the first element
                shifted_prices_cp = cp.roll(price_cp, 1)
                shifted_prices_cp[0] = last_price  # From previous chunk
                
                # Calculate returns, handling division by zero
                # Calculate returns, handling division by zero manually
                returns_cp = (price_cp - shifted_prices_cp) / shifted_prices_cp
                returns_cp = cp.where(cp.isfinite(returns_cp), returns_cp, cp.nan)
                returns_cp = cp.where(shifted_prices_cp == 0, cp.nan, returns_cp)

                
                # Store last price of *this* chunk for the *next* chunk
                last_price = float(price_cp[-1])
                
                target_chunk_np = cp.asnumpy(returns_cp)
                
                del price_series_gpu, price_cp, shifted_prices_cp, returns_cp
            
            else:
                # Original logic: just get the price
                target_chunk_np = partition[price_col].to_numpy()
            
            target_data_list.append(target_chunk_np)
            # --- END OF MODIFICATION ---
            
            for col in feature_cols:
                feature_chunk = partition[col].to_numpy()
                feature_data_dict[col].append(feature_chunk)
            
            # Clear GPU memory for this partition
            del partition
            
            if (part_idx + 1) % 10 == 0:
                print(f"  Processed {part_idx + 1}/{n_partitions} partitions ({total_rows:,} rows)...")
        
        print(f"✓ Loaded {total_rows:,} rows total from {n_partitions} partitions")
        
        # Concatenate all chunks in CPU memory
        print("Concatenating data chunks...")
        target_data = np.concatenate(target_data_list)
        
        for col in feature_cols:
            feature_data_dict[col] = np.concatenate(feature_data_dict[col])
        
        # Clear the lists to free memory
        del target_data_list
        
        target_valid = np.sum(~np.isnan(target_data))
        print(f"✓ {target_name} data: {target_valid:,}/{total_rows:,} valid points ({100*target_valid/total_rows:.2f}%)")
        
        # Analyze each feature
        print("\n" + "="*100)
        print(f"ANALYZING CORRELATIONS FOR EACH FEATURE (Target: {target_name})")
        print("="*100)
        
        results = []
        
        for idx, feature in enumerate(feature_cols, 1):
            feature_data = feature_data_dict[feature]
            
            # Count valid points (non-NaN in both target and feature)
            mask = ~(np.isnan(target_data) | np.isnan(feature_data))
            n_valid = np.sum(mask)
            n_nan = total_rows - n_valid
            pct_valid = 100 * n_valid / total_rows
            
            if n_valid < 10:
                print(f"\n[{idx}/{len(feature_cols)}] {feature}: SKIPPED (only {n_valid} valid points)")
                continue
            
            # Extract valid data
            target_clean = target_data[mask]
            feature_clean = feature_data[mask]
            
            print(f"\n[{idx}/{len(feature_cols)}] {feature}")
            print(f"  Valid points: {n_valid:,}/{total_rows:,} ({pct_valid:.2f}%)")
            print(f"  NaN values: {n_nan:,} ({100-pct_valid:.2f}%)")
            
            # Calculate correlations
            print("  Computing correlations...")
            pearson, _ = gpu_pearson_corr(target_clean, feature_clean)
            spearman = spearman_correlation(target_clean, feature_clean)
            kendall = kendall_correlation(target_clean, feature_clean)
            mi = mutual_information(target_clean, feature_clean)
            best_lag, best_lag_corr = lagged_correlation(target_clean, feature_clean, max_lag=MAX_LAG)
            
            # Granger causality (only for features with enough data)
            if n_valid >= 100:
                print("  Computing Granger causality...")
                granger = granger_causality(feature_clean, target_clean, max_lag=min(5, MAX_LAG))
            else:
                granger = False
            
            granger_col_name = f'Granger→{target_name}'
            
            result = {
                'Feature': feature,
                'Total_Rows': total_rows,
                'Valid_Points': n_valid,
                'NaN_Count': n_nan,
                'Valid_Pct': pct_valid,
                'Pearson': pearson,
                'Spearman': spearman,
                'Kendall': kendall,
                'Mutual_Info': mi,
                'Best_Lag': best_lag,
                'Best_Lag_Corr': best_lag_corr,
                granger_col_name: 'YES' if granger else 'NO'
            }
            
            results.append(result)
            
            # Print results
            print(f"  Pearson: {pearson:+.4f}")
            print(f"  Spearman: {spearman:+.4f}")
            print(f"  Kendall: {kendall:+.4f}")
            print(f"  Mutual Info: {mi:.4f}")
            print(f"  Best Lag: {best_lag:+d} (r={best_lag_corr:+.4f})")
            print(f"  {granger_col_name}: {'YES ✓' if granger else 'NO'}")
        
        if not results:
            print("\nNo valid results found.")
            return None
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Sort by absolute Pearson correlation
        summary_df['Abs_Pearson'] = summary_df['Pearson'].abs()
        summary_df_sorted = summary_df.sort_values('Abs_Pearson', ascending=False).copy()
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV (sorted by correlation)
        csv_file = os.path.join(OUTPUT_DIR, f'correlation_results_{target_name.lower()}_{timestamp}.csv')
        summary_df_sorted.to_csv(csv_file, index=False)
        
        # Save as TXT (detailed per-feature report)
        txt_file = os.path.join(OUTPUT_DIR, f'correlation_results_{target_name.lower()}_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("FEATURE CORRELATION ANALYSIS RESULTS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {file_path}\n")
            f.write(f"Analysis Target: {target_name} (from column '{price_col}')\n")
            f.write(f"Total Rows: {total_rows:,}\n")
            f.write(f"Max Lag Tested: {MAX_LAG}\n\n")
            
            f.write("="*100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*100 + "\n")
            f.write(f"Total features analyzed: {len(results)}\n")
            f.write(f"Average valid points: {summary_df['Valid_Points'].mean():,.0f} ({summary_df['Valid_Pct'].mean():.2f}%)\n")
            f.write(f"Features with |Pearson| > 0.5: {len(summary_df[summary_df['Abs_Pearson'] > 0.5])}\n")
            f.write(f"Features with |Pearson| > 0.3: {len(summary_df[summary_df['Abs_Pearson'] > 0.3])}\n")
            f.write(f"Features with |Pearson| > 0.1: {len(summary_df[summary_df['Abs_Pearson'] > 0.1])}\n")
            
            granger_col_name = f'Granger→{target_name}'
            granger_features = summary_df[summary_df[granger_col_name] == 'YES']
            f.write(f"Features with Granger causality: {len(granger_features)}\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("DETAILED FEATURE REPORTS (in order of analysis)\n")
            f.write("="*100 + "\n\n")
            
            # Write detailed report for each feature in original order (from results list)
            for idx, result in enumerate(results, 1):
                f.write("\n" + "-"*100 + "\n")
                f.write(f"[{idx}/{len(results)}] FEATURE: {result['Feature']}\n")
                f.write("-"*100 + "\n\n")
                
                f.write("DATA QUALITY:\n")
                f.write(f"  Total Rows:         {result['Total_Rows']:,}\n")
                f.write(f"  Valid Points:       {result['Valid_Points']:,} ({result['Valid_Pct']:.2f}%)\n")
                f.write(f"  NaN Count:          {result['NaN_Count']:,} ({100-result['Valid_Pct']:.2f}%)\n\n")
                
                f.write("CORRELATION METRICS:\n")
                f.write(f"  Pearson:            {result['Pearson']:+.6f}\n")
                f.write(f"  Spearman:           {result['Spearman']:+.6f}\n")
                f.write(f"  Kendall:            {result['Kendall']:+.6f}\n")
                f.write(f"  Mutual Info:        {result['Mutual_Info']:.6f}\n\n")
                
                f.write("LAGGED CORRELATION:\n")
                f.write(f"  Best Lag:           {result['Best_Lag']:+d} periods\n")
                f.write(f"  Best Lag Corr:      {result['Best_Lag_Corr']:+.6f}\n\n")
                
                f.write("CAUSALITY TEST:\n")
                f.write(f"  {granger_col_name}:      {result[granger_col_name]}\n")
                
                # Add interpretation
                f.write("\nINTERPRETATION:\n")
                abs_pearson = abs(result['Pearson'])
                if abs_pearson > 0.7:
                    strength = "VERY STRONG"
                elif abs_pearson > 0.5:
                    strength = "STRONG"
                elif abs_pearson > 0.3:
                    strength = "MODERATE"
                elif abs_pearson > 0.1:
                    strength = "WEAK"
                else:
                    strength = "VERY WEAK"
                
                direction = "positive" if result['Pearson'] > 0 else "negative"
                f.write(f"  Correlation Strength: {strength} {direction} correlation\n")
                
                if result['Best_Lag'] > 0:
                    f.write(f"  Temporal Pattern: Feature leads {target_name.lower()} by {result['Best_Lag']} periods\n")
                elif result['Best_Lag'] < 0:
                    f.write(f"  Temporal Pattern: Feature lags {target_name.lower()} by {abs(result['Best_Lag'])} periods\n")
                else:
                    f.write(f"  Temporal Pattern: Best correlation at zero lag (contemporaneous)\n")
                
                if result[granger_col_name] == 'YES':
                    f.write(f"  Predictive Power: Feature has predictive power for future {target_name.lower()} (Granger causality detected)\n")
                else:
                    f.write(f"  Predictive Power: No significant Granger causality detected\n")
                
                f.write("\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("QUICK REFERENCE SUMMARY TABLE (sorted by correlation strength)\n")
            f.write("="*100 + "\n\n")
            f.write(summary_df_sorted.to_string(index=False))
            
            f.write("\n\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        # Drop the helper column for final display
        summary_df_sorted = summary_df_sorted.drop('Abs_Pearson', axis=1)
        
        granger_col_name = f'Granger→{target_name}'
        
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total features analyzed: {len(results)}")
        print(f"Average valid points: {summary_df['Valid_Points'].mean():,.0f} ({summary_df['Valid_Pct'].mean():.2f}%)")
        print(f"Features with |Pearson| > 0.5: {len(summary_df[summary_df['Abs_Pearson'] > 0.5])}")
        print(f"Features with |Pearson| > 0.3: {len(summary_df[summary_df['Abs_Pearson'] > 0.3])}")
        print(f"Features with Granger causality: {len(granger_features)}")
        
        print("\n" + "="*100)
        print(f"TOP 10 FEATURES (by absolute correlation vs {target_name})")
        print("="*100)
        print(summary_df_sorted[['Feature', 'Valid_Pct', 'Pearson', 'Spearman', 'Best_Lag', 'Best_Lag_Corr', granger_col_name]].head(10).to_string(index=False))
        
        print("\n" + "="*100)
        print(f"✓ Results saved to:")
        print(f"  CSV: {csv_file}")
        print(f"  TXT: {txt_file}")
        print("="*100)
        
        return summary_df_sorted
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "="*100)
    print("GPU-ACCELERATED FEATURE CORRELATION ANALYSIS")
    print("="*100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    
    if ANALYZE_RETURNS:
        print(f"Analysis Target: Returns (calculated from '{PRICE_COLUMN}')")
    else:
        print(f"Analysis Target: Price (from '{PRICE_COLUMN}')")
        
    print(f"Max lag: {MAX_LAG}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Show feature selection mode
    if FEATURES_TO_ANALYZE:
        print(f"\nMode: Analyzing SPECIFIC features ({len(FEATURES_TO_ANALYZE)} features)")
        print(f"Features: {FEATURES_TO_ANALYZE}")
    elif FEATURE_PATTERN:
        print(f"\nMode: Analyzing features matching PATTERN: '{FEATURE_PATTERN}'")
    elif FEATURE_INDEX_RANGE:
        print(f"\nMode: Analyzing features by INDEX RANGE: {FEATURE_INDEX_RANGE}")
    else:
        print(f"\nMode: Analyzing ALL features")
    
    print("="*100 + "\n")
    results = load_and_analyze_data(DATA_FILE, BLOCKSIZE)