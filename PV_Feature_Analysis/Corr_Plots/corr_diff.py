import numpy as np
import cupy as cp
import dask_cudf
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
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
OUTPUT_DIR = '.'  # Directory to save results

# FEATURE SELECTION - Choose one of these options:
# Option 1: Analyze specific features (leave empty list to analyze all)
FEATURES_TO_ANALYZE = []

# Option 2: Use pattern matching (set to None to disable, or use patterns like 'BB*', 'RSI*', etc.)
FEATURE_PATTERN = None  # Example: 'BB*' will match all features starting with BB

# Option 3: Analyze features by index range (set to None to disable)
FEATURE_INDEX_RANGE = None  # Example: (0, 50) will analyze first 50 features

# Granger causality parameters
GRANGER_MAX_LAG = 5  # Maximum lag for Granger causality test
GRANGER_SAMPLE_SIZE = 5000  # Sample size for Granger test (to manage computation time)
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


def granger_causality(series1, series2, max_lag=5, sample_size=5000):
    """Test if series1 Granger-causes series2 (sampled for large datasets)."""
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        
        if len(s1) < 50:
            return False, np.nan
        
        # Sample if too large (Granger test is computationally expensive)
        if len(s1) > sample_size:
            sample_idx = np.random.choice(len(s1), sample_size, replace=False)
            sample_idx = np.sort(sample_idx)  # Maintain time order
            s1 = s1[sample_idx]
            s2 = s2[sample_idx]
        
        data = pd.DataFrame({'series2': s2, 'series1': s1})
        
        gc_results = grangercausalitytests(data[['series2', 'series1']], 
                                          maxlag=min(max_lag, len(s1)//4), 
                                          verbose=False)
        
        # Find the minimum p-value across all lags
        min_p_value = 1.0
        for lag in range(1, min(max_lag, len(s1)//4) + 1):
            f_test = gc_results[lag][0]['ssr_ftest']
            p_value = f_test[1]
            if p_value < min_p_value:
                min_p_value = p_value
        
        is_significant = min_p_value < 0.05
        return is_significant, min_p_value
    except:
        return False, np.nan


def calculate_changes(data_array):
    """Calculate first differences (Xt - Xt-1) using CuPy for GPU acceleration."""
    try:
        data_cp = cp.asarray(data_array)
        
        # Calculate differences: current - previous
        changes = cp.diff(data_cp, prepend=cp.nan)
        
        # Convert back to numpy
        changes_np = cp.asnumpy(changes)
        
        del data_cp
        return changes_np
    except:
        # Fallback to numpy if GPU fails
        return np.diff(data_array, prepend=np.nan)


def load_and_analyze_data(file_path, blocksize='256MB'):
    """Load and analyze price changes vs feature changes."""
    
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
            feature_cols = [f for f in FEATURES_TO_ANALYZE if f in all_feature_cols]
            missing_features = [f for f in FEATURES_TO_ANALYZE if f not in all_feature_cols]
            if missing_features:
                print(f"\n⚠ WARNING: These features were not found in the data:")
                for mf in missing_features:
                    print(f"  - {mf}")
        elif FEATURE_PATTERN:
            import fnmatch
            feature_cols = [f for f in all_feature_cols if fnmatch.fnmatch(f, FEATURE_PATTERN)]
            print(f"✓ Pattern '{FEATURE_PATTERN}' matched {len(feature_cols)} features")
        elif FEATURE_INDEX_RANGE:
            start_idx, end_idx = FEATURE_INDEX_RANGE
            feature_cols = all_feature_cols[start_idx:end_idx]
            print(f"✓ Selected features {start_idx} to {end_idx-1} ({len(feature_cols)} features)")
        else:
            feature_cols = all_feature_cols
        
        print(f"✓ Price column: {price_col}")
        print(f"✓ Analysis: ΔPrice (Pt - Pt-1) vs ΔFeature (Ft - Ft-1)")
        if time_col:
            print(f"✓ Time column: {time_col}")
        print(f"✓ Total available features: {len(all_feature_cols)}")
        print(f"✓ Features to analyze: {len(feature_cols)}")
        if len(feature_cols) <= 20:
            print(f"  Features: {feature_cols}")
        else:
            print(f"  First 10: {feature_cols[:10]}")
            print(f"  Last 10: {feature_cols[-10:]}")
        
        # Load only necessary columns
        print("\nLoading data with dask_cudf (GPU accelerated, chunked processing)...")
        cols_to_load = [price_col] + feature_cols
        
        ddf = dask_cudf.read_csv(file_path, blocksize=blocksize, usecols=cols_to_load)
        
        print(f"✓ Data will be processed in chunks of {blocksize}")
        
        # Process data chunk by chunk
        print("Processing chunks and calculating changes...")
        
        price_data_list = []
        feature_data_dict = {col: [] for col in feature_cols}
        
        total_rows = 0
        n_partitions = ddf.npartitions
        
        for part_idx in range(n_partitions):
            partition = ddf.get_partition(part_idx).compute()
            total_rows += len(partition)
            
            # Get price data for this chunk
            price_chunk = partition[price_col].to_numpy()
            price_data_list.append(price_chunk)
            
            # Get feature data for this chunk
            for col in feature_cols:
                feature_chunk = partition[col].to_numpy()
                feature_data_dict[col].append(feature_chunk)
            
            del partition
            
            if (part_idx + 1) % 10 == 0:
                print(f"  Processed {part_idx + 1}/{n_partitions} partitions ({total_rows:,} rows)...")
        
        print(f"✓ Loaded {total_rows:,} rows total from {n_partitions} partitions")
        
        # Concatenate all chunks
        print("Concatenating data chunks...")
        price_data = np.concatenate(price_data_list)
        
        for col in feature_cols:
            feature_data_dict[col] = np.concatenate(feature_data_dict[col])
        
        del price_data_list
        
        # Calculate price changes
        print("Calculating price changes (Pt - Pt-1)...")
        price_changes = calculate_changes(price_data)
        price_changes_valid = np.sum(~np.isnan(price_changes))
        print(f"✓ Price changes: {price_changes_valid:,}/{total_rows:,} valid points ({100*price_changes_valid/total_rows:.2f}%)")
        
        # Analyze each feature
        print("\n" + "="*100)
        print("ANALYZING CORRELATIONS: ΔPrice vs ΔFeature")
        print("="*100)
        
        results = []
        
        for idx, feature in enumerate(feature_cols, 1):
            feature_data = feature_data_dict[feature]
            
            # Calculate feature changes
            feature_changes = calculate_changes(feature_data)
            
            # Count valid points (non-NaN in both price changes and feature changes)
            mask = ~(np.isnan(price_changes) | np.isnan(feature_changes))
            n_valid = np.sum(mask)
            n_nan = total_rows - n_valid
            pct_valid = 100 * n_valid / total_rows
            
            if n_valid < 10:
                print(f"\n[{idx}/{len(feature_cols)}] {feature}: SKIPPED (only {n_valid} valid points)")
                continue
            
            # Extract valid data
            price_changes_clean = price_changes[mask]
            feature_changes_clean = feature_changes[mask]
            
            print(f"\n[{idx}/{len(feature_cols)}] {feature}")
            print(f"  Valid points: {n_valid:,}/{total_rows:,} ({pct_valid:.2f}%)")
            print(f"  NaN values: {n_nan:,} ({100-pct_valid:.2f}%)")
            
            # Calculate Pearson correlation
            print("  Computing Pearson correlation...")
            pearson, _ = gpu_pearson_corr(price_changes_clean, feature_changes_clean)
            
            # Granger causality (only for features with enough data)
            if n_valid >= 100:
                print("  Computing Granger causality (ΔFeature → ΔPrice)...")
                granger, granger_pval = granger_causality(feature_changes_clean, price_changes_clean, 
                                                          max_lag=GRANGER_MAX_LAG,
                                                          sample_size=GRANGER_SAMPLE_SIZE)
            else:
                granger = False
                granger_pval = np.nan
            
            result = {
                'Feature': feature,
                'Total_Rows': total_rows,
                'Valid_Points': n_valid,
                'NaN_Count': n_nan,
                'Valid_Pct': pct_valid,
                'Pearson_Corr': pearson,
                'Granger_Significant': 'YES' if granger else 'NO',
                'Granger_P_Value': granger_pval
            }
            
            results.append(result)
            
            # Print results
            print(f"  Pearson Correlation: {pearson:+.6f}")
            print(f"  Granger Causality (ΔFeature → ΔPrice): {'YES ✓' if granger else 'NO'}")
            if not np.isnan(granger_pval):
                print(f"  Granger P-Value: {granger_pval:.6f}")
        
        if not results:
            print("\nNo valid results found.")
            return None
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Sort by absolute Pearson correlation
        summary_df['Abs_Pearson'] = summary_df['Pearson_Corr'].abs()
        summary_df_sorted = summary_df.sort_values('Abs_Pearson', ascending=False).copy()
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV (sorted by correlation)
        csv_file = os.path.join(OUTPUT_DIR, f'change_correlation_results_{timestamp}.csv')
        summary_df_sorted.drop('Abs_Pearson', axis=1).to_csv(csv_file, index=False)
        
        # Save as TXT (detailed per-feature report)
        txt_file = os.path.join(OUTPUT_DIR, f'change_correlation_results_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("PRICE CHANGE vs FEATURE CHANGE CORRELATION ANALYSIS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {file_path}\n")
            f.write(f"Analysis: ΔPrice (Pt - Pt-1) vs ΔFeature (Ft - Ft-1)\n")
            f.write(f"Price Column: {price_col}\n")
            f.write(f"Total Rows: {total_rows:,}\n")
            f.write(f"Granger Max Lag: {GRANGER_MAX_LAG}\n")
            f.write(f"Granger Sample Size: {GRANGER_SAMPLE_SIZE}\n\n")
            
            f.write("="*100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*100 + "\n")
            f.write(f"Total features analyzed: {len(results)}\n")
            f.write(f"Average valid points: {summary_df['Valid_Points'].mean():,.0f} ({summary_df['Valid_Pct'].mean():.2f}%)\n")
            f.write(f"Features with |Pearson| > 0.5: {len(summary_df[summary_df['Abs_Pearson'] > 0.5])}\n")
            f.write(f"Features with |Pearson| > 0.3: {len(summary_df[summary_df['Abs_Pearson'] > 0.3])}\n")
            f.write(f"Features with |Pearson| > 0.1: {len(summary_df[summary_df['Abs_Pearson'] > 0.1])}\n")
            
            granger_features = summary_df[summary_df['Granger_Significant'] == 'YES']
            f.write(f"Features with Granger causality: {len(granger_features)}\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("DETAILED FEATURE REPORTS (in order of analysis)\n")
            f.write("="*100 + "\n\n")
            
            for idx, result in enumerate(results, 1):
                f.write("\n" + "-"*100 + "\n")
                f.write(f"[{idx}/{len(results)}] FEATURE: {result['Feature']}\n")
                f.write("-"*100 + "\n\n")
                
                f.write("DATA QUALITY:\n")
                f.write(f"  Total Rows:         {result['Total_Rows']:,}\n")
                f.write(f"  Valid Points:       {result['Valid_Points']:,} ({result['Valid_Pct']:.2f}%)\n")
                f.write(f"  NaN Count:          {result['NaN_Count']:,} ({100-result['Valid_Pct']:.2f}%)\n\n")
                
                f.write("CORRELATION ANALYSIS:\n")
                f.write(f"  Pearson Correlation (ΔPrice vs Δ{result['Feature']}): {result['Pearson_Corr']:+.6f}\n\n")
                
                f.write("GRANGER CAUSALITY TEST:\n")
                f.write(f"  Δ{result['Feature']} → ΔPrice: {result['Granger_Significant']}\n")
                if not np.isnan(result['Granger_P_Value']):
                    f.write(f"  P-Value: {result['Granger_P_Value']:.6f}\n")
                f.write("\n")
                
                # Add interpretation
                f.write("INTERPRETATION:\n")
                abs_pearson = abs(result['Pearson_Corr'])
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
                
                direction = "positive" if result['Pearson_Corr'] > 0 else "negative"
                f.write(f"  Correlation Strength: {strength} {direction} correlation between price changes and feature changes\n")
                
                if result['Granger_Significant'] == 'YES':
                    f.write(f"  Predictive Power: Feature changes have predictive power for future price changes (Granger causality detected)\n")
                else:
                    f.write(f"  Predictive Power: No significant Granger causality detected\n")
                
                f.write("\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("QUICK REFERENCE SUMMARY TABLE (sorted by correlation strength)\n")
            f.write("="*100 + "\n\n")
            display_df = summary_df_sorted.drop('Abs_Pearson', axis=1)
            f.write(display_df.to_string(index=False))
            
            f.write("\n\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        # Drop the helper column for final display
        summary_df_sorted = summary_df_sorted.drop('Abs_Pearson', axis=1)
        
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total features analyzed: {len(results)}")
        print(f"Average valid points: {summary_df['Valid_Points'].mean():,.0f} ({summary_df['Valid_Pct'].mean():.2f}%)")
        print(f"Features with |Pearson| > 0.5: {len(summary_df[summary_df['Abs_Pearson'] > 0.5])}")
        print(f"Features with |Pearson| > 0.3: {len(summary_df[summary_df['Abs_Pearson'] > 0.3])}")
        print(f"Features with Granger causality: {len(granger_features)}")
        
        print("\n" + "="*100)
        print("TOP 10 FEATURES (by absolute correlation)")
        print("="*100)
        print(summary_df_sorted[['Feature', 'Valid_Pct', 'Pearson_Corr', 'Granger_Significant', 'Granger_P_Value']].head(10).to_string(index=False))
        
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
    print("GPU-ACCELERATED PRICE CHANGE vs FEATURE CHANGE CORRELATION ANALYSIS")
    print("="*100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    print(f"Analysis: ΔPrice (Pt - Pt-1) vs ΔFeature (Ft - Ft-1)")
    print(f"Granger causality max lag: {GRANGER_MAX_LAG}")
    print(f"Granger causality sample size: {GRANGER_SAMPLE_SIZE}")
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