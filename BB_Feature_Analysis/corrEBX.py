import numpy as np
import cupy as cp
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
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
DATA_FILE = '/data/quant14/EBX/combined.csv'
BLOCKSIZE = '256MB'  # Size of each chunk for processing
TIME_COLUMN = 'Time'  # Name of the time column
PRICE_COLUMN = 'Price'  # Name of the price column
ANALYZE_RETURNS = False  # Set to True to analyze returns, False for price
MAX_LAG = 10  # Maximum lag to test for lagged correlations
OUTPUT_DIR = '.'  # Directory to save results

# DASK CONFIGURATION
USE_DASK_DISTRIBUTED = True  # Set to False to use single-threaded scheduler
N_WORKERS = 4  # Number of Dask workers (adjust based on your GPU/CPU)
THREADS_PER_WORKER = 2  # Threads per worker

# FEATURE SELECTION - Choose one of these options:
FEATURES_TO_ANALYZE = []  # Specific features
FEATURE_PATTERN = 'BB*'  # Pattern matching (set to None to disable)
FEATURE_INDEX_RANGE = None  # Index range (set to None to disable)
# ============================================================================


def gpu_pearson_corr_dask(x, y):
    """
    GPU-accelerated Pearson correlation for Dask arrays.
    Works with numpy arrays (will be called on chunks).
    """
    try:
        x_gpu = cp.asarray(x)
        y_gpu = cp.asarray(y)
        
        mask = ~(cp.isnan(x_gpu) | cp.isnan(y_gpu))
        x_clean = x_gpu[mask]
        y_clean = y_gpu[mask]
        
        if len(x_clean) < 2:
            return np.array([np.nan, 0])
        
        corr_matrix = cp.corrcoef(x_clean, y_clean)
        corr = float(corr_matrix[0, 1])
        n_valid = int(cp.sum(mask))
        
        return np.array([corr, n_valid])
    except Exception as e:
        return np.array([np.nan, 0])


def compute_correlations_chunk(target_chunk, feature_chunk):
    """
    Compute multiple correlation metrics for a single chunk.
    Returns a dictionary of metrics.
    """
    try:
        # Convert to numpy if needed
        target_np = np.asarray(target_chunk)
        feature_np = np.asarray(feature_chunk)
        
        # Create mask for valid values
        mask = ~(np.isnan(target_np) | np.isnan(feature_np))
        target_clean = target_np[mask]
        feature_clean = feature_np[mask]
        
        n_valid = len(target_clean)
        
        if n_valid < 2:
            return {
                'n_valid': 0,
                'pearson': np.nan,
                'spearman': np.nan,
                'kendall': np.nan,
                'mi': np.nan
            }
        
        # Pearson (GPU accelerated)
        try:
            x_gpu = cp.asarray(target_clean)
            y_gpu = cp.asarray(feature_clean)
            corr_matrix = cp.corrcoef(x_gpu, y_gpu)
            pearson = float(corr_matrix[0, 1])
        except:
            pearson = np.nan
        
        # Spearman
        try:
            spearman, _ = spearmanr(target_clean, feature_clean)
        except:
            spearman = np.nan
        
        # Kendall (sample if too large)
        try:
            if len(target_clean) > 10000:
                sample_idx = np.random.choice(len(target_clean), 10000, replace=False)
                kendall, _ = kendalltau(target_clean[sample_idx], feature_clean[sample_idx])
            else:
                kendall, _ = kendalltau(target_clean, feature_clean)
        except:
            kendall = np.nan
        
        # Mutual Information (sample if too large)
        try:
            if len(target_clean) > 50000:
                sample_idx = np.random.choice(len(target_clean), 50000, replace=False)
                target_sample = target_clean[sample_idx]
                feature_sample = feature_clean[sample_idx]
            else:
                target_sample = target_clean
                feature_sample = feature_clean
            
            mi = mutual_info_regression(target_sample.reshape(-1, 1), feature_sample, random_state=42)[0]
        except:
            mi = np.nan
        
        return {
            'n_valid': n_valid,
            'pearson': pearson,
            'spearman': spearman,
            'kendall': kendall,
            'mi': mi
        }
    
    except Exception as e:
        return {
            'n_valid': 0,
            'pearson': np.nan,
            'spearman': np.nan,
            'kendall': np.nan,
            'mi': np.nan
        }


def lagged_correlation_dask(target_data, feature_data, max_lag=10):
    """
    Compute lagged correlations using Dask arrays.
    """
    try:
        # Convert to cupy for GPU acceleration
        target_gpu = cp.asarray(target_data)
        feature_gpu = cp.asarray(feature_data)
        
        # Remove NaNs
        mask = ~(cp.isnan(target_gpu) | cp.isnan(feature_gpu))
        target_clean = target_gpu[mask]
        feature_clean = feature_gpu[mask]
        
        if len(target_clean) < 2:
            return 0, np.nan
        
        best_lag = 0
        best_corr = cp.nan
        
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag > 0:
                    if len(target_clean) - lag < 2:
                        continue
                    corr_matrix = cp.corrcoef(target_clean[:-lag], feature_clean[lag:])
                elif lag < 0:
                    if len(target_clean) + lag < 2:
                        continue
                    corr_matrix = cp.corrcoef(target_clean[-lag:], feature_clean[:lag])
                else:
                    corr_matrix = cp.corrcoef(target_clean, feature_clean)
                
                corr = corr_matrix[0, 1]
                
                if cp.isnan(best_corr) or abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            except:
                continue
        
        return int(best_lag), float(best_corr) if not cp.isnan(best_corr) else np.nan
    
    except Exception as e:
        return 0, np.nan


def granger_causality_test(feature_data, target_data, max_lag=5):
    """
    Test Granger causality (feature -> target).
    """
    try:
        # Convert to numpy and remove NaNs
        feature_np = np.asarray(feature_data)
        target_np = np.asarray(target_data)
        
        mask = ~(np.isnan(feature_np) | np.isnan(target_np))
        feature_clean = feature_np[mask]
        target_clean = target_np[mask]
        
        if len(feature_clean) < 50:
            return False
        
        # Sample if too large
        if len(feature_clean) > 5000:
            sample_idx = np.random.choice(len(feature_clean), 5000, replace=False)
            sample_idx = np.sort(sample_idx)
            feature_clean = feature_clean[sample_idx]
            target_clean = target_clean[sample_idx]
        
        data = pd.DataFrame({'target': target_clean, 'feature': feature_clean})
        
        gc_results = grangercausalitytests(
            data[['target', 'feature']], 
            maxlag=min(max_lag, len(feature_clean)//4), 
            verbose=False
        )
        
        for lag in range(1, min(max_lag, len(feature_clean)//4) + 1):
            f_test = gc_results[lag][0]['ssr_ftest']
            if f_test[1] < 0.05:
                return True
        
        return False
    
    except:
        return False


def analyze_single_feature(feature_name, target_data, feature_data, total_rows, target_name, max_lag):
    """
    Analyze a single feature against the target.
    This function will be called by Dask for parallel processing.
    """
    try:
        # Count valid points
        target_np = np.asarray(target_data)
        feature_np = np.asarray(feature_data)
        
        mask = ~(np.isnan(target_np) | np.isnan(feature_np))
        n_valid = np.sum(mask)
        n_nan = total_rows - n_valid
        pct_valid = 100 * n_valid / total_rows
        
        if n_valid < 10:
            return None
        
        # Extract valid data
        target_clean = target_np[mask]
        feature_clean = feature_np[mask]
        
        # Compute correlations
        pearson_result = gpu_pearson_corr_dask(target_clean, feature_clean)
        pearson = pearson_result[0]
        
        # Spearman
        try:
            spearman, _ = spearmanr(target_clean, feature_clean)
        except:
            spearman = np.nan
        
        # Kendall (sampled)
        try:
            if len(target_clean) > 10000:
                sample_idx = np.random.choice(len(target_clean), 10000, replace=False)
                kendall, _ = kendalltau(target_clean[sample_idx], feature_clean[sample_idx])
            else:
                kendall, _ = kendalltau(target_clean, feature_clean)
        except:
            kendall = np.nan
        
        # Mutual Information
        try:
            if len(target_clean) > 50000:
                sample_idx = np.random.choice(len(target_clean), 50000, replace=False)
                target_sample = target_clean[sample_idx]
                feature_sample = feature_clean[sample_idx]
            else:
                target_sample = target_clean
                feature_sample = feature_clean
            
            mi = mutual_info_regression(target_sample.reshape(-1, 1), feature_sample, random_state=42)[0]
        except:
            mi = np.nan
        
        # Lagged correlation
        best_lag, best_lag_corr = lagged_correlation_dask(target_clean, feature_clean, max_lag=max_lag)
        
        # Granger causality
        if n_valid >= 100:
            granger = granger_causality_test(feature_clean, target_clean, max_lag=min(5, max_lag))
        else:
            granger = False
        
        granger_col_name = f'Granger→{target_name}'
        
        result = {
            'Feature': feature_name,
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
        
        return result
    
    except Exception as e:
        print(f"Error analyzing {feature_name}: {e}")
        return None


def load_and_analyze_data_dask(file_path, blocksize='256MB'):
    """
    Load and analyze data using Dask with proper lazy evaluation.
    """
    
    print(f"Loading data from: {file_path}")
    print(f"Blocksize: {blocksize}")
    print("="*100)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' does not exist!")
        return None
    
    try:
        # Initialize Dask client for distributed computing
        if USE_DASK_DISTRIBUTED:
            print(f"\nInitializing Dask distributed cluster...")
            print(f"  Workers: {N_WORKERS}")
            print(f"  Threads per worker: {THREADS_PER_WORKER}")
            
            cluster = LocalCluster(
                n_workers=N_WORKERS,
                threads_per_worker=THREADS_PER_WORKER,
                processes=True,
                memory_limit='auto'
            )
            client = Client(cluster)
            print(f"✓ Dask dashboard available at: {client.dashboard_link}")
        else:
            print("\nUsing single-threaded Dask scheduler")
            client = None
        
        # Read CSV header to identify columns
        print("\nReading CSV header to identify columns...")
        sample_df = pd.read_csv(file_path, nrows=10)
        
        columns = sample_df.columns.tolist()
        print(f"✓ Found {len(columns)} columns")
        
        # Identify price column
        price_col = None
        for col in columns:
            if col.lower() == PRICE_COLUMN.lower():
                price_col = col
                break
        
        if price_col is None:
            print(f"ERROR: Price column '{PRICE_COLUMN}' not found!")
            return None
        
        # Identify time column (optional)
        time_col = None
        for col in columns:
            if col.lower() == TIME_COLUMN.lower():
                time_col = col
                break
        
        # Get feature columns
        all_feature_cols = [col for col in columns if col != price_col and col != time_col]
        
        # Filter features based on configuration
        if FEATURES_TO_ANALYZE:
            feature_cols = [f for f in FEATURES_TO_ANALYZE if f in all_feature_cols]
            missing_features = [f for f in FEATURES_TO_ANALYZE if f not in all_feature_cols]
            if missing_features:
                print(f"\n⚠ WARNING: These features were not found:")
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
        
        target_name = "Returns" if ANALYZE_RETURNS else "Price"
        
        print(f"✓ Price column (source): {price_col}")
        print(f"✓ Analysis Target: {target_name}")
        if time_col:
            print(f"✓ Time column: {time_col}")
        print(f"✓ Total available features: {len(all_feature_cols)}")
        print(f"✓ Features to analyze: {len(feature_cols)}")
        
        # Load data using Dask DataFrame (lazy evaluation)
        print("\nLoading data with Dask DataFrame (lazy evaluation)...")
        cols_to_load = [price_col] + feature_cols
        
        ddf = dd.read_csv(
            file_path,
            blocksize=blocksize,
            usecols=cols_to_load,
            assume_missing=True
        )
        
        print(f"✓ Data loaded as Dask DataFrame with {ddf.npartitions} partitions")
        
        # Calculate target (price or returns) using Dask operations
        print(f"\nPreparing target data ({target_name})...")
        
        if ANALYZE_RETURNS:
            # Calculate returns using Dask operations (lazy)
            price_series = ddf[price_col]
            shifted_price = price_series.shift(1)
            returns = (price_series - shifted_price) / shifted_price
            target_series = returns
            print("✓ Returns calculation defined (lazy)")
        else:
            target_series = ddf[price_col]
            print("✓ Price data selected")
        
        # Get total rows (this triggers a small computation)
        print("\nComputing total rows...")
        total_rows = len(ddf)
        print(f"✓ Total rows: {total_rows:,}")
        
        # Compute target data and feature data (triggers computation)
        print("\nComputing target and feature data...")
        print("  (This may take a while for large datasets)")
        
        # Compute target
        target_data = target_series.compute()
        target_data_np = target_data.values
        
        print(f"✓ Target data computed")
        
        # Compute all feature data at once (more efficient than one-by-one)
        print(f"✓ Computing {len(feature_cols)} features in parallel...")
        feature_data_dict = {}
        
        for col in feature_cols:
            feature_data_dict[col] = ddf[col]
        
        # Use Dask's compute to evaluate multiple delayed objects in parallel
        feature_results = dask.compute(*[feature_data_dict[col] for col in feature_cols])
        
        # Store results
        for idx, col in enumerate(feature_cols):
            feature_data_dict[col] = feature_results[idx].values
        
        print(f"✓ All feature data computed")
        
        # Analyze features in parallel using Dask delayed
        print("\n" + "="*100)
        print(f"ANALYZING CORRELATIONS FOR EACH FEATURE (Target: {target_name})")
        print("="*100)
        
        # Create delayed tasks for each feature analysis
        delayed_tasks = []
        for feature in feature_cols:
            task = dask.delayed(analyze_single_feature)(
                feature,
                target_data_np,
                feature_data_dict[feature],
                total_rows,
                target_name,
                MAX_LAG
            )
            delayed_tasks.append(task)
        
        print(f"\nProcessing {len(feature_cols)} features in parallel...")
        
        # Compute all analyses in parallel
        results = dask.compute(*delayed_tasks)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        print(f"✓ Completed analysis of {len(results)} features")
        
        if not results:
            print("\nNo valid results found.")
            if client:
                client.close()
                cluster.close()
            return None
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Sort by absolute Pearson correlation
        summary_df['Abs_Pearson'] = summary_df['Pearson'].abs()
        summary_df_sorted = summary_df.sort_values('Abs_Pearson', ascending=False).copy()
        
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as CSV
        csv_file = os.path.join(OUTPUT_DIR, f'correlation_results_{target_name.lower()}_{timestamp}.csv')
        summary_df_sorted.to_csv(csv_file, index=False)
        
        # Save detailed TXT report
        txt_file = os.path.join(OUTPUT_DIR, f'correlation_results_{target_name.lower()}_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("FEATURE CORRELATION ANALYSIS RESULTS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {file_path}\n")
            f.write(f"Analysis Target: {target_name}\n")
            f.write(f"Total Rows: {total_rows:,}\n")
            f.write(f"Max Lag Tested: {MAX_LAG}\n")
            f.write(f"Dask Workers: {N_WORKERS if USE_DASK_DISTRIBUTED else 1}\n\n")
            
            f.write("="*100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*100 + "\n")
            f.write(f"Total features analyzed: {len(results)}\n")
            f.write(f"Average valid points: {summary_df['Valid_Points'].mean():,.0f} ({summary_df['Valid_Pct'].mean():.2f}%)\n")
            f.write(f"Features with |Pearson| > 0.5: {len(summary_df[summary_df['Abs_Pearson'] > 0.5])}\n")
            f.write(f"Features with |Pearson| > 0.3: {len(summary_df[summary_df['Abs_Pearson'] > 0.3])}\n")
            f.write(f"Features with |Pearson| > 0.1: {len(summary_df[summary_df['Abs_Pearson'] > 0.1])}\n\n")
            
            granger_col_name = f'Granger→{target_name}'
            granger_features = summary_df[summary_df[granger_col_name] == 'YES']
            f.write(f"Features with Granger causality: {len(granger_features)}\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 20 FEATURES (sorted by absolute correlation)\n")
            f.write("="*100 + "\n\n")
            f.write(summary_df_sorted.head(20).to_string(index=False))
            
            f.write("\n\n" + "="*100 + "\n")
            f.write("FULL RESULTS TABLE\n")
            f.write("="*100 + "\n\n")
            f.write(summary_df_sorted.to_string(index=False))
            
            f.write("\n\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")
        
        # Drop helper column
        summary_df_sorted = summary_df_sorted.drop('Abs_Pearson', axis=1)
        
        granger_col_name = f'Granger→{target_name}'
        
        # Print summary
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
        
        # Close Dask client
        if client:
            client.close()
            cluster.close()
            print("\n✓ Dask cluster closed")
        
        return summary_df_sorted
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if 'client' in locals() and client:
            client.close()
        if 'cluster' in locals() and cluster:
            cluster.close()
        
        return None


if __name__ == "__main__":
    print("\n" + "="*100)
    print("GPU-ACCELERATED FEATURE CORRELATION ANALYSIS (DASK OPTIMIZED)")
    print("="*100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    print(f"Dask distributed: {USE_DASK_DISTRIBUTED}")
    
    if ANALYZE_RETURNS:
        print(f"Analysis Target: Returns (calculated from '{PRICE_COLUMN}')")
    else:
        print(f"Analysis Target: Price (from '{PRICE_COLUMN}')")
    
    print(f"Max lag: {MAX_LAG}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Show feature selection mode
    if FEATURES_TO_ANALYZE:
        print(f"\nMode: Analyzing SPECIFIC features ({len(FEATURES_TO_ANALYZE)} features)")
    elif FEATURE_PATTERN:
        print(f"\nMode: Analyzing features matching PATTERN: '{FEATURE_PATTERN}'")
    elif FEATURE_INDEX_RANGE:
        print(f"\nMode: Analyzing features by INDEX RANGE: {FEATURE_INDEX_RANGE}")
    else:
        print(f"\nMode: Analyzing ALL features")
    
    print("="*100 + "\n")
    
    results = load_and_analyze_data_dask(DATA_FILE, BLOCKSIZE)