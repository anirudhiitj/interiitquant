import numpy as np
import cupy as cp
import dask_cudf
import cudf
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
import warnings
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
DATA_FILE = '/data/quant14/EBY/combined.csv'
BLOCKSIZE = '256MB'
TIME_COLUMN = 'Time'
PRICE_COLUMN = 'Price'
OUTPUT_DIR = 'BB_Feature_Analysis_EBY/'

# TIME LAGS TO ANALYZE (in seconds)
TIME_LAGS = [0, 1, 5, 10, 15]

# FEATURE SELECTION
FEATURES_TO_ANALYZE = []
FEATURE_PATTERN = 'BB*'
FEATURE_INDEX_RANGE = None

# PROCESSING PARAMETERS
N_PROCESSES = 4  # Number of parallel processes
BATCH_SIZE = 20  # Process this many features at once on GPU

# METRIC SELECTION - Disable slow metrics for speed
CALCULATE_KENDALL = False  # Very slow O(n²) - disable for speed
CALCULATE_MUTUAL_INFO = False  # Slow - disable for speed
CALCULATE_GRANGER = False  # Very slow - disable for speed

GRANGER_MAX_LAG = 3
GRANGER_SUBSAMPLE = 10000

IRREGULAR_TIME_SERIES = False
# ============================================================================

def gpu_batch_pearson_corr(features_matrix, price_vector):
    """
    Calculate Pearson correlation for multiple features at once using GPU batching.
    features_matrix: (n_samples, n_features) array
    price_vector: (n_samples,) array
    Returns: array of correlations for each feature
    """
    try:
        # Move to GPU
        X_gpu = cp.asarray(features_matrix, dtype=cp.float32)
        y_gpu = cp.asarray(price_vector, dtype=cp.float32)
        
        n_samples, n_features = X_gpu.shape
        
        # Handle NaN values - create mask for each feature
        correlations = cp.zeros(n_features, dtype=cp.float32)
        valid_counts = cp.zeros(n_features, dtype=cp.int32)
        
        for i in range(n_features):
            x = X_gpu[:, i]
            mask = ~(cp.isnan(x) | cp.isnan(y_gpu))
            
            x_clean = x[mask]
            y_clean = y_gpu[mask]
            
            valid_counts[i] = cp.sum(mask)
            
            if len(x_clean) < 2:
                correlations[i] = cp.nan
                continue
            
            # Fast correlation using covariance
            x_centered = x_clean - cp.mean(x_clean)
            y_centered = y_clean - cp.mean(y_clean)
            
            cov = cp.sum(x_centered * y_centered)
            std_x = cp.sqrt(cp.sum(x_centered ** 2))
            std_y = cp.sqrt(cp.sum(y_centered ** 2))
            
            if std_x > 0 and std_y > 0:
                correlations[i] = cov / (std_x * std_y)
            else:
                correlations[i] = cp.nan
        
        # Move back to CPU
        result_corr = cp.asnumpy(correlations)
        result_counts = cp.asnumpy(valid_counts)
        
        # Clean up
        del X_gpu, y_gpu, correlations, valid_counts
        cp.get_default_memory_pool().free_all_blocks()
        
        return result_corr, result_counts
        
    except Exception as e:
        print(f"Batch correlation error: {e}")
        return np.full(features_matrix.shape[1], np.nan), np.zeros(features_matrix.shape[1])

def gpu_batch_spearman_corr(features_matrix, price_vector):
    """
    Calculate Spearman correlation for multiple features at once using GPU batching.
    """
    try:
        # Move to GPU
        X_gpu = cp.asarray(features_matrix, dtype=cp.float32)
        y_gpu = cp.asarray(price_vector, dtype=cp.float32)
        
        n_samples, n_features = X_gpu.shape
        correlations = cp.zeros(n_features, dtype=cp.float32)
        
        # Rank price vector once
        y_mask = ~cp.isnan(y_gpu)
        
        for i in range(n_features):
            x = X_gpu[:, i]
            mask = ~(cp.isnan(x) | cp.isnan(y_gpu))
            
            x_clean = x[mask]
            y_clean = y_gpu[mask]
            
            if len(x_clean) < 2:
                correlations[i] = cp.nan
                continue
            
            # Rank both
            x_ranks = cp.argsort(cp.argsort(x_clean)).astype(cp.float32) + 1
            y_ranks = cp.argsort(cp.argsort(y_clean)).astype(cp.float32) + 1
            
            # Pearson on ranks
            x_centered = x_ranks - cp.mean(x_ranks)
            y_centered = y_ranks - cp.mean(y_ranks)
            
            cov = cp.sum(x_centered * y_centered)
            std_x = cp.sqrt(cp.sum(x_centered ** 2))
            std_y = cp.sqrt(cp.sum(y_centered ** 2))
            
            if std_x > 0 and std_y > 0:
                correlations[i] = cov / (std_x * std_y)
            else:
                correlations[i] = cp.nan
        
        result = cp.asnumpy(correlations)
        
        # Clean up
        del X_gpu, y_gpu, correlations
        cp.get_default_memory_pool().free_all_blocks()
        
        return result
        
    except Exception as e:
        print(f"Batch Spearman error: {e}")
        return np.full(features_matrix.shape[1], np.nan)

def fast_kendall_correlation(series1, series2):
    """Fast Kendall's tau using sampling for very large datasets."""
    if not CALCULATE_KENDALL:
        return np.nan
    
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        
        if len(s1) < 10:
            return np.nan
        
        # Always sample for speed
        if len(s1) > 10000:
            # Sample pairs for fast approximation
            n_pairs_sample = 50000
            idx1 = np.random.choice(len(s1), n_pairs_sample, replace=True)
            idx2 = np.random.choice(len(s1), n_pairs_sample, replace=True)
            
            concordant = np.sum((s1[idx1] > s1[idx2]) == (s2[idx1] > s2[idx2]))
            discordant = np.sum((s1[idx1] > s1[idx2]) != (s2[idx1] > s2[idx2]))
            
            tau = (concordant - discordant) / n_pairs_sample
            return tau
        else:
            corr, _ = kendalltau(s1, s2)
            return corr
    except:
        return np.nan

def fast_mutual_information(series1, series2):
    """Fast MI calculation."""
    if not CALCULATE_MUTUAL_INFO:
        return np.nan
    
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        
        if len(s1) < 50:
            return np.nan
        
        s1 = s1.reshape(-1, 1)
        n_neighbors = min(3, len(s1) // 20)  # Reduced neighbors for speed
        mi = mutual_info_regression(s1, s2, random_state=42, n_neighbors=n_neighbors)[0]
        return mi
    except:
        return np.nan

def fast_granger_causality(series1, series2, max_lag=3, subsample=5000):
    """Fast Granger test with aggressive subsampling."""
    if not CALCULATE_GRANGER:
        return False
    
    try:
        series1_np = np.asarray(series1)
        series2_np = np.asarray(series2)
        mask = ~(np.isnan(series1_np) | np.isnan(series2_np))
        s1 = series1_np[mask]
        s2 = series2_np[mask]
        
        if len(s1) < 100:
            return False
        
        # Aggressive subsample for speed
        if len(s1) > subsample:
            indices = np.arange(len(s1))
            sample_idx = np.random.choice(indices, subsample, replace=False)
            sample_idx = np.sort(sample_idx)
            s1 = s1[sample_idx]
            s2 = s2[sample_idx]
        
        data = pd.DataFrame({'series2': s2, 'series1': s1})
        
        max_test_lag = min(max_lag, len(s1) // 10)
        if max_test_lag < 1:
            return False
        
        gc_results = grangercausalitytests(data[['series2', 'series1']], 
                                          maxlag=max_test_lag, 
                                          verbose=False)
        
        for lag in range(1, max_test_lag + 1):
            f_test = gc_results[lag][0]['ssr_ftest']
            if f_test[1] < 0.05:
                return True
        return False
    except:
        return False

def apply_time_lag_gpu(price_data, time_data, lag_seconds):
    """GPU-accelerated time lag application."""
    if lag_seconds == 0:
        return price_data
    
    try:
        time_gpu = cp.asarray(time_data, dtype=cp.float64)
        price_gpu = cp.asarray(price_data, dtype=cp.float32)
        
        if not IRREGULAR_TIME_SERIES:
            # Fast method for regular time series
            if len(time_gpu) < 2:
                return price_data
            
            time_diffs = cp.diff(time_gpu)
            valid_diffs = time_diffs[time_diffs > 0]
            
            if len(valid_diffs) == 0:
                return price_data
            
            avg_time_step = float(cp.median(valid_diffs))
            
            if avg_time_step == 0 or cp.isnan(avg_time_step):
                return price_data
            
            shift_positions = int(round(lag_seconds / avg_time_step))
            
            if shift_positions <= 0 or shift_positions >= len(price_gpu):
                return price_data
            
            lagged_price = cp.full_like(price_gpu, cp.nan)
            lagged_price[:-shift_positions] = price_gpu[shift_positions:]
            
            result = cp.asnumpy(lagged_price)
        else:
            # Precise method for irregular time series
            target_times = time_gpu + lag_seconds
            future_indices = cp.searchsorted(time_gpu, target_times, side='left')
            valid_mask = future_indices < len(price_gpu)
            
            lagged_price = cp.full_like(price_gpu, cp.nan)
            lagged_price[valid_mask] = price_gpu[future_indices[valid_mask]]
            
            result = cp.asnumpy(lagged_price)
        
        del time_gpu, price_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return result
        
    except Exception as e:
        print(f"Time lag error: {e}")
        return price_data

def analyze_batch_features(batch_info):
    """Analyze a batch of features - designed for multiprocessing."""
    
    feature_names, feature_data_dict, price_data, time_data, lag_seconds, total_rows, batch_idx, total_batches = batch_info
    
    try:
        # Apply time lag to price once for this batch
        lagged_price = apply_time_lag_gpu(price_data, time_data, lag_seconds)
        
        # Stack all features into a matrix
        n_features = len(feature_names)
        n_samples = len(price_data)
        
        features_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
        for i, fname in enumerate(feature_names):
            features_matrix[:, i] = feature_data_dict[fname]
        
        # Batch GPU calculations
        print(f"  [Batch {batch_idx}/{total_batches}] Processing {n_features} features on GPU...")
        start_time = time.time()
        
        pearson_corrs, valid_counts = gpu_batch_pearson_corr(features_matrix, lagged_price)
        spearman_corrs = gpu_batch_spearman_corr(features_matrix, lagged_price)
        
        gpu_time = time.time() - start_time
        print(f"  [Batch {batch_idx}/{total_batches}] GPU calculations done in {gpu_time:.2f}s")
        
        # Build results
        results = []
        for i, feature_name in enumerate(feature_names):
            n_valid = valid_counts[i]
            n_nan = total_rows - n_valid
            pct_valid = 100 * n_valid / total_rows
            
            if n_valid < 10:
                continue
            
            # Get clean data for slow metrics
            feature_data = features_matrix[:, i]
            mask = ~(np.isnan(lagged_price) | np.isnan(feature_data))
            price_clean = lagged_price[mask]
            feature_clean = feature_data[mask]
            
            # Calculate slow metrics if enabled
            kendall = fast_kendall_correlation(price_clean, feature_clean) if CALCULATE_KENDALL else np.nan
            mi = fast_mutual_information(price_clean, feature_clean) if CALCULATE_MUTUAL_INFO else np.nan
            granger = fast_granger_causality(feature_clean, price_clean, 
                                            max_lag=GRANGER_MAX_LAG, 
                                            subsample=GRANGER_SUBSAMPLE) if CALCULATE_GRANGER else False
            
            result = {
                'Feature': feature_name,
                'Time_Lag_Sec': lag_seconds,
                'Total_Rows': total_rows,
                'Valid_Points': int(n_valid),
                'NaN_Count': int(n_nan),
                'Valid_Pct': pct_valid,
                'Pearson': float(pearson_corrs[i]),
                'Spearman': float(spearman_corrs[i]),
                'Kendall': float(kendall),
                'Mutual_Info': float(mi),
                'Granger→Price': 'YES' if granger else 'NO'
            }
            
            results.append(result)
        
        print(f"  [Batch {batch_idx}/{total_batches}] Completed {len(results)}/{n_features} features")
        return results
        
    except Exception as e:
        print(f"Error in batch {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
        return []

def analyze_features_parallel(feature_names, feature_data_dict, price_data, time_data, 
                              lag_seconds, total_rows, n_processes=4, batch_size=20):
    """Analyze multiple features in parallel using multiprocessing with GPU batching."""
    
    print(f"\n{'='*100}")
    print(f"ANALYZING TIME LAG: {lag_seconds} seconds")
    print(f"Processing {len(feature_names)} features in batches of {batch_size}")
    print(f"Using {n_processes} parallel processes")
    print(f"{'='*100}")
    
    # Split features into batches
    batches = []
    for i in range(0, len(feature_names), batch_size):
        batch_features = feature_names[i:i+batch_size]
        batches.append(batch_features)
    
    print(f"Split into {len(batches)} batches")
    
    # Prepare batch info for each process
    batch_infos = []
    for batch_idx, batch_features in enumerate(batches, 1):
        batch_info = (
            batch_features,
            feature_data_dict,
            price_data,
            time_data,
            lag_seconds,
            total_rows,
            batch_idx,
            len(batches)
        )
        batch_infos.append(batch_info)
    
    # Process batches in parallel
    all_results = []
    
    if n_processes > 1:
        with Pool(processes=n_processes) as pool:
            batch_results = pool.map(analyze_batch_features, batch_infos)
            for batch_result in batch_results:
                all_results.extend(batch_result)
    else:
        # Single process mode
        for batch_info in batch_infos:
            batch_result = analyze_batch_features(batch_info)
            all_results.extend(batch_result)
    
    return all_results

def load_and_analyze_data(file_path, blocksize='256MB', n_processes=4, batch_size=20):
    """Load and analyze data from combined CSV file with multiple time lags."""
    
    print(f"Loading data from: {file_path}")
    print(f"Blocksize: {blocksize}")
    print(f"Parallel processes: {n_processes}")
    print(f"Batch size: {batch_size}")
    print(f"Time lags to analyze: {TIME_LAGS} seconds")
    print("="*100)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File '{file_path}' does not exist!")
        return None
    
    try:
        # Read header
        print("Reading CSV header...")
        sample_df = pd.read_csv(file_path, nrows=10)
        columns = sample_df.columns.tolist()
        print(f"✓ Found {len(columns)} columns")
        
        # Identify price and time columns
        price_col = None
        for col in columns:
            if col.lower() == PRICE_COLUMN.lower():
                price_col = col
                break
        
        time_col = None
        for col in columns:
            if col.lower() == TIME_COLUMN.lower():
                time_col = col
                break
        
        if not price_col or not time_col:
            print(f"ERROR: Required columns not found!")
            return None
        
        # Get feature columns
        all_feature_cols = [col for col in columns if col != price_col and col != time_col]
        
        # Filter features
        if FEATURES_TO_ANALYZE:
            feature_cols = [f for f in FEATURES_TO_ANALYZE if f in all_feature_cols]
        elif FEATURE_PATTERN:
            import fnmatch
            feature_cols = [f for f in all_feature_cols if fnmatch.fnmatch(f, FEATURE_PATTERN)]
            print(f"✓ Pattern '{FEATURE_PATTERN}' matched {len(feature_cols)} features")
        elif FEATURE_INDEX_RANGE:
            start_idx, end_idx = FEATURE_INDEX_RANGE
            feature_cols = all_feature_cols[start_idx:end_idx]
            print(f"✓ Selected features {start_idx} to {end_idx-1}")
        else:
            feature_cols = all_feature_cols
        
        print(f"✓ Price column: {price_col}")
        print(f"✓ Time column: {time_col}")
        print(f"✓ Features to analyze: {len(feature_cols)}")
        
        if len(feature_cols) == 0:
            print("ERROR: No features selected!")
            return None
        
        # Load data with dask_cudf
        print("\nLoading data with dask_cudf (GPU accelerated)...")
        cols_to_load = [price_col, time_col] + feature_cols
        ddf = dask_cudf.read_csv(file_path, blocksize=blocksize, usecols=cols_to_load)
        
        print(f"✓ Data will be processed in chunks of {blocksize}")
        
        # Process chunks
        print("Processing chunks...")
        price_data_list = []
        time_data_list = []
        feature_data_dict = {col: [] for col in feature_cols}
        
        total_rows = 0
        n_partitions = ddf.npartitions
        
        for part_idx in range(n_partitions):
            partition = ddf.get_partition(part_idx).compute()
            total_rows += len(partition)
            
            price_data_list.append(partition[price_col].to_numpy())
            time_data_list.append(partition[time_col].to_numpy())
            
            for col in feature_cols:
                feature_data_dict[col].append(partition[col].to_numpy())
            
            del partition
            
            if (part_idx + 1) % 10 == 0:
                print(f"  Processed {part_idx + 1}/{n_partitions} partitions ({total_rows:,} rows)...")
        
        print(f"✓ Loaded {total_rows:,} rows from {n_partitions} partitions")
        
        # Concatenate
        print("Concatenating data...")
        price_data = np.concatenate(price_data_list)
        time_data = np.concatenate(time_data_list)
        
        for col in feature_cols:
            feature_data_dict[col] = np.concatenate(feature_data_dict[col])
        
        del price_data_list, time_data_list
        
        print(f"✓ Data ready: {total_rows:,} rows × {len(feature_cols)} features")
        
        # Analyze for each time lag using parallel processing
        all_results = {}
        
        for lag_seconds in TIME_LAGS:
            lag_start = time.time()
            
            lag_results = analyze_features_parallel(
                feature_cols,
                feature_data_dict,
                price_data,
                time_data,
                lag_seconds,
                total_rows,
                n_processes=n_processes,
                batch_size=batch_size
            )
            
            lag_time = time.time() - lag_start
            
            if lag_results:
                results_df = pd.DataFrame(lag_results)
                all_results[lag_seconds] = results_df
                
                print(f"\n✓ Completed {lag_seconds}s lag in {lag_time:.2f}s: {len(lag_results)} features analyzed")
                print(f"  Speed: {len(lag_results)/lag_time:.1f} features/second")
                
                # Show top 5
                results_df['Abs_Pearson'] = results_df['Pearson'].abs()
                top_5 = results_df.nlargest(5, 'Abs_Pearson')
                print(f"\nTop 5 features at {lag_seconds}s lag:")
                for i, row in enumerate(top_5.itertuples(), 1):
                    print(f"  {i}. {row.Feature}: Pearson={row.Pearson:+.4f}, Spearman={row.Spearman:+.4f}")
        
        # Save results to Excel with multiple sheets
        if all_results:
            print("\n" + "="*100)
            print("SAVING RESULTS TO EXCEL")
            print("="*100)
            
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file = os.path.join(OUTPUT_DIR, f'multi_lag_analysis_{timestamp}.xlsx')
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for lag_seconds in TIME_LAGS:
                    if lag_seconds in all_results:
                        results_df = all_results[lag_seconds].copy()
                        results_df['Abs_Pearson'] = results_df['Pearson'].abs()
                        results_df_sorted = results_df.sort_values('Abs_Pearson', ascending=False)
                        results_df_sorted = results_df_sorted.drop('Abs_Pearson', axis=1)
                        
                        sheet_name = f'Lag_{lag_seconds}s'
                        results_df_sorted.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Format the sheet
                        from openpyxl.styles import Font, PatternFill, Alignment
                        worksheet = writer.sheets[sheet_name]
                        
                        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
                        header_font = Font(bold=True, color='FFFFFF')
                        
                        for cell in worksheet[1]:
                            cell.fill = header_fill
                            cell.font = header_font
                            cell.alignment = Alignment(horizontal='center', vertical='center')
                        
                        for column in worksheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)
                            worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"✓ Excel file saved: {excel_file}")
            print(f"✓ Sheets created: {len(all_results)} (one per time lag)")
            
            # Generate text summaries for each lag
            for lag_seconds in TIME_LAGS:
                if lag_seconds in all_results:
                    results_df = all_results[lag_seconds]
                    txt_file = os.path.join(OUTPUT_DIR, f'lag_{lag_seconds}s_summary_{timestamp}.txt')
                    
                    with open(txt_file, 'w') as f:
                        f.write("="*100 + "\n")
                        f.write(f"FEATURE CORRELATION ANALYSIS - {lag_seconds}s LAG\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("="*100 + "\n\n")
                        
                        f.write("STATISTICS\n")
                        f.write("-"*100 + "\n")
                        f.write(f"Total Features: {len(results_df)}\n")
                        f.write(f"Strong (|r| > 0.5): {len(results_df[results_df['Pearson'].abs() > 0.5])}\n")
                        f.write(f"Moderate (|r| > 0.3): {len(results_df[results_df['Pearson'].abs() > 0.3])}\n")
                        f.write(f"Weak (|r| > 0.1): {len(results_df[results_df['Pearson'].abs() > 0.1])}\n")
                        if CALCULATE_GRANGER:
                            f.write(f"Granger Causality: {len(results_df[results_df['Granger→Price'] == 'YES'])}\n")
                        f.write(f"Avg Data Quality: {results_df['Valid_Pct'].mean():.2f}%\n\n")
                        
                        results_df['Abs_Pearson'] = results_df['Pearson'].abs()
                        top_10 = results_df.nlargest(10, 'Abs_Pearson')
                        
                        f.write("TOP 10 FEATURES\n")
                        f.write("-"*100 + "\n")
                        f.write(f"{'Rank':<6} {'Feature':<30} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10} {'MI':<10} {'Granger':<8}\n")
                        f.write("-"*100 + "\n")
                        
                        for i, row in enumerate(top_10.itertuples(), 1):
                            f.write(f"{i:<6} {row.Feature:<30} {row.Pearson:+.4f}    {row.Spearman:+.4f}    "
                                   f"{row.Kendall:+.4f}    {row.Mutual_Info:.4f}    {row._11:<8}\n")
                        
                        f.write("\n" + "="*100 + "\n")
                    
                    print(f"✓ Summary saved: {txt_file}")
            
            # Final summary
            print("\n" + "="*100)
            print("ANALYSIS COMPLETE!")
            print("="*100)
            print(f"✓ Total features: {len(feature_cols)}")
            print(f"✓ Total rows: {total_rows:,}")
            print(f"✓ Time lags analyzed: {len(TIME_LAGS)}")
            print(f"✓ Parallel processes used: {n_processes}")
            print(f"✓ Batch size: {batch_size}")
            print(f"✓ Output directory: {OUTPUT_DIR}")
            print("="*100)
        
        return all_results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "="*100)
    print("ULTRA-FAST GPU-ACCELERATED MULTI-LAG FEATURE ANALYSIS")
    print("="*100)
    print(f"Data file: {DATA_FILE}")
    print(f"Chunk size: {BLOCKSIZE}")
    print(f"Parallel processes: {N_PROCESSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Time lags: {TIME_LAGS} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Time series type: {'IRREGULAR' if IRREGULAR_TIME_SERIES else 'REGULAR'}")
    print(f"\nMetrics enabled:")
    print(f"  Pearson: YES (GPU batched)")
    print(f"  Spearman: YES (GPU batched)")
    print(f"  Kendall: {'YES' if CALCULATE_KENDALL else 'NO (disabled for speed)'}")
    print(f"  Mutual Info: {'YES' if CALCULATE_MUTUAL_INFO else 'NO (disabled for speed)'}")
    print(f"  Granger: {'YES' if CALCULATE_GRANGER else 'NO (disabled for speed)'}")
    
    if FEATURES_TO_ANALYZE:
        print(f"\nMode: Analyzing SPECIFIC features ({len(FEATURES_TO_ANALYZE)} features)")
    elif FEATURE_PATTERN:
        print(f"\nMode: Analyzing features matching PATTERN: '{FEATURE_PATTERN}'")
    elif FEATURE_INDEX_RANGE:
        print(f"\nMode: Analyzing features by INDEX RANGE: {FEATURE_INDEX_RANGE}")
    else:
        print(f"\nMode: Analyzing ALL features")
    
    print("="*100 + "\n")
    
    results = load_and_analyze_data(DATA_FILE, BLOCKSIZE, n_processes=N_PROCESSES, batch_size=BATCH_SIZE)