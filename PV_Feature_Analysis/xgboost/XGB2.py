import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import random
import gc
import xgboost as xgb

# ADDED: For train/test split and accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix # <-- ADDED confusion_matrix

# Try to import Numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python (install with: pip install numba)")
    # Create a dummy decorator if Numba is not available
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# Try to import GPU libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - using CPU only (install with: pip install cupy-cuda12x)")

try:
    import dask_cudf
    import dask.dataframe as dd
    DASK_CUDF_AVAILABLE = True
    print("✓ Dask-cuDF detected - GPU-accelerated dataframe operations enabled")
except ImportError:
    DASK_CUDF_AVAILABLE = False
    print("✗ Dask-cudf not found - using pandas (install with: pip install dask-cudf)")


# ============================================================================
# TRIPLE BARRIER LABELING
# ============================================================================

@jit(nopython=True, fastmath=True)
def create_forward_labels(prices, timestamps_sec, lookahead_seconds=2400.0, price_target=0.25):
    """
    Creates forward-looking labels using the Triple Barrier Method.

    Labels:
    1: Price hit upper barrier (Price + 0.25) first.
    2: Price hit lower barrier (Price - 0.25) first.
    3: Price hit time barrier (2400s) without hitting price barriers.
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        current_price = prices[i]
        current_time = timestamps_sec[i]

        upper_barrier = current_price + price_target
        lower_barrier = current_price - price_target
        time_barrier = current_time + lookahead_seconds
        
        # Default label is 3 (hits time barrier)
        labels[i] = 3 

        for j in range(i + 1, n):
            future_time = timestamps_sec[j]
            
            # 1. Check time barrier
            if future_time > time_barrier:
                break 
            
            future_price = prices[j]

            # 2. Check price barriers
            if future_price >= upper_barrier:
                labels[i] = 1
                break
            
            elif future_price <= lower_barrier:
                labels[i] = 2
                break
        
    return labels


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_day_for_labels(day_num, data_dir):
    """
    Process single day, keep all features, and add the forward label.
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        # Load data
        if DASK_CUDF_AVAILABLE:
            try:
                ddf = dask_cudf.read_parquet(str(file_path))
                df = ddf.compute()
                df = df.to_pandas()
            except:
                df = pd.read_parquet(file_path)
        else:
            df = pd.read_parquet(file_path)

        if 'Price' not in df.columns:
            return None

        # Handle timestamps
        timestamp_col = next((c for c in df.columns if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col:
            try:
                dt_series = pd.to_datetime(df[timestamp_col], format='mixed', errors='coerce')
                df['timestamp_seconds'] = (
                    dt_series.dt.hour * 3600 + 
                    dt_series.dt.minute * 60 + 
                    dt_series.dt.second + 
                    dt_series.dt.microsecond / 1e6
                )
                if df['timestamp_seconds'].isna().any():
                    raise ValueError("Timestamp conversion failed")
            except:
                df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)
        else:
            df['timestamp_seconds'] = np.arange(len(df), dtype=np.float64)

        # Get arrays from cleaned data
        required_cols = ['Price', 'timestamp_seconds']
        original_index = df.index
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 2:
            return None

        prices = df_clean['Price'].values.astype(np.float64)
        timestamps = df_clean['timestamp_seconds'].values.astype(np.float64)
        
        # Generate labels
        labels = create_forward_labels(prices, timestamps, 
                                     lookahead_seconds=2400.0, # <-- Set to 2400s
                                     price_target=0.25)
        
        # Align labels back to original dataframe
        label_series = pd.Series(labels, index=df_clean.index, name='Label')
        df = df.join(label_series)
        df = df.drop(columns=['timestamp_seconds'])

        return df

    except Exception as e:
        return None


def label_wrapper(args):
    """Wrapper for parallel processing"""
    day_num, data_dir = args
    return process_day_for_labels(day_num, data_dir)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    data_dir = '/data/quant14/EBY'
    num_days = 279
    
    # <--- MODIFIED: Path to the POOLED feature importance file
    importance_file_path = 'feature_importance_pooled.csv'
    
    print("="*80)
    print("POOLED TRAINING (OPTION 1) - TOP 50 FEATURES (from Pooled)") # <--- MODIFIED
    print("="*80)
    print(f"Configuration:")
    print(f"   Data directory: {data_dir}")
    print(f"   Total days: {num_days}")
    print(f"   Acceleration: {'GPU (Dask-cuDF + CuPy)' if (DASK_CUDF_AVAILABLE and GPU_AVAILABLE) else 'CPU (Pandas)'}")
    print(f"   JIT Engine: {'Numba' if NUMBA_AVAILABLE else 'Standard Python'}")
    print(f"\n   === FEATURE SELECTION ===") # <--- MODIFIED
    print(f"   Method: Loading Top 50 features from '{importance_file_path}'") # <--- MODIFIED
    print(f"\n   === LABELING ===")
    print(f"   Target: $0.25")
    print(f"   Time Horizon: 2400 seconds")
    print(f"   Label 1: Price >= +0.25")
    print(f"   Label 2: Price <= -0.25")
    print(f"   Label 3: Price stays within range for 2400s")
    print(f"\n   === IMBALANCE HANDLING ===")
    print(f"   Method: Undersampling (on 80% train set)")
    print(f"\n   === EVALUATION ===")
    print(f"   Method: Pooled 80/20 Train/Test Split (shuffle=False)") 
    print(f"   Metric: Final Test Accuracy & Class Recall") 
    print("="*80)
    
    # Check data directory
    print(f"\n[1/5] Checking data directory...")
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    parquet_files = list(data_path.glob("day*.parquet"))
    print(f"   ✓ Found {len(parquet_files)} day*.parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No day*.parquet files found in {data_dir}")
        return
    
    # Process all days in parallel
    print(f"\n[2/5] Generating labels for {num_days} days (in parallel)...")
    print(f"   (Using parallel processing with {min(cpu_count(), 16)} workers)")
    start_time = time.time()
    
    workers = min(cpu_count(), 16)
    all_days = list(range(num_days))
    args_list = [(day_num, data_dir) for day_num in all_days]
    
    with Pool(workers) as pool:
        all_labeled_results = pool.map(label_wrapper, args_list)
    
    # Filter out None results
    all_labeled_dfs = [res for res in all_labeled_results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"   ✓ Data preparation completed in {elapsed:.1f} seconds")
    print(f"   ✓ Successfully processed {len(all_labeled_dfs)} days of data")
    
    if len(all_labeled_dfs) == 0:
        print("\nERROR: No data was processed")
        return

    # ========================================================================
    # [3/5] CALCULATE TOTAL LABEL DISTRIBUTION
    # ========================================================================
    print(f"\n[3/5] Calculating total label distribution...")
    try:
        all_labels_series = pd.concat([df['Label'] for df in all_labeled_dfs])
        all_labels_clean = all_labels_series.dropna()
        all_labels_remapped = all_labels_clean - 1
        label_counts = all_labels_remapped.value_counts().sort_index()
        total_vals = len(all_labels_remapped)
        
        count_0 = label_counts.get(0, 0)
        count_1 = label_counts.get(1, 0)
        count_2 = label_counts.get(2, 0)

        print(f"   Total Valid Labels: {total_vals}")
        print(f"   (0-Up):   {count_0}/{total_vals}  ({count_0/total_vals*100:.2f}%)")
        print(f"   (1-Down): {count_1}/{total_vals}  ({count_1/total_vals*100:.2f}%)")
        print(f"   (2-Time): {count_2}/{total_vals}  ({count_2/total_vals*100:.2f}%)")
        
        del all_labels_series, all_labels_clean, all_labels_remapped, label_counts
        gc.collect()
    except Exception as e:
        print(f"   Could not calculate total label distribution: {e}")
    
    
    # ========================================================================
    # <--- MODIFIED BLOCK: LOAD TOP 50 FEATURES --->
    # ========================================================================
    print(f"\n[PRE-4] Loading top 50 features from '{importance_file_path}'...") # <--- MODIFIED
    try:
        imp_df = pd.read_csv(importance_file_path)
    except FileNotFoundError:
        print(f"   ERROR: '{importance_file_path}' not found.")
        print(f"   Please run the 'pooled' script first to generate this file.")
        return

    if 'Feature' not in imp_df.columns:
        print(f"   ERROR: 'Feature' column not in '{importance_file_path}'.")
        return

    # <--- MODIFIED: Select the top 50 feature names
    top_50_features = imp_df['Feature'].head(50).tolist()
    print(f"   ✓ Loaded {len(top_50_features)} features for retraining.")
    
    # This list now replaces the dynamically generated feature_cols
    feature_cols = top_50_features

    # --- Sanity check: Ensure these features exist in the loaded data ---
    available_cols = set(all_labeled_dfs[0].columns)
    missing_features = [f for f in feature_cols if f not in available_cols]
    
    if missing_features:
         print(f"   ERROR: The following {len(missing_features)} features from your CSV are missing from the parquet files:")
         print(f"   {missing_features}")
         return
    # ========================================================================
    
    
    # ========================================================================
    # <--- MODIFIED BLOCK: [4/5] POOLED TRAINING AND EVALUATION (TOP 50) --->
    # ========================================================================
    print(f"\n[4/5] Starting pooled training (TOP 50 FEATURES)...") 
    
    print(f"   ✓ Using {len(feature_cols)} features from '{importance_file_path}'.")

    # 1. Concatenate all data
    print("   Concatenating all daily dataframes...")
    all_data_df = pd.concat(all_labeled_dfs, ignore_index=True)
    del all_labeled_dfs # Free memory
    gc.collect()

    # 2. Clean final dataframe
    print("   Cleaning pooled dataframe...")
    # <--- MODIFIED: Only select the Top 50 features + Label
    cols_to_load = feature_cols + ['Label']
    df_clean = all_data_df[cols_to_load].dropna()
    del all_data_df # Free memory
    gc.collect()

    if len(df_clean) < 100:
        print(f"   ERROR: Not enough data to train (need > 100, have {len(df_clean)})")
        return

    # 3. Create X and y
    X = df_clean[feature_cols]
    y = df_clean['Label'] - 1 # Remap to [0, 1, 2]
    del df_clean # Free memory
    gc.collect()

    # 4. Create single Train/Test Split
    print("   Creating 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False # DO NOT SHUFFLE time-series data
    )

    # 5. Undersample the Training Set
    print("   Undersampling training set...")
    train_df = X_train.copy()
    train_df['Label'] = y_train
    
    label_1s = train_df[train_df['Label'] == 0] # 0 = Up
    label_2s = train_df[train_df['Label'] == 1] # 1 = Down
    label_3s = train_df[train_df['Label'] == 2] # 2 = Time
    
    rare_count = len(label_1s) + len(label_2s)
    
    if rare_count == 0:
        print(f"   ERROR: No 1s or 2s found in the entire training set. Stopping.")
        return
        
    if len(label_3s) > rare_count:
        label_3s_sampled = label_3s.sample(rare_count, random_state=42)
    else:
        label_3s_sampled = label_3s

    df_balanced = pd.concat([label_1s, label_2s, label_3s_sampled])
    
    X_train_balanced = df_balanced[feature_cols]
    y_train_balanced = df_balanced['Label']
    
    print(f"   Original train size: {len(X_train)}, Balanced train size: {len(X_train_balanced)}")
    
    del X_train, y_train, train_df, label_1s, label_2s, label_3s, label_3s_sampled, df_balanced
    gc.collect()

    # 6. Create DMatrix objects
    print("   Creating DMatrix objects...")
    dtrain = xgb.DMatrix(X_train_balanced, label=y_train_balanced, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols) # Test set is imbalanced
    
    del X_train_balanced, y_train_balanced, X_test, y_test
    gc.collect()

    # 7. Train the Model
    # XGBoost parameters
    xgb_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'device': 'cuda' if GPU_AVAILABLE else 'cpu',
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    if not GPU_AVAILABLE:
        xgb_params['tree_method'] = 'hist'
    
    print("   Starting model training (500 rounds with early stopping)...")
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500, # Train for a fixed number of rounds
        evals=[(dtrain, 'train'), (dtest, 'test')], 
        early_stopping_rounds=50, # Stop if test-mlogloss doesn't improve for 50 rounds
        verbose_eval=100
    )
    
    print(f"   ✓ Training complete. Final trees: {booster.best_iteration}")

    # 8. Evaluate on Test Set
    print(f"\n   === FINAL POOLED PERFORMANCE (TOP 50) ===") 
    preds = booster.predict(dtest)
    acc = accuracy_score(dtest.get_label(), preds)
    print(f"   Final Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(dtest.get_label(), preds, labels=[0, 1, 2])
    
    correct_preds = np.diag(cm)
    total_actuals = cm.sum(axis=1)

    recall_0 = correct_preds[0] / total_actuals[0] if total_actuals[0] > 0 else 0.0
    recall_1 = correct_preds[1] / total_actuals[1] if total_actuals[1] > 0 else 0.0
    recall_2 = correct_preds[2] / total_actuals[2] if total_actuals[2] > 0 else 0.0

    print(f"\n   Final Test Recall (Correct/Total):")
    print(f"   Recall (0-Up):   {correct_preds[0]}/{total_actuals[0]} ({recall_0:.4f})")
    print(f"   Recall (1-Down): {correct_preds[1]}/{total_actuals[1]} ({recall_1:.4f})")
    print(f"   Recall (2-Time): {correct_preds[2]}/{total_actuals[2]} ({recall_2:.4f})")
    
    print("\n   Confusion Matrix (Rows=Actual, Cols=Predicted):")
    print("          Pred 0 (Up) | Pred 1 (Down) | Pred 2 (Time)")
    print(f"Actual 0: {cm[0,0]:<11} | {cm[0,1]:<13} | {cm[0,2]:<13}")
    print(f"Actual 1: {cm[1,0]:<11} | {cm[1,1]:<13} | {cm[1,2]:<13}")
    print(f"Actual 2: {cm[2,0]:<11} | {cm[2,1]:<13} | {cm[2,2]:<13}")

    # ========================================================================
    # [5/5] GET FEATURE IMPORTANCE AND SAVE (FROM TOP 50)
    # ========================================================================
    print(f"\n[5/5] Calculating final feature importance (TOP 50)...") 

    if booster is None:
        print("   ERROR: No model was trained. Skipping feature importance.")
        return

    # 1. Get raw gain scores
    importance = booster.get_score(importance_type='gain')

    # 2. Calculate total gain
    total_gain = sum(importance.values())

    # 3. Create list of features with fractional gain
    importance_data = []
    if total_gain > 0:
        for feature, score in importance.items():
            importance_data.append({
                'Feature': feature,
                'Fractional_Gain': score / total_gain
            })
    else:
        # Handle case where no gain was found
        for feature in feature_cols: # Use the original 50 features
            importance_data.append({
                'Feature': feature,
                'Fractional_Gain': 0.0
            })

    # 4. Sort by fractional gain (descending)
    sorted_importance_data = sorted(importance_data, key=lambda item: item['Fractional_Gain'], reverse=True)

    # 5. APPLY FEATURE IMPORTANCE THRESHOLD
    total_features = len(feature_cols) 
    threshold = 1 / total_features if total_features > 0 else 0
    filtered_importance_data = [item for item in sorted_importance_data if item['Fractional_Gain'] > threshold]

    print(f"\n   Applied Feature Importance Threshold: > {threshold:.4%}")
    print(f"   Features retained (printed below): {len(filtered_importance_data)} / {total_features}")

    # 6. Print final feature importance table (ONLY filtered)
    print(f"\n === FINAL FEATURE IMPORTANCE (Top 20 from TOP 50 Run) ===")
    print(f"   {'Feature':<30} | {'Fractional Gain':<18}")
    print("    " + "-" * 50)
    for item in filtered_importance_data[:20]: # Print top 20 of the filtered list
        print(f"   {item['Feature']:<30} | {item['Fractional_Gain']:<18.4%}")

    # 7. Save the model
    # <--- MODIFIED: Save to new file
    model_path = 'final_pooled_xgb_model_TOP50.json' 
    print(f"\nSaving final model to '{model_path}'...")
    booster.save_model(model_path)

    # 8. Save ALL 50 sorted importance features to CSV
    # <--- MODIFIED: Save to new file
    importance_path = 'feature_importance_pooled_TOP50.csv' 
    print(f"Saving ALL {len(sorted_importance_data)} feature importances to '{importance_path}'...")

    imp_df = pd.DataFrame(sorted_importance_data) 
    if not imp_df.empty:
        imp_df = imp_df[['Feature', 'Fractional_Gain']]

    imp_df.to_csv(importance_path, index=False)

    print("\n" + "="*80)
    print("✓ TOP 50 POOLED TRAINING & FEATURE IMPORTANCE COMPLETED!") # <--- MODIFIED
    print("="*80)


if __name__ == '__main__':
    main()