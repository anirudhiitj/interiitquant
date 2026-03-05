import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
# Import confusion_matrix, removed ParameterGrid and KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings
import os
import gc
import psutil
from datetime import datetime
from generate_labels import LabelGenerator
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS - OPTIMIZED FOR SPEED WITH 40GB RAM
# ============================================================================
DATA_DIR = '/data/quant14/EBY'
DAY_FILE_PATTERN = 'day{}.parquet'
START_DAY = 0
END_DAY = 278

OUTPUT_DIR = './xgboost_results'
MODELS_DIR = './xgboost_models'

FEATURE_PREFIXES = [
    'PB10_', 'PB11_', 'PB6_', 'PB3_', 'PB1_', 'PB2_', 'PB7_', 'PB14_', 'PB15_',
    'BB13_', 'BB14_', 'BB15_', 'BB4_', 'BB5_', 'BB6_', 'BB23', 'VB4_', 'VB5_'
]

# SPEED OPTIMIZATIONS
DAYS_PER_BATCH = 50  # Increased from 20 (use more RAM for speed)
MAX_RAM_GB = 40  # Increased limit
PARALLEL_DAY_LOADING = True  # Load multiple days in parallel
NUM_WORKERS = 4  # Parallel workers for data loading

# Generate all labels at once (faster than one-by-one)
GENERATE_ALL_LABELS_AT_ONCE = True  

USE_UNDERSAMPLING = True
UNDERSAMPLING_STRATEGY = 'not majority'  # Note: space not underscore!
TRAIN_SPLIT = 0.8

# --- REMOVED: Hyperparameter grids are now fixed ---
# XGBOOST_PARAM_GRID_CLASSIFICATION = { ... }
# XGBOOST_PARAM_GRID_REGRESSION = { ... }

BASE_PARAMS_CLASSIFICATION = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1  # Use all CPU cores
}

BASE_PARAMS_REGRESSION = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1
}


def get_memory_usage_gb():
    """Get current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()


def load_single_day_worker(args):
    """
    Worker function for parallel day loading
    Returns: (day_num, df) or (day_num, None) if failed
    """
    day_num, filepath, columns_to_load = args
    
    if not os.path.exists(filepath):
        return day_num, None
    
    try:
        df = pd.read_parquet(filepath, columns=columns_to_load)
        df['Day'] = day_num
        return day_num, df
    except Exception as e:
        print(f"      Error loading day {day_num}: {e}")
        return day_num, None


class FastXGBoostTrainer:
    """
    Speed-optimized XGBoost trainer - uses 40GB RAM for maximum performance
    
    Key optimizations:
    1. Parallel day loading (4 workers)
    2. Generate all 30 labels at once per day (faster than individual)
    3. Larger batches (50 days instead of 20)
    4. Keep data in memory longer
    5. --- REMOVED: Grid search removed for fixed params ---
    """
    
    def __init__(self, feature_prefixes=None, output_dir='./xgboost_results', 
                 models_dir='./xgboost_models'):
        self.feature_prefixes = feature_prefixes
        self.feature_columns = None
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.label_gen = LabelGenerator(use_cpu_fallback=True)
        self.label_results = {}
        
        # Cache for storing DataFrames with all labels
        self.data_cache = {}
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        print("=" * 100)
        print("FAST XGBOOST TRAINER (40GB RAM MODE - FIXED PARAMS)")
        print("=" * 100)
        print(f"Max RAM: {MAX_RAM_GB} GB")
        print(f"Days per batch: {DAYS_PER_BATCH}")
        print(f"Parallel workers: {NUM_WORKERS}")
        print(f"Generate all labels at once: {GENERATE_ALL_LABELS_AT_ONCE}")
        print(f"Current RAM: {get_memory_usage_gb():.2f} GB")
        print("=" * 100)
    
    
    def get_columns_to_load(self, sample_file):
        """Determine which columns to load"""
        try:
            dataset = pq.ParquetDataset(sample_file)
            all_columns = dataset.schema.names
        except Exception as e:
            print(f"❌ Error reading schema: {e}")
            raise
        
        columns_to_load = ['Price', 'Time']
        
        if self.feature_prefixes:
            feature_cols = [col for col in all_columns 
                            if any(col.startswith(prefix) for prefix in self.feature_prefixes)]
            columns_to_load.extend(feature_cols)
            self.feature_columns = feature_cols
        else:
            raise ValueError("No features specified!")
        
        columns_to_load = sorted(list(set(columns_to_load)))
        self.feature_columns = sorted(list(set(self.feature_columns)))
        
        print(f"✓ Loading {len(columns_to_load)} columns")
        print(f"  Features: {len(self.feature_columns)}")
        
        return columns_to_load
    
    
    def load_days_batch_parallel(self, day_numbers, columns_to_load):
        """
        Load multiple days in parallel using ProcessPoolExecutor
        Much faster than sequential loading
        """
        print(f"  Loading {len(day_numbers)} days in parallel ({NUM_WORKERS} workers)...")
        
        # Prepare arguments for parallel processing
        args_list = [
            (day_num, os.path.join(DATA_DIR, DAY_FILE_PATTERN.format(day_num)), columns_to_load)
            for day_num in day_numbers
        ]
        
        batch_dfs = []
        
        # Use ProcessPoolExecutor for true parallel I/O
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(load_single_day_worker, args): args[0] 
                       for args in args_list}
            
            for future in as_completed(futures):
                day_num, df = future.result()
                if df is not None:
                    batch_dfs.append(df)
        
        print(f"  Loaded {len(batch_dfs)} days successfully")
        return batch_dfs
    
    
    def generate_all_labels_batch(self, batch_dfs):
        """
        Generate ALL 30 labels for a batch of days at once
        More efficient than generating one label at a time
        """
        print(f"  Generating all 30 labels for batch...")
        
        labeled_dfs = []
        for df in batch_dfs:
            # Generate all labels at once (optimized in label generator)
            df = self.label_gen.generate_labels(df, price_col='Price', time_col='Time')
            labeled_dfs.append(df)
        
        print(f"  Labels generated for {len(labeled_dfs)} days")
        return labeled_dfs
    
    
    def load_and_cache_all_data(self, day_numbers, columns_to_load):
        """
        Load ALL data with ALL labels and cache in memory
        With 40GB RAM, we can afford to keep everything loaded
        """
        print(f"\n{'='*100}")
        print(f"LOADING ALL DATA INTO MEMORY (with all 30 labels)")
        print(f"{'='*100}")
        
        batches = [day_numbers[i:i+DAYS_PER_BATCH] 
                   for i in range(0, len(day_numbers), DAYS_PER_BATCH)]
        
        all_dfs = []
        
        for batch_idx, batch_days in enumerate(batches):
            print(f"\nBatch {batch_idx+1}/{len(batches)}: days {batch_days[0]}-{batch_days[-1]}")
            
            # Load days in parallel
            if PARALLEL_DAY_LOADING:
                batch_dfs = self.load_days_batch_parallel(batch_days, columns_to_load)
            else:
                batch_dfs = self.load_days_sequential(batch_days, columns_to_load)
            
            if len(batch_dfs) == 0:
                continue
            
            # Generate all labels for this batch
            if GENERATE_ALL_LABELS_AT_ONCE:
                labeled_dfs = self.generate_all_labels_batch(batch_dfs)
            else:
                labeled_dfs = batch_dfs
            
            all_dfs.extend(labeled_dfs)
            
            print(f"  RAM usage: {get_memory_usage_gb():.2f} GB")
            
            # Safety check
            if get_memory_usage_gb() > MAX_RAM_GB * 0.9:
                print(f"  ⚠ Approaching RAM limit, stopping early")
                break
        
        # Concatenate all DataFrames into one big DataFrame
        print(f"\nConcatenating {len(all_dfs)} days into single DataFrame...")
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        del all_dfs
        force_cleanup()
        
        print(f"✓ Full dataset loaded: {len(full_df):,} samples")
        print(f"  RAM usage: {get_memory_usage_gb():.2f} GB")
        
        return full_df
    
    
    def load_days_sequential(self, day_numbers, columns_to_load):
        """Fallback: sequential loading"""
        batch_dfs = []
        for day_num in day_numbers:
            filepath = os.path.join(DATA_DIR, DAY_FILE_PATTERN.format(day_num))
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath, columns=columns_to_load)
                df['Day'] = day_num
                batch_dfs.append(df)
        return batch_dfs
    
    
    def extract_label_data(self, df, label_col, is_classification):
        """
        Extract X, y for a specific label from the full DataFrame
        Much faster than loading from disk
        """
        X = df[self.feature_columns].values
        y = df[label_col].values
        
        # Remove invalid samples
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid]
        y = y[valid]
        
        return X, y
    
    
    def apply_undersampling(self, X, y, strategy='not majority'):
        """Apply undersampling"""
        unique, counts = np.unique(y, return_counts=True)
        print(f"      Before: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        undersampler = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"      After: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        return X_resampled, y_resampled
    
    
    # --- REMOVED: quick_grid_search function is no longer needed ---
    
    
    def train_single_label(self, train_df, test_df, label_col, is_classification):
        """
        Train model for single label using pre-loaded DataFrames
        MUCH faster than loading from disk
        """
        print(f"\n{'='*80}")
        print(f"LABEL: {label_col}")
        print(f"{'='*80}")
        
        # Extract training data (instant - already in memory!)
        print(f"  Extracting training data...")
        X_train, y_train = self.extract_label_data(train_df, label_col, is_classification)
        
        if X_train is None or len(X_train) == 0:
            print(f"  ⚠ No training data")
            return None
        
        print(f"  Train samples: {len(X_train):,}")
        
        # Preprocessing
        if is_classification:
            y_train = (y_train.astype(int) + 1).astype(int)
            
            # Check if we have multiple classes
            unique_classes = np.unique(y_train)
            n_classes = len(unique_classes)
            print(f"  Classes found: {unique_classes.tolist()} (n={n_classes})")
            
            if n_classes < 2:
                print(f"  ⚠ SKIPPED: Only {n_classes} class found. Cannot train classification model.")
                return None
            
            if USE_UNDERSAMPLING:
                print(f"  Undersampling...")
                X_train, y_train = self.apply_undersampling(X_train, y_train, UNDERSAMPLING_STRATEGY)
                
                # Re-check class count after undersampling
                unique_classes_after = np.unique(y_train)
                if len(unique_classes_after) < 2:
                    print(f"  ⚠ SKIPPED: Only {len(unique_classes_after)} class after undersampling.")
                    return None
        
        # --- MODIFIED: Set fixed hyperparameters (Grid Search removed) ---
        best_score = 0 # Placeholder, as we're not CV'ing
        if is_classification:
            print("  Using fixed params (LR=0.1, Depth=6, N=150)")
            best_params = {
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 150
            }
        else:
            print("  Using fixed params (Depth=7, LR=0.1, N=200)")
            best_params = {
                'max_depth': 7,
                'learning_rate': 0.1,
                'n_estimators': 200
            }
        
        print(f"  Using params: {best_params}")
        # --- END MODIFICATION ---
        
        # Train final model
        print(f"  Training final model...")
        if is_classification:
            base_params = {**BASE_PARAMS_CLASSIFICATION, **best_params}
            model = xgb.XGBClassifier(**base_params)
        else:
            base_params = {**BASE_PARAMS_REGRESSION, **best_params}
            model = xgb.XGBRegressor(**base_params)
        
        model.fit(X_train, y_train, verbose=False)
        
        # Training metrics
        y_pred_train = model.predict(X_train)
        train_samples = len(X_train)
        
        if is_classification:
            train_acc = accuracy_score(y_train, y_pred_train)
        else:
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        del X_train, y_train, y_pred_train
        force_cleanup()
        
        # Extract test data (instant!)
        print(f"  Extracting test data...")
        X_test, y_test = self.extract_label_data(test_df, label_col, is_classification)
        
        if X_test is None or len(X_test) == 0:
            print(f"  ⚠ No test data")
            return None
        
        if is_classification:
            y_test = (y_test.astype(int) + 1).astype(int)
        
        y_pred_test = model.predict(X_test)
        
        # Results
        results = {
            'label': label_col,
            'type': 'classification' if is_classification else 'regression',
            'train_samples': train_samples,
            'test_samples': len(y_test),
            'best_params': best_params,
            'best_cv_score': best_score # Note: this is just 0 now
        }
        
        if is_classification:
            results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
            results['train_accuracy'] = train_acc
            print(f" ----------------------------------------------------")
            print(f"  Train Acc: {train_acc:.4f}, Test Acc: {results['test_accuracy']:.4f}")
            
            # --- New Classification Details ---
            # Ensure labels=[0, 1, 2] to get a 3x3 matrix even if one class is missing
            cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2])
            
            # Get actual test set counts
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            test_counts = dict(zip(unique_test, counts_test))
            
            total_0 = test_counts.get(0, 0)
            total_1 = test_counts.get(1, 0)
            total_2 = test_counts.get(2, 0)
            
            correct_0 = cm[0, 0]
            correct_1 = cm[1, 1]
            correct_2 = cm[2, 2]

            print(f"  TEST SET DISTRIBUTION & PREDICTIONS:")
            print(f"  Class 0 (Down): {correct_0: >7,} correctly predicted out of {total_0: >7,} total")
            print(f"  Class 1 (Stay): {correct_1: >7,} correctly predicted out of {total_1: >7,} total")
            print(f"  Class 2 (Up)  : {correct_2: >7,} correctly predicted out of {total_2: >7,} total")
            print(f" ----------------------------------------------------")
            
            results['test_confusion_matrix'] = cm.tolist()
            results['test_class_counts'] = {k: int(v) for k, v in test_counts.items()} # Ensure JSON serializable
            # --- End New Details ---
            
        else:
            results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
            results['test_r2'] = r2_score(y_test, y_pred_test)
            results['train_rmse'] = train_rmse
            print(f"  Train RMSE: {train_rmse:.6f}, Test RMSE: {results['test_rmse']:.6f}")
        
        results['feature_importance'] = {feat: float(imp) for feat, imp in zip(self.feature_columns, model.feature_importances_)}
        
        del X_test, y_test, y_pred_test
        force_cleanup()
        
        return results, model
    
    
    def train_all_labels(self, train_df, test_df):
        """Train all labels using pre-loaded data"""
        print("\n" + "=" * 100)
        print("TRAINING ALL LABELS (FAST MODE)")
        print("=" * 100)
        
        label_columns = self.label_gen.get_label_columns()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for idx, label_col in enumerate(label_columns):
            print(f"\n[{idx+1}/{len(label_columns)}] Processing {label_col}")
            
            is_classification = label_col.startswith('TB_')
            result = self.train_single_label(train_df, test_df, label_col, is_classification)
            
            if result is None:
                continue
            
            results, model = result
            self.label_results[label_col] = results
            
            # Save model
            model_file = f"model_{label_col}_{timestamp}.json"
            model.save_model(os.path.join(self.models_dir, model_file))
            
            del model, result, results
            force_cleanup()
            
            print(f"  RAM: {get_memory_usage_gb():.2f} GB")
        
        print("\n" + "=" * 100)
        print("TRAINING COMPLETED")
        print("=" * 100)
    
    
    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f'results_{timestamp}.txt')
        
        # Also save a JSON version for easier parsing
        filename_json = os.path.join(self.output_dir, f'results_{timestamp}.json')
        with open(filename_json, 'w') as f_json:
            json.dump(self.label_results, f_json, indent=4)
        
        with open(filename, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("XGBOOST TRAINING RESULTS (FAST MODE - 40GB - FIXED PARAMS)\n")
            f.write("=" * 100 + "\n\n")
            
            for label_col, results in self.label_results.items():
                f.write(f"\n{label_col}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Type: {results['type'].upper()}\n")
                f.write(f"Best params: {results['best_params']}\n")
                
                if results['type'] == 'classification':
                    f.write(f"Train Accuracy: {results['train_accuracy']:.4f}\n")
                    f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
                    
                    # Add new classification details to text report
                    counts = results.get('test_class_counts', {})
                    cm = results.get('test_confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])
                    total_0 = counts.get('0', 0)
                    total_1 = counts.get('1', 0)
                    total_2 = counts.get('2', 0)
                    correct_0 = cm[0][0]
                    correct_1 = cm[1][1]
                    correct_2 = cm[2][2]
                    
                    f.write(f"\n  TEST SET BREAKDOWN:\n")
                    f.write(f"  Class 0 (Down): {correct_0:,} / {total_0:,}\n")
                    f.write(f"  Class 1 (Stay): {correct_1:,} / {total_1:,}\n")
                    f.write(f"  Class 2 (Up)  : {correct_2:,} / {total_2:,}\n")
                    
                else:
                    f.write(f"Train RMSE: {results['train_rmse']:.6f}\n")
                    f.write(f"Test RMSE: {results['test_rmse']:.6f}\n")
                    f.write(f"Test R²: {results['test_r2']:.4f}\n")
                
                f.write("\nTOP 10 FEATURES:\n")
                sorted_imp = sorted(results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
                    f.write(f"  {rank}. {feat}: {imp:.6f}\n")
                f.write("\n")
        
        print(f"✓ Results saved: {filename}")
        print(f"✓ JSON Results saved: {filename_json}")
        return filename


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print(f"Initial RAM: {get_memory_usage_gb():.2f} GB")
    
    trainer = FastXGBoostTrainer(
        feature_prefixes=FEATURE_PREFIXES,
        output_dir=OUTPUT_DIR,
        models_dir=MODELS_DIR
    )
    
    sample_file = os.path.join(DATA_DIR, DAY_FILE_PATTERN.format(START_DAY))
    columns_to_load = trainer.get_columns_to_load(sample_file)
    
    all_days = list(range(START_DAY, END_DAY + 1))
    n_train = int(len(all_days) * TRAIN_SPLIT)
    train_days = all_days[:n_train]
    test_days = all_days[n_train:]
    
    print(f"\nTrain: {len(train_days)} days, Test: {len(test_days)} days")
    
    # Load ALL data once with ALL labels
    print("\n" + "="*100)
    print("PHASE 1: LOADING ALL DATA")
    print("="*100)
    train_df = trainer.load_and_cache_all_data(train_days, columns_to_load)
    test_df = trainer.load_and_cache_all_data(test_days, columns_to_load)
    
    # Train all models (super fast now!)
    print("\n" + "="*100)
    print("PHASE 2: TRAINING ALL MODELS")
    print("="*100)
    trainer.train_all_labels(train_df, test_df)
    
    trainer.save_results()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"TOTAL TIME: {elapsed/60:.1f} minutes")
    print(f"Final RAM: {get_memory_usage_gb():.2f} GB")
    print(f"{'='*100}")