import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import your existing backtest function
from backtesting import backtest

# NOTE: No need to import generate_labels - we're using pre-trained model!

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_PATH = './xgboost_models/model_TB_t0.30_l3600_20251108_073322.json'  # Update with actual filename
LABEL_NAME = 'TB_t0.30_l3600'

# Feature Configuration (must match training)
FEATURE_PREFIXES = [
    'PB10_', 'PB11_', 'PB6_', 'PB3_', 'PB1_', 'PB2_', 'PB7_', 'PB14_', 'PB15_',
    'BB13_', 'BB14_', 'BB15_', 'BB4_', 'BB5_', 'BB6_', 'BB23', 'VB4_', 'VB5_'
]

# Trading Parameters
THRESHOLD = 0.3  # Take profit/stop loss (absolute price change)
LOOKAHEAD = 3600  # Max position hold time (seconds)
MIN_GAP_BETWEEN_SIGNALS = 15  # Minimum 15s gap between any two signals

# Data Configuration
DATA_DIR = '/data/quant14/EBX'
START_DAY = 223
END_DAY = 278
MIN_BARS = 2400

# Backtest Configuration
INITIAL_CAPITAL = 100000
TRANSACTION_COST_RATE = 0.02  # 0.02% per trade
SLIPPAGE = 0.0
N_WORKERS = None  # None = auto-detect


# ============================================================================
# XGBOOST SIGNAL GENERATOR WITH POSITION MANAGEMENT
# ============================================================================

def generate_signals(day_df, model, feature_cols, threshold=0.30,
                     lookahead=1200, min_gap=15):
    """
    Generate trading signals from XGBoost predictions with full position management
    
    (Function content is unchanged)
    """
    n = len(day_df)
    signals = np.zeros(n, dtype=np.int8)
    
    # Get prices and times (now guaranteed to be numeric)
    prices = day_df['Price'].values
    times = day_df['Time'].values if 'Time' in day_df.columns else np.arange(n)
    
    # Extract features and get XGBoost predictions
    try:
        X = day_df[feature_cols].values
    except KeyError as e:
        print(f"   Warning: Missing features: {e}")
        return signals
    
    # Handle missing values
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() == 0:
        print(f"   Warning: No valid samples after removing NaNs")
        return signals
    
    # Get XGBoost predictions (0=down, 1=neutral, 2=up)
    predictions = np.ones(n, dtype=np.int8)  # Default: neutral
    X_valid = X[valid_mask]
    predictions[valid_mask] = model.predict(X_valid)
    
    # Position tracking
    current_position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_time = 0.0
    last_signal_time = -min_gap  # Allow first signal immediately
    
    for i in range(n):
        current_price = prices[i]
        current_time = times[i]
        
        # Check minimum gap since last signal
        time_since_last_signal = current_time - last_signal_time
        if time_since_last_signal < min_gap:
            continue  # Skip this bar, too soon since last signal
        
        # === EXIT LOGIC (if in position) ===
        if current_position != 0:
            time_in_position = current_time - entry_time
            price_change = current_price - entry_price
            
            should_exit = False
            
            # Check exit conditions (only after min_gap has passed since entry)
            if time_in_position >= min_gap:
                
                if current_position == 1:  # Long position
                    if price_change >= threshold:
                        should_exit = True  # Take profit
                    elif price_change <= -threshold:
                        should_exit = True  # Stop loss
                
                elif current_position == -1:  # Short position
                    if price_change <= -threshold:
                        should_exit = True  # Take profit
                    elif price_change >= threshold:
                        should_exit = True  # Stop loss
                
                if time_in_position >= lookahead:
                    should_exit = True
            
            if i == n - 1:
                should_exit = True
            
            if should_exit:
                if current_position == 1:
                    signals[i] = -1
                elif current_position == -1:
                    signals[i] = 1
                
                current_position = 0
                last_signal_time = current_time
                continue
        
        # === ENTRY LOGIC (if flat) ===
        elif current_position == 0:
            pred = predictions[i]
            
            if pred == 2:  # Model predicts UP
                signals[i] = 1
                current_position = 1
                entry_price = current_price
                entry_time = current_time
                last_signal_time = current_time
            
            elif pred == 0:  # Model predicts DOWN
                signals[i] = -1
                current_position = -1
                entry_price = current_price
                entry_time = current_time
                last_signal_time = current_time
    
    return signals


# ============================================================================
# DAY PROCESSING
# ============================================================================

# ============================================================================
# DAY PROCESSING
# ============================================================================

# ============================================================================
# DAY PROCESSING
# ============================================================================

# ============================================================================
# DAY PROCESSING
# ============================================================================

def process_day(day_num, data_dir, price_col, min_bars, model_path, feature_cols,
                threshold, lookahead, min_gap):
    """
    Load day data and generate signals with position management
    """
    try:
        # --- Load model inside the worker ---
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path)
        except Exception as e:
            print(f"Error loading model in worker for day {day_num}: {e}")
            return None
        # --- End model loading ---

        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        day_df = pd.read_parquet(file_path)
        len_original = len(day_df)

        if price_col not in day_df.columns:
            return None

        # --- Data Cleaning and Type Conversion ---
        timestamp_col = next((c for c in day_df.columns
                            if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        # Start with just the price column as critical
        subset_cols_critical = [price_col] 
        
        if timestamp_col:
            
            # --- *** THE NEW FIX (HH:MM:SS) *** ---
            # 1. Convert "HH:MM:SS" string to Timedelta object
            #    errors='coerce' handles <NA> or malformed strings
            time_deltas = pd.to_timedelta(day_df[timestamp_col], errors='coerce')
            
            # 2. Convert Timedelta to total seconds (e.g., 34200.0)
            #    NaT (Not-a-Time) from step 1 becomes NaN
            time_numeric = time_deltas.dt.total_seconds()
            # --- *** END NEW FIX *** ---
            
            # Check if conversion failed for ALL rows
            if time_numeric.isna().all():
                print(f"DEBUG (Day {day_num}): Time column '{timestamp_col}' is 100% NaN "
                      f"after trying to parse HH:MM:SS. Dropping it.")
                
                day_df = day_df.drop(columns=[timestamp_col])
                
            else:
                # Success! Store the numeric seconds
                day_df[timestamp_col] = time_numeric 
                if timestamp_col != 'Time':
                     day_df = day_df.rename(columns={timestamp_col: 'Time'})
                
                # Add 'Time' to the list of critical columns
                if 'Time' not in subset_cols_critical:
                    subset_cols_critical.append('Time')
        
        # Convert price column (always critical)
        day_df[price_col] = pd.to_numeric(day_df[price_col], errors='coerce')
        # --- END Data Cleaning ---
        
        
        # Only drop rows if they have NaNs in Price or (a valid) Time.
        day_df = day_df.dropna(subset=subset_cols_critical)

        len_after_dropna = len(day_df)

        if len_after_dropna == 0:
            if len_original > 0:
                # This message now means the PRICE column must be bad
                print(f"DEBUG (Day {day_num}): Dataframe empty after dropping NaNs in Price/Time. "
                      f"Original rows={len_original}. Check PRICE or TIME columns for NaNs.")
            return None
            
        # Store the clean prices/timestamps *after* dropna
        clean_prices = day_df[price_col].values
        
        # This will now use the fallback np.arange() for bad-time-files
        if 'Time' in day_df.columns:
            clean_timestamps = day_df['Time'].values
        else:
            clean_timestamps = np.arange(len(day_df))


        # Generate signals using XGBoost with position management
        signals = generate_signals(
            day_df=day_df,
            model=model,
            feature_cols=feature_cols,
            threshold=threshold,
            lookahead=lookahead,
            min_gap=min_gap
        )
        
        if len(signals) == 0:
            return None
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Time': clean_timestamps,
            'Price': clean_prices,
            'Signal': signals,
        })

        return result_df

    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return None

def process_wrapper(args):
    """Wrapper for parallel processing"""
    return process_day(*args)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(
    data_dir=DATA_DIR,
    num_days=None,
    price_col='Price',
    min_bars=MIN_BARS,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_rate=TRANSACTION_COST_RATE,
    slippage=SLIPPAGE,
    n_workers=N_WORKERS,
    # XGBoost specific params
    model_path=MODEL_PATH,
    feature_prefixes=FEATURE_PREFIXES,
    threshold=THRESHOLD,
    lookahead=LOOKAHEAD,
    min_gap=MIN_GAP_BETWEEN_SIGNALS,
    start_day=START_DAY, # <-- Will now be used
    end_day=END_DAY      # <-- Will now be used
):
    
    # --- FIXED: Define 'workers' BEFORE using it ---
    workers = n_workers if n_workers else min(cpu_count(), 16)
    
    print("="*80)
    print("XGBOOST STRATEGY BACKTEST")
    print("="*80)
    print(f"Model: {LABEL_NAME}")
    print(f"Take Profit/Stop Loss: ±{threshold}")
    print(f"Max Hold: {lookahead}s")
    print(f"Min Gap Between Signals: {min_gap}s")
    print(f"Data Directory: {data_dir}")
    # --- MODIFIED: Print the day range ---
    print(f"Processing days: {start_day} to {end_day}")
    print(f"Initial Capital: {initial_capital:,.0f}")
    print(f"Transaction Cost: {transaction_cost_rate}%")
    print(f"Workers: {workers}") # <-- Now this line is safe
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file not found: {model_path}")
        # (rest of error handling...)
        return None
    
    print(f"\nModel path: {model_path}")
    print(f"✓ Model will be loaded by each worker process to avoid deadlocks.")
    
    
    # Find all days to process
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return None
    
    print(f"\n[1/3] Finding all 'day*.parquet' files in {data_dir}...")
    parquet_files = sorted(data_path.glob("day*.parquet"))
    
    if not parquet_files:
        print(f"ERROR: No 'day*.parquet' files found")
        return None

    day_numbers_all = []
    for f in parquet_files:
        try:
            day_num_str = f.stem[3:]
            day_numbers_all.append(int(day_num_str))
        except ValueError:
            print(f"   Warning: Skipping file with invalid name format: {f.name}")
    
    if not day_numbers_all:
         print(f"ERROR: No valid day files found after parsing filenames.")
         return None
    
    print(f"✓ Found {len(day_numbers_all)} total files (Day {min(day_numbers_all)} to {max(day_numbers_all)})")

    # --- *** MODIFIED: Filter day_numbers based on config *** ---
    day_numbers = [d for d in day_numbers_all if start_day <= d <= end_day]

    if not day_numbers:
        print(f"\n❌ ERROR: No day files found within the specified range {start_day}-{end_day}.")
        return None
    
    total_days = len(day_numbers)
    min_day, max_day = min(day_numbers), max(day_numbers) # This is now the range we process
    print(f"✓ Will process {total_days} days (from {min_day} to {max_day})")
    # --- *** END MODIFICATION *** ---


    # --- MODIFIED: Get feature columns from start_day (or first available day in range) ---
    print(f"\nDetecting feature columns...")
    sample_file_path = Path(data_dir) / f'day{start_day}.parquet'
    
    if not sample_file_path.exists():
        print(f"   Warning: Sample file 'day{start_day}.parquet' not found.")
        print(f"   Using first available day in range: 'day{min_day}.parquet'")
        sample_file_path = Path(data_dir) / f'day{min_day}.parquet'
        
        if not sample_file_path.exists():
             print(f"❌ ERROR: First sample file 'day{min_day}.parquet' also not found!")
             return None
    
    sample_df = pd.read_parquet(sample_file_path)
    # --- END MODIFICATION ---

    feature_cols = [col for col in sample_df.columns 
                    if any(col.startswith(prefix) for prefix in feature_prefixes)]
    feature_cols = sorted(feature_cols)
    print(f"✓ Found {len(feature_cols)} features")
    

    # Process all days (now using the filtered day_numbers list)
    print(f"\n[2/3] Processing days with {workers} parallel workers...")
    start_time = time.time()
    
    args_list = [
        (day_num, data_dir, price_col, min_bars, model_path, feature_cols, 
         threshold, lookahead, min_gap)
        for day_num in day_numbers # <-- This is now the filtered list
    ]
    
    with Pool(workers) as pool:
        results_iter = pool.imap(process_wrapper, args_list)
        
        results = list(tqdm(
            results_iter, 
            total=len(args_list),
            desc="   Processing",
            unit="day"
        ))
    
    all_signals = [res for res in results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"   ✓ Completed in {elapsed:.1f}s")
    print(f"   ✓ Processed {len(all_signals)}/{total_days} days")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated")
        return None
    
    # Combine all days
    print(f"\n[3/3] Running backtest...")
    combined_df = pd.concat(all_signals, ignore_index=True)
    
    # Statistics before backtest
    longs = (combined_df['Signal'] == 1).sum()
    shorts = (combined_df['Signal'] == -1).sum()
    total = len(combined_df)
    
    print(f"\nSignal Statistics:")
    print(f"   Long entries:  {longs:,}")
    print(f"   Short entries: {shorts:,}")
    print(f"   Total bars:    {total:,}")
    print(f"   Signal freq:   {((longs + shorts)/total)*100:.3f}%")
    
    # Run backtest using your existing function
    results = backtest(
        df=combined_df,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        slippage=slippage
    )
    
    # Save results
    if results:
        output_dir = Path('./backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        with open(output_dir / f'xgboost_backtest_{timestamp}.txt', 'w') as f:
            f.write("XGBOOST BACKTEST RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Label: {LABEL_NAME}\n")
            f.write(f"Take Profit/Stop Loss: ±{threshold}\n")
            f.write(f"Max Hold: {lookahead}s\n")
            f.write(f"Min Gap Between Signals: {min_gap}s\n")
            # --- MODIFIED: Report the processed range ---
            f.write(f"Days: {min_day}-{max_day} ({total_days} days total in range {start_day}-{end_day})\n")
            f.write(f"\nSignal Statistics:\n")
            f.write(f"   Long entries:  {longs:,}\n")
            f.write(f"   Short entries: {shorts:,}\n")
            f.write(f"   Total bars:    {total:,}\n")
            f.write(f"   Signal freq:   {((longs + shorts)/total)*100:.3f}%\n")
            f.write(f"\nBacktest Results:\n")
            f.write(f"{results}\n")
        
        # Save signals DataFrame
        combined_df.to_csv(output_dir / f'signals_{timestamp}.csv', index=False)
        
        print(f"\n✓ Results saved to {output_dir}/")
        print(f"   - xgboost_backtest_{timestamp}.txt")
        print(f"   - signals_{timestamp}.csv")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Run XGBoost backtest
    results = main(
        data_dir=DATA_DIR,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        slippage=SLIPPAGE
    )