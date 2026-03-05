import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import warnings

# Import from your existing files
from backtesting import backtest

# Try to import Numba for acceleration
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("✓ Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python")
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

# Try to import LightGBM for significance analysis
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("✓ LightGBM detected - for feature significance")
except ImportError:
    LGBM_AVAILABLE = False
    print("✗ LightGBM not found - 'give_significance' will not work.")
    print("  Please install: pip install lightgbm")

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
try:
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings('ignore', category=DataConversionWarning)
except ImportError:
    pass

# ============================================================================
# STRATEGY FUNCTION - MODIFY THIS FOR YOUR STRATEGY
# ============================================================================

@jit(nopython=True, fastmath=True)
def fast_signals(prices, fast_span, slow_span):
    """
    Numba-accelerated example strategy.
    This must be a separate function for @jit to work.
    """
    n = len(prices)
    signals = np.zeros(n)
    
    # Simple EMA calculation (Numba-compatible)
    fast_ma = np.zeros(n)
    slow_ma = np.zeros(n)
    alpha_fast = 2.0 / (fast_span + 1.0)
    alpha_slow = 2.0 / (slow_span + 1.0)
    
    fast_ma[0] = prices[0]
    slow_ma[0] = prices[0]
    
    for i in range(1, n):
        fast_ma[i] = alpha_fast * prices[i] + (1.0 - alpha_fast) * fast_ma[i-1]
        slow_ma[i] = alpha_slow * prices[i] + (1.0 - alpha_slow) * slow_ma[i-1]
        
        if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
            signals[i] = 1  # Long signal
        elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
            signals[i] = -1 # Short signal
            
    # Ensure last signal is 0 to meet 'IntraDay Flat Requirement'
    signals[n-1] = 0
    return signals

def generate_signals(day_df):
    """
    This is the strategy wrapper.
    !! REPLACE THIS LOGIC WITH YOUR OWN !!
    This is just a placeholder so the backtest runs.
    """
    if 'Price' not in day_df.columns:
        return np.array([])
        
    # Example: Use the Numba-accelerated MA crossover
    price_values = day_df['Price'].values
    signals = fast_signals(price_values, fast_span=10, slow_span=30)
    
    # ---
    # Example using other features (from your significance analysis)
    # ---
    # if 'feature_123' in day_df.columns and 'feature_45' in day_df.columns:
    #    feature1 = day_df['feature_123'].values
    #    feature2 = day_df['feature_45'].values
    #
    #    signals = np.zeros(len(day_df))
    #    signals[feature1 > 0.5] = 1
    #    signals[feature2 < -0.2] = -1
    #    signals[len(signals)-1] = 0 # Square off
    
    return signals

# ============================================================================
# ANALYSIS FUNCTION - FOR DOCUMENTATION & JUSTIFICATION
# ============================================================================

def give_significance(day_df, price_col='Price', lookahead=10):
    """
    Calculates feature importance by predicting future returns.
    This is for ANALYSIS and JUSTIFICATION, not for generating signals.
    
    Returns:
        A pandas DataFrame with 'feature' and 'importance' columns.
    """
    if not LGBM_AVAILABLE:
        return pd.DataFrame(columns=['feature', 'importance'])

    try:
        # 1. Define the Outcome (Y) - The "Effect"
        # We want to predict the future price change
        y = day_df[price_col].shift(-lookahead) - day_df[price_col]
        
        # 2. Define the Features (X) - The "Causes"
        # We drop Price and any timestamp columns
        timestamp_col = next((c for c in day_df.columns 
                            if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        features_to_drop = [price_col]
        if timestamp_col:
            features_to_drop.append(timestamp_col)
            
        X = day_df.drop(columns=features_to_drop)
        
        # 3. Align X and y
        # We combine them and drop NaNs created by the shift
        data = X.copy()
        data['target'] = y
        
        # Drop rows with NaN target (end of day) or NaN features
        data = data.dropna()
        
        if len(data) < 100: # Not enough data to train
            return pd.DataFrame(columns=['feature', 'importance'])

        y_final = data['target']
        X_final = data.drop(columns=['target'])
        
        # 4. Train a fast model
        # n_jobs=1 is crucial when running inside multiprocessing Pool
        model = lgb.LGBMRegressor(
            n_estimators=100, 
            n_jobs=1, 
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_final, y_final)

        # 5. Extract and return importances
        importance_df = pd.DataFrame({
            'feature': X_final.columns,
            'importance': model.feature_importances_
        })
        
        return importance_df
    
    except Exception as e:
        # print(f"Significance error: {e}")
        return pd.DataFrame(columns=['feature', 'importance']) # Return empty on error

# ============================================================================
# DATA PROCESSING (Modified)
# ============================================================================

def process_day(day_num, data_dir, price_col, min_bars):
    """Load day data, generate signals, and calculate significance"""
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None, None # Return tuple

        day_df = pd.read_parquet(file_path)

        if price_col not in day_df.columns:
            return None, None

        # Handle timestamps
        timestamp_col = next((c for c in day_df.columns 
                            if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col:
            original_timestamps = day_df[timestamp_col].values
        else:
            original_timestamps = np.arange(len(day_df))

        # Drop NaNs *before* passing to strategy and significance functions
        day_df = day_df.dropna()

        if len(day_df) < min_bars:
            return None, None

        # --- MODIFICATION ---
        # 1. Calculate Feature Significance (for analysis)
        # We pass a copy to be safe
        importance_df = give_significance(day_df.copy(), price_col=price_col)
        
        # 2. GENERATE TRADING SIGNALS (for backtest)
        signals = generate_signals(day_df)
        # --- END MODIFICATION ---
        
        if len(signals) == 0:
            return None, None
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Time': original_timestamps[original_timestamps.size - len(signals):], # Align timestamps
            'Price': day_df[price_col].values,
            'Signal': signals,
        })
        
        # Return both results
        return result_df, importance_df

    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        return None, None


def process_wrapper(args):
    """Wrapper for parallel processing"""
    return process_day(*args)


# ============================================================================
# MAIN EXECUTION (Modified)
# ============================================================================

def main(
    data_dir='/data/quant14/EBY',  # Change to EBX or EBY
    num_days=279,
    price_col='Price',
    min_bars=2400,
    initial_capital=100000,
    transaction_cost_rate=0.02,
    slippage=0.0,
    n_workers=None
):
    
    print("="*80)
    print("STRATEGY BACKTEST & SIGNIFICANCE ANALYSIS")
    print("="*80)
    print(f"Data Directory: {data_dir}")
    print(f"Processing {num_days} days")
    print(f"Initial Capital: {initial_capital:,.0f}")
    print(f"Transaction Cost: {transaction_cost_rate}%")
    workers = n_workers if n_workers else min(cpu_count(), 16)
    print(f"Workers: {workers}")
    print("="*80)
    
    # Check data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return None
    
    parquet_files = list(data_path.glob("day*.parquet"))
    print(f"\n[1/4] Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No day*.parquet files found")
        return None
    
    # Process all days
    print(f"\n[2/4] Processing days with {workers} parallel workers...")
    start_time = time.time()
    
    args_list = [(day_num, data_dir, price_col, min_bars) for day_num in range(num_days)]
    
    with Pool(workers) as pool:
        results = pool.map(process_wrapper, args_list)
    
    # --- MODIFICATION ---
    # Unpack the results
    all_signals = [res[0] for res in results if res is not None and res[0] is not None]
    all_importances = [res[1] for res in results if res is not None and res[1] is not None]
    # --- END MODIFICATION ---
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f}s")
    print(f"  ✓ Processed {len(all_signals)}/{num_days} days for backtesting")
    print(f"  ✓ Calculated significance for {len(all_importances)}/{num_days} days")
    
    if len(all_signals) == 0:
        print("\nERROR: No signals generated")
        return None
    
    # --- NEW SECTION: AGGREGATE AND SHOW SIGNIFICANCE ---
    print(f"\n[3/4] Aggregating Feature Significance...")
    if len(all_importances) > 0:
        combined_importance = pd.concat(all_importances)
        
        # Average the importance scores across all days
        final_importance = combined_importance.groupby('feature')['importance'].mean()
        final_importance = final_importance.sort_values(ascending=False)
        
        print("\n--- CAUSAL-INSPIRED FEATURE IMPORTANCE ---")
        print(f"(Based on predicting {10}-second forward returns)") # lookahead=10
        print(final_importance.head(10).to_string())
        print("...")
        print(f"  ✓ Total features analyzed: {len(final_importance)}")
        
        # This is your justification!
        print("\nACTION: Use this list to justify your strategy in your documentation.")
        
    else:
        print("  ✗ No feature significance data was calculated.")
    # --- END NEW SECTION ---
    
    # Combine all days for backtest
    print(f"\n[4/4] Running backtest...")
    combined_df = pd.concat(all_signals, ignore_index=True)
    
    # Statistics before backtest
    longs = (combined_df['Signal'] == 1).sum()
    shorts = (combined_df['Signal'] == -1).sum()
    total = len(combined_df)
    
    print(f"\nSignal Statistics:")
    print(f"  Long entries:  {longs:,}")
    print(f"  Short entries: {shorts:,}")
    print(f"  Total bars:    {total:,}")
    print(f"  Signal freq:   {((longs + shorts)/total)*100:.3f}%")
    
    # Run backtest using imported function
    results = backtest(
        df=combined_df,
        initial_capital=initial_capital,
        transaction_cost_rate=transaction_cost_rate,
        slippage=slippage
    )
    
    return results, final_importance.head(10) # Return results and top features


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # CHANGE DATA DIRECTORY HERE TO SWITCH BETWEEN EBX/EBY
    
    # Option 1: EBY
    results, top_features = main(
        data_dir='/data/quant14/EBX',
        num_days=10, # Set to 10 for a quick test
        initial_capital=1000,
        transaction_cost_rate=0.02,
        slippage=0.0
    )
    
    if results:
        print("\n--- Backtest Results ---")
        print(results)
        
    # Option 2: EBX (uncomment to use)
    # results, top_features = main(
    #     data_dir='/data/quant14/EBX',
    #     num_days=279, # Use all days
    #     initial_capital=100000,
    #     transaction_cost_rate=0.02,
    #     slippage=0.0
    # )
    