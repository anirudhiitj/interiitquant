import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import warnings
warnings.filterwarnings('ignore')

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

# ============================================================================
# STRATEGY FUNCTION - MODIFY THIS FOR YOUR STRATEGY
# ============================================================================

def generate_signals(day_df):

    n = len(day_df)
    signals = np.zeros(n, dtype=np.int8)
    
    # Example: Access any feature from day_df
    prices = day_df['Price'].values
    
    # ADD YOUR STRATEGY LOGIC HERE
    # Example: Simple price momentum
    for i in range(1, n):
        if prices[i] > prices[i-1]:
            signals[i] = 1  # Long
        elif prices[i] < prices[i-1]:
            signals[i] = -1  # Short
    
    return signals


# ============================================================================
# DATA PROCESSING (using your existing structure)
# ============================================================================

def process_day(day_num, data_dir, price_col, min_bars):
    """Load day data and generate signals"""
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            return None

        day_df = pd.read_parquet(file_path)

        if price_col not in day_df.columns:
            return None

        # Handle timestamps
        timestamp_col = next((c for c in day_df.columns 
                            if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col:
            original_timestamps = day_df[timestamp_col].values
        else:
            original_timestamps = np.arange(len(day_df))

        day_df = day_df.dropna()

        if len(day_df) < min_bars:
            return None

        # GENERATE SIGNALS - Pass entire DataFrame
        signals = generate_signals(day_df)
        
        if len(signals) == 0:
            return None
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'Time': original_timestamps[:len(signals)],
            'Price': day_df[price_col].values,
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
    print("STRATEGY BACKTEST FRAMEWORK")
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
    print(f"\n[1/3] Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print(f"ERROR: No day*.parquet files found")
        return None
    
    # Process all days
    print(f"\n[2/3] Processing days with {workers} parallel workers...")
    start_time = time.time()
    
    args_list = [(day_num, data_dir, price_col, min_bars) for day_num in range(num_days)]
    
    with Pool(workers) as pool:
        results = pool.map(process_wrapper, args_list)
    
    all_signals = [res for res in results if res is not None]
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f}s")
    print(f"  ✓ Processed {len(all_signals)}/{num_days} days")
    
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
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # CHANGE DATA DIRECTORY HERE TO SWITCH BETWEEN EBX/EBY
    
    # Option 1: EBY
    results = main(
        data_dir='/data/quant14/EBX',
        num_days=10,
        initial_capital=1000,
        transaction_cost_rate=0.02,
        slippage=0.0
    )
    
    # Option 2: EBX (uncomment to use)
    # results = main(
    #     data_dir='/data/quant14/EBX',
    #     num_days=279,
    #     initial_capital=100000,
    #     transaction_cost_rate=0.02,
    #     slippage=0.0
    # )
