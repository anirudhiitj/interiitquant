"""
ENHANCED Mean Reversion Strategy - Strategy 1 (Adaptive Version)

Key Improvements:
1. Adaptive ATR threshold based on daily volatility regime
2. Dynamic BB thresholds based on recent price action
3. Better entry/exit logic with profit targets
4. Trade count optimization (10-20 trades per day)
5. Enhanced risk management
"""

import pandas as pd
import numpy as np
import numba as nb
from numba import jit, prange
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': '/data/quant14/EBX/',
    'output_file': 'test_signals.csv',
    'initial_capital': 100,
    'transaction_cost': 0.002,  # 0.2%
    'min_hold_time': 15,
    'min_cooldown': 15,
    'max_workers': 25,
    
    # Adaptive Strategy Parameters
    'atr_window': 20,
    'atr_multiplier': 1.5,  # Adaptive threshold = mean_atr * multiplier
    
    # BB Parameters (using T3 lookback)
    'bb_lookback': 'T3',
    'bb_window': 50,  # For adaptive bands
    
    # Entry/Exit Thresholds (adaptive)
    'entry_z_threshold': 1.5,  # Z-score for entry
    'exit_z_threshold': 0.5,   # Z-score for exit
    'profit_target': 0.005,     # 0.5% profit target (>2x transaction cost)
    'stop_loss': 0.003,         # 0.3% stop loss
    
    # Trade Management
    'max_trades_per_day': 20,
    'min_trades_per_day': 8,
    'trade_size': 1,  # Position size (-1, 0, +1)
    
    # Volume Filter
    'volume_filter': True,
    'volume_percentile': 55,
}

# ============================================================================
# NUMBA OPTIMIZED CORE FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_atr_fast(prices, window=20):
    """Fast ATR calculation"""
    n = len(prices)
    atr = np.zeros(n)
    
    if n < window:
        return atr
    
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = abs(prices[i] - prices[i-1])
    
    # Rolling average
    for i in range(window, n):
        atr[i] = np.mean(tr[i-window+1:i+1])
    
    return atr

@jit(nopython=True, cache=True)
def calculate_rolling_stats(prices, window=50):
    """Calculate rolling mean and std"""
    n = len(prices)
    means = np.zeros(n)
    stds = np.zeros(n)
    
    for i in range(window, n):
        window_data = prices[i-window+1:i+1]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
    
    return means, stds

@jit(nopython=True, cache=True)
def calculate_bb_zscore(price, bb_lower, bb_middle, bb_upper):
    """Calculate standardized position within BB"""
    band_width = bb_upper - bb_lower
    if band_width < 1e-8:
        return 0.0
    
    # Distance from middle in units of half-band-width
    position = (price - bb_middle) / (band_width / 2.0)
    return position

@jit(nopython=True, cache=True)
def calculate_profit_loss(entry_price, current_price, position):
    """Calculate P&L percentage"""
    if position == 0:
        return 0.0
    
    if position == 1:  # Long
        return (current_price - entry_price) / entry_price
    else:  # Short
        return (entry_price - current_price) / entry_price

@jit(nopython=True, cache=True)
def generate_signals_adaptive(times, prices, atr, bb_lower, bb_middle, bb_upper,
                              volume, volume_threshold, config_dict):
    """
    Adaptive signal generation with profit targets and risk management
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    # Extract config
    min_hold = config_dict['min_hold_time']
    min_cooldown = config_dict['min_cooldown']
    entry_z = config_dict['entry_z_threshold']
    exit_z = config_dict['exit_z_threshold']
    profit_target = config_dict['profit_target']
    stop_loss = config_dict['stop_loss']
    volume_filter = config_dict['volume_filter']
    max_trades = config_dict['max_trades_per_day']
    
    # State variables
    current_position = 0
    entry_price = 0.0
    position_entry_time = 0.0
    last_signal_time = 0.0
    trade_count = 0
    
    # Calculate adaptive ATR threshold
    atr_mean = np.mean(atr[100:]) if n > 100 else np.mean(atr)
    atr_threshold = atr_mean * config_dict['atr_multiplier']
    
    for i in range(100, n):
        current_time = times[i]
        current_price = prices[i]
        
        # Check trade limit
        if trade_count >= max_trades:
            signals[i] = 0
            if current_position != 0:
                current_position = 0
            continue
        
        # Cooldown check
        if current_time - last_signal_time < min_cooldown:
            signals[i] = current_position
            continue
        
        # Minimum hold check
        if current_position != 0:
            hold_time = current_time - position_entry_time
            if hold_time < min_hold:
                signals[i] = current_position
                continue
        
        # Volatility filter
        if atr[i] < atr_threshold:
            if current_position != 0:
                # Exit in low volatility
                signals[i] = 0
                current_position = 0
                last_signal_time = current_time
                trade_count += 1
            else:
                signals[i] = 0
            continue
        
        # Volume filter
        if volume_filter and volume[i] < volume_threshold:
            signals[i] = current_position
            continue
        
        # Calculate BB z-score
        bb_z = calculate_bb_zscore(current_price, bb_lower[i], bb_middle[i], bb_upper[i])
        
        # === POSITION MANAGEMENT ===
        if current_position != 0:
            # Calculate P&L
            pnl = calculate_profit_loss(entry_price, current_price, current_position)
            
            # Profit target hit
            if pnl >= profit_target:
                signals[i] = 0
                current_position = 0
                last_signal_time = current_time
                trade_count += 1
                continue
            
            # Stop loss hit
            if pnl <= -stop_loss:
                signals[i] = 0
                current_position = 0
                last_signal_time = current_time
                trade_count += 1
                continue
            
            # Mean reversion exit
            if current_position == 1:  # Long
                if bb_z > -exit_z:  # Reverted toward mean
                    signals[i] = 0
                    current_position = 0
                    last_signal_time = current_time
                    trade_count += 1
                else:
                    signals[i] = 1
                    
            elif current_position == -1:  # Short
                if bb_z < exit_z:  # Reverted toward mean
                    signals[i] = 0
                    current_position = 0
                    last_signal_time = current_time
                    trade_count += 1
                else:
                    signals[i] = -1
        
        # === ENTRY LOGIC ===
        else:
            # Long entry: Oversold condition
            if bb_z < -entry_z:
                signals[i] = 1
                current_position = 1
                entry_price = current_price
                position_entry_time = current_time
                last_signal_time = current_time
                trade_count += 1
            
            # Short entry: Overbought condition
            elif bb_z > entry_z:
                signals[i] = -1
                current_position = -1
                entry_price = current_price
                position_entry_time = current_time
                last_signal_time = current_time
                trade_count += 1
            
            else:
                signals[i] = 0
    
    return signals, trade_count

@jit(nopython=True, cache=True)
def force_eod_squareoff(signals, times, eod_time):
    """Force square off before EOD"""
    n = len(signals)
    squareoff_time = eod_time - 60
    
    for i in range(n-1, -1, -1):
        if times[i] <= squareoff_time:
            break
        signals[i] = 0
    
    return signals

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_prepare_data(day_file):
    """Load and prepare day data with all required features"""
    try:
        df = pd.read_parquet(day_file)
        
        # Check required columns
        required = ['Time', 'Price', 'PB9_T1']
        if not all(col in df.columns for col in required):
            return None
        
        # Find BB columns
        bb_lookback = CONFIG['bb_lookback']
        bb_cols = []
        for i in [4, 5, 6]:
            col_name = f'BB{i}_{bb_lookback}'
            if col_name in df.columns:
                bb_cols.append(col_name)
        
        if len(bb_cols) < 3:
            # Fallback to any BB columns
            bb_cols = [c for c in df.columns if c.startswith('BB')][:3]
        
        if len(bb_cols) < 3:
            return None
        
        # Sort to ensure lower, middle, upper order
        bb_cols = sorted(bb_cols)
        
        # Volume
        volume_col = None
        for v in ['V', 'Volume', 'V_T1']:
            if v in df.columns:
                volume_col = v
                break
        
        return {
            'df': df,
            'bb_lower': bb_cols[0],
            'bb_middle': bb_cols[1],
            'bb_upper': bb_cols[2],
            'volume_col': volume_col
        }
        
    except Exception as e:
        print(f"Error loading {day_file}: {e}")
        return None

def process_day(day_file):
    """Process single day with enhanced strategy"""
    day_num = int(Path(day_file).stem.split('_')[-1])
    
    data = load_and_prepare_data(day_file)
    if data is None:
        return None
    
    df = data['df']
    
    # Extract features
    times = df['Time'].values
    prices = df['Price'].values
    pb9_t1 = df['PB9_T1'].values
    
    bb_lower = df[data['bb_lower']].values
    bb_middle = df[data['bb_middle']].values
    bb_upper = df[data['bb_upper']].values
    
    # Volume
    if data['volume_col']:
        volume = df[data['volume_col']].values
        volume_threshold = np.percentile(volume, CONFIG['volume_percentile'])
    else:
        volume = np.ones(len(df))
        volume_threshold = 0.0
    
    # Calculate ATR from PB9_T1 (previous second price)
    atr = calculate_atr_fast(pb9_t1, CONFIG['atr_window'])
    
    # Config dict for numba
    config_dict = {
        'min_hold_time': CONFIG['min_hold_time'],
        'min_cooldown': CONFIG['min_cooldown'],
        'entry_z_threshold': CONFIG['entry_z_threshold'],
        'exit_z_threshold': CONFIG['exit_z_threshold'],
        'profit_target': CONFIG['profit_target'],
        'stop_loss': CONFIG['stop_loss'],
        'atr_multiplier': CONFIG['atr_multiplier'],
        'volume_filter': CONFIG['volume_filter'],
        'max_trades_per_day': CONFIG['max_trades_per_day']
    }
    
    # Generate signals
    signals, trade_count = generate_signals_adaptive(
        times, prices, atr, bb_lower, bb_middle, bb_upper,
        volume, volume_threshold, config_dict
    )
    
    # EOD square off
    eod_time = times[-1]
    signals = force_eod_squareoff(signals, times, eod_time)
    
    # Extract signal transitions
    transitions = []
    prev_signal = 0
    
    for i in range(len(signals)):
        if signals[i] != prev_signal:
            transitions.append({
                'Time': times[i],
                'Signal': int(signals[i]),
                'Price': prices[i]
            })
            prev_signal = signals[i]
    
    if transitions:
        result = pd.DataFrame(transitions)
        result['Day'] = day_num
        return result
    
    return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute strategy with parallel processing"""
    print("=" * 80)
    print("ENHANCED MEAN REVERSION STRATEGY - Strategy 1 (Adaptive)")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        if key != 'data_dir':
            print(f"  {key:.<40} {value}")
    print()
    
    # Get day files
    data_dir = Path(CONFIG['data_dir'])
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"Please ensure data is available at: {data_dir}")
        return
    
    day_files = sorted(data_dir.glob('day_*.parquet'))
    if not day_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(day_files)} days to process")
    print(f"Using {CONFIG['max_workers']} parallel workers\n")
    
    # Parallel processing
    all_results = []
    
    print("Processing days...")
    with ProcessPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {executor.submit(process_day, str(f)): f for f in day_files}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == len(day_files):
                print(f"  Progress: {completed}/{len(day_files)} days")
            
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\nProcessing complete!")
    print(f"  Total days: {len(day_files)}")
    print(f"  Days with signals: {len(all_results)}")
    
    if not all_results:
        print("\nWARNING: No signals generated!")
        return
    
    # Combine and sort
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values(['Day', 'Time']).reset_index(drop=True)
    
    # Keep only required columns
    output_df = final_df[['Time', 'Signal', 'Price']].copy()
    
    # Save output
    output_path = CONFIG['output_file']
    output_df.to_csv(output_path, index=False)
    
    # Statistics
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total signal transitions: {len(output_df)}")
    print(f"\nSignal distribution:")
    signal_counts = output_df['Signal'].value_counts().sort_index()
    for sig, count in signal_counts.items():
        sig_name = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}.get(sig, sig)
        print(f"  {sig_name:.<15} {count:>8} ({count/len(output_df)*100:.1f}%)")
    
    # Estimate trades per day
    trades_per_day = final_df.groupby('Day').size()
    print(f"\nTrades per day:")
    print(f"  Average: {trades_per_day.mean():.1f}")
    print(f"  Median: {trades_per_day.median():.1f}")
    print(f"  Min: {trades_per_day.min()}")
    print(f"  Max: {trades_per_day.max()}")
    
    print(f"\nOutput saved to: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()