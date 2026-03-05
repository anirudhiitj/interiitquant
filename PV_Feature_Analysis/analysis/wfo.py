import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
from numba import jit
import itertools

# ============================================================================
# IMPORT REFERENCE FUNCTIONS
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_directional_cusum(changes, drift=0.02, decay=0.9):
    """Calculate directional CUSUM with decay (from reference_cusum.py)"""
    n = len(changes)
    cusum_up = 0.0
    cusum_down = 0.0
    cusum_up_values = np.zeros(n, dtype=np.float64)
    cusum_down_values = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = changes[i]
        if change > drift:
            cusum_up += change
            cusum_down = max(0.0, cusum_down * decay)
        elif change < -drift:
            cusum_down += abs(change)
            cusum_up = max(0.0, cusum_up * decay)
        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)
        
        cusum_up_values[i] = cusum_up
        cusum_down_values[i] = cusum_down
    
    return cusum_up_values, cusum_down_values


def classify_regime_by_difference(cusum_up, cusum_down, 
                                  choppy_up_thresh, choppy_down_thresh,
                                  trending_up_thresh, trending_down_thresh):
    """Classify regime based on CUSUM difference (from reference_cusum.py)"""
    n = len(cusum_up)
    cusum_diff = cusum_up - cusum_down
    regimes = np.full(n, 'Transition', dtype='<U15')
    
    is_trending_up = cusum_diff > trending_up_thresh
    is_trending_down = cusum_diff < trending_down_thresh
    regimes[is_trending_up] = 'Trending Up'
    regimes[is_trending_down] = 'Trending Down'
    
    is_choppy = (cusum_diff <= choppy_up_thresh) & (cusum_diff >= choppy_down_thresh)
    is_not_trending = ~is_trending_up & ~is_trending_down
    regimes[is_choppy & is_not_trending] = 'Choppy'
    
    regime_map = {'Trending Up': 1, 'Trending Down': -1, 'Choppy': 0, 'Transition': 0}
    regime_int = np.array([regime_map[r] for r in regimes], dtype=np.int8)
    
    return regime_int, cusum_diff


@jit(nopython=True, fastmath=True)
def find_price_jump_pairs(prices, jump_threshold):
    """Find pairs where price moves >= jump_threshold (from reference_profitable_pairs.py)"""
    n = len(prices)
    pairs = []
    i = 0
    while i < n:
        start_price = prices[i]
        for j in range(i + 1, n):
            price_diff = abs(prices[j] - start_price)
            if price_diff >= jump_threshold:
                pairs.append((i, j))
                i = j + 1
                break
        else:
            i += 1
    return pairs


# ============================================================================
# COMPONENT 1: GROUND TRUTH GENERATOR
# ============================================================================

def get_ideal_trades_for_day(prices):
    """
    Generate ideal profitable trades for the entire day
    Returns: List of (start_index, end_index) tuples
    """
    ideal_trades = find_price_jump_pairs(prices, 0.3)
    return ideal_trades


# ============================================================================
# COMPONENT 2: STRATEGY SIMULATION ENGINE
# ============================================================================

@jit(nopython=True, fastmath=True)
def run_strategy_simulation(prices, timestamps, cusum_decay, trending_up_thresh, trending_down_thresh):
    """
    Simulate the strategy on a chunk of data with given parameters
    
    Returns: 2D numpy array where each row is [entry_index, exit_index, pnl_with_costs]
    """
    n = len(prices)
    
    # Calculate CUSUM regime signals
    changes = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        changes[i] = prices[i] - prices[i-1]
    
    cusum_up, cusum_down = calculate_directional_cusum(changes, drift=0.005, decay=cusum_decay)
    cusum_diff = cusum_up - cusum_down
    
    # Simple regime classification using raw threshold values from grid search
    regime_signals = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if cusum_diff[i] > trending_up_thresh:
            regime_signals[i] = 1
        elif cusum_diff[i] < trending_down_thresh:
            regime_signals[i] = -1
        else:
            regime_signals[i] = 0
    
    # State machine - use arrays to collect trades
    max_trades = n // 2
    trade_entries = np.zeros(max_trades, dtype=np.int32)
    trade_exits = np.zeros(max_trades, dtype=np.int32)
    trade_pnls = np.zeros(max_trades, dtype=np.float64)
    trade_count = 0
    
    state = 0  # 0=flat, 1=long, -1=short
    entry_idx = 0
    entry_time = 0.0
    entry_price = 0.0
    peak_value = 0.0
    
    TRANSACTION_COST = 0.001  # 10 bps
    TIME_STOP_SECONDS = 15.0
    
    for i in range(n):
        current_time = timestamps[i]
        current_price = prices[i]
        
        if state == 0:  # Flat - looking for entry
            if regime_signals[i] == 1:  # Enter long
                state = 1
                entry_idx = i
                entry_time = current_time
                entry_price = current_price
                peak_value = current_price
            elif regime_signals[i] == -1:  # Enter short
                state = -1
                entry_idx = i
                entry_time = current_time
                entry_price = current_price
                peak_value = current_price
        
        elif state == 1:  # In long position
            # Update peak
            if current_price > peak_value:
                peak_value = current_price
            
            # Check exit conditions
            trailing_stop_hit = current_price <= 0.9 * peak_value
            time_stop_hit = (current_time - entry_time) > TIME_STOP_SECONDS
            
            if trailing_stop_hit or time_stop_hit:
                # Exit long
                exit_price = current_price
                raw_pnl = exit_price - entry_price
                pnl_with_costs = raw_pnl - (TRANSACTION_COST * entry_price) - (TRANSACTION_COST * exit_price)
                
                trade_entries[trade_count] = entry_idx
                trade_exits[trade_count] = i
                trade_pnls[trade_count] = pnl_with_costs
                trade_count += 1
                state = 0
        
        elif state == -1:  # In short position
            # Update peak (lowest for short)
            if current_price < peak_value:
                peak_value = current_price
            
            # Check exit conditions
            # Inside run_strategy_simulation, in the `elif state == -1:` block
            trailing_stop_hit = current_price >= 1.1 * peak_value
            time_stop_hit = (current_time - entry_time) > TIME_STOP_SECONDS
            
            if trailing_stop_hit or time_stop_hit:
                # Exit short
                exit_price = current_price
                raw_pnl = entry_price - exit_price
                pnl_with_costs = raw_pnl - (TRANSACTION_COST * entry_price) - (TRANSACTION_COST * exit_price)
                
                trade_entries[trade_count] = entry_idx
                trade_exits[trade_count] = i
                trade_pnls[trade_count] = pnl_with_costs
                trade_count += 1
                state = 0
    
    # Force exit at end if still in position
    if state != 0:
        exit_price = prices[n-1]
        if state == 1:
            raw_pnl = exit_price - entry_price
        else:
            raw_pnl = entry_price - exit_price
        pnl_with_costs = raw_pnl - (TRANSACTION_COST * entry_price) - (TRANSACTION_COST * exit_price)
        
        trade_entries[trade_count] = entry_idx
        trade_exits[trade_count] = n - 1
        trade_pnls[trade_count] = pnl_with_costs
        trade_count += 1
    
    # Return only the filled portion as a 2D array
    if trade_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    
    trades_array = np.zeros((trade_count, 3), dtype=np.float64)
    for i in range(trade_count):
        trades_array[i, 0] = trade_entries[i]
        trades_array[i, 1] = trade_exits[i]
        trades_array[i, 2] = trade_pnls[i]
    
    return trades_array


# ============================================================================
# COMPONENT 3: CUSTOM OBJECTIVE FUNCTION
# ============================================================================

def calculate_objective_score(simulated_trades, ideal_trades_in_chunk, prices):
    """
    Calculate custom objective score with first-trade penalty and missed opportunity penalty
    
    Args:
        simulated_trades: 2D numpy array where each row is [entry_idx, exit_idx, pnl]
        ideal_trades_in_chunk: list of (start_idx, end_idx) tuples
        prices: numpy array of prices
    """
    # A. Calculate Strategy Score (First-Trade Penalty Rule)
    strategy_score = 0.0
    
    if len(simulated_trades) > 0:
        pnl_list = simulated_trades[:, 2]
        
        # First trade penalty/reward
        pnl_1 = pnl_list[0]
        if pnl_1 < 0:
            strategy_score = pnl_1
        else:
            strategy_score = pnl_1
        
        # Subsequent trades (only add profits)
        for i in range(1, len(pnl_list)):
            if pnl_list[i] > 0:
                strategy_score += pnl_list[i]
    
    # B. Calculate Missed Opportunity Penalty
    missed_penalty = 0.0
    TRANSACTION_COST = 0.001
    
    for ideal_trade in ideal_trades_in_chunk:
        ideal_start, ideal_end = ideal_trade
        
        # Check if any profitable simulated trade overlaps
        has_overlap = False
        for i in range(len(simulated_trades)):
            sim_start = int(simulated_trades[i, 0])
            sim_end = int(simulated_trades[i, 1])
            sim_pnl = simulated_trades[i, 2]
            
            if sim_pnl > 0:
                # Check for overlap
                if not (sim_end < ideal_start or sim_start > ideal_end):
                    has_overlap = True
                    break
        
        if not has_overlap:
            # Calculate missed profit
            ideal_profit = abs(prices[ideal_end] - prices[ideal_start])
            cost_deduction = TRANSACTION_COST * (prices[ideal_start] + prices[ideal_end])
            missed_profit = ideal_profit - cost_deduction
            missed_penalty += missed_profit
    
    final_score = strategy_score - missed_penalty
    return final_score


# ============================================================================
# PARAMETER GRID
# ============================================================================

CUSUM_DECAY_GRID = [0.7, 0.8, 0.9, 0.95, 0.98]
TREND_UP_THRESH_GRID = [0.02, 0.03, 0.04, 0.05, 0.06]
TREND_DOWN_THRESH_GRID = [-0.02, -0.03, -0.04, -0.05, -0.06]


# ============================================================================
# PARALLELIZATION WRAPPER
# ============================================================================

def run_one_param_set(args):
    """
    Wrapper for parallel execution
    Returns: (score, params)
    """
    params, train_prices, train_timestamps, ideal_trades_in_chunk = args
    decay, up_thresh, down_thresh = params
    
    # Run simulation
    simulated_trades = run_strategy_simulation(
        train_prices, train_timestamps, decay, up_thresh, down_thresh
    )
    
    # Calculate score
    score = calculate_objective_score(simulated_trades, ideal_trades_in_chunk, train_prices)
    
    return (score, params)


# ============================================================================
# MAIN WFO PROCESS
# ============================================================================

def main():
    # Configuration
    DATA_DIR = '/data/quant14/EBY'
    NUM_DAYS = 279
    CHUNK_SIZE = 512
    
    print("="*80)
    print("ROLLING INTRADAY WALK-FORWARD OPTIMIZATION")
    print("="*80)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Number of Days: {NUM_DAYS}")
    print(f"Chunk Size: {CHUNK_SIZE} bars")
    print(f"Parameter Grid:")
    print(f"  CUSUM Decay: {CUSUM_DECAY_GRID}")
    print(f"  Trend Up Threshold: {TREND_UP_THRESH_GRID}")
    print(f"  Trend Down Threshold: {TREND_DOWN_THRESH_GRID}")
    print(f"Total parameter combinations: {len(CUSUM_DECAY_GRID) * len(TREND_UP_THRESH_GRID) * len(TREND_DOWN_THRESH_GRID)}")
    print("="*80)
    
    # Initialize tracking
    all_daily_profits = []
    all_best_params_log = []
    
    # Outer Loop: Iterate over days
    for day_num in range(NUM_DAYS):
        print(f"\n[Day {day_num}/{NUM_DAYS-1}] Processing...")
        
        # Load day data
        file_path = Path(DATA_DIR) / f'day{day_num}.parquet'
        if not file_path.exists():
            print(f"  ⚠ File not found, skipping")
            all_daily_profits.append(0.0)
            continue
        
        try:
            df = pd.read_parquet(file_path)
            
            # Extract prices and timestamps
            if 'Price' not in df.columns:
                print(f"  ⚠ Price column not found, skipping")
                all_daily_profits.append(0.0)
                continue
            
            prices = df['Price'].values.astype(np.float64)
            
            # Handle timestamps
            timestamp_col = next((c for c in df.columns if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
            if timestamp_col:
                try:
                    dt_series = pd.to_datetime(df[timestamp_col], format='mixed', errors='coerce')
                    timestamps = (dt_series.dt.hour * 3600 + 
                                dt_series.dt.minute * 60 + 
                                dt_series.dt.second + 
                                dt_series.dt.microsecond / 1e6).values.astype(np.float64)
                except:
                    timestamps = np.arange(len(prices), dtype=np.float64)
            else:
                timestamps = np.arange(len(prices), dtype=np.float64)
            
        except Exception as e:
            print(f"  ⚠ Error loading data: {e}")
            all_daily_profits.append(0.0)
            continue
        
        # Get ideal trades for the day
        all_ideal_trades_for_day = get_ideal_trades_for_day(prices)
        
        # Calculate number of chunks
        num_bars = len(prices)
        num_chunks = num_bars // CHUNK_SIZE
        
        if num_chunks < 2:
            print(f"  ⚠ Not enough data for WFO (need at least 2 chunks), skipping")
            all_daily_profits.append(0.0)
            continue
        
        print(f"  Total bars: {num_bars}, Chunks: {num_chunks}")
        print(f"  Ideal trades for day: {len(all_ideal_trades_for_day)}")
        
        Total_Daily_Profit = 0.0
        total_train_trades = 0
        total_predict_trades = 0
        chunks_with_trades = 0
        
        # Inner Loop: Rolling chunks
        for chunk_idx in range(num_chunks - 1):
            # A. TRAIN STEP (Optimize on Chunk i)
            train_start = chunk_idx * CHUNK_SIZE
            train_end = (chunk_idx + 1) * CHUNK_SIZE
            
            train_prices = prices[train_start:train_end]
            train_timestamps = timestamps[train_start:train_end]
            
            # Filter ideal trades for this chunk
            ideal_trades_in_chunk = [
                (start - train_start, end - train_start) 
                for start, end in all_ideal_trades_for_day
                if train_start <= start < train_end and train_start <= end < train_end
            ]
            
            if chunk_idx == 0:
                print(f"    Chunk 0 - Train range: [{train_start}:{train_end}]")
                print(f"    Ideal trades in chunk: {len(ideal_trades_in_chunk)}")
                if len(ideal_trades_in_chunk) > 0:
                    print(f"    First ideal trade: {ideal_trades_in_chunk[0]}")
                print(f"    Price range: [{train_prices.min():.2f}, {train_prices.max():.2f}]")
            
            # Create parameter combinations
            param_combinations = list(itertools.product(
                CUSUM_DECAY_GRID, TREND_UP_THRESH_GRID, TREND_DOWN_THRESH_GRID
            ))
            
            # Prepare arguments for parallel processing
            args_list = [
                (params, train_prices, train_timestamps, ideal_trades_in_chunk)
                for params in param_combinations
            ]
            
            # Parallel grid search
            with Pool(processes=min(cpu_count(), 16)) as pool:
                results = pool.map(run_one_param_set, args_list)
            
            # Find best parameters
            best_score, best_params = max(results, key=lambda x: x[0])
            
            # Count trades in training
            decay, up_thresh, down_thresh = best_params
            train_trades = run_strategy_simulation(
                train_prices, train_timestamps, decay, up_thresh, down_thresh
            )
            total_train_trades += len(train_trades)
            
            if chunk_idx == 0:
                print(f"    Best params: decay={decay}, up={up_thresh:.1f}, down={down_thresh:.1f}")
                print(f"    Best score: {best_score:.4f}")
                print(f"    Trades in training: {len(train_trades)}")
                if len(train_trades) > 0:
                    print(f"    First 3 train trades PnL: {train_trades[:min(3, len(train_trades)), 2]}")
                
                # Debug CUSUM values
                changes_debug = np.zeros(len(train_prices))
                for i in range(1, len(train_prices)):
                    changes_debug[i] = train_prices[i] - train_prices[i-1]
                cusum_up_debug, cusum_down_debug = calculate_directional_cusum(changes_debug, drift=0.005, decay=decay)
                cusum_diff_debug = cusum_up_debug - cusum_down_debug
                print(f"    CUSUM diff range: [{cusum_diff_debug.min():.4f}, {cusum_diff_debug.max():.4f}], mean: {cusum_diff_debug.mean():.4f}, std: {cusum_diff_debug.std():.4f}")
            
            # Log best parameters
            all_best_params_log.append({
                'day': day_num,
                'chunk_index': chunk_idx,
                'best_score': best_score,
                'best_params': best_params
            })
            
            # B. PREDICT STEP (Trade on Chunk i+1)
            predict_start = (chunk_idx + 1) * CHUNK_SIZE
            predict_end = (chunk_idx + 2) * CHUNK_SIZE
            
            if predict_end > num_bars:
                predict_end = num_bars
            
            predict_prices = prices[predict_start:predict_end]
            predict_timestamps = timestamps[predict_start:predict_end]
            
            # Run simulation with best params
            simulated_trades = run_strategy_simulation(
                predict_prices, predict_timestamps, decay, up_thresh, down_thresh
            )
            
            total_predict_trades += len(simulated_trades)
            
            # Calculate standard P&L
            if len(simulated_trades) > 0:
                chunk_pnl = np.sum(simulated_trades[:, 2])
                chunks_with_trades += 1
            else:
                chunk_pnl = 0.0
            Total_Daily_Profit += chunk_pnl
            
            if chunk_idx == 0:
                print(f"    Trades in prediction: {len(simulated_trades)}")
                if len(simulated_trades) > 0:
                    print(f"    First 3 predict trades PnL: {simulated_trades[:min(3, len(simulated_trades)), 2]}")
                print(f"    Chunk PnL: {chunk_pnl:.4f}")
            
            # Debug occasional chunks
            if chunk_idx in [10, 20, 30, 40] and len(simulated_trades) > 0:
                print(f"    Chunk {chunk_idx}: {len(simulated_trades)} trades, PnL: {chunk_pnl:.4f}")
        
        # Store daily profit
        all_daily_profits.append(Total_Daily_Profit)
        print(f"  ✓ Day {day_num} Complete | Profit: {Total_Daily_Profit:.4f} | Train trades: {total_train_trades} | Predict trades: {total_predict_trades} | Chunks with trades: {chunks_with_trades}/{num_chunks-1}")
    
    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # 1. Save Parameter Log
    params_df = pd.DataFrame(all_best_params_log)
    params_df.to_csv('best_params_log.csv', index=False)
    print(f"\n✓ Parameter log saved: best_params_log.csv ({len(params_df)} records)")
    
    # 2. Financial Summary
    total_profit = sum(all_daily_profits)
    avg_daily_profit = np.mean(all_daily_profits)
    std_daily_profit = np.std(all_daily_profits)
    
    # Sharpe Ratio
    if std_daily_profit > 0:
        sharpe_ratio = avg_daily_profit / std_daily_profit
    else:
        sharpe_ratio = 0.0
    
    print("\n" + "="*80)
    print("FINANCIAL SUMMARY")
    print("="*80)
    print(f"Total Profit: {total_profit:.4f}")
    print(f"Average Daily Profit: {avg_daily_profit:.4f}")
    print(f"Std Dev of Daily Profits: {std_daily_profit:.4f}")
    print(f"Sharpe Ratio (Daily): {sharpe_ratio:.4f}")
    print(f"\nNumber of Trading Days: {len(all_daily_profits)}")
    print(f"Profitable Days: {sum(1 for p in all_daily_profits if p > 0)}")
    print(f"Losing Days: {sum(1 for p in all_daily_profits if p < 0)}")
    print("="*80)
    
    # Save daily profits
    daily_profits_df = pd.DataFrame({
        'day': range(len(all_daily_profits)),
        'profit': all_daily_profits
    })
    daily_profits_df.to_csv('daily_profits.csv', index=False)
    print(f"\n✓ Daily profits saved: daily_profits.csv")
    
    print("\n" + "="*80)
    print("ALL OUTPUTS SAVED SUCCESSFULLY")
    print("="*80)


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed/60:.2f} minutes")