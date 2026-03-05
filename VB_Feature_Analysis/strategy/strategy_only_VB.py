import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# Configuration
DATA_FOLDER = '/data/quant14/EBY/'
PLOT_DIR = 'daily_strategy_plots_EBY/'
WEIGHTS_OUTPUT_DIR = 'strategy_weights_EBY/'
TIME_COLUMN = 'Time'
PRICE_COLUMN = 'Price'

# ONLY VB4 AND VB5 FEATURES (Best performers)
FEATURES = ['VB5_T12', 'VB4_T12', 'VB4_T11']

# OPTIMIZED PARAMETERS
INITIAL_CAPITAL = 100000
MIN_TRADE_DURATION = 15
STOP_LOSS_PCT = 0.003  # 0.3%
PROFIT_TARGET_PCT = 0.006  # 0.6%
TRANSACTION_COST_PCT = 0.0001

Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
Path(WEIGHTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


@jit(nopython=True)
def calculate_rolling_stats(arr, window):
    """Calculate rolling mean and std"""
    n = len(arr)
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    
    for i in range(window, n):
        window_data = arr[i-window:i]
        rolling_mean[i] = np.mean(window_data)
        rolling_std[i] = np.std(window_data)
    
    return rolling_mean, rolling_std


@jit(nopython=True)
def calculate_ema(arr, window):
    """Calculate exponential moving average"""
    n = len(arr)
    ema = np.full(n, np.nan)
    alpha = 2.0 / (window + 1.0)
    
    # Initialize with first valid value
    first_valid = 0
    for i in range(n):
        if not np.isnan(arr[i]):
            ema[i] = arr[i]
            first_valid = i
            break
    
    # Calculate EMA
    for i in range(first_valid + 1, n):
        if not np.isnan(arr[i]):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
        else:
            ema[i] = ema[i-1]
    
    return ema


@jit(nopython=True)
def vb4_vb5_strategy_signals(vb5_t12, vb4_t12, vb4_t11, prices):
    """
    Optimized strategy using only VB4 and VB5 features.
    
    Key Insights:
    - VB4 (T11, T12): Tracks volatility bursts accurately, low in calm zones
    - VB5 (T12): Slight lag but smooth and stable
    
    Strategy:
    1. VB4 for fast burst detection (entry triggers)
    2. VB5 for trend confirmation (filter false signals)
    3. VB4_T11 for regime context (longer timeframe)
    """
    n = len(prices)
    signals = np.zeros(n)
    
    # === FAST BURST DETECTOR: VB4_T12 ===
    # Use short window (5 min) to catch bursts quickly
    window_fast = 300
    vb4_t12_mean, vb4_t12_std = calculate_rolling_stats(vb4_t12, window_fast)
    
    # === SMOOTH TREND: VB5_T12 ===
    # Use medium window (10 min) for stable signals
    window_med = 600
    vb5_t12_mean, vb5_t12_std = calculate_rolling_stats(vb5_t12, window_med)
    
    # === REGIME CONTEXT: VB4_T11 ===
    # Use longer window (15 min) for regime
    window_long = 900
    vb4_t11_mean, vb4_t11_std = calculate_rolling_stats(vb4_t11, window_long)
    
    # Calculate price momentum at multiple timeframes
    for i in range(window_long, n):
        # Skip if no valid data
        if (vb4_t12_std[i] <= 0 or vb5_t12_std[i] <= 0 or 
            vb4_t11_std[i] <= 0):
            continue
        
        # === CALCULATE Z-SCORES ===
        z_vb4_t12 = (vb4_t12[i] - vb4_t12_mean[i]) / vb4_t12_std[i]
        z_vb5_t12 = (vb5_t12[i] - vb5_t12_mean[i]) / vb5_t12_std[i]
        z_vb4_t11 = (vb4_t11[i] - vb4_t11_mean[i]) / vb4_t11_std[i]
        
        # === PRICE MOMENTUM ===
        mom_2min = 0
        mom_5min = 0
        if i >= 120:
            mom_2min = (prices[i] - prices[i-120]) / prices[i-120]
        if i >= 300:
            mom_5min = (prices[i] - prices[i-300]) / prices[i-300]
        
        # === STRATEGY LOGIC ===
        # Use VB4 as primary trigger (fast bursts)
        # Use VB5 as confirmation (stable trend)
        # Use VB4_T11 as regime filter
        
        # LONG CONDITIONS
        # Path 1: VB4 burst detected + VB5 confirms + positive momentum
        if (z_vb4_t12 > 1.0 and z_vb5_t12 > 0.7 and 
            z_vb4_t11 > 0.5 and mom_2min > 0.0001):
            signals[i] = 1
        
        # Path 2: Strong VB4 burst + VB5 alignment (even without momentum)
        elif z_vb4_t12 > 1.5 and z_vb5_t12 > 0.8:
            if z_vb4_t11 > 0.3:
                signals[i] = 1
        
        # Path 3: Moderate burst + strong positive momentum
        elif (z_vb4_t12 > 0.8 and z_vb5_t12 > 0.5 and 
              mom_2min > 0.0003 and mom_5min > 0):
            signals[i] = 1
        
        # Path 4: Lower threshold during favorable regime
        elif z_vb4_t11 > 1.0:  # Strong regime
            if z_vb4_t12 > 0.6 and z_vb5_t12 > 0.4 and mom_2min > 0.0001:
                signals[i] = 1
        
        # SHORT CONDITIONS
        # Path 1: VB4 burst + VB5 confirms + negative momentum
        elif (z_vb4_t12 > 1.0 and z_vb5_t12 > 0.7 and 
              z_vb4_t11 > 0.5 and mom_2min < -0.0001):
            signals[i] = -1
        
        # Path 2: Strong burst + VB5 alignment
        elif z_vb4_t12 > 1.5 and z_vb5_t12 > 0.8:
            if z_vb4_t11 > 0.3:
                signals[i] = -1
        
        # Path 3: Moderate burst + strong negative momentum
        elif (z_vb4_t12 > 0.8 and z_vb5_t12 > 0.5 and 
              mom_2min < -0.0003 and mom_5min < 0):
            signals[i] = -1
        
        # Path 4: Lower threshold during favorable regime
        elif z_vb4_t11 > 1.0:
            if z_vb4_t12 > 0.6 and z_vb5_t12 > 0.4 and mom_2min < -0.0001:
                signals[i] = -1
        
        # EXIT: All features normalize
        elif (z_vb4_t12 < 0.3 and z_vb5_t12 < 0.3 and 
              z_vb4_t11 < 0.4):
            signals[i] = 0
    
    return signals


@jit(nopython=True)
def manage_positions_optimized(signals, prices, timestamps, min_duration,
                               stop_pct, target_pct):
    """
    Optimized position management with trailing stop.
    """
    n = len(signals)
    positions = np.zeros(n)
    
    current_pos = 0
    entry_price = np.nan
    entry_time = np.nan
    highest_price = 0.0
    lowest_price = 999999.0
    
    for i in range(1, n):
        time_held = timestamps[i] - entry_time if not np.isnan(entry_time) else 0
        
        # === ENTRY ===
        if current_pos == 0:
            if signals[i] == 1:
                current_pos = 1
                entry_price = prices[i]
                entry_time = timestamps[i]
                highest_price = prices[i]
            elif signals[i] == -1:
                current_pos = -1
                entry_price = prices[i]
                entry_time = timestamps[i]
                lowest_price = prices[i]
        
        # === EXIT ===
        elif current_pos != 0:
            current_return = (prices[i] - entry_price) / entry_price
            
            # Update trailing levels
            if current_pos == 1:
                highest_price = max(highest_price, prices[i])
                trailing_stop_level = highest_price * (1 - stop_pct * 1.5)
            else:
                lowest_price = min(lowest_price, prices[i])
                trailing_stop_level = lowest_price * (1 + stop_pct * 1.5)
            
            exit_trade = False
            
            # LONG exits
            if current_pos == 1:
                # Profit target
                if current_return >= target_pct:
                    exit_trade = True
                # Stop loss
                elif current_return <= -stop_pct:
                    exit_trade = True
                # Trailing stop (after 0.3% profit)
                elif current_return >= 0.003 and prices[i] < trailing_stop_level:
                    exit_trade = True
                # Max hold (15 minutes)
                elif time_held >= 900:
                    exit_trade = True
                # Signal exit
                elif signals[i] == 0 and time_held >= min_duration:
                    exit_trade = True
            
            # SHORT exits
            elif current_pos == -1:
                # Profit target
                if current_return <= -target_pct:
                    exit_trade = True
                # Stop loss
                elif current_return >= stop_pct:
                    exit_trade = True
                # Trailing stop
                elif current_return <= -0.003 and prices[i] > trailing_stop_level:
                    exit_trade = True
                # Max hold
                elif time_held >= 900:
                    exit_trade = True
                # Signal exit
                elif signals[i] == 0 and time_held >= min_duration:
                    exit_trade = True
            
            if exit_trade:
                current_pos = 0
                entry_price = np.nan
                entry_time = np.nan
                highest_price = 0.0
                lowest_price = 999999.0
        
        positions[i] = current_pos
    
    # EOD square-off
    positions[-1] = 0
    
    return positions


@jit(nopython=True)
def calculate_performance_metrics(positions, prices):
    """Calculate performance metrics"""
    n = len(positions)
    
    trades = 0
    wins = 0
    total_pnl = 0.0
    win_pnl = 0.0
    loss_pnl = 0.0
    
    entry_price = 0.0
    entry_pos = 0
    
    for i in range(1, n):
        # Entry
        if positions[i-1] == 0 and positions[i] != 0:
            entry_price = prices[i]
            entry_pos = positions[i]
            trades += 1
        
        # Exit
        elif positions[i-1] != 0 and positions[i] == 0:
            exit_price = prices[i]
            trade_return = (exit_price - entry_price) / entry_price * entry_pos
            trade_pnl = trade_return * 500  # $500 per trade
            
            total_pnl += trade_pnl
            
            if trade_pnl > 0:
                wins += 1
                win_pnl += trade_pnl
            else:
                loss_pnl += abs(trade_pnl)
    
    win_rate = wins / trades if trades > 0 else 0
    profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 0
    expectancy = total_pnl / trades if trades > 0 else 0
    
    return total_pnl, trades, wins, win_rate, profit_factor, expectancy


def process_day(day_num):
    """Process single day"""
    filename = f"day{day_num}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return None
    
    print(f"Processing {filename}...", end=' ')
    
    # Load data
    df = pd.read_csv(filepath)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    df['timestamp_sec'] = (df[TIME_COLUMN] - df[TIME_COLUMN].iloc[0]).dt.total_seconds()
    
    # Check features
    for feat in FEATURES:
        if feat not in df.columns:
            print(f"Missing {feat}")
            return None
        df[feat] = df[feat].fillna(method='ffill').fillna(method='bfill')
    
    prices = df[PRICE_COLUMN].values
    timestamps = df['timestamp_sec'].values
    
    # Generate signals
    signals = vb4_vb5_strategy_signals(
        df['VB5_T12'].values,
        df['VB4_T12'].values,
        df['VB4_T11'].values,
        prices
    )
    
    # Shift to prevent forward bias
    signals = np.roll(signals, 1)
    signals[0] = 0
    
    # Manage positions
    positions = manage_positions_optimized(
        signals, prices, timestamps,
        MIN_TRADE_DURATION, STOP_LOSS_PCT, PROFIT_TARGET_PCT
    )
    
    # Calculate metrics
    pnl, trades, wins, win_rate, pf, expectancy = calculate_performance_metrics(
        positions, prices
    )
    
    position_changes = np.diff(positions, prepend=0)
    long_entries = np.sum(position_changes == 1)
    short_entries = np.sum(position_changes == -1)
    
    print(f"Trades: {trades}, Long: {long_entries}, Short: {short_entries}, "
          f"WR: {win_rate:.1%}, P&L: ${pnl:.2f}")
    
    # Save signals for backtester (DIFF format)
    signals_df = pd.DataFrame({
        'Time': df[TIME_COLUMN],
        'Signal': position_changes,  # Use diff for backtester
        'Price': prices
    })
    
    # Create plot
    plot_day(df, positions, day_num, trades, pnl, win_rate, pf)
    
    gc.collect()
    
    return signals_df, (trades, wins, pnl, win_rate, pf, expectancy)


def plot_day(df, positions, day_num, trades, pnl, win_rate, pf):
    """Create visualization"""
    try:
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # Downsample if needed
        if len(df) > 10000:
            sample_rate = len(df) // 10000 + 1
            df_plot = df.iloc[::sample_rate].copy()
            positions_plot = positions[::sample_rate]
        else:
            df_plot = df.copy()
            positions_plot = positions
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        
        # Plot price
        ax1.plot(df_plot[TIME_COLUMN], df_plot[PRICE_COLUMN], 
                'b-', alpha=0.3, linewidth=0.5)
        
        # Mark entries
        position_changes = np.diff(positions_plot, prepend=0)
        long_idx = np.where(position_changes == 1)[0]
        short_idx = np.where(position_changes == -1)[0]
        
        if len(long_idx) > 0:
            ax1.scatter(df_plot[TIME_COLUMN].iloc[long_idx],
                       df_plot[PRICE_COLUMN].iloc[long_idx],
                       c='green', marker='^', s=50, alpha=0.6, label='Long')
        
        if len(short_idx) > 0:
            ax1.scatter(df_plot[TIME_COLUMN].iloc[short_idx],
                       df_plot[PRICE_COLUMN].iloc[short_idx],
                       c='red', marker='v', s=50, alpha=0.6, label='Short')
        
        pnl_color = 'green' if pnl > 0 else 'red'
        ax1.set_title(f"Day {day_num} - VB4/VB5 Strategy | {trades} Trades | "
                     f"WR: {win_rate:.1%} | P&L: ${pnl:.2f} | PF: {pf:.2f}",
                     fontsize=14, fontweight='bold', color=pnl_color)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot positions
        ax2.plot(df_plot[TIME_COLUMN], positions_plot, 'k-', drawstyle='steps-post')
        ax2.fill_between(df_plot[TIME_COLUMN], 0, positions_plot, 
                        where=positions_plot > 0, facecolor='green', 
                        alpha=0.3, step='post')
        ax2.fill_between(df_plot[TIME_COLUMN], 0, positions_plot,
                        where=positions_plot < 0, facecolor='red',
                        alpha=0.3, step='post')
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_yticks([-1, 0, 1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'day{day_num}.png'), dpi=100)
        plt.close('all')
        
        gc.collect()
    except Exception as e:
        print(f"Plot error: {e}")
        plt.close('all')


def main():
    """Main execution"""
    print("="*80)
    print("VB4/VB5 OPTIMIZED STRATEGY")
    print("="*80)
    print("Features: VB5_T12 (smooth trend), VB4_T12 (fast bursts), VB4_T11 (regime)")
    print("="*80)
    
    all_signals = []
    all_stats = []
    
    for day in range(510):
        try:
            result = process_day(day)
            if result is not None:
                signals_df, stats = result
                all_signals.append(signals_df)
                all_stats.append(stats)
        except Exception as e:
            print(f"Error day {day}: {e}")
            gc.collect()
            continue
    
    if not all_signals:
        print("No data processed")
        return
    
    # Combine signals
    portfolio = pd.concat(all_signals, ignore_index=True)
    portfolio.to_csv('test_signals.csv', index=False)
    print(f"\n✓ Signals saved to test_signals.csv")
    
    # Calculate overall stats
    total_trades = sum(s[0] for s in all_stats)
    total_wins = sum(s[1] for s in all_stats)
    total_pnl = sum(s[2] for s in all_stats)
    
    daily_pnls = [s[2] for s in all_stats]
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252) 
              if np.std(daily_pnls) > 0 else 0)
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Days Processed: {len(all_stats)}")
    print(f"Total Trades: {total_trades:,}")
    print(f"Avg Trades/Day: {total_trades/len(all_stats):.1f}")
    print(f"Win Rate: {total_wins/total_trades:.1%}" if total_trades > 0 else "N/A")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Avg P&L/Day: ${total_pnl/len(all_stats):.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*80)


if __name__ == "__main__":
    main()