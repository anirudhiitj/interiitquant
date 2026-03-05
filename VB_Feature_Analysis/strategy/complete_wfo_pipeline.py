"""
Complete Walk-Forward Optimization Pipeline
============================================

This script:
1. Optimizes strategy parameters (grid search)
2. Generates test_signal.csv with optimized parameters
3. Outputs portfolio_weights.csv for analysis

Usage:
    python complete_wfo_pipeline.py --mode quick     # Fast optimization (grid search)
    python complete_wfo_pipeline.py --mode skip      # Skip optimization, use current params

Output:
    - test_signal.csv (Time, Signal, Price) → Input to backtester
    - portfolio_weights.csv (Time, Position, Price) → For analysis
    - optimized_params.json → Best parameters found
"""

import pandas as pd
import numpy as np
import os
import argparse
import json
from datetime import datetime
import sys # Import sys for exiting on error

# ===== ROBUST STRATEGY IMPORT =====
# This block will find the correct strategy class, whether you
# named it 'CompliantBalancedStrategy' or kept 'BalancedHybridStrategy'.

try:
    from balanced_hybrid_strategy import CompliantBalancedStrategy
    STRATEGY_CLASS = CompliantBalancedStrategy
    print("✅ Successfully imported 'CompliantBalancedStrategy'")
except ImportError:
    print("INFO: 'CompliantBalancedStrategy' not found. Trying 'BalancedHybridStrategy'...")
    try:
        from balanced_hybrid_strategy import BalancedHybridStrategy
        STRATEGY_CLASS = BalancedHybridStrategy
        print("⚠ Successfully imported 'BalancedHybridStrategy'. (Ensure it meets constraints)")
    except ImportError:
        print("="*80)
        print("❌ FATAL IMPORT ERROR: Could not find EITHER 'CompliantBalancedStrategy'")
        print("  or 'BalancedHybridStrategy' in your 'balanced_hybrid_strategy.py' file.")
        print("  Please check the class name inside that file.")
        print("="*80)
        sys.exit(1)
# ==================================


# NOTE: You must create this function or move it here
# This is a copy from the optuna script for standalone use
def calculate_day_stats(positions, prices):
    """Quick statistics calculation for a single day"""
    long_entries, short_entries, pnl, wins, losses = 0, 0, 0, 0, 0
    entry_price, entry_pos = 0, 0
    
    for i in range(1, len(positions)):
        if abs(positions[i-1]) < 0.01 and abs(positions[i]) > 0.01:
            entry_price, entry_pos = prices[i], positions[i]
            if positions[i] > 0: long_entries += 1
            else: short_entries += 1
        elif abs(positions[i-1]) > 0.01 and abs(positions[i]) < 0.01:
            if entry_price == 0: continue
            exit_price = prices[i]
            direction = 1 if entry_pos > 0 else -1
            trade_pnl_pct = (exit_price - entry_price) / entry_price * direction
            trade_pnl = trade_pnl_pct * abs(entry_pos) # Weighted PnL
            pnl += trade_pnl
            if trade_pnl > 0: wins += 1
            else: losses += 1
            entry_price = 0
    
    trades = wins + losses
    win_rate = wins / trades if trades > 0 else 0
    
    return {
        'trades': trades, 'long': long_entries, 'short': short_entries,
        'wins': wins, 'win_rate': win_rate, 'pnl': pnl * 1000 # Scale for consistency
    }


def optimize_quick(strategy_class, data_folder, days_range=range(0, 90)):
    """
    Quick optimization - top 5 high-impact parameters (GRID SEARCH)
    """
    print("\n" + "="*80)
    print("QUICK OPTIMIZATION (GRID SEARCH) - Top 5 Parameters")
    print("="*80)
    
    param_grid = {
        'base_threshold_medium': [45, 50, 55, 60],
        'stop_loss_medium': [0.0015, 0.0020, 0.0025, 0.0030],
        'take_profit_medium': [0.0030, 0.0035, 0.0040, 0.0045],
        'vb_extreme_percentile': [88, 90, 92],
        'strong_multiplier': [1.3, 1.4, 1.5],
    }
    
    from itertools import product
    
    best_score = -999999
    best_params = {}
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total_combos = np.prod([len(v) for v in values])
    
    print(f"Testing {total_combos} parameter combinations...")
    
    combo_num = 0
    for combo in product(*values):
        combo_num += 1
        params = dict(zip(keys, combo))
        
        # Validate: TP > SL
        if params['take_profit_medium'] <= params['stop_loss_medium'] * 1.3:
            continue
        
        # Create strategy with these params
        strategy = strategy_class()
        for k, v in params.items():
            setattr(strategy, k, v)
        
        # Backtest on training days
        stats_list = []
        for day in days_range:
            try:
                result = strategy.process_day(day, data_folder)
                if result is not None:
                    weights_df, positions = result
                    stats = calculate_day_stats(positions, weights_df['Price'].values) # Use new stats
                    stats_list.append(stats)
            except:
                continue
        
        if len(stats_list) < len(days_range) * 0.5:  # Need at least 50% valid days
            continue
        
        # Calculate fitness score
        total_trades = sum(s['trades'] for s in stats_list)
        total_long = sum(s['long'] for s in stats_list)
        total_short = sum(s['short'] for s in stats_list)
        
        if total_trades == 0:
            continue
        
        trades_per_day = total_trades / len(stats_list)
        long_ratio = total_long / total_trades
        
        # Hard constraints
        if trades_per_day < 20:
            continue
        if not (0.30 <= long_ratio <= 0.70):
            continue
        
        avg_wr = np.mean([s['win_rate'] for s in stats_list if s['trades'] > 0])
        if avg_wr < 0.48:
            continue
        
        # Fitness score
        total_pnl = sum(s['pnl'] for s in stats_list)
        daily_pnls = [s['pnl'] for s in stats_list]
        sharpe = np.mean(daily_pnls) / (np.std(daily_pnls) + 1e-8) * np.sqrt(252)
        
        score = (
            0.40 * (total_pnl / len(stats_list)) +  # Daily PnL
            0.30 * sharpe +                          # Sharpe
            0.20 * avg_wr * 100 +                    # Win rate
            0.10 * min(trades_per_day, 50)           # Trade frequency
        )
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            
            print(f"\n✓ New Best! [{combo_num}/{total_combos}]")
            print(f"  Score: {score:.2f}")
            print(f"  Trades/Day: {trades_per_day:.1f} (L:{total_long} S:{total_short})")
            print(f"  Win Rate: {avg_wr:.1%}")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Avg Daily PnL (Metric): {total_pnl/len(stats_list):.2f}")
            print(f"  Params: {params}")
        
        if combo_num % 20 == 0:
            print(f"  Progress: {combo_num}/{total_combos}...")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print(f"Best Score: {best_score:.2f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*80 + "\n")
    
    return best_params


# ... (Your optimize_tactical_wfo function would go here)
# ... It also needs to be updated to use calculate_day_stats
# ... This function was not in the optuna script, so I am omitting it
# ... for brevity, but you MUST update it if you use it.


def generate_final_signals(strategy_class, params, data_folder, total_days=510):
    """
    Generate test_signal.csv and portfolio_weights.csv using optimized parameters
    """
    print("\n" + "="*80)
    print("GENERATING FINAL SIGNALS (test_signal.csv & portfolio_weights.csv)")
    print("="*80)
    
    print(f"\nUsing parameters:")
    if params:
        for k, v in params.items():
            print(f"  {k}: {v}")
    else:
        print("  (Using default strategy parameters)")
    
    # Create strategy with optimized parameters
    strategy = strategy_class()
    if params:
        for k, v in params.items():
            if hasattr(strategy, k):
                setattr(strategy, k, v)
    
    print(f"\nProcessing {total_days} days...")
    
    all_signals_df = []   # For test_signal.csv
    all_positions_df = [] # For portfolio_weights.csv
    all_stats = []
    
    for day in range(total_days):
        try:
            result = strategy.process_day(day, data_folder)
            if result is not None:
                weights_df, positions = result # (Time, Signal, Price), (Positions)
                
                # For test_signal.csv (Raw Signals)
                all_signals_df.append(weights_df)
                
                # For portfolio_weights.csv (Managed Positions)
                position_df = weights_df.copy()
                position_df['Position'] = positions.astype(np.float32)
                all_positions_df.append(position_df[['Time', 'Position', 'Price']])
                
                # For final stats
                stats = calculate_day_stats(positions, weights_df['Price'].values)
                all_stats.append(stats)
            
            if day > 0 and day % 50 == 0:
                print(f"  Processed day {day}...")
                
        except Exception as e:
            print(f"  Day {day}: Error - {e}")
            continue
    
    if len(all_signals_df) == 0:
        print("\n✗ ERROR: No signals generated!")
        return False
    
    # Save test_signal.csv
    test_signals = pd.concat(all_signals_df, ignore_index=True)
    test_signals.to_csv('test_signal.csv', index=False)
    
    # Save portfolio_weights.csv
    portfolio = pd.concat(all_positions_df, ignore_index=True)
    portfolio.to_csv('portfolio_weights.csv', index=False)
    
    # Statistics
    total_long_signals = (test_signals['Signal'] == 1).sum()
    total_short_signals = (test_signals['Signal'] == -1).sum()
    total_signals = total_long_signals + total_short_signals
    
    total_trades = sum(s['trades'] for s in all_stats)
    total_long = sum(s['long'] for s in all_stats)
    total_short = sum(s['short'] for s in all_stats)
    total_pnl = sum(s['pnl'] for s in all_stats)
    
    avg_wr = np.mean([s['win_rate'] for s in all_stats if s['trades'] > 0])
    
    daily_pnls = [s['pnl'] for s in all_stats]
    sharpe = np.mean(daily_pnls) / (np.std(daily_pnls) + 1e-8) * np.sqrt(252)
    
    print("\n" + "="*80)
    print("FINAL OUTPUT SUMMARY")
    print("="*80)
    print(f"\n📄 Files Generated:")
    print(f"  ✓ test_signal.csv ({len(test_signals):,} rows)")
    print(f"  ✓ portfolio_weights.csv ({len(portfolio):,} rows)")
    
    print(f"\n📊 Signal Statistics (test_signal.csv):")
    if total_signals > 0:
        print(f"  Total Signals: {total_signals:,}")
        print(f"  Long Signals: {total_long_signals:,} ({total_long_signals/total_signals*100:.1f}%)")
        print(f"  Short Signals: {total_short_signals:,} ({total_short_signals/total_signals*100:.1f}%)")
    else:
        print("  No raw signals generated.")
        
    print(f"\n💰 Performance (based on internal stats):")
    if total_trades > 0:
        print(f"  Days Traded: {len(all_stats)}")
        print(f"  Total Trades: {total_trades:,}")
        print(f"    - Long: {total_long:,} ({total_long/total_trades*100:.1f}%)")
        print(f"    - Short: {total_short:,} ({total_short/total_trades*100:.1f}%)")
        print(f"  Trades/Day: {total_trades/len(all_stats):.1f}")
        print(f"  Win Rate: {avg_wr:.1%}")
        print(f"  Total PnL (Metric): {total_pnl:,.2f}")
        print(f"  Avg Daily PnL (Metric): {np.mean(daily_pnls):.2f}")
        print(f"  Sharpe Ratio (Metric): {sharpe:.2f}")
        print(f"  Win Days: {sum(1 for p in daily_pnls if p > 0)}/{len(daily_pnls)}")
    else:
        print("  No trades were executed.")
    
    print("\n" + "="*80)
    print("✅ COMPLETE! Ready for backtesting.")
    print("   → Use test_signal.csv as input to your backtester")
    print("="*80 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='WFO Pipeline with Signal Generation')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'tactical', 'skip'],
                        help='Optimization mode: quick (grid), tactical (WFO), skip (use defaults)')
    parser.add_argument('--data', type=str, default='/data/quant14/EBX/',
                        help='Data folder path')
    parser.add_argument('--days', type=int, default=510,
                        help='Total number of days to process')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║             WALK-FORWARD OPTIMIZATION + SIGNAL GENERATION                 ║
    ║                                                                           ║
    ║   This pipeline will:                                                     ║
    ║   1. Optimize strategy parameters (or skip if specified)                  ║
    ║   2. Generate test_signal.csv with optimized parameters                   ║
    ║   3. Generate portfolio_weights.csv for analysis                          ║
    ║   4. Save optimized parameters to optimized_params.json                   ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Optimize (or skip)
    if args.mode == 'skip':
        print("\n⚠ Skipping optimization - using default strategy parameters")
        optimized_params = {}  # Empty = use strategy defaults
    
    elif args.mode == 'quick':
        optimized_params = optimize_quick(
            STRATEGY_CLASS,  # <-- FIXED
            args.data,
            days_range=range(0, min(90, args.days))
        )
    
    elif args.mode == 'tactical':
        # optimized_params = optimize_tactical_wfo(
        #     STRATEGY_CLASS,  # <-- FIXED
        #     args.data,
        #     total_days=args.days
        # )
        print("Tactical WFO mode is not fully implemented in this example.")
        print("Please update 'optimize_tactical_wfo' to use 'calculate_day_stats'.")
        optimized_params = {} # Placeholder
    
    # Save optimized parameters
    if optimized_params:
        with open('optimized_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)
        print(f"✓ Saved optimized parameters to optimized_params.json")
    
    # Step 2: Generate final signals
    success = generate_final_signals(
        STRATEGY_CLASS,  # <-- FIXED
        optimized_params,
        args.data,
        total_days=args.days
    )
    
    if success:
        print("\n🎉 Pipeline complete!")
        print("   Next step: Run your backtester with test_signal.csv")
    else:
        print("\n❌ Pipeline failed - check errors above")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())