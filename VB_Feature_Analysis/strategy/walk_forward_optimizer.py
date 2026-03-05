"""
===============================================================================
COMPLETE WALK-FORWARD OPTIMIZATION SYSTEM - FINAL VERSION
===============================================================================

FILE 1: walk_forward_optimizer.py
(FIXED: Transaction cost set to 0.05% and score logic improved)
"""

import pandas as pd
import numpy as np
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime
import warnings
from functools import partial
import sys

warnings.filterwarnings('ignore')

# Import modules
try:
    from backtester import backtest
    from adaptive_strategy import AdaptiveRegimeTradingStrategy
    print("✓ Backtester and Strategy imported successfully\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)


class WalkForwardOptimizer:
    """
    WFO: Generate signals (25 CPUs) → Run backtester → Optimize parameters
    """
    
    def __init__(self, 
                 train_days=170,
                 test_days=170,
                 anchor=False):
        
        self.train_days = train_days
        self.test_days = test_days
        self.anchor = anchor
        
        # Parameter grid - 128 combinations
        self.param_grid = {
            'strong_z_threshold': [1.0, 1.2],
            'weak_z_threshold': [0.50, 0.60],
            'sl_strong_mult': [0.55, 0.65],
            'tp_strong_mult': [1.1, 1.3],
            'sl_weak_mult': [0.40, 0.50],
            'tp_weak_mult': [0.85, 0.95],
            'max_trade_duration': [540, 660],
        }
        
        # Optimization objective: MAXIMIZE CAGR ONLY
        print("🎯 OPTIMIZATION OBJECTIVE: MAXIMIZE CAGR (with fallback to Total Returns)")
        print("="*80 + "\n")
        
        print("="*80)
        print("PARAMETER GRID")
        print("="*80)
        total = 1
        for key, values in self.param_grid.items():
            print(f"  {key:25s}: {values}")
            total *= len(values)
        print(f"\n  Total combinations: {total}")
        print("="*80 + "\n")
    
    def run_single_day_with_params(self, day, params, data_folder):
        """Run strategy on ONE day with specific parameters"""
        try:
            strategy = AdaptiveRegimeTradingStrategy()
            
            # Apply parameters
            for param_name, param_value in params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
            
            # Suppress output
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            result_df = strategy.process_day(day, data_folder=data_folder)
            
            sys.stdout = old_stdout
            
            return result_df
            
        except Exception as e:
            sys.stdout = old_stdout
            return None
    
    def generate_signals_for_days(self, days_list, params, data_folder, max_workers=25):
        """Generate signals for multiple days in PARALLEL (25 CPUs)"""
        print(f"    Generating signals for {len(days_list)} days (25 CPUs)...", end='', flush=True)
        
        process_func = partial(
            self.run_single_day_with_params,
            params=params,
            data_folder=data_folder
        )
        
        day_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_day = {executor.submit(process_func, day): day for day in days_list}
            
            for future in as_completed(future_to_day):
                try:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        day_results.append(result)
                except:
                    pass
        
        if len(day_results) == 0:
            print(f" ✗ No signals")
            return None
        
        combined_df = pd.concat(day_results, ignore_index=True)
        print(f" ✓ {len(day_results)}/{len(days_list)} days")
        
        return combined_df
    
    def run_backtest_get_metrics(self, signals_df):
        """Run backtester and extract key metrics"""
        try:
            temp_csv = 'temp_wfo_signals.csv'
            signals_df.to_csv(temp_csv, index=False)
            
            # Run backtester (suppress output)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            # ====================
            # === CRITICAL FIX ===
            # ====================
            # Set a realistic transaction cost. 0.1 is 10%. 
            # 0.0005 is 0.05% (or 5 basis points), which is more realistic.
            backtest_results = backtest(
                pd.read_csv(temp_csv),
                initial_capital=100000,
                transaction_cost_rate=0.0005, # CHANGED FROM 0.1
                slippage=0.0
            )
            
            sys.stdout = old_stdout
            
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            # Extract metrics
            sharpe = backtest_results.get('Annualized Sharpe Ratio', 0)
            calmar = backtest_results.get('Calmar Ratio', 0)
            max_dd = backtest_results.get('Maximum Drawdown', 0)
            returns = backtest_results.get('Total Returns', 0)
            cagr = backtest_results.get('CAGR', 0)
            trades = backtest_results.get('Total Trades Executed', 0)
            
            return {
                'sharpe': sharpe,
                'calmar': calmar,
                'max_dd': max_dd,
                'returns': returns,
                'cagr': cagr,
                'trades': trades,
                'full_results': backtest_results
            }
            
        except Exception as e:
            sys.stdout = old_stdout
            print(f" ✗ Backtest error: {e}")
            return None
    
    def calculate_score(self, metrics):
        """
        Calculate optimization score: MAXIMIZE CAGR.
        FIX: Removed -999 penalty. Will now use 'Total Returns' as the
        score if 'CAGR' is nan. This creates a smooth optimization
        surface even for negative-performing parameters.
        """
        if metrics is None:
            return -np.inf
        
        cagr = metrics['cagr']
        returns = metrics['returns']
        
        # Check if CAGR is valid and finite
        if np.isfinite(cagr):
            # CAGR is valid, use it as the score
            score = cagr
        else:
            # CAGR is 'nan'. This usually means returns were negative.
            # Use 'Total Returns' as the score instead.
            # A 20% loss (score: -20.0) is better than a 50% loss (score: -50.0).
            if np.isfinite(returns):
                score = returns
            else:
                # If both are nan, it's a total failure
                score = -np.inf
                
        return score
    
    def test_parameter_set(self, params, days_list, data_folder, max_workers=25):
        """Test one parameter set: Generate signals → Run backtester → Score"""
        signals_df = self.generate_signals_for_days(
            days_list, params, data_folder, max_workers
        )
        
        if signals_df is None:
            return None, None
        
        print(f"    Running backtester...", end='', flush=True)
        metrics = self.run_backtest_get_metrics(signals_df)
        
        if metrics is None:
            print(f" ✗")
            return None, None
        
        print(f" ✓")
        
        score = self.calculate_score(metrics)
        
        return metrics, score
    
    def optimize_on_training_period(self, days_list, data_folder, max_workers=25):
        """Test all parameter combinations on training period"""
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION: Days {days_list[0]}-{days_list[-1]} ({len(days_list)} days)")
        print(f"{'='*80}")
        
        # Generate all combinations
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]
        all_combinations = list(product(*param_values))
        
        print(f"\nTesting {len(all_combinations)} parameter sets\n")
        
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Test each combination
        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            
            print(f"[{i+1}/{len(all_combinations)}] Testing:")
            print(f"  strong_z={params['strong_z_threshold']:.1f}, "
                  f"weak_z={params['weak_z_threshold']:.2f}, "
                  f"sl_strong={params['sl_strong_mult']:.2f}, "
                  f"tp_strong={params['tp_strong_mult']:.1f}, "
                  f"max_dur={params['max_trade_duration']}")
            
            metrics, score = self.test_parameter_set(
                params, days_list, data_folder, max_workers
            )
            
            if metrics is None:
                print(f"  ⚠️  Test failed\n")
                continue
            
            # Use 'returns' for printing if cagr is nan
            score_metric_val = metrics['cagr'] if np.isfinite(metrics['cagr']) else metrics['returns']
            score_metric_name = "CAGR" if np.isfinite(metrics['cagr']) else "Returns"
            
            print(f"  Sharpe={metrics['sharpe']:.2f}, "
                  f"Calmar={metrics['calmar']:.2f}, "
                  f"CAGR={metrics['cagr']:.2f}%, "
                  f"DD={metrics['max_dd']:.2f}%, "
                  f"Score({score_metric_name})={score:.4f}%")
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_metrics = metrics.copy()
                print(f"  🔥 NEW BEST! Score={score:.4f}%")
            
            print()
        
        # Summary
        print(f"{'='*80}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        
        if best_params:
            best_score_metric_val = best_metrics['cagr'] if np.isfinite(best_metrics['cagr']) else best_metrics['returns']
            best_score_metric_name = "CAGR" if np.isfinite(best_metrics['cagr']) else "Returns"

            print(f"\n✓ Best Score ({best_score_metric_name}): {best_score:.4f}%")
            print(f"\nBest Parameters:")
            for k, v in sorted(best_params.items()):
                print(f"  {k:25s} = {v}")
            print(f"\nBest Metrics:")
            print(f"  CAGR         : {best_metrics['cagr']:.4f}% ⭐ OPTIMIZED (fallback {best_metrics['returns']:.2f}%)")
            print(f"  Sharpe       : {best_metrics['sharpe']:.4f}")
            print(f"  Calmar       : {best_metrics['calmar']:.4f}")
            print(f"  Max Drawdown : {best_metrics['max_dd']:.2f}%")
            print(f"  Returns      : {best_metrics['returns']:.2f}%")
            print(f"  Trades       : {best_metrics['trades']:.0f}")
        else:
            print("\n⚠️  No valid parameters found")
        
        return best_params, best_metrics, best_score
    
    def walk_forward_optimize(self, total_days=510, data_folder='/data/quant14/EBX/', max_workers=25):
        """
        Walk-Forward Optimization main loop
        NO strategy_class argument - imports directly
        """
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*80)
        print(f"Train window  : {self.train_days} days")
        print(f"Test window   : {self.test_days} days")
        print(f"Total days    : {total_days}")
        print(f"Window type   : {'ANCHORED' if self.anchor else 'ROLLING'}")
        print(f"CPUs per test : {max_workers}")
        print(f"Data folder   : {data_folder}")
        print("="*80)
        
        import time
        start_time = time.time()
        
        wfo_results = []
        current_start = 0
        iteration = 0
        
        while current_start + self.train_days + self.test_days <= total_days:
            iteration += 1
            iter_start = time.time()
            
            print(f"\n{'#'*80}")
            print(f"# WFO ITERATION {iteration}")
            print(f"{'#'*80}")
            
            # Define periods
            train_start = 0 if self.anchor else current_start
            train_end = current_start + self.train_days
            test_start = train_end
            test_end = min(test_start + self.test_days, total_days)
            
            train_days = list(range(train_start, train_end))
            test_days = list(range(test_start, test_end))
            
            print(f"\nTraining : days {train_start:3d}-{train_end-1:3d}")
            print(f"Testing  : days {test_start:3d}-{test_end-1:3d}")
            
            # TRAINING
            print(f"\n{'─'*80}")
            print("TRAINING PHASE")
            print(f"{'─'*80}")
            
            best_params, train_metrics, train_score = self.optimize_on_training_period(
                train_days, data_folder, max_workers
            )
            
            if best_params is None:
                print("⚠️  Training failed - skipping iteration")
                current_start += self.test_days
                continue
            
            # TESTING
            print(f"\n{'─'*80}")
            print("TESTING PHASE")
            print(f"{'─'*80}")
            print(f"\nTesting best parameters on out-of-sample days...")
            
            test_metrics, test_score = self.test_parameter_set(
                best_params, test_days, data_folder, max_workers
            )
            
            if test_metrics:
                score_metric_val = test_metrics['cagr'] if np.isfinite(test_metrics['cagr']) else test_metrics['returns']
                score_metric_name = "CAGR" if np.isfinite(test_metrics['cagr']) else "Returns"

                print(f"\n{'='*80}")
                print("OUT-OF-SAMPLE RESULTS")
                print(f"{'='*80}")
                print(f"  Score ({score_metric_name}) : {test_score:.4f}% ⭐")
                print(f"  CAGR         : {test_metrics['cagr']:.4f}%")
                print(f"  Sharpe       : {test_metrics['sharpe']:.4f}")
                print(f"  Calmar       : {test_metrics['calmar']:.4f}")
                print(f"  Max Drawdown : {test_metrics['max_dd']:.2f}%")
                print(f"  Returns      : {test_metrics['returns']:.2f}%")
                print(f"  Trades       : {test_metrics['trades']:.0f}")
                print(f"{'='*80}")
            else:
                print("⚠️  Testing failed")
                test_score = None
            
            iter_time = (time.time() - iter_start) / 60
            print(f"\n⏱️  Iteration {iteration} completed in {iter_time:.1f} minutes")
            
            # Store results
            wfo_results.append({
                'iteration': iteration,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_score': train_score,
                'test_score': test_score,
                'iteration_time_min': iter_time
            })
            
            # Move window
            current_start += self.test_days
        
        total_time = (time.time() - start_time) / 3600
        print(f"\n{'='*80}")
        print(f"WFO COMPLETE - {total_time:.2f} hours")
        print(f"{'='*80}")
        
        # Save and summarize
        self.save_results(wfo_results)
        final_params = self.generate_summary(wfo_results)
        
        return wfo_results, final_params
    
    def save_results(self, results):
        """Save WFO results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wfo_results_{timestamp}.json"
        
        serializable = []
        for r in results:
            serializable.append({
                'iteration': r['iteration'],
                'train_start': r['train_start'],
                'train_end': r['train_end'],
                'test_start': r['test_start'],
                'test_end': r['test_end'],
                'best_params': r['best_params'],
                'train_metrics': r['train_metrics'],
                'test_metrics': r['test_metrics'],
                'train_score': float(r['train_score']) if r['train_score'] and np.isfinite(r['train_score']) else None,
                'test_score': float(r['test_score']) if r['test_score'] and np.isfinite(r['test_score']) else None,
                'iteration_time_min': r['iteration_time_min']
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n💾 Detailed results saved: {filename}")
    
    def generate_summary(self, results):
        """Generate final summary with recommended parameters"""
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION SUMMARY")
        print("="*80)
        
        valid_results = [r for r in results if r['test_metrics'] is not None]
        
        if not valid_results:
            print("No valid out-of-sample results")
            return None
        
        # Aggregate OOS metrics
        oos_scores = [self.calculate_score(r['test_metrics']) for r in valid_results]
        oos_cagrs = [r['test_metrics']['cagr'] for r in valid_results if np.isfinite(r['test_metrics']['cagr'])]
        oos_sharpes = [r['test_metrics']['sharpe'] for r in valid_results]
        oos_calmars = [r['test_metrics']['calmar'] for r in valid_results if np.isfinite(r['test_metrics']['calmar'])]
        oos_dds = [r['test_metrics']['max_dd'] for r in valid_results]
        oos_returns = [r['test_metrics']['returns'] for r in valid_results]
        
        print(f"\nOUT-OF-SAMPLE PERFORMANCE ({len(valid_results)} iterations):")
        print(f"  Avg Score    : {np.mean(oos_scores):.4f} (±{np.std(oos_scores):.4f}) ⭐ OPTIMIZED")
        print(f"  Avg CAGR     : {np.mean(oos_cagrs):.4f}% (from {len(oos_cagrs)} valid runs)")
        print(f"  Avg Sharpe   : {np.mean(oos_sharpes):.4f} (±{np.std(oos_sharpes):.4f})")
        print(f"  Avg Calmar   : {np.mean(oos_calmars):.4f} (from {len(oos_calmars)} valid runs)")
        print(f"  Avg Max DD   : {np.mean(oos_dds):.2f}%")
        print(f"  Avg Returns  : {np.mean(oos_returns):.2f}%")
        
        # Most stable parameters
        print(f"\n🎯 RECOMMENDED PARAMETERS (Most Stable):")
        all_params = {}
        for r in valid_results:
            for param, value in r['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        final_params = {}
        for param, values in all_params.items():
            most_common = max(set(values), key=values.count)
            frequency = values.count(most_common) / len(values)
            final_params[param] = most_common
            print(f"  {param:25s} = {most_common:8} (used {frequency:.0%} of iterations)")
        
        # Best iteration (highest score)
        best_iter = max(valid_results, key=lambda x: self.calculate_score(x['test_metrics']) if x['test_metrics'] else -np.inf)
        best_iter_score = self.calculate_score(best_iter['test_metrics'])
        best_iter_score_name = "CAGR" if np.isfinite(best_iter['test_metrics']['cagr']) else "Returns"
        
        print(f"\n🏆 BEST SINGLE ITERATION (#{best_iter['iteration']}) - Highest OOS Score:")
        print(f"  OOS Score ({best_iter_score_name}) : {best_iter_score:.4f}% ⭐")
        print(f"  OOS CAGR     : {best_iter['test_metrics']['cagr']:.4f}%")
        print(f"  OOS Sharpe   : {best_iter['test_metrics']['sharpe']:.4f}")
        print(f"  OOS Calmar   : {best_iter['test_metrics']['calmar']:.4f}")
        print(f"  OOS Returns  : {best_iter['test_metrics']['returns']:.2f}%")
        print(f"  OOS Max DD   : {best_iter['test_metrics']['max_dd']:.2f}%")
        
        # Save final parameters
        with open('wfo_final_parameters.json', 'w') as f:
            json.dump({
                'recommended_stable_params': final_params,
                'best_iteration_params': best_iter['best_params'],
                'summary': {
                    'avg_oos_score': float(np.mean(oos_scores)),
                    'avg_oos_cagr': float(np.mean(oos_cagrs)) if oos_cagrs else None,
                    'avg_oos_sharpe': float(np.mean(oos_sharpes)),
                    'avg_oos_calmar': float(np.mean(oos_calmars)) if oos_calmars else None,
                    'avg_oos_returns': float(np.mean(oos_returns)),
                    'avg_oos_max_dd': float(np.mean(oos_dds)),
                    'best_iteration': int(best_iter['iteration']),
                    'best_oos_score': float(best_iter_score)
                }
            }, f, indent=2)
        
        print(f"\n💾 Final parameters saved: wfo_final_parameters.json")
        
        # Ready to copy
        print(f"\n{'='*80}")
        print("COPY THESE TO adaptive_strategy.py __init__():")
        print(f"{'='*80}")
        for param, value in sorted(final_params.items()):
            print(f"        self.{param} = {value}")
        print(f"{'='*80}\n")
        
        return final_params


"""
===============================================================================
FILE 2: run_wfo.py (This file is correct, no changes needed)
===============================================================================
"""

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING WALK-FORWARD OPTIMIZATION")
    print("="*80)
    
    wfo = WalkForwardOptimizer(
        train_days=170,      # ~6 months training
        test_days=170,       # ~6 months testing
        anchor=False         # Rolling window
    )
    
    # NO strategy_class argument - it's imported directly in the class
    results, final_params = wfo.walk_forward_optimize(
        total_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    print("\n✅ WFO COMPLETE!")
    print("\nFiles generated:")
    print("  - wfo_results_TIMESTAMP.json (full details)")
    print("  - wfo_final_parameters.json (recommended params)")
    print("\nNext steps:")
    print("  1. Copy the parameters shown above to adaptive_strategy.py")
    print("  2. Run: python adaptive_strategy.py")
    print("  3. Run: python backtester.py")