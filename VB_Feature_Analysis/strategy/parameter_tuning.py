"""
Parameter Tuning Script
========================
Quick testing of different parameter combinations
"""

import pandas as pd
import numpy as np
from optimized_strategy import HighConfidenceStrategy
import itertools


def test_parameter_set(params, num_days=50, data_folder='/data/quant14/EBX/'):
    """
    Test a single parameter combination
    Returns: (params, total_trades, avg_trades_per_day)
    """
    strategy = HighConfidenceStrategy()
    
    # Apply parameters
    strategy.strong_z_threshold = params['z_threshold']
    strategy.max_trades_per_day = params['max_trades']
    strategy.tp_mult = params['tp_mult']
    strategy.sl_mult = params['sl_mult']
    strategy.min_expected_profit_pct = params['min_profit']  # Now percentage
    strategy.min_profit_target_pct = params['min_profit']     # Now percentage
    strategy.volume_surge_threshold = params['vol_surge']
    
    print(f"\nTesting: z={params['z_threshold']}, max_trades={params['max_trades']}, "
          f"tp={params['tp_mult']}, sl={params['sl_mult']}")
    
    try:
        result = strategy.run_strategy(
            num_days=num_days,
            data_folder=data_folder,
            max_workers=10
        )
        
        if result is not None:
            # Count trades
            total_trades = 0
            position_changes = np.sum(np.abs(np.diff(result['Signal'].values)))
            total_trades = position_changes // 2
            avg_trades = total_trades / num_days
            
            return {
                'params': params,
                'total_trades': total_trades,
                'avg_trades_per_day': avg_trades,
                'success': True
            }
        else:
            return {'params': params, 'success': False}
    
    except Exception as e:
        print(f"Error: {e}")
        return {'params': params, 'success': False}


def grid_search(num_test_days=50):
    """
    Run grid search over parameter space
    """
    
    # Define parameter grid - VERY RELAXED TO GENERATE TRADES
    param_grid = {
        'z_threshold': [0.4, 0.5, 0.6],           # Very low thresholds
        'max_trades': [18, 20, 25],                # Higher limits
        'tp_mult': [1.4, 1.5, 1.6],                # Take profit multipliers
        'sl_mult': [0.50, 0.55, 0.60],             # Stop loss multipliers
        'min_profit': [0.0012, 0.0015, 0.0018],    # Min profit % (0.12%, 0.15%, 0.18%)
        'vol_surge': [1.0, 1.05, 1.1]              # Very low volume requirements
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} parameter combinations...")
    print(f"This may take a while...\n")
    
    results = []
    
    for i, params in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Combination {i}/{len(combinations)}")
        print(f"{'='*60}")
        
        result = test_parameter_set(params, num_days=num_test_days)
        results.append(result)
        
        if result['success']:
            print(f"✓ Avg trades/day: {result['avg_trades_per_day']:.1f}")
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) > 0:
        df = pd.DataFrame([{
            **r['params'],
            'total_trades': r['total_trades'],
            'avg_trades_per_day': r['avg_trades_per_day']
        } for r in successful_results])
        
        # Sort by trades closest to target (10/day)
        df['trade_deviation'] = abs(df['avg_trades_per_day'] - 10)
        df = df.sort_values('trade_deviation')
        
        print(f"\n{'='*80}")
        print("TOP 10 PARAMETER COMBINATIONS")
        print(f"{'='*80}\n")
        
        print(df.head(10).to_string(index=False))
        
        # Save results
        df.to_csv('parameter_tuning_results.csv', index=False)
        print(f"\n✓ Saved to parameter_tuning_results.csv")
        
        # Best parameters
        best = df.iloc[0]
        print(f"\n{'='*80}")
        print("RECOMMENDED PARAMETERS")
        print(f"{'='*80}")
        print(f"z_threshold: {best['z_threshold']}")
        print(f"max_trades: {int(best['max_trades'])}")
        print(f"tp_mult: {best['tp_mult']}")
        print(f"sl_mult: {best['sl_mult']}")
        print(f"min_profit: {best['min_profit']}")
        print(f"vol_surge: {best['vol_surge']}")
        print(f"\nExpected trades/day: {best['avg_trades_per_day']:.1f}")
        print(f"{'='*80}")


def quick_test():
    """
    Quick test with 3 parameter sets
    """
    
    test_sets = [
        {
            'name': 'Conservative',
            'z_threshold': 0.6,
            'max_trades': 18,
            'tp_mult': 1.6,
            'sl_mult': 0.55,
            'min_profit': 0.0018,  # 0.18%
            'vol_surge': 1.1
        },
        {
            'name': 'Balanced',
            'z_threshold': 0.5,
            'max_trades': 20,
            'tp_mult': 1.5,
            'sl_mult': 0.50,
            'min_profit': 0.0015,   # 0.15%
            'vol_surge': 1.05
        },
        {
            'name': 'Aggressive',
            'z_threshold': 0.4,
            'max_trades': 25,
            'tp_mult': 1.4,
            'sl_mult': 0.50,
            'min_profit': 0.0012,  # 0.12%
            'vol_surge': 1.0
        }
    ]
    
    print("="*80)
    print("QUICK PARAMETER TEST (3 sets)")
    print("="*80)
    
    results = []
    
    for test_set in test_sets:
        name = test_set.pop('name')
        print(f"\n\nTesting: {name}")
        print("-"*80)
        
        result = test_parameter_set(test_set, num_days=50)
        
        if result['success']:
            results.append({
                'name': name,
                'avg_trades_per_day': result['avg_trades_per_day'],
                'total_trades': result['total_trades'],
                **test_set
            })
    
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        print(f"\n\n{'='*80}")
        print("QUICK TEST RESULTS")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        
        # Recommend best
        df['trade_deviation'] = abs(df['avg_trades_per_day'] - 10)
        best = df.loc[df['trade_deviation'].idxmin()]
        
        print(f"\n✓ Best configuration: {best['name']}")
        print(f"  Avg trades/day: {best['avg_trades_per_day']:.1f}")


def regime_specific_test(num_days=100):
    """
    Test if trading only one regime is better
    """
    
    print("="*80)
    print("REGIME-SPECIFIC TESTING")
    print("="*80)
    
    configs = [
        {'name': 'Both Regimes', 'trade_both': True},
        {'name': 'Mean Reversion Only', 'trade_both': False, 'regime': 0},
        {'name': 'Momentum Only', 'trade_both': False, 'regime': 1}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print("-"*80)
        
        strategy = HighConfidenceStrategy()
        
        # Standard parameters
        strategy.strong_z_threshold = 2.0
        strategy.max_trades_per_day = 10
        
        # TODO: Implement regime filtering in strategy
        # This would require modifying the strategy class
        
        print("(Implementation needed in strategy class)")
    
    print("\nTo implement regime-specific trading:")
    print("1. Add 'trade_only_regime' parameter to strategy __init__")
    print("2. In generate_ultra_selective_signals, check regime")
    print("3. Return empty signals if regime doesn't match")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'quick':
            print("Running quick test (3 parameter sets, 50 days)...")
            quick_test()
        
        elif mode == 'grid':
            print("Running full grid search...")
            grid_search(num_test_days=50)
        
        elif mode == 'regime':
            print("Running regime-specific test...")
            regime_specific_test(num_days=100)
        
        else:
            print("Unknown mode. Use: quick, grid, or regime")
    
    else:
        print("\nParameter Tuning Script")
        print("="*60)
        print("\nUsage:")
        print("  python parameter_tuning.py quick   # Test 3 configs (fast)")
        print("  python parameter_tuning.py grid    # Full grid search (slow)")
        print("  python parameter_tuning.py regime  # Test regime-specific")
        print("\nRecommendation: Start with 'quick' mode")
        print("\nRunning QUICK TEST by default...\n")
        
        quick_test()