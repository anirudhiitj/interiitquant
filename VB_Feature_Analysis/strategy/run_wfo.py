"""
Run Walk-Forward Optimization - FIXED VERSION
==============================================
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the strategy and WFO
from adaptive_strategy import AdaptiveRegimeTradingStrategy
from walk_forward_optimizer import WalkForwardOptimizer

if __name__ == "__main__":
    print("\n" + "="*80)
    print("WALK-FORWARD OPTIMIZATION - STARTING")
    print("="*80)
    print("Configuration:")
    print("  - 170 day training windows (~6 months)")
    print("  - 170 day testing windows (~6 months)")
    print("  - Rolling windows (non-anchored)")
    print("  - 128 parameter combinations (2^7)")
    print("  - Sequential processing (more stable)")
    print("  - Data folder: /data/quant14/EBX/")
    print("  - Estimated time: 5-6 hours")
    print("="*80 + "\n")
    
    # Initialize WFO with optimized settings
    wfo = WalkForwardOptimizer(
        train_days=170,   # ~6 months for robust learning
        test_days=170,    # ~6 months for validation  
        anchor=False      # Rolling window (adapts to regime changes)
    )
    
    # Run WFO
    print("🚀 Starting optimization...")
    print("💾 Progress will be shown in real-time")
    print("⚠️  This will take several hours - be patient!\n")
    
    try:
        results, final_params = wfo.walk_forward_optimize(
            strategy_class=AdaptiveRegimeTradingStrategy,
            total_days=510,
            data_folder='/data/quant14/EBX/'
        )
        
        print("\n" + "="*80)
        print("✅ WFO COMPLETE!")
        print("="*80)
        print("\nFiles generated:")
        print("  1. wfo_results_TIMESTAMP.json - Full optimization results")
        print("  2. wfo_final_parameters.json - Recommended parameters")
        
        if final_params:
            print("\n📊 FINAL RECOMMENDED PARAMETERS:")
            for param, value in final_params.items():
                print(f"  {param}: {value}")
            
            print("\n" + "="*80)
            print("NEXT STEPS:")
            print("="*80)
            print("1. Review wfo_final_parameters.json")
            print("2. Update your adaptive_strategy.py with recommended_stable_params")
            print("3. Example:")
            print("   self.strong_z_threshold = {}".format(final_params.get('strong_z_threshold', 1.1)))
            print("   self.weak_z_threshold = {}".format(final_params.get('weak_z_threshold', 0.55)))
            print("   # ... etc for all parameters")
            print("4. Run: python adaptive_strategy.py")
            print("5. Then: python backtester.py")
            print("="*80)
        else:
            print("\n⚠️  No valid parameters found. Check the error messages above.")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80)
        print("TROUBLESHOOTING:")
        print("="*80)
        print("1. Check that /data/quant14/EBX/ contains day0.parquet through day509.parquet")
        print("2. Verify your strategy file has no syntax errors")
        print("3. Try running with fewer days first (change total_days=100)")
        print("4. Check memory/CPU availability")
        print("="*80)