import numpy as np
import pandas as pd
from numba import jit, prange
import time

# ============================================================================
# OPTIMIZED CPU FUNCTIONS USING NUMBA
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_triple_barrier_numba(prices, threshold, lookahead):
    """
    Ultra-fast CPU implementation using Numba
    MODIFIED: Uses ABSOLUTE price thresholds.
    e.g., threshold=0.07 means a $0.07 change.
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    
    for i in prange(n):  # Parallel across all cores
        start_price = prices[i]
        if start_price == 0:
            continue
        
        # --- MODIFIED CALCULATION (ABSOLUTE) ---
        upper_barrier = start_price + threshold  # Was: start_price * (1.0 + threshold)
        lower_barrier = start_price - threshold  # Was: start_price * (1.0 - threshold)
        # --- END MODIFICATION ---
        
        end_j = min(i + lookahead, n - 1)
        
        for j in range(i + 1, end_j + 1):
            future_price = prices[j]
            
            if future_price >= upper_barrier:
                labels[i] = 1
                break
            
            if future_price <= lower_barrier:
                labels[i] = -1
                break
    
    return labels


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_slope_numba(prices, lookahead):
    """
    Ultra-fast CPU slope computation (ORIGINAL VERSION)
    Computes the rate of change: (p_future - p_now) / lookahead
    """
    n = len(prices)
    slopes = np.full(n, np.nan, dtype=np.float32)
    
    for i in prange(n - lookahead):
        # --- ORIGINAL SLOPE CALCULATION ---
        slopes[i] = (prices[i + lookahead] - prices[i]) / lookahead
        # --- END ORIGINAL ---
    
    return slopes


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_all_triple_barriers_batch(prices, thresholds, lookaheads):
    """
    Compute multiple triple barrier labels in one pass (ABSOLUTE thresholds)
    """
    n = len(prices)
    n_labels = len(thresholds)
    labels = np.zeros((n, n_labels), dtype=np.int8)
    
    for label_idx in range(n_labels):
        threshold = thresholds[label_idx]
        lookahead = lookaheads[label_idx]
        
        for i in prange(n):
            start_price = prices[i]
            if start_price == 0:
                continue
            
            # --- MODIFIED CALCULATION (ABSOLUTE) ---
            upper_barrier = start_price + threshold  # Was: start_price * (1.0 + threshold)
            lower_barrier = start_price - threshold  # Was: start_price * (1.0 - threshold)
            # --- END MODIFICATION ---
            
            end_j = min(i + lookahead, n - 1)
            
            for j in range(i + 1, end_j + 1):
                future_price = prices[j]
                
                if future_price >= upper_barrier:
                    labels[i, label_idx] = 1
                    break
                
                if future_price <= lower_barrier:
                    labels[i, label_idx] = -1
                    break
    
    return labels


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_all_slopes_batch(prices, lookaheads):
    """
    Compute multiple slope labels in one pass (ORIGINAL VERSION)
    """
    n = len(prices)
    n_labels = len(lookaheads)
    slopes = np.full((n, n_labels), np.nan, dtype=np.float32)
    
    for label_idx in range(n_labels):
        lookahead = lookaheads[label_idx]
        
        for i in prange(n - lookahead):
            # --- ORIGINAL SLOPE CALCULATION ---
            slopes[i, label_idx] = (prices[i + lookahead] - prices[i]) / lookahead
            # --- END ORIGINAL ---
    
    return slopes


# ============================================================================
# LABEL GENERATOR CLASS - (HYBRID: ABSOLUTE TB, ORIGINAL SLOPE)
# ============================================================================

class LabelGenerator:
    """
    High-performance Label Generator (HYBRID MODE)
    
    *** WARNING ***
    This class now calculates:
    1. Triple Barriers (TB_): Uses ABSOLUTE price changes.
       e.g., 'threshold': 0.07 means a $0.07 price move, NOT 7%.
       You will likely need to adjust your config thresholds.
       
    2. Slopes (Slope_): Uses the ORIGINAL slope calculation.
       (future_price - current_price) / lookahead
    """
    
    def __init__(self, use_cpu_fallback=True):
        """Initialize with configurations"""
        self.use_cpu_only = use_cpu_fallback
        
        # --- WARNING: These thresholds are now ABSOLUTE price values ---
        # e.g., 0.07 means $0.07. You may want 7.0 for a $7 move.
        self.triple_barrier_configs = [
            {'threshold': 0.07, 'lookahead': 30},
            {'threshold': 0.07, 'lookahead': 60},
            {'threshold': 0.07, 'lookahead': 120},
            {'threshold': 0.07, 'lookahead': 300},
            {'threshold': 0.07, 'lookahead': 600},
            {'threshold': 0.07, 'lookahead': 900},
            {'threshold': 0.07, 'lookahead': 1200},
            {'threshold': 0.07, 'lookahead': 2000},
            {'threshold': 0.07, 'lookahead': 2400},
            {'threshold': 0.07, 'lookahead': 3600},
            {'threshold': 0.10, 'lookahead': 600},
            {'threshold': 0.10, 'lookahead': 900},
            {'threshold': 0.10, 'lookahead': 1200},
            {'threshold': 0.10, 'lookahead': 2000},
            {'threshold': 0.10, 'lookahead': 2400},
            {'threshold': 0.10, 'lookahead': 3600},
            {'threshold': 0.30, 'lookahead': 1200},
            {'threshold': 0.30, 'lookahead': 2000},
            {'threshold': 0.30, 'lookahead': 2400},
            {'threshold': 0.30, 'lookahead': 3600},
        ]
        
        # Slope Configurations (Original)
        self.slope_lookaheads = [5, 30, 60, 90, 120, 180, 300, 600, 900, 1200, 1800, 2400, 3600]
        
        # Pre-compute arrays for batch processing
        self.tb_thresholds = np.array([c['threshold'] for c in self.triple_barrier_configs], dtype=np.float32)
        self.tb_lookaheads = np.array([c['lookahead'] for c in self.triple_barrier_configs], dtype=np.int32)
        self.slope_lookaheads_array = np.array(self.slope_lookaheads, dtype=np.int32)
        
        # Precompute label names
        self._label_columns = self._compute_label_columns()
        
        print(f"LabelGenerator initialized (Batch mode: CPU-only, HYBRID)")
        print(f"  WARNING: Triple Barrier (TB_) thresholds are ABSOLUTE values (e.g., $0.07).")
        print(f"  WARNING: Slope (Slope_) labels are ORIGINAL slopes (rate of change).")
        print(f"  Triple barriers: {len(self.triple_barrier_configs)}")
        print(f"  Slopes: {len(self.slope_lookaheads)}")
        print(f"  Total labels: {len(self._label_columns)}")
        
        # Trigger JIT compilation with dummy data
        self._warmup_jit()
    
    
    def _warmup_jit(self):
        """
        Pre-compile Numba functions with small dataset
        """
        print("  Warming up JIT compiler...", end=" ")
        dummy_prices = np.random.randn(1000).cumsum().astype(np.float32) + 100
        
        # Compile all functions
        _ = compute_triple_barrier_numba(dummy_prices, 0.07, 300) # Absolute TB
        _ = compute_slope_numba(dummy_prices, 300)                # Original Slope
        _ = compute_all_triple_barriers_batch(dummy_prices, self.tb_thresholds, self.tb_lookaheads) # Absolute TB
        _ = compute_all_slopes_batch(dummy_prices, self.slope_lookaheads_array) # Original Slope
        
        print("✓ Done")
    
    
    def _compute_label_columns(self):
        """Precompute all label column names"""
        cols = []
        
        for config in self.triple_barrier_configs:
            # Note: The 't' value still comes from your config, but it's used as an absolute value
            cols.append(f"TB_t{config['threshold']:.2f}_l{config['lookahead']}")
        
        for lookahead in self.slope_lookaheads:
            cols.append(f"Slope_l{lookahead}") # Kept original name
        
        return cols
    
    
    def get_label_columns(self):
        """Get list of all label column names"""
        return self._label_columns
    
    
    def generate_labels(self, df, price_col='Price', time_col='Time', use_gpu=False):
        """
        Generate ALL labels at once (OPTIMIZED FOR SPEED)
        """
        prices = df[price_col].values.astype(np.float32)
        
        # Generate all triple barrier labels in one batch
        tb_labels = compute_all_triple_barriers_batch(
            prices, self.tb_thresholds, self.tb_lookaheads
        )
        
        # Add triple barrier columns
        for idx, config in enumerate(self.triple_barrier_configs):
            col_name = f"TB_t{config['threshold']:.2f}_l{config['lookahead']}"
            df[col_name] = tb_labels[:, idx]
        
        # Generate all slope labels in one batch
        slope_labels = compute_all_slopes_batch(prices, self.slope_lookaheads_array)
        
        # Add slope columns
        for idx, lookahead in enumerate(self.slope_lookaheads):
            col_name = f"Slope_l{lookahead}"
            df[col_name] = slope_labels[:, idx]
        
        return df
    
    
    def generate_single_label(self, df, label_col, price_col='Price'):
        """
        Generate ONLY ONE label (for backwards compatibility)
        """
        prices = df[price_col].values.astype(np.float32)
        
        if label_col.startswith('TB_'):
            # Triple barrier label (ABSOLUTE)
            parts = label_col.split('_')
            threshold = float(parts[1][1:])
            lookahead = int(parts[2][1:])
            
            labels = compute_triple_barrier_numba(prices, threshold, lookahead)
            df[label_col] = labels
            
        elif label_col.startswith('Slope_'):
            # Slope label (ORIGINAL)
            lookahead = int(label_col.split('_')[1][1:])
            
            slopes = compute_slope_numba(prices, lookahead)
            df[label_col] = slopes
        
        else:
            raise ValueError(f"Unknown label format: {label_col}. Must start with 'TB_' or 'Slope_'.")
        
        return df


# ============================================================================
# PERFORMANCE BENCHMARK
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LABEL GENERATOR PERFORMANCE BENCHMARK (HYBRID MODE)")
    print(" (Absolute Triple-Barrier, Original Slope) ")
    print("=" * 80)
    
    # Test different dataset sizes
    sizes = [10000, 50000, 100000, 500000]
    
    label_gen = LabelGenerator(use_cpu_fallback=True)
    
    print("\nBenchmark: Generate all labels at once (BATCH MODE)")
    print("-" * 80)
    
    total_labels = len(label_gen.get_label_columns())
    
    for n_samples in sizes:
        test_prices = np.random.randn(n_samples).cumsum() + 100
        test_df = pd.DataFrame({'Price': test_prices, 'Time': range(n_samples)})
        
        start = time.time()
        test_df = label_gen.generate_labels(test_df.copy())
        elapsed = time.time() - start
        
        samples_per_sec = n_samples / elapsed
        labels_per_sec = n_samples * total_labels / elapsed
        
        print(f"  {n_samples:>7,} samples: {elapsed:>6.2f}s  "
              f"({samples_per_sec:>10,.0f} samples/s, {labels_per_sec:>12,.0f} label-samples/s)")
    
    print("\n" + "=" * 80)
    print("Comparison: Single label vs Batch mode (100K samples)")
    print("-" * 80)
    
    test_prices = np.random.randn(100000).cumsum() + 100
    test_df = pd.DataFrame({'Price': test_prices, 'Time': range(100000)})
    
    # Test 1: Generate all labels one-by-one
    print(f"\n1. Generate {total_labels} labels ONE-BY-ONE:")
    start = time.time()
    df_single = test_df.copy()
    for label_col in label_gen.get_label_columns():
        df_single = label_gen.generate_single_label(df_single, label_col)
    time_single = time.time() - start
    print(f"   Time: {time_single:.2f}s")
    
    # Test 2: Generate all labels at once (batch)
    print(f"\n2. Generate {total_labels} labels ALL-AT-ONCE (batch):")
    start = time.time()
    df_batch = label_gen.generate_labels(test_df.copy())
    time_batch = time.time() - start
    print(f"   Time: {time_batch:.2f}s")
    
    print(f"\n✓ Batch mode is {time_single/time_batch:.1f}x FASTER!")
    
    # Verify results are identical
    print("\n3. Verification:")
    all_match = True
    for col in label_gen.get_label_columns():
        if not np.allclose(df_single[col].fillna(-999), df_batch[col].fillna(-999), equal_nan=True):
            print(f"   ❌ Mismatch in {col}")
            all_match = False
    
    if all_match:
        print("   ✓ All labels match perfectly!")
    
    print("\n" + "=" * 80)