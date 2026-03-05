"""
Optimized Constraint-Compliant Trading Strategy
================================================

Architecture:
- ProcessPoolExecutor with 25 workers for parallel day processing
- CPU-only (avoids CUDA multiprocessing issues)
- Numba JIT compilation with parallel=True for speed
- Memory-efficient batch processing

OPTIMIZATION 1:
- Added 'vb_quiet_percentile' to filter out low-volatility chop.

LOGIC FIX (2025-10-30):
- Corrected 'generate_balanced_signals' to remove the "double inversion" error.
- With negative weights, standard logic (combined_z > 0 -> LONG) is now correct.
"""

import pandas as pd
import numpy as np
import os
import warnings
from numba import jit, prange
import concurrent.futures
from functools import partial
import gc

warnings.filterwarnings('ignore')


class OptimizedParallelStrategy:
    """
    CPU-optimized trading strategy with parallel processing
    
    Uses ProcessPoolExecutor (25 workers) for true parallel execution
    All computations optimized with Numba JIT compilation
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # Feature definitions
        self.bb_features = [
            'BB1_T10', 'BB1_T11', 'BB1_T12',
            'BB4_T10', 'BB4_T11', 'BB4_T12',
            'BB5_T10', 'BB5_T11', 'BB5_T12',
            'PB10_T11', 'PB11_T11'
        ]
        
        # PB Features (0.90 corr) - MOMENTUM & DIRECTION
        self.pb_features = [
            'PB2_T10', 'PB2_T11', 'PB2_T12',
            'PB5_T10', 'PB5_T11', 'PB5_T12', 
            'PB6_T11', 'PB6_T12',
            'PB7_T11', 'PB7_T12',
            'PB3_T7', 'PB3_T10', 'PB3_T8'
        ]
        
        # PV Features (0.70 corr) - VOLUME CONFIRMATION
        self.pv_features = [
            'PV3_B3_T12', 'PV3_B4_T12', 'PV3_B5_T12'
        ]
        
        # VB Features (0.30 corr) - VOLATILITY REGIME
        self.vb_features = [
            'VB4_T11', 'VB4_T12',
            'VB5_T11', 'VB5_T12'
        ]
        
        # V Features (0.10 corr) - VOLUME SURGE
        self.v_features = [
            'V5', 'V2_T8', 'V1_T4', 'V8_T9_T12', 'V8_T7_T11'
        ]
        
        # BB Weights (negative = mean reversion)
        self.bb_weights = {
            'BB1_T10': -0.22, 'BB1_T11': -0.22, 'BB1_T12': -0.18,
            'BB4_T10': -0.18, 'BB4_T11': -0.15, 'BB4_T12': -0.03,
            'BB5_T10': -0.01, 'BB5_T11': -0.005, 'BB5_T12': -0.005,
            'PB10_T11': -0.1, 'PB11_T11': 0.1
        }
        
        # PB Weights (positive = momentum)
        self.pb_weights = {
            'PB2_T10': -0.05, 'PB2_T11': -0.05, 'PB2_T12': -0.05,
            'PB5_T10': -0.07, 'PB5_T11': -0.07, 'PB5_T12': -0.07, 
            'PB6_T11': -0.08, 'PB6_T12': -0.08,
            'PB7_T11': -0.06, 'PB7_T12': -0.06,
            'PB3_T7': -0.05, 'PB3_T10': -0.05, 'PB3_T8': -0.05
        }
        
        self.pv_weights = {
            'PV3_B3_T6': 0.33, 
            'PV3_B4_T6': 0.40, 
            'PV3_B5_T6' : 0.33
        }
        
        self.vb_weights = {
            'VB4_T11': 0.35, 'VB4_T12': 0.35,
            'VB5_T11': 0.25, 'VB5_T12': 0.3
        }
        self.v_weights = {
            'V5': 0.40, 'V2_T8': 0.3, 'V1_T4': 0.3, 'V8_T9_T12': 0.3, 'V8_T7_T11': 0.3
        }
        
        # Parameters
        self.base_threshold_strong = 30
        self.base_threshold_medium = 50
        self.base_threshold_weak = 70
        self.vb_extreme_percentile = 90
        self.vb_favorable_percentile = 40
        self.vb_quiet_percentile = 20     # <-- OPTIMIZATION 1
        self.volume_surge_threshold = 1.15
        self.stop_loss_strong = 0.0025
        self.stop_loss_medium = 0.0020
        self.stop_loss_weak = 0.0015
        self.take_profit_strong = 0.0050
        self.take_profit_medium = 0.0035
        self.take_profit_weak = 0.0025
        self.trailing_stop_pct = 0.0020
        self.min_trade_duration = 15
        self.max_trade_duration = 600
        self.min_hold_time = 30
        
        self.plot_output_dir = plot_output_dir
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # ==================== NUMBA-OPTIMIZED FUNCTIONS ====================
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def calculate_rolling_percentile(arr, window, percentile):
        """Optimized rolling percentile"""
        n = len(arr)
        result = np.zeros(n, dtype=np.float32)
        for i in range(window, n):
            result[i] = np.percentile(arr[i-window:i], percentile)
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def calculate_rolling_mean(arr, window):
        """Parallel rolling mean"""
        n = len(arr)
        result = np.zeros(n, dtype=np.float32)
        for i in prange(window, n):
            result[i] = np.mean(arr[i-window:i])
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def normalize_signal(signal, window=300):
        """Parallel z-score normalization"""
        n = len(signal)
        normalized = np.zeros(n, dtype=np.float32)
        
        for i in prange(window, n):
            window_data = signal[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 1e-8:
                normalized[i] = (signal[i] - mean) / std
        
        return normalized
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def generate_balanced_signals(bb_signal, pb_signal, pv_signal, vb_filter, v_surge,
                                  prices, bb_norm, pb_norm, combined_norm,
                                  strong_pct, med_pct, weak_pct,
                                  vb_extreme, vb_favorable, vb_quiet, vol_surge_thresh): # <-- Param added
        """Generate binary trading signals"""
        n = len(bb_signal)
        signals = np.zeros(n, dtype=np.int32)
        signal_quality = np.zeros(n, dtype=np.int32)
        lookback_thresh = 600
        
        for i in range(300, n):
            # Adaptive thresholds
            if i >= lookback_thresh:
                recent_signals = np.abs(combined_norm[i-lookback_thresh:i])
                thresh_strong = np.percentile(recent_signals, strong_pct)
                thresh_medium = np.percentile(recent_signals, med_pct)
                thresh_weak = np.percentile(recent_signals, weak_pct)
            else:
                thresh_strong = 1.5
                thresh_medium = 1.0
                thresh_weak = 0.6
            
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            combined_z = combined_norm[i]
            abs_signal = abs(combined_z)
            
            # Context
            is_extreme_vol = vb_filter[i] > vb_extreme[i]
            is_favorable_vol = vb_filter[i] < vb_favorable[i]
            is_quiet_vol = vb_filter[i] < vb_quiet[i]      # <-- OPTIMIZATION 1
            has_vol_surge = v_surge[i] > vol_surge_thresh
            
            pv_confirms_long = pv_signal[i] > 0.0001
            pv_confirms_short = pv_signal[i] < -0.0001
            
            # <-- OPTIMIZATION 1 FILTER LOGIC
            # Stop if volatility is EITHER too high OR too low
            if is_extreme_vol or is_quiet_vol:
                continue
            # <-- END OPTIMIZATION
            
            # ======================== LOGIC FIX ========================
            #
            # Weights are negative, so a positive z-score means the
            # underlying (now-negative) signal is positive -> LONG
            #
            # Long signals
            if combined_z > 0: # <-- CORRECTED LOGIC
                bb_pb_agree = (bb_z > 0) and (pb_z > 0)
                
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = 1
                    signal_quality[i] = 1
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_long:
                        signals[i] = 1
                        signal_quality[i] = 2
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree:
                        confirmations += 1
                    if pv_confirms_long:
                        confirmations += 1
                    if has_vol_surge:
                        confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = 1
                        signal_quality[i] = 3
            
            # Short signals
            elif combined_z < 0: # <-- CORRECTED LOGIC
                bb_pb_agree = (bb_z < 0) and (pb_z < 0)
                
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = -1
                    signal_quality[i] = 1
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_short:
                        signals[i] = -1
                        signal_quality[i] = 2
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree:
                        confirmations += 1
                    if pv_confirms_short:
                        confirmations += 1
                    if has_vol_surge:
                        confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = -1
                        signal_quality[i] = 3
            # ====================== END LOGIC FIX ======================

        return signals, signal_quality
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def manage_positions(signals, signal_quality, prices, timestamps,
                         min_duration, min_hold, max_duration,
                         sl_strong, sl_med, sl_weak,
                         tp_strong, tp_med, tp_weak, trailing_stop):
        """Manage binary positions with risk management"""
        n = len(signals)
        positions = np.zeros(n, dtype=np.int32)
        
        current_pos = 0
        entry_price = 0.0
        entry_time = 0.0
        entry_quality = 0
        highest_price = 0.0
        lowest_price = 999999.0
        max_favorable = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            # Entry logic
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_quality = signal_quality[i]
                    highest_price = prices[i]
                    lowest_price = prices[i]
                    max_favorable = 0.0
            
            # Exit logic
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pct = (prices[i] - entry_price) / entry_price * direction
                
                # Track extremes
                if current_pos > 0:
                    highest_price = max(highest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = highest_price * (1.0 - trailing_stop)
                else:
                    lowest_price = min(lowest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = lowest_price * (1.0 + trailing_stop)
                
                # Quality-based parameters
                if entry_quality == 1:
                    stop_loss = sl_strong
                    take_profit = tp_strong
                elif entry_quality == 2:
                    stop_loss = sl_med
                    take_profit = tp_med
                else:
                    stop_loss = sl_weak
                    take_profit = tp_weak
                
                should_exit = False
                
                # Min duration check
                if time_in_trade < min_duration:
                    pass
                # Stop loss
                elif pnl_pct <= -stop_loss:
                    should_exit = True
                # Take profit
                elif pnl_pct >= take_profit:
                    should_exit = True
                # Trailing stop
                elif max_favorable >= take_profit * 0.4:
                    if current_pos > 0 and prices[i] < trailing_level:
                        should_exit = True
                    elif current_pos < 0 and prices[i] > trailing_level:
                        should_exit = True
                # Max duration
                elif time_in_trade >= max_duration:
                    should_exit = True
                # Signal reversal
                elif time_in_trade >= min_hold:
                    if current_pos > 0 and signals[i] < 0:
                        should_exit = True
                    elif current_pos < 0 and signals[i] > 0:
                        should_exit = True
                # Weak signal fade
                elif entry_quality == 3 and time_in_trade >= min_duration * 2:
                    if signals[i] == 0:
                        should_exit = True
                
                if should_exit:
                    current_pos = 0
                    entry_price = 0.0
                    entry_time = 0.0
                    entry_quality = 0
                    highest_price = 0.0
                    lowest_price = 999999.0
                    max_favorable = 0.0
            
            positions[i] = current_pos
        
        # Force close at EOD
        positions[-1] = 0
        
        return positions
    
    def _calculate_weighted_signal(self, df, features, weights):
        """Calculate weighted signal from features"""
        signal = np.zeros(len(df), dtype=np.float32)
        total_weight = 0.0
        
        for feat in features:
            if feat in df.columns and feat in weights:
                signal += df[feat].values.astype(np.float32) * weights[feat]
                total_weight += weights[feat]
        
        if total_weight > 0:
            signal /= total_weight
        
        return signal
    
    def process_day(self, day_num, data_folder='/data/quant14/EBX/'):
        """
        Process single day
        Runs in parallel via ProcessPoolExecutor
        """
        filename = f"day{day_num}.csv"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Check features
            all_features = (self.bb_features + self.pb_features + self.pv_features +
                            self.vb_features + self.v_features)
            missing = [f for f in all_features if f not in df.columns]
            
            if len(missing) > len(all_features) * 0.3:
                return None
            
            # Fill missing
            for feat in all_features:
                if feat in df.columns:
                    df[feat] = df[feat].fillna(method='ffill').fillna(0)
                else:
                    df[feat] = 0
            
            df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
            
            prices = df['Price'].values.astype(np.float32)
            timestamps = df['timestamp_sec'].values.astype(np.float32)
            
            # Calculate weighted signals
            bb_signal = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights)
            pb_signal = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights)
            pv_signal = self._calculate_weighted_signal(df, self.pv_features, self.pv_weights)
            vb_filter = self._calculate_weighted_signal(df, self.vb_features, self.vb_weights)
            v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
            
            combined_signal = 0.50 * bb_signal + 0.35 * pb_signal + 0.15 * pv_signal
            
            # Normalize signals (Numba optimized)
            bb_normalized = self.normalize_signal(bb_signal, window=300)
            pb_normalized = self.normalize_signal(pb_signal, window=300)
            combined_normalized = self.normalize_signal(combined_signal, window=300)
            
            # Volatility regime (Numba optimized)
            vb_extreme = self.calculate_rolling_percentile(vb_filter, 600, self.vb_extreme_percentile)
            vb_favorable = self.calculate_rolling_percentile(vb_filter, 600, self.vb_favorable_percentile)
            vb_quiet = self.calculate_rolling_percentile(vb_filter, 600, self.vb_quiet_percentile) # <-- OPTIMIZATION 1
            
            # Volume surge (Numba optimized)
            v_mean = self.calculate_rolling_mean(v_signal, 600)
            v_surge = v_signal / (v_mean + 1e-8)
            
            # Generate signals (Numba optimized)
            signals, signal_quality = self.generate_balanced_signals(
                bb_signal, pb_signal, pv_signal, vb_filter, v_surge, prices,
                bb_normalized, pb_normalized, combined_normalized,
                self.base_threshold_strong, self.base_threshold_medium, self.base_threshold_weak,
                vb_extreme, vb_favorable, vb_quiet, self.volume_surge_threshold # <-- OPTIMIZATION 1
            )
            
            # Manage positions (Numba optimized)
            positions = self.manage_positions(
                signals, signal_quality, prices, timestamps,
                self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
                self.stop_loss_strong, self.stop_loss_medium, self.stop_loss_weak,
                self.take_profit_strong, self.take_profit_medium, self.take_profit_weak,
                self.trailing_stop_pct
            )
            
            # Statistics
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
            total_signals = long_signals + short_signals
            
            print(f"Day {day_num:3d}: Signals: {total_signals:4d} (L:{long_signals:4d}, S:{short_signals:4d})")
            
            # Return results
            result_df = pd.DataFrame({
                'Time': df['Time'],
                'Signal': positions.astype(np.int32), 
                'Price': prices,
                'Day': day_num                          
            })
            
            return result_df
            
        except Exception as e:
            print(f"Day {day_num}: Error - {e}")
            return None
    
    def run_strategy(self, num_days=510, data_folder='/data/quant14/EBX/', max_workers=25):
        """
        Run strategy with parallel processing
        
        Uses ProcessPoolExecutor for true parallel execution
        FIXED: Maintains day-wise chronological order in output
        """
        print("="*80)
        print("OPTIMIZED PARALLEL STRATEGY - DAY-ORDERED OUTPUT")
        print("="*80)
        print(f"✓ CPU Workers: {max_workers} (ProcessPoolExecutor)")
        print(f"✓ Numba JIT: ENABLED (parallel=True, fastmath=True)")
        print(f"✓ Binary positions: {-1, 0, 1}")
        print(f"✓ Min trade duration: 15 seconds")
        print(f"✓ Output: Day-wise chronological order (day0 → day509)")
        print("="*80 + "\n")
        
        # Create partial function
        process_func = partial(self.process_day, data_folder=data_folder)
        
        # Dictionary to store results with day number as key
        day_results = {}
        
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all days
            future_to_day = {executor.submit(process_func, day): day for day in range(num_days)}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_day):
                day = future_to_day[future]
                try:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        day_results[day] = result  # Store with day number as key
                except Exception as e:
                    print(f"Day {day}: Error - {e}")
        
        if len(day_results) > 0:
            print(f"\n✓ Processed {len(day_results)} days successfully")
            print("Combining results in day-wise order...")
            
            # Sort by day number and concatenate in order
            sorted_days = sorted(day_results.keys())
            all_results = [day_results[day] for day in sorted_days]
            
            portfolio_weights = pd.concat(all_results, ignore_index=True)
            
            # Verify chronological order
            print(f"\n✓ Verifying chronological order...")
            print(f"  First timestamp: {portfolio_weights['Time'].iloc[0]}")
            print(f"  Last timestamp: {portfolio_weights['Time'].iloc[-1]}")
            
            # Check for any time reversals (shouldn't happen now)
            time_diffs = portfolio_weights['Time'].diff()
            negative_diffs = (time_diffs < pd.Timedelta(0)).sum()
            if negative_diffs > 0:
                print(f"  ⚠ Warning: {negative_diffs} time reversals detected (day boundaries)")
            else:
                print(f"  ✓ Perfect chronological order maintained")
            
            # Remove Day column before saving (backtester doesn't need it)
            portfolio_weights_output = portfolio_weights[['Time', 'Signal', 'Price']].copy()
            
            # Save to CSV
            portfolio_weights_output.to_csv('portfolio_weights.csv', index=False)
            
            print(f"\n✓ Saved portfolio_weights.csv ({len(portfolio_weights_output):,} rows)")
            
            # Statistics by day
            days_processed = portfolio_weights['Day'].nunique()
            signals_per_day = portfolio_weights.groupby('Day')['Signal'].count()
            
            total_signals = len(portfolio_weights_output) # Use total signals from output file
            total_long = (portfolio_weights_output['Signal'] > 0).sum()
            total_short = (portfolio_weights_output['Signal'] < 0).sum()
            
            print(f"\n✅ SUMMARY:")
            print(f"  Days Processed: {days_processed} (day{min(sorted_days)} to day{max(sorted_days)})")
            print(f"  Total Signals in CSV: {total_signals:,}")
            
            # Avoid division by zero if total_signals is 0
            if total_signals > 0:
                print(f"  Long Signals: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short Signals: {total_short:,} ({total_short/total_signals*100:.1f}%)")
            else:
                print("  Long Signals: 0 (0.0%)")
                print("  Short Signals: 0 (0.0%)")
                
            print(f"  Average Signals/Day: {signals_per_day.mean():.1f}")
            print(f"  Min Signals/Day: {signals_per_day.min()}")
            print(f"  Max Signals/Day: {signals_per_day.max()}")
            
            # Show first few days for verification
            print(f"\n  First 5 days signal counts:")
            for day in sorted_days[:5]:
                day_count = len(day_results[day])
                print(f"    Day {day}: {day_count} signals")
            
            print("="*80)
            
            gc.collect()
            
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = OptimizedParallelStrategy()
    
    print("\n" + "="*80)
    print("STARTING PARALLEL PROCESSING")
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✓ Strategy complete!")
        print("✓ portfolio_weights.csv ready (Time, Signal, Price)")
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")