import pandas as pd
import numpy as np
import warnings
import gc
import os
from numba import jit
from itertools import product
from datetime import datetime
import time
warnings.filterwarnings('ignore')

class WFOBalancedHybridStrategy:
    """
    Walk-Forward Optimized Trading Strategy - SIGNAL GENERATOR ONLY
    Outputs: test_signal.csv (Time, Signal, Price)
    Performance evaluation handled by external backtester
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # Feature groups (unchanged from original)
        self.bb_features = [
            'BB1_T10', 'BB1_T11', 'BB1_T12',
            'BB4_T10', 'BB4_T11', 'BB4_T12',
            'BB5_T10', 'BB5_T11', 'BB5_T12'
        ]
        
        self.pb_features = [
            'PB2_T10', 'PB2_T11', 'PB2_T12',
            'PB5_T10', 'PB5_T11', 'PB5_T12',
            'PB6_T10', 'PB6_T11', 'PB6_T12',
            'PB7_T11', 'PB7_T12',
            'PB8_T11', 'PB8_T12',
            'PB12_T11', 'PB12_T12',
            'PB13_T10', 'PB13_T11',
            'PB16_T11', 'PB16_T12',
            'PB17_T11', 'PB17_T12'
        ]
        
        self.pv_features = [
            'PV1_T10', 'PV1_T11', 'PV1_T12',
            'PV3_T10', 'PV3_T11', 'PV3_T12'
        ]
        
        self.vb_features = [
            'VB4_T11', 'VB4_T12',
            'VB5_T11', 'VB5_T12'
        ]
        
        self.v_features = [
            'V5_T10', 'V5_T11', 'V5_T12'
        ]
        
        # Feature weights (unchanged)
        self.bb_weights = {
            'BB1_T10': 0.22, 'BB1_T11': 0.22, 'BB1_T12': 0.18,
            'BB4_T10': 0.18, 'BB4_T11': 0.15, 'BB4_T12': 0.03,
            'BB5_T10': 0.01, 'BB5_T11': 0.005, 'BB5_T12': 0.005
        }
        
        self.pb_weights = {
            'PB2_T10': 0.08, 'PB2_T11': 0.09, 'PB2_T12': 0.08,
            'PB5_T10': 0.07, 'PB5_T11': 0.08, 'PB5_T12': 0.08,
            'PB6_T10': 0.06, 'PB6_T11': 0.07, 'PB6_T12': 0.06,
            'PB7_T11': 0.06, 'PB7_T12': 0.05,
            'PB8_T11': 0.05, 'PB8_T12': 0.04,
            'PB12_T11': 0.04, 'PB12_T12': 0.03,
            'PB13_T10': 0.03, 'PB13_T11': 0.03,
            'PB16_T11': 0.03, 'PB16_T12': 0.03,
            'PB17_T11': 0.02, 'PB17_T12': 0.02
        }
        
        self.pv_weights = {
            'PV1_T10': 0.40, 'PV1_T11': 0.35, 'PV1_T12': 0.15,
            'PV3_T10': 0.03, 'PV3_T11': 0.04, 'PV3_T12': 0.03
        }
        
        self.vb_weights = {
            'VB4_T11': 0.45, 'VB4_T12': 0.35,
            'VB5_T11': 0.10, 'VB5_T12': 0.10
        }
        
        self.v_weights = {
            'V5_T10': 0.40, 'V5_T11': 0.35, 'V5_T12': 0.25
        }
        
        # DEFAULT parameters (will be overridden by WFO)
        self.base_threshold_strong = 0.30
        self.base_threshold_medium = 0.50
        self.base_threshold_weak = 0.70
        
        self.vb_extreme_percentile = 90
        self.vb_favorable_percentile = 40
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
    
    def set_parameters(self, params):
        """Update strategy parameters from dict"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @staticmethod
    @jit(nopython=True)
    def calculate_rolling_percentile(arr, window, percentile):
        n = len(arr)
        result = np.zeros(n)
        for i in range(window, n):
            window_data = arr[i-window:i]
            result[i] = np.percentile(window_data, percentile)
        return result
    
    @staticmethod
    @jit(nopython=True)
    def calculate_rolling_mean(arr, window):
        n = len(arr)
        result = np.zeros(n)
        for i in range(window, n):
            result[i] = np.mean(arr[i-window:i])
        return result
    
    @staticmethod
    @jit(nopython=True)
    def calculate_rolling_std(arr, window):
        n = len(arr)
        result = np.zeros(n)
        for i in range(window, n):
            result[i] = np.std(arr[i-window:i])
        return result
    
    @staticmethod
    @jit(nopython=True)
    def normalize_signal(signal, window=300):
        n = len(signal)
        normalized = np.zeros(n)
        for i in range(window, n):
            window_data = signal[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 1e-8:
                normalized[i] = (signal[i] - mean) / std
            else:
                normalized[i] = 0.0
        return normalized
    
    @staticmethod
    @jit(nopython=True)
    def generate_balanced_signals(bb_signal, pb_signal, pv_signal, vb_filter, v_surge,
                                  prices, bb_norm, pb_norm, combined_norm,
                                  strong_pct, med_pct, weak_pct,
                                  vb_extreme, vb_favorable, vol_surge_thresh):
        n = len(bb_signal)
        signals = np.zeros(n, dtype=np.int32)
        trade_tiers = np.zeros(n, dtype=np.int32)
        
        lookback_thresh = 600
        
        for i in range(300, n):
            if i >= lookback_thresh:
                recent_signals = np.abs(combined_norm[i-lookback_thresh:i])
                thresh_strong = np.percentile(recent_signals, strong_pct * 100) # Convert fraction to percentile
                thresh_medium = np.percentile(recent_signals, med_pct * 100)
                thresh_weak = np.percentile(recent_signals, weak_pct * 100)
            else:
                thresh_strong = 1.5
                thresh_medium = 1.0
                thresh_weak = 0.6
                
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            combined_z = combined_norm[i]
            abs_signal = abs(combined_z)
            
            is_extreme_vol = vb_filter[i] > vb_extreme[i]
            is_favorable_vol = vb_filter[i] < vb_favorable[i]
            has_vol_surge = v_surge[i] > vol_surge_thresh
            
            pv_confirms_long = pv_signal[i] > 0.0001
            pv_confirms_short = pv_signal[i] < -0.0001
            
            if is_extreme_vol:
                continue
                
            if combined_z > 0:
                bb_pb_agree = (bb_z > 0) and (pb_z > 0)
                
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = 1
                    trade_tiers[i] = 1
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_long:
                        signals[i] = 1
                        trade_tiers[i] = 2
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree: confirmations += 1
                    if pv_confirms_long: confirmations += 1
                    if has_vol_surge: confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = 1
                        trade_tiers[i] = 3
            
            elif combined_z < 0:
                bb_pb_agree = (bb_z < 0) and (pb_z < 0)
                
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = -1
                    trade_tiers[i] = 1
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_short:
                        signals[i] = -1
                        trade_tiers[i] = 2
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree: confirmations += 1
                    if pv_confirms_short: confirmations += 1
                    if has_vol_surge: confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = -1
                        trade_tiers[i] = 3
        
        return signals, trade_tiers
    
    @staticmethod
    @jit(nopython=True)
    def manage_positions_fast(signals, trade_tiers, prices, timestamps,
                              min_duration, min_hold, max_duration,
                              sl_strong, sl_med, sl_weak,
                              tp_strong, tp_med, tp_weak, trailing_stop):
        n = len(signals)
        positions = np.zeros(n, dtype=np.int32)
        
        current_pos = 0
        entry_price = 0.0
        entry_time = 0.0
        entry_tier = 0
        highest_price = 0.0
        lowest_price = 999999.0
        max_favorable = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_tier = trade_tiers[i]
                    highest_price = prices[i]
                    lowest_price = prices[i]
                    max_favorable = 0.0
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pct = (prices[i] - entry_price) / entry_price * direction
                
                if current_pos > 0:
                    highest_price = max(highest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = highest_price * (1 - trailing_stop)
                else:
                    lowest_price = min(lowest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = lowest_price * (1 + trailing_stop)
                
                if entry_tier == 1:
                    stop_loss, take_profit = sl_strong, tp_strong
                elif entry_tier == 2:
                    stop_loss, take_profit = sl_med, tp_med
                else:
                    stop_loss, take_profit = sl_weak, tp_weak
                
                should_exit = False
                
                # Check for exit conditions ONLY after min_duration
                if time_in_trade >= min_duration:
                    if pnl_pct <= -stop_loss: should_exit = True
                    elif pnl_pct >= take_profit: should_exit = True
                    # Activate trailing stop only after PnL has reached 40% of TP
                    elif max_favorable >= take_profit * 0.4:
                        if (current_pos > 0 and prices[i] < trailing_level) or \
                           (current_pos < 0 and prices[i] > trailing_level):
                            should_exit = True
                    # Check for reversal signal only after min_hold time
                    elif time_in_trade >= min_hold:
                        if (current_pos > 0 and signals[i] < 0) or \
                           (current_pos < 0 and signals[i] > 0):
                            should_exit = True

                # Time-based exits (hard stop)
                if time_in_trade >= max_duration: should_exit = True
                
                # Special exit for weak (tier 3) signals
                elif entry_tier == 3 and time_in_trade >= min_duration * 2:
                    if signals[i] == 0: should_exit = True
                            
                if should_exit:
                    current_pos = 0
                    entry_price = 0.0
                    entry_time = 0.0
                    entry_tier = 0
                    highest_price = 0.0
                    lowest_price = 999999.0
                    max_favorable = 0.0
            
            positions[i] = current_pos
        
        # EOD square-off
        positions[-1] = 0
        
        return positions
    
    @staticmethod
    @jit(nopython=True)
    def calculate_simple_fitness(positions, prices):
        """
        Simple fitness metric for optimization: total position changes weighted by magnitude
        This avoids full PnL calculation but still guides optimization
        """
        n = len(positions)
        total_activity = 0.0
        profitable_moves = 0
        total_moves = 0
        
        for i in range(1, n):
            if positions[i-1] != 0:
                # Calculate return for holding the position from t-1 to t
                ret = (prices[i] - prices[i-1]) / prices[i-1] * positions[i-1]
                if ret > 0:
                    profitable_moves += 1
                    total_activity += ret
                else:
                    total_activity += ret
                total_moves += 1
        
        if total_moves == 0:
            return 0.0
        
        # Win rate as simple fitness
        win_rate = profitable_moves / total_moves if total_moves > 0 else 0
        avg_return = total_activity / total_moves if total_moves > 0 else 0
        
        # Combined fitness: favor higher win rate and positive returns
        # (1 + avg_return) scales returns to be positive (assuming avg_return > -1)
        fitness = win_rate * 0.6 + (1 + avg_return) * 0.4
        
        return fitness
    
    def process_day_for_optimization(self, day_num, data_folder='/data/quant14/EBX/', verbose=False):
        """Lightweight day processing for optimization (no saving)"""
        filename = f"day{day_num}.csv"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                return None
        except pd.errors.EmptyDataError:
            return None

        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        all_features = (self.bb_features + self.pb_features + self.pv_features +
                        self.vb_features + self.v_features)
        missing = [f for f in all_features if f not in df.columns]
        
        if len(missing) > len(all_features) * 0.3:
            return None
        
        for feat in all_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(method='ffill').fillna(0)
            else:
                df[feat] = 0
        
        df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
        
        prices = df['Price'].values.astype(np.float32)
        timestamps = df['timestamp_sec'].values.astype(np.float32)
        times = df['Time'].values
        
        bb_signal = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights)
        pb_signal = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights)
        pv_signal = self._calculate_weighted_signal(df, self.pv_features, self.pv_weights)
        vb_filter = self._calculate_weighted_signal(df, self.vb_features, self.vb_weights)
        v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
        
        combined_signal = 0.50 * bb_signal + 0.35 * pb_signal + 0.15 * pv_signal
        
        bb_normalized = self.normalize_signal(bb_signal, window=300)
        pb_normalized = self.normalize_signal(pb_signal, window=300)
        combined_normalized = self.normalize_signal(combined_signal, window=300)
        
        vb_extreme = self.calculate_rolling_percentile(vb_filter, 600, self.vb_extreme_percentile)
        vb_favorable = self.calculate_rolling_percentile(vb_filter, 600, self.vb_favorable_percentile)
        
        v_mean = self.calculate_rolling_mean(v_signal, 600)
        v_surge = v_signal / (v_mean + 1e-8)
        
        signals, trade_tiers = self.generate_balanced_signals(
            bb_signal, pb_signal, pv_signal, vb_filter, v_surge, prices,
            bb_normalized, pb_normalized, combined_normalized,
            self.base_threshold_strong, self.base_threshold_medium, self.base_threshold_weak,
            vb_extreme, vb_favorable, self.volume_surge_threshold
        )
        
        positions = self.manage_positions_fast(
            signals, trade_tiers, prices, timestamps,
            self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
            self.stop_loss_strong, self.stop_loss_medium, self.stop_loss_weak,
            self.take_profit_strong, self.take_profit_medium, self.take_profit_weak,
            self.trailing_stop_pct
        )
        
        if verbose:
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
            print(f"       Day {day_num:3d}: {long_signals + short_signals} signals (L:{long_signals} S:{short_signals})")
        
        return {
            'times': times,
            'prices': prices,
            'positions': positions,
            'signals': signals
        }
    
    def _calculate_weighted_signal(self, df, features, weights):
        signal = np.zeros(len(df), dtype=np.float32)
        total_weight = 0
        for feat in features:
            if feat in df.columns and feat in weights:
                signal += df[feat].values * weights[feat]
                total_weight += weights[feat]
        if total_weight > 0:
            signal /= total_weight
        return signal
    
    def optimize_parameters(self, day_range, data_folder='/data/quant14/EBX/'):
        """
        Optimize parameters on a specific range of days
        Returns best parameters based on simple fitness metric
        """
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║        OPTIMIZATION PHASE - PARAMETER SEARCH           ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        
        # Reduced parameter grid for efficiency
        # NOTE: This is the grid I designed.
        param_grid = {
            'base_threshold_strong': [0.25, 0.35],
            'base_threshold_medium': [0.45, 0.55],
            'stop_loss_strong': [0.0020, 0.0030],
            'take_profit_strong': [0.0040, 0.0060],
            'min_trade_duration': [15] # Only test the constraint
        }
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        total_combinations = len(combinations)
        print(f"  → Total parameter combinations: {total_combinations}")
        print(f"  → Training days: {len(day_range)}")
        print(f"  → Starting grid search...\n")
        
        best_fitness = -np.inf
        best_params = None
        
        start_time = time.time()
        
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            # --- Auto-scaling related parameters ---
            # Set medium/weak params relative to the strong ones
            params['stop_loss_medium'] = params['stop_loss_strong'] * 0.8
            params['stop_loss_weak'] = params['stop_loss_strong'] * 0.6
            params['take_profit_medium'] = params['take_profit_strong'] * 0.7
            params['take_profit_weak'] = params['take_profit_strong'] * 0.5
            params['base_threshold_weak'] = params['base_threshold_medium'] * 1.4
            
            self.set_parameters(params)
            
            # Test on training days
            all_positions = []
            all_prices = []
            valid_days = 0
            
            for day in day_range:
                result = self.process_day_for_optimization(day, data_folder, verbose=False)
                if result is not None:
                    all_positions.extend(result['positions'])
                    all_prices.extend(result['prices'])
                    valid_days += 1
            
            if len(all_positions) > 100:
                positions_array = np.array(all_positions, dtype=np.int32)
                prices_array = np.array(all_prices, dtype=np.float32)
                
                fitness = self.calculate_simple_fitness(positions_array, prices_array)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = params.copy()
            
            # Progress update every 10 combinations or last one
            if (idx + 1) % 10 == 0 or (idx + 1) == total_combinations:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (total_combinations - idx - 1)
                print(f"    Progress: {idx+1:3d}/{total_combinations} | "
                      f"Best Fitness: {best_fitness:.4f} | "
                      f"ETA: {remaining/60:.1f}min", end='\r')
        
        print() # Newline after progress bar
        total_time = time.time() - start_time
        print(f"\n  ✓ Optimization complete in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"  ✓ Best fitness score: {best_fitness:.4f}")
        
        return best_params, best_fitness
    
    def walk_forward_optimize(self, num_days=510, train_window=120, test_window=30, 
                              data_folder='/data/quant14/EBX/'):
        """
        Walk-Forward Optimization Main Loop
        Generates test_signal.csv with optimized parameters for each period
        """
        print("\n" + "="*80)
        print("   WALK-FORWARD OPTIMIZATION - SIGNAL GENERATOR")
        print("="*80)
        print(f"   Configuration:")
        print(f"     • Total Days: {num_days}")
        print(f"     • Training Window: {train_window} days")
        print(f"     • Testing Window: {test_window} days")
        print(f"     • Roll Forward: {test_window} days")
        print("="*80 + "\n")
        
        wfo_results = []
        all_test_signals = []
        
        start_day = 0
        wfo_period = 1
        
        total_periods = (num_days - train_window) // test_window
        if total_periods <= 0:
            print("  Error: Not enough days for even one WFO period. Check num_days, train_window.")
            return
            
        print(f"   → Estimated WFO Periods: {total_periods}\n")
        
        while start_day + train_window + test_window <= num_days:
            print(f"\n{'█'*80}")
            print(f"   WFO PERIOD {wfo_period}/{total_periods}")
            print(f"{'█'*80}")
            
            train_start = start_day
            train_end = start_day + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            print(f"\n  ┌─────────────────────────────────────────────────────┐")
            print(f"  │  Training Days: {train_start:3d} → {train_end-1:3d} (Window: {train_window} days)  │")
            print(f"  │  Testing Days:  {test_start:3d} → {test_end-1:3d} (Window: {test_window} days)   │")
            print(f"  └─────────────────────────────────────────────────────┘")
            
            # OPTIMIZATION PHASE
            train_range = range(train_start, train_end)
            best_params, train_fitness = self.optimize_parameters(train_range, data_folder)
            
            if best_params is None:
                print(f"\n  ! WARNING: No suitable parameters found for period {wfo_period}. Skipping.")
                start_day += test_window
                wfo_period += 1
                gc.collect()
                continue

            print(f"\n  ╔══════════════════════════════════════════════════════════╗")
            print(f"  ║                OPTIMAL PARAMETERS FOUND                ║")
            print(f"  ╚══════════════════════════════════════════════════════════╝")
            for key, value in best_params.items():
                if key in self.optimize_parameters.__defaults__[0]: # Only print optimized keys
                    print(f"    • {key:25s}: {value}")
            
            # TESTING PHASE - Generate signals with best parameters
            print(f"\n  ╔══════════════════════════════════════════════════════════╗")
            print(f"  ║         TESTING PHASE - GENERATING SIGNALS           ║")
            print(f"  ╚══════════════════════════════════════════════════════════╝")
            
            self.set_parameters(best_params)
            
            test_period_signals = []
            test_days_processed = 0
            
            for day in range(test_start, test_end):
                result = self.process_day_for_optimization(day, data_folder, verbose=True)
                if result is not None:
                    # Create signal dataframe for this day
                    day_df = pd.DataFrame({
                        'Time': result['times'],
                        'Signal': result['signals'],
                        'Price': result['prices']
                    })
                    test_period_signals.append(day_df)
                    test_days_processed += 1
            
            if len(test_period_signals) > 0:
                period_signals_df = pd.concat(test_period_signals, ignore_index=True)
                all_test_signals.append(period_signals_df)
                
                total_signals = (period_signals_df['Signal'] != 0).sum()
                long_signals = (period_signals_df['Signal'] == 1).sum()
                short_signals = (period_signals_df['Signal'] == -1).sum()
                
                print(f"\n  ✓ Period {wfo_period} Complete:")
                print(f"      • Days Processed: {test_days_processed}/{test_window}")
                if total_signals > 0:
                    print(f"      • Total Signals: {total_signals:,}")
                    print(f"      • Long: {long_signals:,} ({long_signals/total_signals*100:.1f}%)")
                    print(f"      • Short: {short_signals:,} ({short_signals/total_signals*100:.1f}%)")
                else:
                    print(f"      • Total Signals: 0")

                
                wfo_results.append({
                    'period': wfo_period,
                    'train_range': f"{train_start}-{train_end-1}",
                    'test_range': f"{test_start}-{test_end-1}",
                    'best_params': best_params,
                    'train_fitness': train_fitness,
                    'test_signals': total_signals
                })
            
            start_day += test_window
            wfo_period += 1
            gc.collect()
        
        # SAVE FINAL RESULTS
        print(f"\n\n{'='*80}")
        print(f"   WALK-FORWARD OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        
        if len(all_test_signals) > 0:
            final_signals = pd.concat(all_test_signals, ignore_index=True)
            final_signals.to_csv('test_signal.csv', index=False)
            
            print(f"\n  ✓ Saved: test_signal.csv")
            print(f"      • Total Rows: {len(final_signals):,}")
            print(f"      • Columns: Time, Signal, Price")
            print(f"      • WFO Periods: {len(wfo_results)}")
            
            total_signals = (final_signals['Signal'] != 0).sum()
            long_signals = (final_signals['Signal'] == 1).sum()
            short_signals = (final_signals['Signal'] == -1).sum()
            
            print(f"\n  📊 Final Signal Statistics:")
            print(f"      • Total Signals: {total_signals:,}")
            # --- THIS IS THE COMPLETED SECTION ---
            if total_signals > 0:
                print(f"      • Long Signals: {long_signals:,} ({long_signals/total_signals*100:.1f}%)")
                print(f"      • Short Signals: {short_signals:,} ({short_signals/total_signals*100:.1f}%)")
            else:
                print("      • No signals were generated in the final output.")
            
            print(f"\n  📑 WFO Period Summary:")
            for res in wfo_results:
                print(f"    • P{res['period']:<2} (Train: {res['train_range']:<7} | Test: {res['test_range']:<7}) "
                      f"-> Fit: {res['train_fitness']:.3f}, Signals: {res['test_signals']:<5,}")
                
            print(f"\n  Next step: Run your external backtester on 'test_signal.csv' to get PnL and Sharpe.")
            
        else:
            print(f"\n  ! No signals were generated during the entire WFO process.")
        
        print("="*80)
        return final_signals, wfo_results


# --- Main execution block to run the WFO ---
# --- Main execution block to run the WFO ---
if __name__ == "__main__":
    
    # --- WFO Configuration ---
    TOTAL_DAYS = 510
    TRAIN_WINDOW_DAYS = 120  # Use 120 days of data to find best params
    TEST_WINDOW_DAYS = 30    # Apply those params to the next 30 days
    
    DATA_FOLDER = '/data/quant14/EBX/' # As specified in your code
    PLOT_DIR = 'daily_plots'
    
    print(f"Starting WFO Process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data folder: {DATA_FOLDER}") # <-- This is the corrected line
    
    try:
        # 1. Instantiate the strategy class
        strategy = WFOBalancedHybridStrategy(plot_output_dir=PLOT_DIR)
        
        # 2. Run the full Walk-Forward Optimization
        final_signals, wfo_summary = strategy.walk_forward_optimize(
            num_days=TOTAL_DAYS,
            train_window=TRAIN_WINDOW_DAYS,
            test_window=TEST_WINDOW_DAYS,
            data_folder=DATA_FOLDER
        )
        
        print(f"\nProcess finished successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Data folder not found: {DATA_FOLDER}")
        print("Please check the DATA_FOLDER path and ensure the dayX.csv files are present.")
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(e)
        import traceback
        traceback.print_exc()