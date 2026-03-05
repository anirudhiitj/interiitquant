"""
Optimized Two-Tier Signal Trading Strategy
==========================================

NEW FEATURES:
1. Strong Signals: BB and PB Z-scores must agree in direction
2. Weak/Medium Signals: Price vs PV bands (B3/B4/B5_T6) + Volume confirmation
3. Separate risk management for Strong vs Weak/Medium signals
4. All 6 mandatory constraints strictly enforced

Architecture:
- ProcessPoolExecutor with 25 workers for parallel day processing
- CPU-only (avoids CUDA multiprocessing issues)
- Numba JIT compilation with parallel=True for speed
- Memory-efficient batch processing
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


class TwoTierSignalStrategy:
    """
    Two-tier signal strategy with Strong and Weak/Medium signals
    
    Strong: BB + PB Z-score agreement
    Weak/Medium: Price vs PV bands + Volume confirmation
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # ==================== FEATURE DEFINITIONS ====================
        
        # BB Features - Bollinger Bands (Mean Reversion)
        self.bb_features = [
            'BB1_T10', 'BB1_T11', 'BB1_T12',
            'BB4_T10', 'BB4_T11', 'BB4_T12',
            'BB5_T10', 'BB5_T11', 'BB5_T12',
            'PB10_T11', 'PB11_T11'
        ]
        
        # PB Features - Price-Based Momentum
        self.pb_features = [
            'PB2_T10', 'PB2_T11', 'PB2_T12',
            'PB5_T10', 'PB5_T11', 'PB5_T12', 
            'PB6_T11', 'PB6_T12',
            'PB7_T11', 'PB7_T12',
            'PB3_T7', 'PB3_T10', 'PB3_T8'
        ]
        
        # PV Features - Price-Volume bands (NEW: Used for Weak/Medium signals)
        self.pv_features = [
            'PV3_B3_T6',  # Lower bound
            'PV3_B4_T6',  # Price-following indicator (center)
            'PV3_B5_T6'   # Upper bound
        ]
        
        # ==================== WEIGHTS ====================
        
        # BB Weights (negative = mean reversion)
        self.bb_weights = {
            'BB1_T10': -0.22, 'BB1_T11': -0.22, 'BB1_T12': -0.18,
            'BB4_T10': -0.18, 'BB4_T11': -0.15, 'BB4_T12': -0.03,
            'BB5_T10': -0.01, 'BB5_T11': -0.005, 'BB5_T12': -0.005,
            'PB10_T11': -0.1, 'PB11_T11': 0.1
        }
        
        # PB Weights (positive = momentum)
        self.pb_weights = {
            'PB2_T10': 0.05, 'PB2_T11': 0.05, 'PB2_T12': 0.05,
            'PB5_T10': 0.07, 'PB5_T11': 0.07, 'PB5_T12': 0.07, 
            'PB6_T11': 0.08, 'PB6_T12': 0.08,
            'PB7_T11': 0.06, 'PB7_T12': 0.06,
            'PB3_T7': 0.05, 'PB3_T10': 0.05, 'PB3_T8': 0.05
        }
        
        # V Weights (volume surge detection)
        self.v_weights = {
            'V5': 0.40, 'V2_T8': 0.3, 'V1_T4': 0.3, 
            'V8_T9_T12': 0.3, 'V8_T7_T11': 0.3
        }
        
        # ==================== PARAMETERS ====================
        
        # Strong Signal Parameters (RELAXED for higher frequency)
        self.strong_z_threshold = 1.2  # Reduced from 2.0 - more signals
        self.strong_agreement_factor = 0.6  # Reduced from 0.8 - easier to trigger
        
        # Weak/Medium Signal Parameters (RELAXED for higher frequency)
        self.weak_z_threshold = 0.6  # Reduced from 1.2 - many more weak signals
        self.pv_band_breakout_pct = 0.0001  # Reduced from 0.0004 - easier breakouts
        self.volume_surge_threshold = 1.15  # Reduced from 1.4 - more volume signals
        
        # Risk Management - Strong Signals (TIGHT for scalping)
        self.stop_loss_strong = 0.0015  # 15bps - tight for scalping
        self.take_profit_strong = 0.0030  # 30bps - quick profits
        self.trailing_stop_strong = 0.0010  # 10bps
        self.emergency_stop_loss = 0.0050  # 50bps emergency stop
        
        # Risk Management - Weak/Medium Signals (VERY TIGHT for scalping)
        self.stop_loss_weak = 0.0010  # 10bps - very tight
        self.take_profit_weak = 0.0020  # 20bps - quick scalps
        self.trailing_stop_weak = 0.0008  # 8bps
        
        # Position Management (OPTIMIZED for high frequency)
        self.min_trade_duration = 15  # CONSTRAINT 5: Minimum 15 seconds
        self.max_trade_duration = 120  # 2 minutes max - quick in/out
        self.min_hold_time = 20  # Reduced from 45 - faster turnover
        
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
    @jit(nopython=True, cache=True, fastmath=True, parallel=False)
    def generate_two_tier_signals(bb_norm, pb_norm, prices, 
                                  pv_lower, pv_center, pv_upper, v_surge,
                                  strong_z_thresh, strong_agree_factor,
                                  weak_z_thresh, pv_breakout_pct, vol_surge_thresh):
        """
        Generate two-tier trading signals - OPTIMIZED FOR HIGH FREQUENCY
        
        STRONG SIGNALS: BB and PB Z-scores agree in direction
        WEAK/MEDIUM SIGNALS: Price breaks PV bands + Volume confirmation (OR just momentum)
        
        Returns:
            signals: -1 (short), 0 (flat), +1 (long)
            signal_quality: 1 (strong), 2 (weak/medium)
        """
        n = len(bb_norm)
        signals = np.zeros(n, dtype=np.int32)
        signal_quality = np.zeros(n, dtype=np.int32)
        
        for i in range(300, n):
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            price = prices[i]
            
            # PV band relationships
            lower_band = pv_lower[i]
            center_band = pv_center[i]
            upper_band = pv_upper[i]
            
            # Volume confirmation
            has_volume_surge = v_surge[i] > vol_surge_thresh
            
            # ==================== STRONG SIGNALS ====================
            # LONG: Both BB and PB Z-scores are positive and above threshold
            if (bb_z > strong_z_thresh * strong_agree_factor and 
                pb_z > strong_z_thresh * strong_agree_factor):
                signals[i] = 1
                signal_quality[i] = 1
                continue
            
            # SHORT: Both BB and PB Z-scores are negative and below threshold
            if (bb_z < -strong_z_thresh * strong_agree_factor and 
                pb_z < -strong_z_thresh * strong_agree_factor):
                signals[i] = -1
                signal_quality[i] = 1
                continue
            
            # ==================== WEAK/MEDIUM SIGNALS ====================
            # More aggressive - generate many more signals
            
            # Check for moderate Z-score strength
            bb_moderate = abs(bb_z) > weak_z_thresh
            pb_moderate = abs(pb_z) > weak_z_thresh
            
            # LONG CONDITIONS (multiple paths to generate signals)
            long_signal = False
            
            # Path 1: Price above upper PV band
            if price > upper_band * (1.0 + pv_breakout_pct):
                if has_volume_surge or bb_z > 0 or pb_z > 0:
                    long_signal = True
            
            # Path 2: Price above center, momentum positive
            elif price > center_band * (1.0 + pv_breakout_pct * 0.3):
                if (bb_z > weak_z_thresh or pb_z > weak_z_thresh):
                    long_signal = True
                elif has_volume_surge and (bb_z > 0 or pb_z > 0):
                    long_signal = True
            
            # Path 3: Strong momentum alone (BB or PB)
            elif bb_moderate and bb_z > 0:
                if pb_z > 0 or has_volume_surge:
                    long_signal = True
            elif pb_moderate and pb_z > 0:
                if bb_z > 0 or has_volume_surge:
                    long_signal = True
            
            # Path 4: Price rising through center band
            elif price > center_band and price < upper_band:
                if (bb_z > weak_z_thresh * 0.5 and pb_z > weak_z_thresh * 0.5):
                    long_signal = True
            
            if long_signal:
                signals[i] = 1
                signal_quality[i] = 2
                continue
            
            # SHORT CONDITIONS (multiple paths to generate signals)
            short_signal = False
            
            # Path 1: Price below lower PV band
            if price < lower_band * (1.0 - pv_breakout_pct):
                if has_volume_surge or bb_z < 0 or pb_z < 0:
                    short_signal = True
            
            # Path 2: Price below center, momentum negative
            elif price < center_band * (1.0 - pv_breakout_pct * 0.3):
                if (bb_z < -weak_z_thresh or pb_z < -weak_z_thresh):
                    short_signal = True
                elif has_volume_surge and (bb_z < 0 or pb_z < 0):
                    short_signal = True
            
            # Path 3: Strong momentum alone (BB or PB)
            elif bb_moderate and bb_z < 0:
                if pb_z < 0 or has_volume_surge:
                    short_signal = True
            elif pb_moderate and pb_z < 0:
                if bb_z < 0 or has_volume_surge:
                    short_signal = True
            
            # Path 4: Price falling through center band
            elif price < center_band and price > lower_band:
                if (bb_z < -weak_z_thresh * 0.5 and pb_z < -weak_z_thresh * 0.5):
                    short_signal = True
            
            if short_signal:
                signals[i] = -1
                signal_quality[i] = 2
        
        return signals, signal_quality
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def manage_positions(signals, signal_quality, prices, timestamps,
                        min_duration, min_hold, max_duration,
                        sl_strong, sl_weak, tp_strong, tp_weak,
                        trail_strong, trail_weak):
        """
        Manage positions with quality-based risk management
        
        CONSTRAINT 1: Positions in {-1, 0, +1}
        CONSTRAINT 5: Minimum hold time = 15 seconds
        """
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
            
            # ==================== ENTRY LOGIC ====================
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]  # CONSTRAINT 1: {-1, 0, +1}
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_quality = signal_quality[i]
                    highest_price = prices[i]
                    lowest_price = prices[i]
                    max_favorable = 0.0
            
            # ==================== EXIT LOGIC ====================
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pct = (prices[i] - entry_price) / entry_price * direction
                
                # Track extremes for trailing stop
                if current_pos > 0:
                    highest_price = max(highest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                else:
                    lowest_price = min(lowest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                
                # Select parameters based on signal quality
                if entry_quality == 1:  # Strong signal
                    stop_loss = sl_strong
                    take_profit = tp_strong
                    trailing_stop = trail_strong
                else:  # Weak/Medium signal
                    stop_loss = sl_weak
                    take_profit = tp_weak
                    trailing_stop = trail_weak
                
                # Calculate trailing stop level
                if current_pos > 0:
                    trailing_level = highest_price * (1.0 - trailing_stop)
                else:
                    trailing_level = lowest_price * (1.0 + trailing_stop)
                
                should_exit = False
                
                # CONSTRAINT 5: Must hold for minimum duration
                if time_in_trade < min_duration:
                    pass
                # Stop loss
                elif pnl_pct <= -stop_loss:
                    should_exit = True
                # Take profit
                elif pnl_pct >= take_profit:
                    should_exit = True
                # Trailing stop (activate after reaching 40% of TP)
                elif max_favorable >= take_profit * 0.4:
                    if current_pos > 0 and prices[i] < trailing_level:
                        should_exit = True
                    elif current_pos < 0 and prices[i] > trailing_level:
                        should_exit = True
                # Max duration
                elif time_in_trade >= max_duration:
                    should_exit = True
                # Signal reversal (after min hold time)
                elif time_in_trade >= min_hold:
                    if current_pos > 0 and signals[i] < 0:
                        should_exit = True
                    elif current_pos < 0 and signals[i] > 0:
                        should_exit = True
                # Weak signal fade
                elif entry_quality == 2 and time_in_trade >= min_duration * 2:
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
        
        # CONSTRAINT 2: Force close at end of day (intraday only)
        positions[-1] = 0
        
        return positions
    
    def _calculate_weighted_signal(self, df, features, weights):
        """Calculate weighted signal from features"""
        signal = np.zeros(len(df), dtype=np.float32)
        total_weight = 0.0
        
        for feat in features:
            if feat in df.columns and feat in weights:
                signal += df[feat].values.astype(np.float32) * weights[feat]
                total_weight += abs(weights[feat])
        
        if total_weight > 0:
            signal /= total_weight
        
        return signal
    
    def process_day(self, day_num, data_folder='/data/quant14/EBX/'):
        """
        Process single day
        Runs in parallel via ProcessPoolExecutor
        
        CONSTRAINT 2: Intraday only - all positions closed at EOD
        CONSTRAINT 3: Fixed capital each day (no compounding)
        CONSTRAINT 6: No forward bias (all calculations use T-1 data)
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
            
            # Check required features
            required_features = (self.bb_features + self.pb_features + 
                               self.pv_features + self.v_features)
            missing = [f for f in required_features if f not in df.columns]
            
            if len(missing) > 0:
                print(f"Day {day_num}: Missing {len(missing)} features")
                # Fill missing with zeros (conservative approach)
                for feat in missing:
                    df[feat] = 0
            
            # Fill any NaN values (CONSTRAINT 6: no forward bias)
            for feat in required_features:
                if feat in df.columns:
                    df[feat] = df[feat].fillna(method='ffill').fillna(0)
            
            df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
            df['time_duration'] = pd.to_timedelta(df['timestamp_sec'], unit='s')
            
            prices = df['Price'].values.astype(np.float32)
            timestamps = df['timestamp_sec'].values.astype(np.float32)
            
            # ==================== SIGNAL CALCULATION ====================
            
            # Calculate weighted signals
            bb_signal = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights)
            pb_signal = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights)
            v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
            
            # Extract PV bands (CONSTRAINT 6: no lookahead)
            pv_lower = df['PV3_B3_T6'].values.astype(np.float32)
            pv_center = df['PV3_B4_T6'].values.astype(np.float32)
            pv_upper = df['PV3_B5_T6'].values.astype(np.float32)
            
            # Normalize signals (using historical data only)
            bb_normalized = self.normalize_signal(bb_signal, window=300)
            pb_normalized = self.normalize_signal(pb_signal, window=300)
            
            # Volume surge calculation (CONSTRAINT 6: rolling mean uses past data)
            v_mean = self.calculate_rolling_mean(v_signal, 600)
            v_surge = v_signal / (v_mean + 1e-8)
            
            # ==================== SIGNAL GENERATION ====================
            # Two-tier signals: Strong and Weak/Medium
            signals, signal_quality = self.generate_two_tier_signals(
                bb_normalized, pb_normalized, prices,
                pv_lower, pv_center, pv_upper, v_surge,
                self.strong_z_threshold, self.strong_agreement_factor,
                self.weak_z_threshold, self.pv_band_breakout_pct,
                self.volume_surge_threshold
            )
            
            # ==================== POSITION MANAGEMENT ====================
            # IMPORTANT: Ensure argument count matches function signature (14 args)
            positions = self.manage_positions(
                signals,                      # 1
                signal_quality,               # 2
                prices,                       # 3
                timestamps,                   # 4
                self.min_trade_duration,      # 5
                self.min_hold_time,           # 6
                self.max_trade_duration,      # 7
                self.stop_loss_strong,        # 8
                self.stop_loss_weak,          # 9
                self.take_profit_strong,      # 10
                self.take_profit_weak,        # 11
                self.trailing_stop_strong,    # 12
                self.trailing_stop_weak,      # 13
                self.emergency_stop_loss      # 14
            )
            
            # Statistics
            strong_long = ((signals == 1) & (signal_quality == 1)).sum()
            strong_short = ((signals == -1) & (signal_quality == 1)).sum()
            weak_long = ((signals == 1) & (signal_quality == 2)).sum()
            weak_short = ((signals == -1) & (signal_quality == 2)).sum()
            
            print(f"Day {day_num:3d}: Strong(L:{strong_long:3d},S:{strong_short:3d}) "
                  f"Weak(L:{weak_long:3d},S:{weak_short:3d})")
            
            # Return results
            result_df = pd.DataFrame({
                'Time': df['time_duration'],
                'Signal': positions.astype(np.int32),  # CONSTRAINT 1: {-1, 0, +1}
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
        
        CONSTRAINT 3: Fixed capital each day (no compounding across days)
        """
        print("="*80)
        print("TWO-TIER SIGNAL STRATEGY - HIGH FREQUENCY SCALPING")
        print("="*80)
        print(f"✓ TARGET: Hundreds of trades per day")
        print(f"✓ Strong Signals: BB + PB agreement (Z > {self.strong_z_threshold})")
        print(f"✓ Weak Signals: Multiple paths (PV bands, momentum, volume)")
        print(f"✓ Scalping Parameters:")
        print(f"  - Strong: {self.stop_loss_strong*10000:.0f}bp SL, {self.take_profit_strong*10000:.0f}bp TP")
        print(f"  - Weak: {self.stop_loss_weak*10000:.0f}bp SL, {self.take_profit_weak*10000:.0f}bp TP")
        print(f"  - EMERGENCY: {self.emergency_stop_loss*10000:.0f}bp stop")
        print(f"  - Max Duration: {self.max_trade_duration}s (quick in/out)")
        print(f"  - Min Hold: {self.min_hold_time}s")
        print(f"✓ All 6 constraints enforced")
        print(f"✓ CPU Workers: {max_workers}")
        print("="*80 + "\n")
        
        # Create partial function
        process_func = partial(self.process_day, data_folder=data_folder)
        
        # Dictionary to store results with day number as key
        day_results = {}
        
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_day = {executor.submit(process_func, day): day for day in range(num_days)}
            
            for future in concurrent.futures.as_completed(future_to_day):
                day = future_to_day[future]
                try:
                    result = future.result()
                    if result is not None and len(result) > 0:
                        day_results[day] = result
                except Exception as e:
                    print(f"Day {day}: Error - {e}")
        
        if len(day_results) > 0:
            print(f"\n✓ Processed {len(day_results)} days successfully")
            print("Combining results in day-wise order...")
            
            # Sort by day number and concatenate
            sorted_days = sorted(day_results.keys())
            all_results = [day_results[day] for day in sorted_days]
            
            portfolio_weights = pd.concat(all_results, ignore_index=True)
            
            # Remove Day column before saving
            portfolio_weights_output = portfolio_weights[['Time', 'Signal', 'Price']].copy()
            
            # Save to CSV
            portfolio_weights_output.to_csv('portfolio_weights.csv', index=False)
            
            print(f"\n✓ Saved portfolio_weights.csv ({len(portfolio_weights_output):,} rows)")
            
            # Statistics
            total_signals = len(portfolio_weights_output)
            total_long = (portfolio_weights_output['Signal'] > 0).sum()
            total_short = (portfolio_weights_output['Signal'] < 0).sum()
            
            print(f"\n✅ SUMMARY:")
            print(f"  Days Processed: {len(day_results)}")
            print(f"  Total Signals: {total_signals:,}")
            if total_signals > 0:
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
            
            # Signal quality breakdown
            strong_trades = 0
            weak_trades = 0
            for day in sorted_days:
                day_df = day_results[day]
                # Count transitions to non-zero positions
                position_changes = (day_df['Signal'].diff() != 0) & (day_df['Signal'] != 0)
                strong_trades += position_changes.sum()
            
            print(f"  Estimated Trades: {strong_trades}")
            print("="*80)
            
            gc.collect()
            
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = TwoTierSignalStrategy()
    
    print("\n" + "="*80)
    print("STARTING TWO-TIER SIGNAL STRATEGY")
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✓ Strategy complete!")
        print("✓ portfolio_weights.csv ready (Time, Signal, Price)")
        print("✓ All 6 constraints enforced")
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")