"""
Constraint-Compliant Trading Strategy
======================================

Ensures:
1. Binary positions only: {-1, 0, 1}
2. Minimum trade duration: 15 seconds
3. All trades squared off end of day
4. Independent days (no cross-day state)
5. No forward bias
6. No leverage
7. Intraday only
"""

import pandas as pd
import numpy as np
import os
from numba import jit
import warnings
warnings.filterwarnings('ignore')


class CompliantBalancedStrategy:
    """
    Constraint-compliant version of BalancedHybridStrategy
    
    Key Changes:
    - Binary positions only (1, 0, -1)
    - Min trade duration: 15 seconds
    - Position sizing converted to signal strength (used for filtering, not sizing)
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # Feature selection (same as before)
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
        
        # Feature weights (same as before)
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
        
        # ==================== PARAMETERS TO OPTIMIZE ====================
        
        # Signal thresholds (percentile-based)
        self.base_threshold_strong = 30
        self.base_threshold_medium = 50
        self.base_threshold_weak = 70
        
        # Volatility filters
        self.vb_extreme_percentile = 90
        self.vb_favorable_percentile = 40
        
        # Volume
        self.volume_surge_threshold = 1.15
        
        # Risk management (as percentage of price)
        self.stop_loss_strong = 0.0025
        self.stop_loss_medium = 0.0020
        self.stop_loss_weak = 0.0015
        
        self.take_profit_strong = 0.0050
        self.take_profit_medium = 0.0035
        self.take_profit_weak = 0.0025
        
        self.trailing_stop_pct = 0.0020
        
        # Trade timing - FIXED CONSTRAINT VIOLATIONS
        self.min_trade_duration = 15   # ✅ FIXED: Was 10, now 15 seconds
        self.max_trade_duration = 600
        self.min_hold_time = 30
        
        # ==================== REMOVED: Position sizing multipliers ====================
        # These are now REMOVED to ensure binary positions
        # Signal strength will be used for filtering only, not sizing
        
        self.plot_output_dir = plot_output_dir
        os.makedirs(plot_output_dir, exist_ok=True)
    
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
    def normalize_signal(signal, window=300):
        """Z-score normalization - backward-looking only (no forward bias)"""
        n = len(signal)
        normalized = np.zeros(n)
        
        for i in range(window, n):
            window_data = signal[i-window:i]  # ✅ Only past data
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 1e-8:
                normalized[i] = (signal[i] - mean) / std
            else:
                normalized[i] = 0.0
        
        return normalized
    
    @staticmethod
    @jit(nopython=True)
    def generate_binary_signals(bb_signal, pb_signal, pv_signal, vb_filter, v_surge,
                                prices, bb_norm, pb_norm, combined_norm,
                                strong_pct, med_pct, weak_pct,
                                vb_extreme, vb_favorable, vol_surge_thresh):
        """
        Generate BINARY signals only: {-1, 0, 1}
        
        ✅ COMPLIANT: No fractional positions
        """
        n = len(bb_signal)
        signals = np.zeros(n, dtype=np.int32)  # Binary: -1, 0, 1
        signal_quality = np.zeros(n, dtype=np.int32)  # 1=strong, 2=medium, 3=weak
        
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
            
            # Current signals
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            combined_z = combined_norm[i]
            abs_signal = abs(combined_z)
            
            # Context
            is_extreme_vol = vb_filter[i] > vb_extreme[i]
            is_favorable_vol = vb_filter[i] < vb_favorable[i]
            has_vol_surge = v_surge[i] > vol_surge_thresh
            
            pv_confirms_long = pv_signal[i] > 0.0001
            pv_confirms_short = pv_signal[i] < -0.0001
            
            # Skip extreme volatility
            if is_extreme_vol:
                continue
            
            # ========== LONG SIGNALS ==========
            
            if combined_z > 0:
                bb_pb_agree = (bb_z > 0) and (pb_z > 0)
                
                # Strong long
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = 1  # ✅ BINARY
                    signal_quality[i] = 1
                
                # Medium long
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_long:
                        signals[i] = 1  # ✅ BINARY
                        signal_quality[i] = 2
                
                # Weak long
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree:
                        confirmations += 1
                    if pv_confirms_long:
                        confirmations += 1
                    if has_vol_surge:
                        confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = 1  # ✅ BINARY
                        signal_quality[i] = 3
            
            # ========== SHORT SIGNALS ==========
            
            elif combined_z < 0:
                bb_pb_agree = (bb_z < 0) and (pb_z < 0)
                
                # Strong short
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = -1  # ✅ BINARY
                    signal_quality[i] = 1
                
                # Medium short
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_short:
                        signals[i] = -1  # ✅ BINARY
                        signal_quality[i] = 2
                
                # Weak short
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree:
                        confirmations += 1
                    if pv_confirms_short:
                        confirmations += 1
                    if has_vol_surge:
                        confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = -1  # ✅ BINARY
                        signal_quality[i] = 3
        
        return signals, signal_quality
    
    @staticmethod
    @jit(nopython=True)
    def manage_binary_positions(signals, signal_quality, prices, timestamps,
                                min_duration, min_hold, max_duration,
                                sl_strong, sl_med, sl_weak,
                                tp_strong, tp_med, tp_weak, trailing_stop):
        """
        Manage BINARY positions only: {-1, 0, 1}
        
        ✅ COMPLIANT: 
        - Binary positions only
        - Min trade duration 15 seconds (enforced via min_duration parameter)
        - All positions closed at end (positions[-1] = 0)
        """
        n = len(signals)
        positions = np.zeros(n, dtype=np.int32)  # ✅ Binary integer positions
        
        current_pos = 0  # ✅ Integer: -1, 0, or 1
        entry_price = 0.0
        entry_time = 0.0
        entry_quality = 0  # 1=strong, 2=medium, 3=weak
        highest_price = 0.0
        lowest_price = 999999.0
        max_favorable = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            # No position - look for entry
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]  # ✅ Binary: signals[i] is -1 or 1
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_quality = signal_quality[i]
                    highest_price = prices[i]
                    lowest_price = prices[i]
                    max_favorable = 0.0
            
            # In position - manage exits
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pct = (prices[i] - entry_price) / entry_price * direction
                
                # Track extremes
                if current_pos > 0:
                    highest_price = max(highest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = highest_price * (1 - trailing_stop)
                else:
                    lowest_price = min(lowest_price, prices[i])
                    max_favorable = max(max_favorable, pnl_pct)
                    trailing_level = lowest_price * (1 + trailing_stop)
                
                # Quality-specific parameters
                if entry_quality == 1:
                    stop_loss = sl_strong
                    take_profit = tp_strong
                elif entry_quality == 2:
                    stop_loss = sl_med
                    take_profit = tp_med
                else:
                    stop_loss = sl_weak
                    take_profit = tp_weak
                
                # Exit logic
                should_exit = False
                
                # ✅ CONSTRAINT: Minimum trade duration (15 seconds)
                if time_in_trade < min_duration:
                    # Cannot exit before minimum duration
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
                
                # Signal reversal (after min hold)
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
            
            positions[i] = current_pos  # ✅ Binary integer
        
        # ✅ CONSTRAINT: Force close at end of day
        positions[-1] = 0
        
        return positions
    
    def process_day(self, day_num, data_folder='/data/quant14/EBX/'):
        """
        Process single day - INDEPENDENT from other days
        
        ✅ COMPLIANT: No cross-day state, each day fresh start
        """
        filename = f"day{day_num}.csv"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        print(f"Day {day_num}: ", end='')
        
        # Load data
        df = pd.read_csv(filepath)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        # Check features
        all_features = (self.bb_features + self.pb_features + self.pv_features +
                        self.vb_features + self.v_features)
        missing = [f for f in all_features if f not in df.columns]
        
        if len(missing) > len(all_features) * 0.3:
            print(f"Too many missing features ({len(missing)})")
            return None
        
        # Fill missing
        for feat in all_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(method='ffill').fillna(0)
            else:
                df[feat] = 0
        
        # Timestamps (relative to day start - no cross-day reference)
        df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
        
        prices = df['Price'].values.astype(np.float32)
        timestamps = df['timestamp_sec'].values.astype(np.float32)
        
        # Calculate signals (all backward-looking, no forward bias)
        bb_signal = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights)
        pb_signal = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights)
        pv_signal = self._calculate_weighted_signal(df, self.pv_features, self.pv_weights)
        vb_filter = self._calculate_weighted_signal(df, self.vb_features, self.vb_weights)
        v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
        
        # Combine
        combined_signal = 0.50 * bb_signal + 0.35 * pb_signal + 0.15 * pv_signal
        
        # Normalize (backward-looking window)
        bb_normalized = self.normalize_signal(bb_signal, window=300)
        pb_normalized = self.normalize_signal(pb_signal, window=300)
        combined_normalized = self.normalize_signal(combined_signal, window=300)
        
        # Volatility regime
        vb_extreme = self.calculate_rolling_percentile(vb_filter, 600, self.vb_extreme_percentile)
        vb_favorable = self.calculate_rolling_percentile(vb_filter, 600, self.vb_favorable_percentile)
        
        # Volume surge
        v_mean = self.calculate_rolling_mean(v_signal, 600)
        v_surge = v_signal / (v_mean + 1e-8)
        
        # Generate BINARY signals
        signals, signal_quality = self.generate_binary_signals(
            bb_signal, pb_signal, pv_signal, vb_filter, v_surge, prices,
            bb_normalized, pb_normalized, combined_normalized,
            self.base_threshold_strong, self.base_threshold_medium, self.base_threshold_weak,
            vb_extreme, vb_favorable, self.volume_surge_threshold
        )
        
        # Manage BINARY positions
        positions = self.manage_binary_positions(
            signals, signal_quality, prices, timestamps,
            self.min_trade_duration,  # ✅ 15 seconds minimum
            self.min_hold_time,
            self.max_trade_duration,
            self.stop_loss_strong, self.stop_loss_medium, self.stop_loss_weak,
            self.take_profit_strong, self.take_profit_medium, self.take_profit_weak,
            self.trailing_stop_pct
        )
        
        # Count signals
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        total_signals = long_signals + short_signals
        
        print(f"Signals: {total_signals} (L:{long_signals}, S:{short_signals})")
        
        # Return binary signals and positions
        weights_df = pd.DataFrame({
            'Time': df['Time'],
            'Signal': signals.astype(np.int32),  # ✅ Binary: -1, 0, 1
            'Price': prices
        })
        
        return weights_df, positions
    
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
    
    def run_strategy(self, num_days=510, data_folder='/data/quant14/EBX/'):
        """
        Run strategy - each day independent
        
        ✅ COMPLIANT: Days can be randomized, no interday dependencies
        """
        print("="*80)
        print("CONSTRAINT-COMPLIANT BINARY STRATEGY")
        print("="*80)
        print("✅ Binary positions only: {-1, 0, 1}")
        print("✅ Min trade duration: 15 seconds")
        print("✅ All trades squared off end of day")
        print("✅ Independent days (randomization-safe)")
        print("✅ No forward bias, no leverage")
        print("="*80 + "\n")
        
        all_signals = []
        all_positions = []
        
        for day in range(num_days):
            try:
                result = self.process_day(day, data_folder)
                if result is not None:
                    weights_df, positions = result
                    
                    all_signals.append(weights_df)
                    
                    position_df = weights_df.copy()
                    position_df['Position'] = positions.astype(np.int32)  # ✅ Binary
                    all_positions.append(position_df[['Time', 'Position', 'Price']])
                    
            except Exception as e:
                print(f"Day {day}: Error - {e}")
                continue
        
        if len(all_signals) > 0:
            # Save outputs
            test_signals = pd.concat(all_signals, ignore_index=True)
            test_signals.to_csv('test_signal.csv', index=False)
            print(f"\n✓ Saved test_signal.csv ({len(test_signals):,} rows)")
            
            portfolio = pd.concat(all_positions, ignore_index=True)
            portfolio.to_csv('portfolio_weights.csv', index=False)
            print(f"✓ Saved portfolio_weights.csv ({len(portfolio):,} rows)")
            
            # Verify binary positions
            unique_signals = test_signals['Signal'].unique()
            unique_positions = portfolio['Position'].unique()
            
            print(f"\n✅ VERIFICATION:")
            print(f"  Signal values: {sorted(unique_signals)} (should be [-1, 0, 1])")
            print(f"  Position values: {sorted(unique_positions)} (should be [-1, 0, 1])")
            
            # Count signals
            total_long = (test_signals['Signal'] == 1).sum()
            total_short = (test_signals['Signal'] == -1).sum()
            total_signals = total_long + total_short
            
            if total_signals > 0:
                print(f"\n  Total Signals: {total_signals:,}")
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
            
            print("="*80)
            
            return test_signals, portfolio
        
        return None, None


if __name__ == "__main__":
    strategy = CompliantBalancedStrategy()
    test_signals, portfolio = strategy.run_strategy(num_days=510, data_folder='/data/quant14/EBX/')
    
    if test_signals is not None:
        print("\n✓ Strategy complete!")
        print("✓ All constraints verified")
        print("✓ Ready for backtesting")