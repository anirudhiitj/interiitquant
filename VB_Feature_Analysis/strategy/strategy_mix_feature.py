import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
import os
from numba import jit
warnings.filterwarnings('ignore')

class BalancedHybridStrategy:
    """
    Balanced Multi-Feature Trading Strategy (Signal/Weight Generator) - STRICT 1-BIT VERSION
    
    Role: Generates raw entry signals (-1, 0, 1) and final risk-managed positions (-1, 0, 1).
    Does NOT calculate any PnL or performance metrics. Enforces no fractional positions or leverage.
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # ==================== FEATURE SELECTION ====================
        
        # BB Features (0.99 corr) - PRIMARY DIRECTION
        self.bb_features = [
            'BB1_T10', 'BB1_T11', 'BB1_T12',
            'BB4_T10', 'BB4_T11', 'BB4_T12',
            'BB5_T10', 'BB5_T11', 'BB5_T12'
        ]
        
        # PB Features (0.90 corr) - MOMENTUM & DIRECTION
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
        
        # ==================== OPTIMIZED WEIGHTS ====================
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
            'PV3_B3_T12': 0.33, 'PV3_B4_T12': 0.40, 'PV3_B5_T12': 0.33
        }
        self.vb_weights = {
            'VB4_T11': 0.45, 'VB4_T12': 0.35,
            'VB5_T11': 0.10, 'VB5_T12': 0.10
        }
        self.v_weights = {
            'V5': 0.40, 'V2_T8': 0.3, 'V1_T4': 0.3, 'V8_T9_T12': 0.3, 'V8_T7_T11': 0.3
        }
        
        # ==================== STRATEGY PARAMETERS ====================
        
        # Adaptive signal thresholds
        self.base_threshold_strong = 0.30     # Top 30%
        self.base_threshold_medium = 0.50     # Top 50%
        self.base_threshold_weak = 0.70       # Top 70%
        
        # Volatility regime filters
        self.vb_extreme_percentile = 90     # Block extreme volatility
        self.vb_favorable_percentile = 40   # Favorable low vol regime
        
        # Volume confirmation
        self.volume_surge_threshold = 1.15    # 15% above average
        
        # Risk management - tiered approach (Now only affects SL/TP, not size)
        self.stop_loss_strong = 0.0025        # 0.25%
        self.stop_loss_medium = 0.0020        # 0.20%
        self.stop_loss_weak = 0.0015          # 0.15%
        
        self.take_profit_strong = 0.0050      # 0.50%
        self.take_profit_medium = 0.0035      # 0.35%
        self.take_profit_weak = 0.0025        # 0.25%
        
        self.trailing_stop_pct = 0.0020       # 0.20%
        
        # Trade timing
        self.min_trade_duration = 15          # 15 seconds minimum
        self.max_trade_duration = 600         # 10 minutes maximum
        self.min_hold_time = 15               # 30 seconds before reversal
        
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
                                # === REMOVED multipliers from signature ===
                                #   strong_mult, med_mult, weak_mult):
        """
        Generate balanced long/short signals (-1, 0, 1) using normalized features.
        Returns signals and trade tiers (for SL/TP selection).
        """
        n = len(bb_signal)
        signals = np.zeros(n, dtype=np.int32)
        # === REMOVED position_sizes array ===
        # position_sizes = np.ones(n, dtype=np.float32) 
        trade_tiers = np.zeros(n, dtype=np.int32) # Tier determines SL/TP used
        
        lookback_thresh = 600
        
        for i in range(300, n):
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
            
            is_extreme_vol = vb_filter[i] > vb_extreme[i]
            is_favorable_vol = vb_filter[i] < vb_favorable[i] # Keep for potential filtering logic if needed
            has_vol_surge = v_surge[i] > vol_surge_thresh     # Keep for confirmation logic
            
            pv_confirms_long = pv_signal[i] > 0.0001
            pv_confirms_short = pv_signal[i] < -0.0001
            
            if is_extreme_vol:
                continue
                
            # === REMOVED base_mult calculation ===
            
            # ========== LONG SIGNAL GENERATION ==========
            if combined_z > 0:
                bb_pb_agree = (bb_z > 0) and (pb_z > 0)
                
                # Strong long signal
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = 1
                    trade_tiers[i] = 1
                    # === REMOVED position_sizes calculation ===
                
                # Medium long signal
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_long:
                        signals[i] = 1
                        trade_tiers[i] = 2
                        # === REMOVED position_sizes calculation ===
                
                # Weak long signal - needs strong confirmation
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree: confirmations += 1
                    if pv_confirms_long: confirmations += 1
                    if has_vol_surge: confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = 1
                        trade_tiers[i] = 3
                        # === REMOVED position_sizes calculation ===
            
            # ========== SHORT SIGNAL GENERATION ==========
            elif combined_z < 0:
                bb_pb_agree = (bb_z < 0) and (pb_z < 0)
                
                # Strong short signal
                if abs_signal > thresh_strong and bb_pb_agree:
                    signals[i] = -1
                    trade_tiers[i] = 1
                    # === REMOVED position_sizes calculation ===
                
                # Medium short signal
                elif abs_signal > thresh_medium:
                    if bb_pb_agree or pv_confirms_short:
                        signals[i] = -1
                        trade_tiers[i] = 2
                        # === REMOVED position_sizes calculation ===
                
                # Weak short signal - needs strong confirmation
                elif abs_signal > thresh_weak:
                    confirmations = 0
                    if bb_pb_agree: confirmations += 1
                    if pv_confirms_short: confirmations += 1
                    if has_vol_surge: confirmations += 1
                    
                    if confirmations >= 2:
                        signals[i] = -1
                        trade_tiers[i] = 3
                        # === REMOVED position_sizes calculation ===
        
        # === CHANGED RETURN VALUE ===
        return signals, trade_tiers
    
    @staticmethod
    @jit(nopython=True)
    def manage_positions_fast(signals, trade_tiers, prices, timestamps, # === REMOVED position_sizes ===
                              min_duration, min_hold, max_duration,
                              sl_strong, sl_med, sl_weak,
                              tp_strong, tp_med, tp_weak, trailing_stop):
        """Enhanced position management with dynamic exits - STRICT 1-BIT VERSION"""
        n = len(signals)
        # === CHANGED DTYPE to int32 ===
        positions = np.zeros(n, dtype=np.int32) 
        
        current_pos = 0 # Use integer for position
        entry_price = 0.0
        entry_time = 0.0
        entry_tier = 0
        highest_price = 0.0
        lowest_price = 999999.0
        max_favorable = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            # No position - look for entry
            if current_pos == 0:
                if signals[i] != 0:
                    # === POSITION IS NOW DIRECTLY THE SIGNAL ===
                    current_pos = signals[i] 
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_tier = trade_tiers[i]
                    highest_price = prices[i]
                    lowest_price = prices[i]
                    max_favorable = 0.0
            
            # In position - manage exits
            else:
                direction = 1.0 if current_pos > 0 else -1.0 # Keep float for calculation
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
                
                # ===== START OF CORRECTED EXIT BLOCK =====
                should_exit = False
                
                if time_in_trade >= min_duration: 
                    if pnl_pct <= -stop_loss: should_exit = True
                    elif pnl_pct >= take_profit: should_exit = True
                    elif max_favorable >= take_profit * 0.4:
                        if (current_pos > 0 and prices[i] < trailing_level) or \
                           (current_pos < 0 and prices[i] > trailing_level):
                            should_exit = True
                    elif time_in_trade >= min_hold:
                        if (current_pos > 0 and signals[i] < 0) or \
                           (current_pos < 0 and signals[i] > 0):
                            should_exit = True

                if time_in_trade >= max_duration: should_exit = True
                
                elif entry_tier == 3 and time_in_trade >= min_duration * 2: 
                    if signals[i] == 0: should_exit = True
                # ===== END OF CORRECTED EXIT BLOCK =====
                        
                if should_exit:
                    current_pos = 0 # Reset to integer 0
                    entry_price = 0.0
                    entry_time = 0.0
                    entry_tier = 0
                    highest_price = 0.0
                    lowest_price = 999999.0
                    max_favorable = 0.0
            
            positions[i] = current_pos
        
        positions[-1] = 0 # Force close final position
        
        return positions
    
    def process_day(self, day_num, data_folder='/data/quant14/EBX/'):
        filename = f"day{day_num}.csv"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath): return None
        
        print(f"Day {day_num}: ", end='')
        
        df = pd.read_csv(filepath)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time').reset_index(drop=True)
        
        all_features = (self.bb_features + self.pb_features + self.pv_features +
                        self.vb_features + self.v_features)
        missing = [f for f in all_features if f not in df.columns]
        
        if len(missing) > len(all_features) * 0.3:
            print(f"Too many missing features ({len(missing)})")
            return None
        
        for feat in all_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(method='ffill').fillna(0)
            else: df[feat] = 0
            
        df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
        
        prices = df['Price'].values.astype(np.float32)
        timestamps = df['timestamp_sec'].values.astype(np.float32)
        
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
        
        # === UPDATED CALL: Removed multipliers ===
        signals, trade_tiers = self.generate_balanced_signals(
            bb_signal, pb_signal, pv_signal, vb_filter, v_surge, prices,
            bb_normalized, pb_normalized, combined_normalized,
            self.base_threshold_strong, self.base_threshold_medium, self.base_threshold_weak,
            vb_extreme, vb_favorable, self.volume_surge_threshold
            # Removed: self.strong_multiplier, self.medium_multiplier, self.weak_multiplier
        )
        
        # === UPDATED CALL: Removed position_sizes ===
        positions = self.manage_positions_fast(
            signals, trade_tiers, prices, timestamps,
            self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
            self.stop_loss_strong, self.stop_loss_medium, self.stop_loss_weak,
            self.take_profit_strong, self.take_profit_medium, self.take_profit_weak,
            self.trailing_stop_pct
        )
        
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        total_signals = long_signals + short_signals
        print(f"Processed. Signals: {total_signals} (L: {long_signals}, S: {short_signals})")
        
        weights_df = pd.DataFrame({
            'Time': df['Time'],
            'Signal': signals.astype(np.int32),
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
        if total_weight > 0: signal /= total_weight
        return signal

    def run_strategy(self, num_days=510, data_folder='/data/quant14/EBX/'):
        print("="*80)
        print("BALANCED HYBRID STRATEGY (1-BIT) - SIGNAL GENERATOR")
        print("="*80)
        print("Generating raw signals (test_signal.csv) and")
        print("managed positions (portfolio_weights.csv)...")
        print("="*80 + "\n")
        
        all_signals_df = []
        all_positions_df = [] 
        
        for day in range(num_days):
            try:
                result = self.process_day(day, data_folder)
                if result is not None:
                    weights_df, positions = result 
                    
                    all_signals_df.append(weights_df) 
                    
                    position_df = weights_df.copy()
                    # === CHANGED DTYPE to int32 ===
                    position_df['Position'] = positions.astype(np.int32) 
                    all_positions_df.append(position_df[['Time', 'Position', 'Price']])
                    
            except Exception as e:
                print(f"Day {day}: Error - {e}")
                continue
        
        if len(all_signals_df) > 0:
            test_signals = pd.concat(all_signals_df, ignore_index=True)
            test_signals.to_csv('test_signal.csv', index=False)
            print(f"\n✓ Saved test_signal.csv ({len(test_signals):,} rows)")
            
            portfolio = pd.concat(all_positions_df, ignore_index=True)
            portfolio.to_csv('portfolio_weights.csv', index=False)
            print(f"✓ Saved portfolio_weights.csv ({len(portfolio):,} rows)")
            
            total_long_signals = (test_signals['Signal'] == 1).sum()
            total_short_signals = (test_signals['Signal'] == -1).sum()
            total_signals = total_long_signals + total_short_signals
            
            print("\n" + "="*80)
            print("SIGNAL GENERATION SUMMARY")
            print("="*80)
            print(f"Days Processed: {len(all_signals_df)}") # Corrected variable name
            print(f"\nRaw Signals Generated (in test_signal.csv):")
            if total_signals > 0:
                print(f"  - Total Signals: {total_signals:,}")
                print(f"  - Long Signals: {total_long_signals:,} ({total_long_signals/total_signals*100:.1f}%)")
                print(f"  - Short Signals: {total_short_signals:,} ({total_short_signals/total_signals*100:.1f}%)")
            else:
                print("  - No signals generated.")
            print("="*80)
            
            return test_signals, portfolio
        
        return None, None


if __name__ == "__main__":
    strategy = BalancedHybridStrategy(plot_output_dir='daily_plots')
    test_signals, portfolio = strategy.run_strategy(num_days=510, data_folder='/data/quant14/EBX/')
    
    if test_signals is not None:
        print("\n✓ Strategy complete!")
        print("✓ test_signal.csv - Raw signals (Time, Signal, Price)")
        print("✓ portfolio_weights.csv - Managed positions (Time, Position, Price)")
        print("✓ Run backtester.py to evaluate performance")