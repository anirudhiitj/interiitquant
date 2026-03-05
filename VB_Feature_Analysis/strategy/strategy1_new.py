"""
Adaptive Regime-Switching Two-Tier Signal Trading Strategy
===========================================================
CRITICAL FIX: Weak SHORT signals now properly converted to positions
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


class AdaptiveRegimeTradingStrategy:
    """
    Adaptive strategy with WORKING weak signal SHORT trades
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # ==================== OPTIMIZED FEATURE SET ====================
        
        self.bb_features = [
            'BB4_T10', 'BB4_T11', 'BB4_T12', 'BB3_T10', 'BB5_T11'
        ]
        
        self.pb_features = [
            'PB1_T10', 'PB1_T11', 'PB6_T11', 'PB6_T12',
            'PB10_T11', 'PB11_T11', 'PB13_T10', 'PB13_T11'
        ]
        
        self.pv_features = [
            'PV3_B3_T6', 'PV3_B4_T6', 'PV3_B5_T6'
        ]
        
        self.v_features = ['V5']
        
        # ==================== BASE WEIGHTS ====================
        
        # MEAN REVERSION (negative = fade extremes)
        self.bb_weights_reversion = {
            'BB4_T10': -0.35, 'BB4_T11': -0.30, 'BB4_T12': -0.20,
            'BB3_T10': -0.10, 'BB5_T11': -0.05
        }
        
        self.pb_weights_reversion = {
            'PB1_T10': -0.15, 'PB1_T11': -0.15, 'PB6_T11': -0.15,
            'PB6_T12': -0.10, 'PB10_T11': -0.15, 'PB11_T11': 0.15,
            'PB13_T10': -0.10, 'PB13_T11': -0.05
        }
        
        # MOMENTUM (positive = follow trends)
        self.bb_weights_momentum = {
            'BB4_T10': 0.25, 'BB4_T11': 0.25, 'BB4_T12': 0.20,
            'BB3_T10': 0.15, 'BB5_T11': 0.15
        }
        
        self.pb_weights_momentum = {
            'PB1_T10': 0.15, 'PB1_T11': 0.10, 'PB6_T11': 0.20,
            'PB6_T12': 0.15, 'PB10_T11': 0.05, 'PB11_T11': 0.05,
            'PB13_T10': 0.20, 'PB13_T11': 0.10
        }
        
        self.v_weights = {'V5': 1.0}
        
        # ==================== REGIME DETECTION ====================
        self.regime_test_duration = 2400
        self.regime_test_min_trades = 3
        
        # ==================== SIGNAL PARAMETERS ====================
        
        # Strong Signals (keep as is)
        self.strong_z_threshold = 1.1
        self.strong_agreement_threshold = 0.6
        
        # CRITICAL: More aggressive weak signal thresholds
<<<<<<< HEAD
        self.weak_z_threshold = 0.6  # Even lower to catch more signals
=======
        self.weak_z_threshold = 0.55  # Even lower to catch more signals
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        self.pv_band_breakout_pct = 0.0001  # More sensitive
        self.volume_surge_threshold = 1.15  # Lower barrier
        
        # ==================== RISK MANAGEMENT ====================
        
        # Strong Signals (unchanged)
<<<<<<< HEAD
        self.sl_strong_mult = 0.55
        self.tp_strong_mult = 1.1
        self.trail_strong_mult = 0.4
        
        # Weak signals - balanced risk/reward
        self.sl_weak_mult = 0.4
        self.tp_weak_mult = 0.85
=======
        self.sl_strong_mult = 0.6
        self.tp_strong_mult = 1.2
        self.trail_strong_mult = 0.4
        
        # Weak signals - balanced risk/reward
        self.sl_weak_mult = 0.45
        self.tp_weak_mult = 0.9
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        self.trail_weak_mult = 0.35
        
        # Position Management
        self.min_trade_duration = 15
<<<<<<< HEAD
        self.max_trade_duration = 540
=======
        self.max_trade_duration = 600
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        self.min_hold_time = 15
        
        self.plot_output_dir = plot_output_dir
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # ==================== NUMBA-OPTIMIZED FUNCTIONS ====================
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def calculate_rolling_mean(arr, window):
        n = len(arr)
        result = np.zeros(n, dtype=np.float32)
        for i in prange(window, n):
            result[i] = np.mean(arr[i-window:i])
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def normalize_signal(signal, window=300):
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
    def detect_regime(bb_norm, pb_norm, prices, timestamps, test_duration, min_trades):
        n = len(prices)
        test_end_idx = 0
        
        test_end_time = timestamps[0] + test_duration
        for i in range(n):
            if timestamps[i] >= test_end_time:
                test_end_idx = i
                break
        
        if test_end_idx < 300:
            return 1, 0.5
        
        reversion_pnl = 0.0
        reversion_trades = 0
        position_rev = 0
        entry_price_rev = 0.0
        
        momentum_pnl = 0.0
        momentum_trades = 0
        position_mom = 0
        entry_price_mom = 0.0
        
        for i in range(300, test_end_idx):
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            price = prices[i]
            
            # Mean Reversion
            if position_rev == 0:
                if bb_z < -1.0 and pb_z < -1.0:
                    position_rev = 1
                    entry_price_rev = price
                    reversion_trades += 1
                elif bb_z > 1.0 and pb_z > 1.0:
                    position_rev = -1
                    entry_price_rev = price
                    reversion_trades += 1
            else:
                pnl = (price - entry_price_rev) * position_rev
                if abs(pnl / entry_price_rev) > 0.002:
                    reversion_pnl += pnl
                    position_rev = 0
                elif (position_rev > 0 and bb_z > 0) or (position_rev < 0 and bb_z < 0):
                    reversion_pnl += pnl
                    position_rev = 0
            
            # Momentum
            if position_mom == 0:
                if bb_z > 1.0 and pb_z > 1.0:
                    position_mom = 1
                    entry_price_mom = price
                    momentum_trades += 1
                elif bb_z < -1.0 and pb_z < -1.0:
                    position_mom = -1
                    entry_price_mom = price
                    momentum_trades += 1
            else:
                pnl = (price - entry_price_mom) * position_mom
                if abs(pnl / entry_price_mom) > 0.002:
                    momentum_pnl += pnl
                    position_mom = 0
                elif (position_mom > 0 and bb_z < 0) or (position_mom < 0 and bb_z > 0):
                    momentum_pnl += pnl
                    position_mom = 0
        
        if position_rev != 0:
            reversion_pnl += (prices[test_end_idx] - entry_price_rev) * position_rev
        if position_mom != 0:
            momentum_pnl += (prices[test_end_idx] - entry_price_mom) * position_mom
        
        if reversion_trades < min_trades and momentum_trades < min_trades:
            return 1, 0.5
        
        avg_reversion = reversion_pnl / max(reversion_trades, 1)
        avg_momentum = momentum_pnl / max(momentum_trades, 1)
        
        if avg_reversion > avg_momentum:
            regime = 0
            confidence = min(1.0, abs(avg_reversion - avg_momentum) / (abs(avg_reversion) + 1e-8))
        else:
            regime = 1
            confidence = min(1.0, abs(avg_momentum - avg_reversion) / (abs(avg_momentum) + 1e-8))
        
        return regime, confidence
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def generate_adaptive_signals(bb_norm, pb_norm, prices,
                                   pv_lower, pv_center, pv_upper, v_surge,
                                   regime, strong_z_thresh, strong_agree_thresh,
                                   weak_z_thresh, pv_breakout_pct, vol_surge_thresh):
        """
        FIXED: Aggressive weak SHORT signal generation
        """
        n = len(bb_norm)
        signals = np.zeros(n, dtype=np.int32)
        signal_quality = np.zeros(n, dtype=np.int32)
        
        for i in range(300, n):
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            price = prices[i]
            
            lower_band = pv_lower[i]
            center_band = pv_center[i]
            upper_band = pv_upper[i]
            
            # More lenient volume check
            has_volume_surge = v_surge[i] > vol_surge_thresh
            moderate_volume = v_surge[i] > vol_surge_thresh * 0.8
            
            # Check agreement
            same_sign = (bb_z > 0 and pb_z > 0) or (bb_z < 0 and pb_z < 0)
            both_strong = (abs(bb_z) > strong_z_thresh * strong_agree_thresh and 
                          abs(pb_z) > strong_z_thresh * strong_agree_thresh)
            
            # ==================== STRONG SIGNALS (UNCHANGED) ====================
            if same_sign and both_strong:
                if regime == 0:  # MEAN REVERSION
                    if bb_z < 0 and pb_z < 0:
                        signals[i] = 1
                        signal_quality[i] = 1
                        continue
                    elif bb_z > 0 and pb_z > 0:
                        signals[i] = -1
                        signal_quality[i] = 1
                        continue
                else:  # MOMENTUM
                    if bb_z > 0 and pb_z > 0:
                        signals[i] = 1
                        signal_quality[i] = 1
                        continue
                    elif bb_z < 0 and pb_z < 0:
                        signals[i] = -1
                        signal_quality[i] = 1
                        continue
            
            # ==================== WEAK SIGNALS - CRITICAL FIX ====================
            
            # Calculate thresholds
            upper_breakout = upper_band * (1.0 + pv_breakout_pct)
            lower_breakout = lower_band * (1.0 - pv_breakout_pct)
            center_upper = center_band * (1.0 + pv_breakout_pct * 0.5)
            center_lower = center_band * (1.0 - pv_breakout_pct * 0.5)
            
            # Check for ANY signal strength (not just moderate)
            has_bb_signal = abs(bb_z) > weak_z_thresh * 0.7
            has_pb_signal = abs(pb_z) > weak_z_thresh * 0.7
            
            if not (has_bb_signal or has_pb_signal):
                continue
            
            # ==================== PRIORITY 1: Band Breakouts ====================
            
            # LONG: Upper band breakout
            if price > upper_breakout:
                if moderate_volume and (bb_z > 0 or pb_z > 0):
                    signals[i] = 1
                    signal_quality[i] = 2
                    continue
            
            # SHORT: Lower band breakout (CRITICAL)
            if price < lower_breakout:
                if moderate_volume and (bb_z < 0 or pb_z < 0):
                    signals[i] = -1
                    signal_quality[i] = 2
                    continue
            
            # ==================== PRIORITY 2: Center Band Crossovers ====================
            
            # LONG: Above center with momentum
            if price > center_upper:
                if moderate_volume and (bb_z > weak_z_thresh * 0.6 or pb_z > weak_z_thresh * 0.6):
                    signals[i] = 1
                    signal_quality[i] = 2
                    continue
            
            # SHORT: Below center with momentum (CRITICAL)
            if price < center_lower:
                if moderate_volume and (bb_z < -weak_z_thresh * 0.6 or pb_z < -weak_z_thresh * 0.6):
                    signals[i] = -1
                    signal_quality[i] = 2
                    continue
            
            # ==================== PRIORITY 3: Pure Indicator Signals ====================
            
            # Both indicators pointing same direction (no volume requirement)
            both_negative = (bb_z < -weak_z_thresh and pb_z < -weak_z_thresh)
            both_positive = (bb_z > weak_z_thresh and pb_z > weak_z_thresh)
            
            # SHORT: Strong bearish indicators
            if both_negative:
                if price <= center_band:  # Below or at center
                    signals[i] = -1
                    signal_quality[i] = 2
                    continue
            
            # LONG: Strong bullish indicators
            if both_positive:
                if price >= center_band:  # Above or at center
                    signals[i] = 1
                    signal_quality[i] = 2
                    continue
            
            # ==================== PRIORITY 4: Single Strong Indicator ====================
            
            # Very strong single indicator (relaxed)
            very_strong_bb = abs(bb_z) > weak_z_thresh * 1.3
            very_strong_pb = abs(pb_z) > weak_z_thresh * 1.3
            
            if very_strong_bb or very_strong_pb:
                # SHORT: Price below center + strong negative signal
                if (bb_z < -weak_z_thresh * 1.3 or pb_z < -weak_z_thresh * 1.3):
                    if price < center_band:
                        signals[i] = -1
                        signal_quality[i] = 2
                        continue
                
                # LONG: Price above center + strong positive signal
                if (bb_z > weak_z_thresh * 1.3 or pb_z > weak_z_thresh * 1.3):
                    if price > center_band:
                        signals[i] = 1
                        signal_quality[i] = 2
                        continue
        
        return signals, signal_quality
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def manage_positions(signals, signal_quality, prices, timestamps,
                         vol_proxy, min_duration, min_hold, max_duration,
                         sl_strong_mult, sl_weak_mult,
                         tp_strong_mult, tp_weak_mult,
                         trail_strong_mult, trail_weak_mult):
        """Position management - NO filtering of weak signals"""
        n = len(signals)
        positions = np.zeros(n, dtype=np.int32)
        
        current_pos = 0
        entry_price = 0.0
        entry_time = 0.0
        entry_quality = 0
        entry_vol = 0.0
        highest_price_in_trade = 0.0
        lowest_price_in_trade = 999999.0
        max_favorable_pts = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            # Entry: Accept ALL signals (both strong and weak)
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_quality = signal_quality[i]
                    entry_vol = vol_proxy[i]
                    highest_price_in_trade = prices[i]
                    lowest_price_in_trade = prices[i]
                    max_favorable_pts = 0.0
            else:
                # Exit logic
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pts = (prices[i] - entry_price) * direction
                max_favorable_pts = max(max_favorable_pts, pnl_pts)
                
                if current_pos > 0:
                    highest_price_in_trade = max(highest_price_in_trade, prices[i])
                else:
                    lowest_price_in_trade = min(lowest_price_in_trade, prices[i])
                
                # Risk parameters based on quality
                if entry_quality == 1:  # Strong
                    stop_loss_pts = entry_vol * sl_strong_mult
                    take_profit_pts = entry_vol * tp_strong_mult
                    trailing_stop_pts = entry_vol * trail_strong_mult
                else:  # Weak
                    stop_loss_pts = entry_vol * sl_weak_mult
                    take_profit_pts = entry_vol * tp_weak_mult
                    trailing_stop_pts = entry_vol * trail_weak_mult
                
                # Trailing level
                if current_pos > 0:
                    trailing_level = highest_price_in_trade - trailing_stop_pts
                else:
                    trailing_level = lowest_price_in_trade + trailing_stop_pts
                
                should_exit = False
                
                # Exit conditions
                if time_in_trade < min_duration:
                    pass
                elif pnl_pts <= -stop_loss_pts:
                    should_exit = True
                elif pnl_pts >= take_profit_pts:
                    should_exit = True
                elif max_favorable_pts >= take_profit_pts * 0.4:
                    if current_pos > 0 and prices[i] < trailing_level:
                        should_exit = True
                    elif current_pos < 0 and prices[i] > trailing_level:
                        should_exit = True
                elif time_in_trade >= max_duration:
                    should_exit = True
                elif time_in_trade >= min_hold:
                    if current_pos > 0 and signals[i] < 0:
                        should_exit = True
                    elif current_pos < 0 and signals[i] > 0:
                        should_exit = True
                
                if should_exit:
                    current_pos = 0
                    entry_price = 0.0
                    entry_time = 0.0
                    entry_quality = 0
                    entry_vol = 0.0
                    highest_price_in_trade = 0.0
                    lowest_price_in_trade = 999999.0
                    max_favorable_pts = 0.0
            
            positions[i] = current_pos
        
        positions[-1] = 0
        return positions
    
    def _calculate_weighted_signal(self, df, features, weights):
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
        filename = f"day{day_num}.parquet"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_parquet(filepath)
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            
            required_features = (self.bb_features + self.pb_features + 
                                self.pv_features + self.v_features)
            missing = [f for f in required_features if f not in df.columns]
            
            if len(missing) > 0:
                for feat in missing:
                    df[feat] = 0
            
            for feat in required_features:
                if feat in df.columns:
                    df[feat] = df[feat].fillna(method='ffill').fillna(0)
            
            df['timestamp_sec'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds()
            df['time_duration'] = pd.to_timedelta(df['timestamp_sec'], unit='s')
            
            prices = df['Price'].values.astype(np.float32)
            timestamps = df['timestamp_sec'].values.astype(np.float32)
            
            pv_lower_raw = df['PV3_B3_T6'].values.astype(np.float32)
            pv_upper_raw = df['PV3_B5_T6'].values.astype(np.float32)
            pv_band_width = pd.Series(pv_upper_raw - pv_lower_raw).rolling(window=1800, min_periods=1).mean()
            pv_band_width = pv_band_width.fillna(method='ffill').fillna(method='bfill')
            pv_band_width = pv_band_width.replace(0, np.nan).fillna(method='ffill').fillna(0.0001)
            vol_proxy = pv_band_width.values.astype(np.float32)
            
            bb_signal_init = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights_momentum)
            pb_signal_init = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights_momentum)
            
            bb_norm_init = self.normalize_signal(bb_signal_init, window=300)
            pb_norm_init = self.normalize_signal(pb_signal_init, window=300)
            
            regime, confidence = self.detect_regime(
                bb_norm_init, pb_norm_init, prices, timestamps,
                self.regime_test_duration, self.regime_test_min_trades
            )
            
            regime_name = "MEAN REVERSION" if regime == 0 else "MOMENTUM"
            
            if regime == 0:
                bb_weights = self.bb_weights_reversion
                pb_weights = self.pb_weights_reversion
            else:
                bb_weights = self.bb_weights_momentum
                pb_weights = self.pb_weights_momentum
            
            bb_signal = self._calculate_weighted_signal(df, self.bb_features, bb_weights)
            pb_signal = self._calculate_weighted_signal(df, self.pb_features, pb_weights)
            v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
            
            pv_lower = df['PV3_B3_T6'].values.astype(np.float32)
            pv_center = df['PV3_B4_T6'].values.astype(np.float32)
            pv_upper = df['PV3_B5_T6'].values.astype(np.float32)
            
            bb_normalized = self.normalize_signal(bb_signal, window=300)
            pb_normalized = self.normalize_signal(pb_signal, window=300)
            
            v_mean = self.calculate_rolling_mean(v_signal, 600)
            v_surge = v_signal / (v_mean + 1e-8)
            
            signals, signal_quality = self.generate_adaptive_signals(
                bb_normalized, pb_normalized, prices,
                pv_lower, pv_center, pv_upper, v_surge,
                regime,
                self.strong_z_threshold, self.strong_agreement_threshold,
                self.weak_z_threshold, self.pv_band_breakout_pct,
                self.volume_surge_threshold
            )
            
            positions = self.manage_positions(
                signals, signal_quality, prices, timestamps,
                vol_proxy,
                self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
                self.sl_strong_mult, self.sl_weak_mult,
                self.tp_strong_mult, self.tp_weak_mult,
                self.trail_strong_mult, self.trail_weak_mult
            )
            
            strong_long = ((signals == 1) & (signal_quality == 1)).sum()
            strong_short = ((signals == -1) & (signal_quality == 1)).sum()
            weak_long = ((signals == 1) & (signal_quality == 2)).sum()
            weak_short = ((signals == -1) & (signal_quality == 2)).sum()
            
            print(f"Day {day_num:3d} [{regime_name:15s} conf:{confidence:.2f}]: "
                  f"Strong(L:{strong_long:3d},S:{strong_short:3d}) "
                  f"Weak(L:{weak_long:3d},S:{weak_short:3d})")
            
            result_df = pd.DataFrame({
                'Time': df['time_duration'],
                'Signal': positions.astype(np.int32),
                'Price': prices,
                'Day': day_num,
                'Regime': regime_name
            })
            
            return result_df
            
        except Exception as e:
            print(f"Day {day_num}: Error - {e}")
            return None
    
    def run_strategy(self, num_days=510, data_folder='/data/quant14/EBX/', max_workers=25):
        print("="*80)
        print("ADAPTIVE REGIME STRATEGY - WEAK SHORTS FULLY ENABLED")
        print("="*80)
        print(f"✓ CRITICAL FIX: 4 priority levels for weak SHORT detection")
        print(f"✓ Lower thresholds: weak_z=0.55, volume=1.15")
        print(f"✓ No signal filtering in position management")
        print(f"✓ INPUT: {data_folder}")
        print(f"✓ CPU Workers: {max_workers}")
        print("="*80 + "\n")
        
        process_func = partial(self.process_day, data_folder=data_folder)
        day_results = {}
        
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
            
            sorted_days = sorted(day_results.keys())
            all_results = [day_results[day] for day in sorted_days]
            portfolio_weights = pd.concat(all_results, ignore_index=True)
            
            regime_counts = portfolio_weights.groupby('Regime')['Day'].nunique()
            print(f"\n📊 REGIME DISTRIBUTION:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} days ({count/len(day_results)*100:.1f}%)")
            
            portfolio_weights_output = portfolio_weights[['Time', 'Signal', 'Price']].copy()
            portfolio_weights_output.to_csv('portfolio_weights.csv', index=False)
            
            print(f"\n✓ Saved portfolio_weights.csv ({len(portfolio_weights_output):,} rows)")
            
            total_signals = len(portfolio_weights_output)
            total_long = (portfolio_weights_output['Signal'] > 0).sum()
            total_short = (portfolio_weights_output['Signal'] < 0).sum()
            
            print(f"\n✅ SUMMARY:")
            print(f"  Days: {len(day_results)}")
            print(f"  Total: {total_signals:,}")
            if total_signals > 0:
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
            print("="*80)
            
            gc.collect()
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = AdaptiveRegimeTradingStrategy()
    
    print("\n" + "="*80)
    print("STARTING STRATEGY - WEAK SHORTS AGGRESSIVELY ENABLED")
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✓ Strategy complete!")
        print("✓ portfolio_weights.csv ready")
        print("✓ Weak SHORT trades NOW ACTIVE")
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")