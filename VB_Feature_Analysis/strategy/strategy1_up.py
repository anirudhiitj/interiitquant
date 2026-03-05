"""
Adaptive Regime-Switching Two-Tier Signal Trading Strategy
===========================================================
<<<<<<< HEAD
CRITICAL FIX: Weak SHORT signals now properly converted to positions
=======
ULTRA-SELECTIVE: 20-30 high-conviction trades/day with maximum profit potential
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
<<<<<<< HEAD
    Adaptive strategy with WORKING weak signal SHORT trades
=======
    ULTRA-SELECTIVE strategy: Only the absolute best setups
    Target: 20-30 trades/day with 70%+ win rate and minimal drawdown
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
        
<<<<<<< HEAD
        # MEAN REVERSION (negative = fade extremes)
=======
        # MEAN REVERSION
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        self.bb_weights_reversion = {
            'BB4_T10': -0.35, 'BB4_T11': -0.30, 'BB4_T12': -0.20,
            'BB3_T10': -0.10, 'BB5_T11': -0.05
        }
        
        self.pb_weights_reversion = {
            'PB1_T10': -0.15, 'PB1_T11': -0.15, 'PB6_T11': -0.15,
            'PB6_T12': -0.10, 'PB10_T11': -0.15, 'PB11_T11': 0.15,
            'PB13_T10': -0.10, 'PB13_T11': -0.05
        }
        
<<<<<<< HEAD
        # MOMENTUM (positive = follow trends)
=======
        # MOMENTUM
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
        
<<<<<<< HEAD
        # ==================== SIGNAL PARAMETERS ====================
        
        # Strong Signals (keep as is)
        self.strong_z_threshold = 1.1
        self.strong_agreement_threshold = 0.6
        
        # CRITICAL: More aggressive weak signal thresholds
        self.weak_z_threshold = 0.6  # Even lower to catch more signals
        self.pv_band_breakout_pct = 0.0001  # More sensitive
        self.volume_surge_threshold = 1.15  # Lower barrier
        
        # ==================== RISK MANAGEMENT ====================
        
        # Strong Signals (unchanged)
        self.sl_strong_mult = 0.55
        self.tp_strong_mult = 1.1
        self.trail_strong_mult = 0.4
        
        # Weak signals - balanced risk/reward
        self.sl_weak_mult = 0.4
        self.tp_weak_mult = 0.85
        self.trail_weak_mult = 0.35
        
        # Position Management
        self.min_trade_duration = 15
        self.max_trade_duration = 540
        self.min_hold_time = 15
=======
        # ==================== BALANCED SELECTIVE SIGNAL PARAMETERS ====================
        
        # STRONG SIGNALS: High conviction but achievable
        self.strong_z_threshold = 1.1  # Balanced selectivity
        self.strong_agreement_threshold = 0.6  # Both strong, not extreme
        
        # WEAK SIGNALS: High quality weak signals only
        self.weak_signals_enabled = True  # Enable for more opportunities
        self.weak_z_threshold = 0.7  # Higher than normal for quality
        self.weak_agreement_threshold = 0.65  # Require some agreement
        
        # Volume: Significant but achievable
        self.volume_surge_threshold = 1.1  # Strong volume, not extreme
        
        # Band extremes: Reasonable extremes
        self.band_extreme_threshold = 0.1  # Top/bottom 20% (not 5%)
        
        # ==================== AGGRESSIVE RISK MANAGEMENT ====================
        
        # Strong Signals: Great risk/reward
        self.sl_strong_mult = 0.55   # Tight stop loss
        self.tp_strong_mult = 1.8    # 3.27:1 R/R ratio
        self.trail_strong_mult = 0.35  # Aggressive trailing
        
        # Weak Signals: Good risk/reward
        self.sl_weak_mult = 0.50
        self.tp_weak_mult = 1.4
        self.trail_weak_mult = 0.30
        
        # Position Management: Hold for profits
        self.min_trade_duration = 20   # Hold at least 20 seconds
        self.max_trade_duration = 800  # Max ~13 minutes
        self.min_hold_time = 30  # Minimum hold before reversal exit
        
        # Time filters: Avoid volatile periods
        self.avoid_first_minutes = 180  # Skip first 3 min
        self.avoid_last_minutes = 180   # Skip last 3 min
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        
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
<<<<<<< HEAD
    def generate_adaptive_signals(bb_norm, pb_norm, prices,
                                   pv_lower, pv_center, pv_upper, v_surge,
                                   regime, strong_z_thresh, strong_agree_thresh,
                                   weak_z_thresh, pv_breakout_pct, vol_surge_thresh):
        """
        FIXED: Aggressive weak SHORT signal generation
=======
    def generate_ultra_selective_signals(bb_norm, pb_norm, prices,
                                         pv_lower, pv_center, pv_upper, v_surge,
                                         timestamps, regime, 
                                         strong_z_thresh, strong_agree_thresh,
                                         weak_z_thresh, weak_agree_thresh,
                                         vol_surge_thresh, band_extreme_thresh,
                                         avoid_first, avoid_last):
        """
        BALANCED SELECTIVE: High-quality trades with 20-30/day target
        
        Strong signals: Perfect setups
        Weak signals: Very good setups (higher threshold than normal)
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        """
        n = len(bb_norm)
        signals = np.zeros(n, dtype=np.int32)
        signal_quality = np.zeros(n, dtype=np.int32)
        
<<<<<<< HEAD
        for i in range(300, n):
=======
        # Find valid trading window
        start_time = timestamps[0]
        end_time = timestamps[-1]
        valid_start = start_time + avoid_first
        valid_end = end_time - avoid_last
        
        for i in range(300, n):
            # Time filter
            if timestamps[i] < valid_start or timestamps[i] > valid_end:
                continue
            
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            price = prices[i]
            
            lower_band = pv_lower[i]
<<<<<<< HEAD
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
=======
            upper_band = pv_upper[i]
            center_band = pv_center[i]
            
            # Band width check
            band_width = upper_band - lower_band
            if band_width < 1e-8:
                continue
            
            # Normalize price position
            price_position = (price - lower_band) / band_width
            
            # Volume requirement (relaxed for more opportunities)
            has_strong_volume = v_surge[i] > vol_surge_thresh
            has_moderate_volume = v_surge[i] > vol_surge_thresh * 0.85
            
            # ==================== STRONG SIGNALS ====================
            
            both_extremely_bullish = (bb_z > strong_z_thresh * strong_agree_thresh and 
                                     pb_z > strong_z_thresh * strong_agree_thresh)
            
            both_extremely_bearish = (bb_z < -strong_z_thresh * strong_agree_thresh and 
                                     pb_z < -strong_z_thresh * strong_agree_thresh)
            
            if both_extremely_bullish or both_extremely_bearish:
                if has_strong_volume:
                    if regime == 0:  # MEAN REVERSION
                        if both_extremely_bearish and price_position < (1.0 - band_extreme_thresh):
                            signals[i] = 1
                            signal_quality[i] = 1
                            continue
                        
                        if both_extremely_bullish and price_position > band_extreme_thresh:
                            signals[i] = -1
                            signal_quality[i] = 1
                            continue
                    
                    else:  # MOMENTUM
                        if both_extremely_bullish and price_position > band_extreme_thresh:
                            signals[i] = 1
                            signal_quality[i] = 1
                            continue
                        
                        if both_extremely_bearish and price_position < (1.0 - band_extreme_thresh):
                            signals[i] = -1
                            signal_quality[i] = 1
                            continue
            
            # ==================== WEAK SIGNALS (HIGH QUALITY) ====================
            
            # Check for good (not perfect) agreement
            both_bullish = (bb_z > weak_z_thresh * weak_agree_thresh and 
                           pb_z > weak_z_thresh * weak_agree_thresh)
            
            both_bearish = (bb_z < -weak_z_thresh * weak_agree_thresh and 
                           pb_z < -weak_z_thresh * weak_agree_thresh)
            
            if not (both_bullish or both_bearish):
                continue
            
            # Require at least moderate volume
            if not has_moderate_volume:
                continue
            
            # WEAK LONG conditions
            if both_bullish:
                # Near upper band OR strong breakout
                if price_position > 0.70:  # Top 30%
                    if regime == 1:  # Momentum favors this
                        signals[i] = 1
                        signal_quality[i] = 2
                        continue
                    elif has_strong_volume:  # Mean reversion needs strong volume
                        signals[i] = 1
                        signal_quality[i] = 2
                        continue
                
                # Very strong single indicator
                if bb_z > weak_z_thresh * 1.5 or pb_z > weak_z_thresh * 1.5:
                    if price_position > 0.60:
                        signals[i] = 1
                        signal_quality[i] = 2
                        continue
            
            # WEAK SHORT conditions (symmetric)
            if both_bearish:
                # Near lower band OR strong breakdown
                if price_position < 0.30:  # Bottom 30%
                    if regime == 1:  # Momentum favors this
                        signals[i] = -1
                        signal_quality[i] = 2
                        continue
                    elif has_strong_volume:  # Mean reversion needs strong volume
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
                        signals[i] = -1
                        signal_quality[i] = 2
                        continue
                
<<<<<<< HEAD
                # LONG: Price above center + strong positive signal
                if (bb_z > weak_z_thresh * 1.3 or pb_z > weak_z_thresh * 1.3):
                    if price > center_band:
                        signals[i] = 1
=======
                # Very strong single indicator
                if bb_z < -weak_z_thresh * 1.5 or pb_z < -weak_z_thresh * 1.5:
                    if price_position < 0.40:
                        signals[i] = -1
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
<<<<<<< HEAD
        """Position management - NO filtering of weak signals"""
=======
        """
        AGGRESSIVE position management with quality-based risk parameters
        """
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
            
<<<<<<< HEAD
            # Entry: Accept ALL signals (both strong and weak)
=======
            # Entry
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
<<<<<<< HEAD
                # Exit logic
=======
                # Track performance
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pts = (prices[i] - entry_price) * direction
                max_favorable_pts = max(max_favorable_pts, pnl_pts)
                
                if current_pos > 0:
                    highest_price_in_trade = max(highest_price_in_trade, prices[i])
                else:
                    lowest_price_in_trade = min(lowest_price_in_trade, prices[i])
                
<<<<<<< HEAD
                # Risk parameters based on quality
=======
                # Select risk parameters based on quality
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
                
<<<<<<< HEAD
                # Exit conditions
=======
                # Exit logic
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
                if time_in_trade < min_duration:
                    pass
                elif pnl_pts <= -stop_loss_pts:
                    should_exit = True
                elif pnl_pts >= take_profit_pts:
                    should_exit = True
<<<<<<< HEAD
                elif max_favorable_pts >= take_profit_pts * 0.4:
=======
                elif max_favorable_pts >= take_profit_pts * 0.5:
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
            
<<<<<<< HEAD
=======
            # Volatility proxy
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            pv_lower_raw = df['PV3_B3_T6'].values.astype(np.float32)
            pv_upper_raw = df['PV3_B5_T6'].values.astype(np.float32)
            pv_band_width = pd.Series(pv_upper_raw - pv_lower_raw).rolling(window=1800, min_periods=1).mean()
            pv_band_width = pv_band_width.fillna(method='ffill').fillna(method='bfill')
            pv_band_width = pv_band_width.replace(0, np.nan).fillna(method='ffill').fillna(0.0001)
            vol_proxy = pv_band_width.values.astype(np.float32)
            
<<<<<<< HEAD
=======
            # Regime detection
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            bb_signal_init = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights_momentum)
            pb_signal_init = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights_momentum)
            
            bb_norm_init = self.normalize_signal(bb_signal_init, window=300)
            pb_norm_init = self.normalize_signal(pb_signal_init, window=300)
            
            regime, confidence = self.detect_regime(
                bb_norm_init, pb_norm_init, prices, timestamps,
                self.regime_test_duration, self.regime_test_min_trades
            )
            
            regime_name = "MEAN REVERSION" if regime == 0 else "MOMENTUM"
            
<<<<<<< HEAD
=======
            # Apply regime-specific weights
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
            
<<<<<<< HEAD
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
            
=======
            # Volume surge
            v_mean = self.calculate_rolling_mean(v_signal, 600)
            v_surge = v_signal / (v_mean + 1e-8)
            
            # Generate balanced selective signals
            signals, signal_quality = self.generate_ultra_selective_signals(
                bb_normalized, pb_normalized, prices,
                pv_lower, pv_center, pv_upper, v_surge,
                timestamps, regime,
                self.strong_z_threshold, self.strong_agreement_threshold,
                self.weak_z_threshold, self.weak_agreement_threshold,
                self.volume_surge_threshold, self.band_extreme_threshold,
                self.avoid_first_minutes, self.avoid_last_minutes
            )
            
            # Manage positions with quality-based parameters
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            positions = self.manage_positions(
                signals, signal_quality, prices, timestamps,
                vol_proxy,
                self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
                self.sl_strong_mult, self.sl_weak_mult,
                self.tp_strong_mult, self.tp_weak_mult,
                self.trail_strong_mult, self.trail_weak_mult
            )
            
<<<<<<< HEAD
=======
            # Count signals by quality
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            strong_long = ((signals == 1) & (signal_quality == 1)).sum()
            strong_short = ((signals == -1) & (signal_quality == 1)).sum()
            weak_long = ((signals == 1) & (signal_quality == 2)).sum()
            weak_short = ((signals == -1) & (signal_quality == 2)).sum()
<<<<<<< HEAD
            
            print(f"Day {day_num:3d} [{regime_name:15s} conf:{confidence:.2f}]: "
                  f"Strong(L:{strong_long:3d},S:{strong_short:3d}) "
                  f"Weak(L:{weak_long:3d},S:{weak_short:3d})")
=======
            total_trades = strong_long + strong_short + weak_long + weak_short
            
            print(f"Day {day_num:3d} [{regime_name:15s} conf:{confidence:.2f}]: "
                  f"Strong(L:{strong_long:2d},S:{strong_short:2d}) "
                  f"Weak(L:{weak_long:2d},S:{weak_short:2d}) = {total_trades:2d} trades")
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            
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
<<<<<<< HEAD
        print("ADAPTIVE REGIME STRATEGY - WEAK SHORTS FULLY ENABLED")
        print("="*80)
        print(f"✓ CRITICAL FIX: 4 priority levels for weak SHORT detection")
        print(f"✓ Lower thresholds: weak_z=0.55, volume=1.15")
        print(f"✓ No signal filtering in position management")
        print(f"✓ INPUT: {data_folder}")
=======
        print("BALANCED SELECTIVE TRADING STRATEGY")
        print("="*80)
        print(f"🎯 TARGET: 20-30 high-quality trades/day")
        print(f"✓ Strong Z-Score: {self.strong_z_threshold} (Agreement: {self.strong_agreement_threshold*100:.0f}%)")
        print(f"✓ Weak Z-Score: {self.weak_z_threshold} (Agreement: {self.weak_agreement_threshold*100:.0f}%)")
        print(f"✓ Volume Surge: >{self.volume_surge_threshold}x average")
        print(f"✓ Band Extremes: Top/Bottom {(1-self.band_extreme_threshold)*100:.0f}%")
        print(f"✓ Strong R/R: 1:{self.tp_strong_mult/self.sl_strong_mult:.1f}")
        print(f"✓ Weak R/R: 1:{self.tp_weak_mult/self.sl_weak_mult:.1f}")
        print(f"✓ Time Filters: Skip first/last {self.avoid_first_minutes}s")
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
            
<<<<<<< HEAD
            print(f"\n✅ SUMMARY:")
            print(f"  Days: {len(day_results)}")
            print(f"  Total: {total_signals:,}")
            if total_signals > 0:
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
=======
            print(f"\n✅ FINAL SUMMARY:")
            print(f"  Days: {len(day_results)}")
            print(f"  Total Signals: {total_signals:,}")
            if total_signals > 0:
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
                avg_per_day = total_signals / len(day_results)
                print(f"  Avg/Day: {avg_per_day:.1f} trades")
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            print("="*80)
            
            gc.collect()
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = AdaptiveRegimeTradingStrategy()
    
    print("\n" + "="*80)
<<<<<<< HEAD
    print("STARTING STRATEGY - WEAK SHORTS AGGRESSIVELY ENABLED")
=======
    print("ULTRA-SELECTIVE STRATEGY - QUALITY OVER QUANTITY")
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✓ Strategy complete!")
        print("✓ portfolio_weights.csv ready")
<<<<<<< HEAD
        print("✓ Weak SHORT trades NOW ACTIVE")
=======
        print("✓ Only highest-conviction trades executed")
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")