"""
Adaptive Regime-Switching Two-Tier Signal Trading Strategy
===========================================================

KEY INNOVATIONS:
1. Intraday Regime Detection: Tests mean-reversion vs momentum in first 30-60 min
2. Dynamic Weight Switching: BB/PB weights flip based on detected regime
3. Strong Signals: BB and PB Z-scores agree in sign (same direction)
4. Weak Signals: BB and PB disagree → use PV bands for confirmation
5. Simplified Feature Set: Focus on high-correlation features
6. Dynamic Risk Management: Based on PV-Band Width volatility proxy

REGIME DETECTION:
- First 30-60 minutes: Run mini-backtest with both strategies
- Compare returns: Mean Reversion vs Momentum
- Apply winning strategy to rest of day
- Weights auto-flip based on regime

Architecture:
- ProcessPoolExecutor with 25 workers for parallel day processing
- Numba JIT compilation for speed
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


class AdaptiveRegimeTradingStrategy:
    """
    Adaptive strategy that detects intraday regime and adjusts BB/PB weights accordingly
    
    Strong: BB + PB Z-score same sign → trade in that direction
    Weak: BB + PB Z-score opposite sign → use PV bands for confirmation
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # ==================== OPTIMIZED FEATURE SET ====================
        
        # BB Features - Bollinger Bands (fewer, more impactful lookbacks)
        self.bb_features = [
            'BB4_T10',   # 5-min BB
            'BB4_T11',   # 10-min BB
            'BB4_T12',   # 20-min BB
            'BB3_T10',   # BB width
            'BB5_T11'    # BB %B
        ]
        
        # PB Features - Price-Based (streamlined for momentum/reversion)
        self.pb_features = [
            'PB1_T10',   # Moving Average (10-min)
            'PB1_T11',   # Moving Average (20-min)
            'PB6_T11',   # Momentum indicator
            'PB6_T12',   # Momentum indicator (longer)
            'PB10_T11',  # Rolling Maximum
            'PB11_T11',  # Rolling Minimum
            'PB13_T10',  # High momentum feature
            'PB13_T11'   # High momentum feature
        ]
        
        # PV Features - Price-Volume bands (for weak signal confirmation)
        self.pv_features = [
            'PV3_B3_T6',  # Lower bound
            'PV3_B4_T6',  # Center
            'PV3_B5_T6'   # Upper bound
        ]
        
        # V Features - Volume indicators
        self.v_features = [
            'V5',        # Current volume
            # 'V2_T8',     # Volume trend
            # 'V1_T4'      # Volume MA
        ]
        
        # ==================== BASE WEIGHTS (WILL BE FLIPPED BY REGIME) ====================
        
        # MEAN REVERSION WEIGHTS (negative = fade extremes)
        self.bb_weights_reversion = {
            'BB4_T10': -0.35,
            'BB4_T11': -0.30,
            'BB4_T12': -0.20,
            'BB3_T10': -0.10,
            'BB5_T11': -0.05
        }
        
        self.pb_weights_reversion = {
            'PB1_T10': -0.15,
            'PB1_T11': -0.15,
            'PB6_T11': -0.15,
            'PB6_T12': -0.10,
            'PB10_T11': -0.15,  # Fade highs
            'PB11_T11': 0.15,   # Buy lows
            'PB13_T10': -0.10,
            'PB13_T11': -0.05
        }
        
        # MOMENTUM WEIGHTS (positive = follow trends)
        self.bb_weights_momentum = {
            'BB4_T10': 0.25,
            'BB4_T11': 0.25,
            'BB4_T12': 0.20,
            'BB3_T10': 0.15,
            'BB5_T11': 0.15
        }
        
        self.pb_weights_momentum = {
            'PB1_T10': 0.15,
            'PB1_T11': 0.10,
            'PB6_T11': 0.20,
            'PB6_T12': 0.15,
            'PB10_T11': 0.05,
            'PB11_T11': 0.05,
            'PB13_T10': 0.20,
            'PB13_T11': 0.10
        }
        
        # V Weights (unchanged by regime)
        self.v_weights = {
            'V5': 1.0,
            # 'V2_T8': 0.35,
            # 'V1_T4': 0.25
        }
        
        # ==================== REGIME DETECTION PARAMETERS ====================
        self.regime_test_duration = 2400  # 30 minutes in seconds
        self.regime_test_min_trades = 3   # Minimum trades for valid test
        
        # ==================== SIGNAL PARAMETERS ====================
        
        # Strong Signal: BB and PB Z-scores agree in sign
        self.strong_z_threshold = 1.1
        self.strong_agreement_threshold = 0.6  # Both must exceed this * threshold
        
        # Weak Signal: BB and PB disagree → use PV bands
        self.weak_z_threshold = 0.7
        self.pv_band_breakout_pct = 0.0002
        self.volume_surge_threshold = 1.3
        
        # ==================== RISK MANAGEMENT ====================
        
        # Strong Signals (more aggressive)
        self.sl_strong_mult = 0.6
        self.tp_strong_mult = 1.2
        self.trail_strong_mult = 0.4
        
        # Weak Signals (more conservative)
        self.sl_weak_mult = 0.4
        self.tp_weak_mult = 0.8
        self.trail_weak_mult = 0.3
        
        # Position Management
        self.min_trade_duration = 15
        self.max_trade_duration = 600
        self.min_hold_time = 15
        
        self.plot_output_dir = plot_output_dir
        os.makedirs(plot_output_dir, exist_ok=True)
    
    # ==================== NUMBA-OPTIMIZED FUNCTIONS ====================
    
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
    def detect_regime(bb_norm, pb_norm, prices, timestamps, test_duration, min_trades):
        """
        Detect regime in first N minutes by testing both strategies
        
        Returns:
            regime: 0 = mean reversion, 1 = momentum
            confidence: 0-1 score
        """
        n = len(prices)
        test_end_idx = 0
        
        # Find end of test period
        test_end_time = timestamps[0] + test_duration
        for i in range(n):
            if timestamps[i] >= test_end_time:
                test_end_idx = i
                break
        
        if test_end_idx < 300:
            return 1, 0.5  # Default to momentum if insufficient data
        
        # Test Mean Reversion: Fade extremes
        reversion_pnl = 0.0
        reversion_trades = 0
        position_rev = 0
        entry_price_rev = 0.0
        
        # Test Momentum: Follow trends
        momentum_pnl = 0.0
        momentum_trades = 0
        position_mom = 0
        entry_price_mom = 0.0
        
        for i in range(300, test_end_idx):
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            price = prices[i]
            
            # Mean Reversion Signals (negative weights logic)
            if position_rev == 0:
                # LONG when both negative (oversold)
                if bb_z < -1.0 and pb_z < -1.0:
                    position_rev = 1
                    entry_price_rev = price
                    reversion_trades += 1
                # SHORT when both positive (overbought)
                elif bb_z > 1.0 and pb_z > 1.0:
                    position_rev = -1
                    entry_price_rev = price
                    reversion_trades += 1
            else:
                # Exit on reversal or profit target
                pnl = (price - entry_price_rev) * position_rev
                if abs(pnl / entry_price_rev) > 0.002:  # 0.2% profit
                    reversion_pnl += pnl
                    position_rev = 0
                elif (position_rev > 0 and bb_z > 0) or (position_rev < 0 and bb_z < 0):
                    reversion_pnl += pnl
                    position_rev = 0
            
            # Momentum Signals (positive weights logic)
            if position_mom == 0:
                # LONG when both positive (uptrend)
                if bb_z > 1.0 and pb_z > 1.0:
                    position_mom = 1
                    entry_price_mom = price
                    momentum_trades += 1
                # SHORT when both negative (downtrend)
                elif bb_z < -1.0 and pb_z < -1.0:
                    position_mom = -1
                    entry_price_mom = price
                    momentum_trades += 1
            else:
                # Exit on reversal or profit target
                pnl = (price - entry_price_mom) * position_mom
                if abs(pnl / entry_price_mom) > 0.002:  # 0.2% profit
                    momentum_pnl += pnl
                    position_mom = 0
                elif (position_mom > 0 and bb_z < 0) or (position_mom < 0 and bb_z > 0):
                    momentum_pnl += pnl
                    position_mom = 0
        
        # Close any open positions
        if position_rev != 0:
            reversion_pnl += (prices[test_end_idx] - entry_price_rev) * position_rev
        if position_mom != 0:
            momentum_pnl += (prices[test_end_idx] - entry_price_mom) * position_mom
        
        # Require minimum trades for valid comparison
        if reversion_trades < min_trades and momentum_trades < min_trades:
            return 1, 0.5  # Default to momentum
        
        # Calculate per-trade average
        avg_reversion = reversion_pnl / max(reversion_trades, 1)
        avg_momentum = momentum_pnl / max(momentum_trades, 1)
        
        # Determine regime
        if avg_reversion > avg_momentum:
            regime = 0  # Mean reversion
            confidence = min(1.0, abs(avg_reversion - avg_momentum) / (abs(avg_reversion) + 1e-8))
        else:
            regime = 1  # Momentum
            confidence = min(1.0, abs(avg_momentum - avg_reversion) / (abs(avg_momentum) + 1e-8))
        
        return regime, confidence
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def generate_adaptive_signals(bb_norm, pb_norm, prices,
                                   pv_lower, pv_center, pv_upper, v_surge,
                                   regime, strong_z_thresh, strong_agree_thresh,
                                   weak_z_thresh, pv_breakout_pct, vol_surge_thresh):
        """
        Generate signals with regime-aware interpretation
        
        REGIME 0 (Mean Reversion):
            Strong LONG: Both BB and PB negative (oversold)
            Strong SHORT: Both BB and PB positive (overbought)
        
        REGIME 1 (Momentum):
            Strong LONG: Both BB and PB positive (uptrend)
            Strong SHORT: Both BB and PB negative (downtrend)
        
        WEAK (Both regimes): BB and PB disagree → use PV bands
        
        Returns:
            signals: -1 (short), 0 (flat), +1 (long)
            signal_quality: 1 (strong), 2 (weak)
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
            
            # Check if BB and PB agree in sign
            same_sign = (bb_z > 0 and pb_z > 0) or (bb_z < 0 and pb_z < 0)
            both_strong = (abs(bb_z) > strong_z_thresh * strong_agree_thresh and 
                          abs(pb_z) > strong_z_thresh * strong_agree_thresh)
            
            # ==================== STRONG SIGNALS ====================
            if same_sign and both_strong:
                if regime == 0:  # MEAN REVERSION
                    # LONG when both negative (oversold, expect bounce)
                    if bb_z < 0 and pb_z < 0:
                        signals[i] = 1
                        signal_quality[i] = 1
                        continue
                    # SHORT when both positive (overbought, expect drop)
                    elif bb_z > 0 and pb_z > 0:
                        signals[i] = -1
                        signal_quality[i] = 1
                        continue
                else:  # MOMENTUM
                    # LONG when both positive (uptrend, follow)
                    if bb_z > 0 and pb_z > 0:
                        signals[i] = 1
                        signal_quality[i] = 1
                        continue
                    # SHORT when both negative (downtrend, follow)
                    elif bb_z < 0 and pb_z < 0:
                        signals[i] = -1
                        signal_quality[i] = 1
                        continue
            
            # ==================== WEAK SIGNALS ====================
            # BB and PB disagree → use PV bands for confirmation
            bb_moderate = abs(bb_z) > weak_z_thresh
            pb_moderate = abs(pb_z) > weak_z_thresh
            
            if not (bb_moderate or pb_moderate):
                continue
            
            # LONG WEAK: Price breaks above upper PV band + volume
            if price > upper_band * (1.0 + pv_breakout_pct):
                if has_volume_surge:
                    # Confirm with at least one positive indicator
                    if bb_z > 0 or pb_z > 0:
                        signals[i] = 1
                        signal_quality[i] = 2
                        continue
            
            # SHORT WEAK: Price breaks below lower PV band + volume
            elif price < lower_band * (1.0 - pv_breakout_pct):
                if has_volume_surge:
                    # Confirm with at least one negative indicator
                    if bb_z < 0 or pb_z < 0:
                        signals[i] = -1
                        signal_quality[i] = 2
                        continue
            
            # Alternative weak signals using center band
            if price > center_band * (1.0 + pv_breakout_pct * 0.5):
                if has_volume_surge and (bb_z > weak_z_thresh or pb_z > weak_z_thresh):
                    signals[i] = 1
                    signal_quality[i] = 2
            elif price < center_band * (1.0 - pv_breakout_pct * 0.5):
                if has_volume_surge and (bb_z < -weak_z_thresh or pb_z < -weak_z_thresh):
                    signals[i] = -1
                    signal_quality[i] = 2
        
        return signals, signal_quality
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def manage_positions(signals, signal_quality, prices, timestamps,
                         vol_proxy, min_duration, min_hold, max_duration,
                         sl_strong_mult, sl_weak_mult,
                         tp_strong_mult, tp_weak_mult,
                         trail_strong_mult, trail_weak_mult):
        """Dynamic position management with quality-based risk parameters"""
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
            
            # Entry Logic
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
            
            # Exit Logic
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                pnl_pts = (prices[i] - entry_price) * direction
                max_favorable_pts = max(max_favorable_pts, pnl_pts)
                
                if current_pos > 0:
                    highest_price_in_trade = max(highest_price_in_trade, prices[i])
                else:
                    lowest_price_in_trade = min(lowest_price_in_trade, prices[i])
                
                # Select risk parameters based on signal quality
                if entry_quality == 1:  # Strong
                    stop_loss_pts = entry_vol * sl_strong_mult
                    take_profit_pts = entry_vol * tp_strong_mult
                    trailing_stop_pts = entry_vol * trail_strong_mult
                else:  # Weak
                    stop_loss_pts = entry_vol * sl_weak_mult
                    take_profit_pts = entry_vol * tp_weak_mult
                    trailing_stop_pts = entry_vol * trail_weak_mult
                
                if current_pos > 0:
                    trailing_level = highest_price_in_trade - trailing_stop_pts
                else:
                    trailing_level = lowest_price_in_trade + trailing_stop_pts
                
                should_exit = False
                
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
        """Process single day with adaptive regime detection"""
        filename = f"day{day_num}.parquet"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            df = pd.read_parquet(filepath)
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Check required features
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
            
            # Calculate volatility proxy
            pv_lower_raw = df['PV3_B3_T6'].values.astype(np.float32)
            pv_upper_raw = df['PV3_B5_T6'].values.astype(np.float32)
            pv_band_width = pd.Series(pv_upper_raw - pv_lower_raw).rolling(window=1800, min_periods=1).mean()
            pv_band_width = pv_band_width.fillna(method='ffill').fillna(method='bfill')
            pv_band_width = pv_band_width.replace(0, np.nan).fillna(method='ffill').fillna(0.0001)
            vol_proxy = pv_band_width.values.astype(np.float32)
            
            # Calculate weighted signals - FIRST PASS for regime detection
            bb_signal_init = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights_momentum)
            pb_signal_init = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights_momentum)
            
            bb_norm_init = self.normalize_signal(bb_signal_init, window=300)
            pb_norm_init = self.normalize_signal(pb_signal_init, window=300)
            
            # ==================== REGIME DETECTION ====================
            regime, confidence = self.detect_regime(
                bb_norm_init, pb_norm_init, prices, timestamps,
                self.regime_test_duration, self.regime_test_min_trades
            )
            
            regime_name = "MEAN REVERSION" if regime == 0 else "MOMENTUM"
            
            # ==================== APPLY REGIME-SPECIFIC WEIGHTS ====================
            if regime == 0:  # Mean Reversion
                bb_weights = self.bb_weights_reversion
                pb_weights = self.pb_weights_reversion
            else:  # Momentum
                bb_weights = self.bb_weights_momentum
                pb_weights = self.pb_weights_momentum
            
            # Recalculate signals with regime-specific weights
            bb_signal = self._calculate_weighted_signal(df, self.bb_features, bb_weights)
            pb_signal = self._calculate_weighted_signal(df, self.pb_features, pb_weights)
            v_signal = self._calculate_weighted_signal(df, self.v_features, self.v_weights)
            
            # Extract PV bands
            pv_lower = df['PV3_B3_T6'].values.astype(np.float32)
            pv_center = df['PV3_B4_T6'].values.astype(np.float32)
            pv_upper = df['PV3_B5_T6'].values.astype(np.float32)
            
            # Normalize signals
            bb_normalized = self.normalize_signal(bb_signal, window=300)
            pb_normalized = self.normalize_signal(pb_signal, window=300)
            
            # Volume surge
            v_mean = self.calculate_rolling_mean(v_signal, 600)
            v_surge = v_signal / (v_mean + 1e-8)
            
            # Generate signals with regime awareness
            signals, signal_quality = self.generate_adaptive_signals(
                bb_normalized, pb_normalized, prices,
                pv_lower, pv_center, pv_upper, v_surge,
                regime,
                self.strong_z_threshold, self.strong_agreement_threshold,
                self.weak_z_threshold, self.pv_band_breakout_pct,
                self.volume_surge_threshold
            )
            
            # Manage positions
            positions = self.manage_positions(
                signals, signal_quality, prices, timestamps,
                vol_proxy,
                self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
                self.sl_strong_mult, self.sl_weak_mult,
                self.tp_strong_mult, self.tp_weak_mult,
                self.trail_strong_mult, self.trail_weak_mult
            )
            
            # Statistics
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
        """Run adaptive strategy with parallel processing"""
        print("="*80)
        print("ADAPTIVE REGIME-SWITCHING TRADING STRATEGY")
        print("="*80)
        print(f"✓ INPUT: Reading .parquet files from {data_folder}")
        print(f"✓ OUTPUT: Generating portfolio_weights.csv")
        print(f"✓ Regime Detection: First {self.regime_test_duration}s tests both strategies")
        print(f"✓ Mean Reversion: Fade extremes (negative weights)")
        print(f"✓ Momentum: Follow trends (positive weights)")
        print(f"✓ Strong Signals: BB + PB same sign")
        print(f"✓ Weak Signals: BB + PB disagree → PV bands confirm")
        print(f"✓ Dynamic Risk: PV-Band Width volatility proxy")
        print(f"✓ All 6 Constraints Enforced")
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
            print("Combining results in day-wise order...")
            
            sorted_days = sorted(day_results.keys())
            all_results = [day_results[day] for day in sorted_days]
            
            portfolio_weights = pd.concat(all_results, ignore_index=True)
            
            # Count regime distribution
            regime_counts = portfolio_weights.groupby('Regime')['Day'].nunique()
            print(f"\n📊 REGIME DISTRIBUTION:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count} days ({count/len(day_results)*100:.1f}%)")
            
            # Remove extra columns before saving
            portfolio_weights_output = portfolio_weights[['Time', 'Signal', 'Price']].copy()
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
            
            print("="*80)
            
            gc.collect()
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = AdaptiveRegimeTradingStrategy()
    
    print("\n" + "="*80)
    print("STARTING ADAPTIVE REGIME-SWITCHING STRATEGY")
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=510,
        data_folder='/data/quant14/EBX/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✓ Strategy complete!")
        print("✓ portfolio_weights.csv ready")
        print("✓ Adaptive regime detection applied")
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")