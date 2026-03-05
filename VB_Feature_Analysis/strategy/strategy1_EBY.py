"""
Optimized Strong Signal Trading Strategy (BB + PB Only)
========================================================

FOCUS: Strong Signals Only - BB and PB Z-scores must agree in direction
- Removed: Weak/Medium signals (no PV, V features)
- Kept: Same thought process for strong signal generation
- Simplified: Single-tier risk management optimized for strong signals

FEATURES:
1. BB Features: Bollinger Bands (Mean Reversion indicators)
2. PB Features: Price-Based Momentum indicators
3. Strong Signal Rule: Both BB and PB Z-scores must agree in direction
4. Dynamic Risk Management based on price volatility proxy
5. All 6 mandatory constraints strictly enforced

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


class StrongSignalStrategy:
    """
    Strong signal strategy using BB + PB Z-score agreement only
    
    Signal Generation:
    - LONG: Both BB and PB Z-scores are strongly positive (agree)
    - SHORT: Both BB and PB Z-scores are strongly negative (agree)
    - FLAT: Otherwise (no agreement or weak signals)
    """
    
    def __init__(self, plot_output_dir='daily_plots'):
        # ==================== FEATURE DEFINITIONS ====================
        
        # BB Features - Bollinger Bands (Mean Reversion)
        self.bb_features = [
            'BB4_T10', 'BB4_T11', 'BB4_T12',
            'BB5_T10', 'BB5_T11', 'BB5_T12',
        ]   
        
        # PB Features - Price-Based Momentum
        self.pb_features = [
            'PB2_T10', 'PB2_T11', 'PB2_T12',
            'PB5_T10', 'PB5_T11', 'PB5_T12', 
            'PB6_T11', 'PB6_T12',
            'PB7_T11', 'PB7_T12',
            'PB3_T7', 'PB3_T10', 'PB3_T8'
        ]
        
        # ==================== WEIGHTS ====================
        
        # BB Weights (negative = mean reversion)
        self.bb_weights = {
            # 'BB1_T10': 0.22, 'BB1_T11': 0.22, 'BB1_T12': 0.18,
            'BB4_T10': 0.18, 'BB4_T11': 0.15, 'BB4_T12': 0.03,
            'BB5_T10': 0.01, 'BB5_T11': 0.005, 'BB5_T12': 0.005,
        }
        
        # PB Weights (momentum indicators)
        self.pb_weights = {
            'PB2_T10': 0.05, 'PB2_T11': 0.05, 'PB2_T12': 0.05,
            'PB5_T10': 0.07, 'PB5_T11': 0.07, 'PB5_T12': 0.07, 
            'PB6_T11': 0.08, 'PB6_T12': 0.08,
            'PB7_T11': 0.06, 'PB7_T12': 0.06,
            'PB3_T7': 0.05, 'PB3_T10': 0.05, 'PB3_T8': 0.05
        }
        
        # ==================== PARAMETERS ====================
        
        # Strong Signal Thresholds
        self.strong_z_threshold = 1.05  # Z-score threshold for strong signals
        self.strong_agreement_factor = 0.6  # BB and PB must both exceed this * threshold
        
        # Risk Management - Optimized for Strong Signals
        self.sl_mult = 0.5       # Stop-loss: 50% of volatility proxy
        self.tp_mult = 1.0       # Take-profit: 100% of volatility proxy (1:2 R:R)
        self.trail_mult = 0.4    # Trailing stop: 40% of volatility proxy
        
        # Position Management
        self.min_trade_duration = 15  # CONSTRAINT 5: Minimum 15 seconds
        self.max_trade_duration = 600  # Max 10 minutes
        self.min_hold_time = 15  # Time before considering reversal
        
        # Volatility proxy parameters
        self.volatility_window = 600  # 30-minute rolling window for band width
        
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
    def generate_strong_signals(bb_norm, pb_norm, strong_z_thresh, strong_agree_factor):
        """
        Generate strong trading signals based on BB and PB Z-score agreement
        
        STRONG SIGNALS ONLY:
        - LONG (+1): Both BB and PB Z-scores are strongly positive
        - SHORT (-1): Both BB and PB Z-scores are strongly negative
        - FLAT (0): No agreement or weak signals
        
        Returns:
            signals: -1 (short), 0 (flat), +1 (long)
        """
        n = len(bb_norm)
        signals = np.zeros(n, dtype=np.int32)
        
        # Calculate agreement threshold
        agree_thresh = strong_z_thresh * strong_agree_factor
        
        for i in range(300, n):
            bb_z = bb_norm[i]
            pb_z = pb_norm[i]
            
            # LONG: Both BB and PB Z-scores are strongly positive
            if bb_z > agree_thresh and pb_z > agree_thresh:
                signals[i] = 1
            
            # SHORT: Both BB and PB Z-scores are strongly negative
            elif bb_z < -agree_thresh and pb_z < -agree_thresh:
                signals[i] = -1
            
            # Otherwise: No signal (flat)
            else:
                signals[i] = 0
        
        return signals
    
    @staticmethod
    @jit(nopython=True, cache=True, fastmath=True)
    def manage_positions(signals, prices, timestamps, vol_proxy,
                         min_duration, min_hold, max_duration,
                         sl_mult, tp_mult, trail_mult):
        """
        Manage positions with dynamic risk management
        
        CONSTRAINT 1: Positions in {-1, 0, +1}
        CONSTRAINT 5: Minimum hold time = 15 seconds
        """
        n = len(signals)
        positions = np.zeros(n, dtype=np.int32)
        
        current_pos = 0
        entry_price = 0.0
        entry_time = 0.0
        entry_vol = 0.0
        highest_price_in_trade = 0.0
        lowest_price_in_trade = 999999.0
        max_favorable_pts = 0.0
        
        for i in range(1, n):
            time_in_trade = timestamps[i] - entry_time
            
            # ==================== ENTRY LOGIC ====================
            if current_pos == 0:
                if signals[i] != 0:
                    current_pos = signals[i]  # CONSTRAINT 1: {-1, 0, +1}
                    entry_price = prices[i]
                    entry_time = timestamps[i]
                    entry_vol = vol_proxy[i]
                    highest_price_in_trade = prices[i]
                    lowest_price_in_trade = prices[i]
                    max_favorable_pts = 0.0
            
            # ==================== EXIT LOGIC ====================
            else:
                direction = 1.0 if current_pos > 0 else -1.0
                
                # Calculate P&L
                pnl_pts = (prices[i] - entry_price) * direction
                max_favorable_pts = max(max_favorable_pts, pnl_pts)
                
                # Track extremes for trailing stop
                if current_pos > 0:
                    highest_price_in_trade = max(highest_price_in_trade, prices[i])
                else:
                    lowest_price_in_trade = min(lowest_price_in_trade, prices[i])
                
                # Calculate dynamic stop-loss and take-profit levels
                stop_loss_pts = entry_vol * sl_mult
                take_profit_pts = entry_vol * tp_mult
                trailing_stop_pts = entry_vol * trail_mult
                
                # Calculate trailing stop level
                if current_pos > 0:
                    trailing_level = highest_price_in_trade - trailing_stop_pts
                else:
                    trailing_level = lowest_price_in_trade + trailing_stop_pts
                
                should_exit = False
                
                # CONSTRAINT 5: Must hold for minimum duration
                if time_in_trade < min_duration:
                    pass  # Cannot exit yet
                
                # Stop-loss hit
                elif pnl_pts <= -stop_loss_pts:
                    should_exit = True
                
                # Take-profit hit
                elif pnl_pts >= take_profit_pts:
                    should_exit = True
                
                # Trailing stop (only after 40% of TP reached)
                elif max_favorable_pts >= take_profit_pts * 0.4:
                    if current_pos > 0 and prices[i] < trailing_level:
                        should_exit = True
                    elif current_pos < 0 and prices[i] > trailing_level:
                        should_exit = True
                
                # Maximum trade duration
                elif time_in_trade >= max_duration:
                    should_exit = True
                
                # Signal reversal (after minimum hold time)
                elif time_in_trade >= min_hold:
                    if current_pos > 0 and signals[i] < 0:
                        should_exit = True
                    elif current_pos < 0 and signals[i] > 0:
                        should_exit = True
                
                if should_exit:
                    current_pos = 0
                    entry_price = 0.0
                    entry_time = 0.0
                    entry_vol = 0.0
                    highest_price_in_trade = 0.0
                    lowest_price_in_trade = 999999.0
                    max_favorable_pts = 0.0
            
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
        Process single day - READS FROM .PARQUET FILES
        Runs in parallel via ProcessPoolExecutor
        
        CONSTRAINT 2: Intraday only - all positions closed at EOD
        CONSTRAINT 3: Fixed capital each day (no compounding)
        CONSTRAINT 6: No forward bias (all calculations use T-1 data)
        """
        filename = f"day{day_num}.parquet"
        filepath = os.path.join(data_folder, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            # Read parquet file
            df = pd.read_parquet(filepath)
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.sort_values('Time').reset_index(drop=True)
            
            # Check required features (BB and PB only)
            required_features = self.bb_features + self.pb_features
            missing = [f for f in required_features if f not in df.columns]
            
            if len(missing) > 0:
                print(f"Day {day_num}: Missing {len(missing)} features")
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
            
            # ==================== VOLATILITY PROXY CALCULATION ====================
            # Use rolling standard deviation of price as volatility proxy
            price_series = pd.Series(prices)
            rolling_std = price_series.rolling(window=self.volatility_window, min_periods=1).std()
            
            # Fill NaNs and ensure minimum value
            rolling_std = rolling_std.fillna(method='ffill').fillna(method='bfill')
            rolling_std = rolling_std.replace(0, np.nan).fillna(method='ffill')
            rolling_std = rolling_std.fillna(0.0001)
            
            vol_proxy = rolling_std.values.astype(np.float32)
            
            # ==================== SIGNAL CALCULATION ====================
            
            # Calculate weighted signals (BB and PB only)
            bb_signal = self._calculate_weighted_signal(df, self.bb_features, self.bb_weights)
            pb_signal = self._calculate_weighted_signal(df, self.pb_features, self.pb_weights)
            
            # Normalize signals (using historical data only - CONSTRAINT 6)
            bb_normalized = self.normalize_signal(bb_signal, window=300)
            pb_normalized = self.normalize_signal(pb_signal, window=300)
            
            # ==================== STRONG SIGNAL GENERATION ====================
            signals = self.generate_strong_signals(
                bb_normalized, pb_normalized,
                self.strong_z_threshold, self.strong_agreement_factor
            )
            
            # ==================== POSITION MANAGEMENT ====================
            positions = self.manage_positions(
                signals, prices, timestamps, vol_proxy,
                self.min_trade_duration, self.min_hold_time, self.max_trade_duration,
                self.sl_mult, self.tp_mult, self.trail_mult
            )
            
            # Statistics
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
            
            print(f"Day {day_num:3d}: Signals(Long:{long_signals:4d}, Short:{short_signals:4d})")
            
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
        print("STRONG SIGNAL STRATEGY (BB + PB Z-SCORE AGREEMENT ONLY)")
        print("="*80)
        print(f"✓ INPUT: Reading .parquet files from {data_folder}")
        print(f"✓ OUTPUT: Generating portfolio_weights.csv")
        print(f"\n📊 SIGNAL GENERATION:")
        print(f"  - BB Features: {len(self.bb_features)} Bollinger Band indicators")
        print(f"  - PB Features: {len(self.pb_features)} Price-Based momentum indicators")
        print(f"  - Z-score Threshold: {self.strong_z_threshold}")
        print(f"  - Agreement Factor: {self.strong_agreement_factor}")
        print(f"  - Rule: LONG when both BB_z > {self.strong_z_threshold * self.strong_agreement_factor:.2f}")
        print(f"         SHORT when both BB_z < -{self.strong_z_threshold * self.strong_agreement_factor:.2f}")
        print(f"\n⚙️  RISK MANAGEMENT (Dynamic):")
        print(f"  - Stop-Loss: {self.sl_mult:.2f}x volatility proxy")
        print(f"  - Take-Profit: {self.tp_mult:.2f}x volatility proxy")
        print(f"  - Trailing Stop: {self.trail_mult:.2f}x volatility proxy")
        print(f"  - Risk:Reward Ratio: 1:{self.tp_mult/self.sl_mult:.1f}")
        print(f"\n✅ CONSTRAINTS:")
        print(f"  [1] Positions in {{-1, 0, +1}}")
        print(f"  [2] Intraday only (EOD close)")
        print(f"  [3] Fixed capital per day")
        print(f"  [4] No leverage (1 unit)")
        print(f"  [5] Min hold time = {self.min_trade_duration} seconds")
        print(f"  [6] No forward bias")
        print(f"\n🔧 EXECUTION:")
        print(f"  - CPU Workers: {max_workers}")
        print(f"  - Days to Process: {num_days}")
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
            total_flat = (portfolio_weights_output['Signal'] == 0).sum()
            
            print(f"\n📈 SUMMARY:")
            print(f"  Days Processed: {len(day_results)}")
            print(f"  Total Rows: {total_signals:,}")
            if total_signals > 0:
                print(f"  Long: {total_long:,} ({total_long/total_signals*100:.1f}%)")
                print(f"  Short: {total_short:,} ({total_short/total_signals*100:.1f}%)")
                print(f"  Flat: {total_flat:,} ({total_flat/total_signals*100:.1f}%)")
            
            print("="*80)
            
            gc.collect()
            
            return portfolio_weights_output
        else:
            print("\n✗ No valid data generated")
            return None


if __name__ == "__main__":
    strategy = StrongSignalStrategy()
    
    print("\n" + "="*80)
    print("STARTING STRONG SIGNAL STRATEGY (BB + PB AGREEMENT)")
    print("="*80)
    
    portfolio_weights = strategy.run_strategy(
        num_days=279,
        data_folder='/data/quant14/EBY/',
        max_workers=25
    )
    
    if portfolio_weights is not None:
        print("\n✅ Strategy complete!")
        print("✓ portfolio_weights.csv ready (Time, Signal, Price)")
        print("✓ All 6 constraints enforced")
        print("✓ Strong signals only (BB + PB Z-score agreement)")
        print("✓ Run: python backtester.py")
    else:
        print("\n✗ No data generated")