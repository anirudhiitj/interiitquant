import os
import glob
import re
import shutil
import sys
import pathlib
import pandas as pd
import numpy as np
import multiprocessing
import gc
import time
import math
from pykalman import KalmanFilter

# Try to import Numba for acceleration
try:
    from numba import jit, njit, int64, float64, config as numba_config
    NUMBA_AVAILABLE = True
    # Configure Numba for threading as per your source
    numba_config.THREADING_LAYER = "threadsafe"
    print("✓ Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("✗ Numba not found - using standard Python")
    def jit(nopython=True, fastmath=True, cache=False):
        def decorator(func):
            return func
        return decorator
    # Mock njit for compatibility if numba missing
    njit = jit 

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'OUTPUT_DIR': '/data/quant14/signals/',
    'OUTPUT_FILE': 'combined_signals_EBX.csv',
    'TEMP_DIR': '/data/quant14/signals/temp_manager_v5_3strat_locked',
    
    # Manager Settings
    'UNIVERSAL_COOLDOWN': 5.0,
    'MAX_WORKERS': 64,
    
    # Required Columns
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    
    # ========================================================================
    # STRATEGY 1 (KALMAN FILTER) - PRIORITY 1
    # ========================================================================
    'P1_FEATURE_COLUMN': 'PB9_T1',
    'P1_KF_FAST_TRANSITION_COV': 0.24,
    'P1_KF_FAST_OBSERVATION_COV': 0.7,
    'P1_KF_FAST_INITIAL_STATE_COV': 100,
    'P1_TAKE_PROFIT': 0.4,   
    'P1_TP1_MULTIPLIER': 1.0,
    'P1_TP2_MULTIPLIER': 1.75,
    'P1_TP3_MULTIPLIER': 2.0,
    'P1_TP1_POS': 0.7,     
    'P1_TP2_POS': 0.5,   
    'P1_STOP_LOSS': 0.5,
    'P1_KAMA_WINDOW': 30,   
    'P1_KAMA_FAST_PERIOD': 60, 
    'P1_KAMA_SLOW_PERIOD': 300, 
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_KAMA_SLOPE_ENTRY_THRESHOLD': 0.001,
    'P1_KAMA_SLOPE_EXIT_THRESHOLD': 0.001,
    'P1_KAMA_SUM_WINDOW': 5,
    'P1_UCM_ALPHA': 0.8,
    'P1_UCM_BETA': 0.08,
    'P1_ATR_SPAN': 30,
    'P1_ATR_HIGH_THRESHOLD': 0.01,
    'P1_MIN_HOLD_SECONDS': 5.0,
    'P1_START_TIME': 300,

    # ========================================================================
    # STRATEGY 2 (HAWKES + EAMA) - PRIORITY 2
    # ========================================================================
    'P2_SUPER_SMOOTHER_PERIOD': 25,  #25
    'P2_EAMA_PERIOD': 15,
    'P2_EAMA_FAST_PERIOD': 25,     #25
    'P2_EAMA_SLOW_PERIOD': 100,   #100
    'P2_HAWKES_LOOKBACK': 1800,
    'P2_HAWKES_THRESHOLD_ABS': 0.00005,   #
    'P2_HAWKES_ALPHA': 1.25, #1.2
    'P2_HAWKES_BETA': 1.75, #1.8
    'P2_HAWKES_SIGNAL_THRESHOLD': 0.08,   #0.08
    'P2_HARD_WALL_UPPER': 101.06,
    'P2_HARD_WALL_LOWER': 98.94,
    'P2_WALL_MIN_PERIODS': 75,   #
    'P2_WALL_STD_MULT': 2.25,   #
    'P2_ATR_LOOKBACK': 28,
    'P2_ATR_MIN_TRADABLE': 0.015,
    'P2_TRAIL_STOP_ABS': 0.005,
    'P2_TRADE_COST_ABS': 0.005,
    
    # ========================================================================
    # STRATEGY 3 (VOLATILITY GATED MU SPLIT) - PRIORITY 3 [UPDATED]
    # ========================================================================
    'P3_DECISION_WINDOW_SECONDS': 1800, # 30 * 60
    'P3_SIGMA_THRESHOLD': 0.10,
    'P3_MU_THRESHOLD_SPLIT': 100.0,
    'P3_COOLDOWN_PERIOD_SECONDS': 5,

    # P3 - Sub-Strategy 1 (Low Mu < 100)
    'P3_S1_ENTRY_DROP_THRESHOLD': 0.8,
    'P3_S1_PRE_DROP_SHORT_SL': 0.4,
    'P3_S1_LONG_STATIC_SL': 0.7,
    'P3_S1_LONG_TP_ACTIVATION': 1.0,
    'P3_S1_LONG_TRAIL_CALLBACK': 0.1,
    'P3_S1_REVERSAL_RISE_TRIGGER': 0.2,
    'P3_S1_SHORT_STATIC_SL': 4.4,
    'P3_S1_SHORT_TP_ACTIVATION': 6.2,
    'P3_S1_SHORT_TRAIL_CALLBACK': 0.04,

    # P3 - Sub-Strategy 2 (High Mu > 100)
    'P3_S2_ENTRY_RISE_THRESHOLD': 0.9,
    'P3_S2_PRE_RISE_SL': 0.9,
    'P3_S2_PRIMARY_SHORT_SL': 0.6,
    'P3_S2_PRIMARY_TP_ACTIVATION': 1.0,
    'P3_S2_PRIMARY_TRAIL_CALLBACK': 0.1,
    'P3_S2_REVERSAL_DROP_TRIGGER': 0.2,
    'P3_S2_SECONDARY_LONG_SL': 0.15,
    'P3_S2_SECONDARY_TP_ACTIVATION': 0.8,
    'P3_S2_SECONDARY_TRAIL_CALLBACK': 0.1,
}

# ============================================================================
# SHARED HELPER FUNCTIONS (GENERAL)
# ============================================================================

@jit(nopython=True, fastmath=True)
def compute_atr_fast(prices, period):
    n = len(prices)
    atr = np.zeros(n)
    if n < 2:
        return atr
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = abs(prices[i] - prices[i - 1])
    for i in range(n):
        if i < period - 1:
            w = i + 1
            s = 0
        else:
            w = period
            s = i - period + 1
        a = 0.0
        for j in range(s, i + 1):
            a += tr[j]
        atr[i] = a / w
    return atr

# ============================================================================
# HELPER FUNCTIONS (SPECIFIC TO P2: HAWKES/EAMA)
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_super_smoother_numba(prices, period):
    n = len(prices)
    ss = np.zeros(n, dtype=np.float64)
    if n == 0: return ss
    pi = 3.14159265358979323846
    a1 = np.exp(-1.414 * pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * pi / period)
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1
    ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]
    for i in range(2, n):
        ss[i] = c1 * prices[i] + c2 * ss[i-1] + c3 * ss[i-2]
    return ss

@jit(nopython=True, fastmath=True)
def calculate_eama_numba(ss_values, eama_period, eama_fast, eama_slow):
    n = len(ss_values)
    eama = np.zeros(n, dtype=np.float64)
    if n == 0: return eama
    eama[0] = ss_values[0]
    fast_sc = 2.0 / (eama_fast + 1.0)
    slow_sc = 2.0 / (eama_slow + 1.0)
    for i in range(1, n):
        if i >= eama_period:
            direction = abs(ss_values[i] - ss_values[i - eama_period])
            volatility = 0.0
            for j in range(eama_period):
                if i - j > 0:
                    volatility += abs(ss_values[i - j] - ss_values[i - j - 1])
            if volatility > 1e-10:
                er = direction / volatility
            else:
                er = 0.0
        else:
            er = 0.0
        sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
        eama[i] = eama[i-1] + sc * (ss_values[i] - eama[i-1])
    return eama

@jit(nopython=True, fastmath=True)
def calculate_expanding_walls_numba(prices, min_periods, std_mult, fallback_upper, fallback_lower):
    n = len(prices)
    upper_walls = np.zeros(n, dtype=np.float64)
    lower_walls = np.zeros(n, dtype=np.float64)
    sum_x = 0.0
    sum_sq_x = 0.0
    sum_std = 0.0
    std_count = 0.0
    for i in range(n):
        val = prices[i]
        sum_x += val
        sum_sq_x += (val * val)
        count = i + 1.0
        if count >= min_periods:
            mean = sum_x / count
            variance = (sum_sq_x / count) - (mean * mean)
            if variance < 0: variance = 0.0
            std = np.sqrt(variance)
            sum_std += std
            std_count += 1.0
            mean_std = sum_std / std_count
            upper_walls[i] = mean + (std)
            lower_walls[i] = mean - (std)
        else:
            upper_walls[i] = fallback_upper
            lower_walls[i] = fallback_lower
    return upper_walls, lower_walls

@jit(nopython=True, fastmath=True)
def calculate_recursive_hawkes_on_eama(eama_prices, alpha, beta, time_delta, threshold):
    N = len(eama_prices)
    H_buy = np.zeros(N, dtype=np.float64)
    H_sell = np.zeros(N, dtype=np.float64)
    decay_factor = np.exp(-beta * time_delta)
    for i in range(1, N):
        current_H_buy = H_buy[i-1] * decay_factor
        current_H_sell = H_sell[i-1] * decay_factor
        eama_change = eama_prices[i] - eama_prices[i-1]
        if eama_change > threshold:
            current_H_buy += alpha 
        elif eama_change < -threshold:
            current_H_sell += alpha
        H_buy[i] = current_H_buy
        H_sell[i] = current_H_sell
    return H_buy, H_sell

@jit(nopython=True, fastmath=True)
def generate_hawkes_signals_numba(raw_prices, h_pos, h_neg, atr, upper_walls, lower_walls, ss_slope, 
                                 trade_cost, cooldown, hawkes_lookback, hawkes_thresh, 
                                 atr_min_tradable, trail_stop_abs):
    N = len(raw_prices)
    signals = np.zeros(N, dtype=np.int32)
    position = 0
    entry_price = 0.0
    traded_today = False
    day_tradable = False
    tp_activated = False
    tp_anchor = 0.0
    last_trade_time = -cooldown 
    
    # Using hard walls as per source fallback, can swap to dynamic if needed
    HARD_WALL_UPPER = 100.19
    HARD_WALL_LOWER = 99.55

    for i in range(N):
        p = raw_prices[i]
        curr_upper = HARD_WALL_UPPER
        curr_lower = HARD_WALL_LOWER
        
        in_cooldown = i - last_trade_time < cooldown
        if (not day_tradable) and (atr[i] > atr_min_tradable):
            day_tradable = True
        
        if i < hawkes_lookback:
            continue
        
        h_buy_i = h_pos[i]
        h_sell_i = h_neg[i]
        
        distance_lower = p - curr_lower
        if distance_lower < 0.0: distance_lower = 0.0 
        long_signal_score = h_sell_i * distance_lower
        
        distance_upper = curr_upper - p
        if distance_upper < 0.0: distance_upper = 0.0 
        short_signal_score = h_buy_i * distance_upper
        
        long_sig = long_signal_score > hawkes_thresh
        short_sig = short_signal_score > hawkes_thresh
        
        # --- ENTRY LOGIC ---
        if position == 0 and (not traded_today) and day_tradable and (not in_cooldown):
            current_slope = ss_slope[i]
            # Long Entry
            if long_sig and current_slope > 0.01:
                signals[i] = 1 
                position = 1
                entry_price = p + trade_cost
                traded_today = False 
                tp_activated = False
                tp_anchor = 0.0
                last_trade_time = i
            # Short Entry
            elif short_sig and current_slope < -0.01:
                signals[i] = -1 
                position = -1
                entry_price = p - trade_cost
                traded_today = False 
                tp_activated = False
                tp_anchor = 0.0
                last_trade_time = i
            
        # --- EXIT / MANAGEMENT LOGIC ---
        if position != 0 and entry_price > 0:
            exit_now = False
            exit_signal = -position
            
            if position == 1:
                abs_move = p - entry_price
            else:
                abs_move = entry_price - p
            
            if (not tp_activated) and abs_move >= 0.2:
                tp_activated = False
                tp_anchor = p
            
            if tp_activated:
                if position == 1:
                    if p > tp_anchor: tp_anchor = p
                    if p < tp_anchor - trail_stop_abs: exit_now = True
                else:
                    if p < tp_anchor: tp_anchor = p
                    if p > tp_anchor + trail_stop_abs: exit_now = True

            if i == N - 1 and position != 0:
                exit_now = True
                exit_signal = -position
            
            if exit_now and (not in_cooldown):
                signals[i] = exit_signal
                position = 0
                entry_price = 0.0
                last_trade_time = i
                
    return signals

# ============================================================================
# STRATEGY 1 (KALMAN FILTER) - PRIORITY 1 (UPDATED)
# ============================================================================

def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare all derived features for Kalman Filter strategy"""
    initial_state_mean = df[config['PRICE_COLUMN']].iloc[0]
    kf_fast = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_FAST_TRANSITION_COV'],
        observation_covariance=config['P1_KF_FAST_OBSERVATION_COV'],
        initial_state_mean=initial_state_mean,
        initial_state_covariance=config['P1_KF_FAST_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf_fast.filter(df[config['P1_FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['P1_FEATURE_COLUMN']].dropna().index)
    
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["Take_Profit"] = config['P1_TAKE_PROFIT']
    
    price = df[config['P1_FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) > 0:
        start = valid_start[0]
        kama[start] = price[start]
        for i in range(start + 1, len(price)):
            if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
            
    df["KAMA"] = kama
    df["KAMA_Slope"] = df["KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    
    # UCM
    series = df[config['P1_FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None:
        alpha = config['P1_UCM_ALPHA']
        beta = config['P1_UCM_BETA']
        n = len(series)
        mu = np.zeros(n)
        beta_slope = np.zeros(n)
        filtered = np.zeros(n)
        
        mu[first_valid_idx] = series.loc[first_valid_idx]
        beta_slope[first_valid_idx] = 0
        filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
        
        for t in range(first_valid_idx + 1, n):
            if np.isnan(series.iloc[t]):
                mu[t] = mu[t-1]
                beta_slope[t] = beta_slope[t-1]
                filtered[t] = filtered[t-1]
                continue
            mu_pred = mu[t-1] + beta_slope[t-1]
            beta_pred = beta_slope[t-1]
            mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
            beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
            filtered[t] = mu[t] + beta_slope[t]
        
        df["UCM"] = filtered
        df.loc[0, "UCM"] = 100
    
    tr = df[config['PRICE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P1_ATR_SPAN']).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()
    
    return df

def generate_p1_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Generate Priority 1 signals"""
    df = prepare_p1_features(df, config)
    position = 0.0
    entry_price = None
    last_signal_time = -config['P1_MIN_HOLD_SECONDS']
    signals = [0.0] * len(df)
    
    tp1, tp2, tp3 = 0.0, 0.0, 0.0
    tp_stage = 0
    apply_sl = False
    apply_tp = True
    sl = config['P1_STOP_LOSS']
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_time = row['Time_sec']
        price = row[config['PRICE_COLUMN']]
        is_last_tick = (i == len(df) - 1)
        signal = 0.0
        position = round(position, 2)
        
        if is_last_tick:
            if position != 0: signal = -position
        else:
            cooldown_over = (current_time - last_signal_time) >= config['P1_MIN_HOLD_SECONDS']
            if cooldown_over and position == 0:
                if (row["KFTrendFast"] <= row["UCM"] and 
                    row["KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                    row["ATR_High"] > config['P1_ATR_HIGH_THRESHOLD']):
                    signal = 1
                    apply_tp = True
                elif (row["KFTrendFast"] >= row["UCM"] and 
                      row["KAMA_Slope_abs"] > config['P1_KAMA_SLOPE_ENTRY_THRESHOLD'] and 
                      row["ATR_High"] > config['P1_ATR_HIGH_THRESHOLD']):
                    signal = -1
                    apply_tp = True
                if signal != 0:
                    entry_price = price
                    tp1 = row["Take_Profit"] * config['P1_TP1_MULTIPLIER']
                    tp2 = row["Take_Profit"] * config['P1_TP2_MULTIPLIER']
                    tp3 = row["Take_Profit"] * config['P1_TP3_MULTIPLIER']
                    tp_stage = 0
                    apply_sl = False
                    sl = 0.5
            elif cooldown_over and position != 0 and signal == 0:
                if apply_tp:
                    if position > 0:
                        price_diff = price - entry_price
                        tp1_pos, tp2_pos = config['P1_TP1_POS'], config['P1_TP2_POS']
                    else:
                        price_diff = entry_price - price
                        tp1_pos, tp2_pos = -config['P1_TP1_POS'], -config['P1_TP2_POS']
                    
                    if price_diff >= tp3 and tp_stage < 3:
                        tp_stage = 3
                        apply_sl = False
                        signal = -position
                    elif price_diff >= tp2 and tp_stage < 2:
                        tp_stage = 2
                        apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp2_pos
                    elif price_diff >= tp1 and tp_stage < 1:
                        tp_stage = 1
                        apply_sl = True
                        if row["KAMA_Slope_abs"] < config['P1_KAMA_SLOPE_EXIT_THRESHOLD']:
                            signal = -position + tp1_pos
                    
                    if apply_sl:
                        if tp_stage == 1 and (price_diff - tp1 < -sl): signal = -position
                        elif tp_stage == 2 and (price_diff - tp2 < -sl): signal = -position

        if signal != 0:
            if (position + signal > 1): signal = 1 - position; position = 1
            elif (position + signal < -1): signal = -1 - position; position = -1
            else: position += signal; last_signal_time = current_time; 
            if round(position, 2) == 0: entry_price = None
        signals[i] = signal
        
    return np.array(signals, dtype=np.float64), df

# ============================================================================
# STRATEGY 2 (HAWKES + EAMA) - PRIORITY 2 [NEW]
# ============================================================================

def generate_p2_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Wrapper for Priority 2: Hawkes Process with EAMA"""
    raw_price = df[config['PRICE_COLUMN']].astype(float).values
    raw_price = np.where(np.isnan(raw_price) | (np.abs(raw_price) < 1e-10), 1e-10, raw_price)

    # 1. Super Smoother
    super_smoother = calculate_super_smoother_numba(raw_price, config['P2_SUPER_SMOOTHER_PERIOD'])
    
    # 2. Slope of SS
    ss_slope = np.zeros_like(super_smoother)
    if len(super_smoother) > 1:
        ss_slope[1:] = super_smoother[1:] - super_smoother[:-1]

    # 3. EAMA
    eama_price = calculate_eama_numba(
        super_smoother, 
        config['P2_EAMA_PERIOD'], 
        config['P2_EAMA_FAST_PERIOD'], 
        config['P2_EAMA_SLOW_PERIOD']
    )
    
    # 4. ATR
    atr = compute_atr_fast(raw_price, config['P2_ATR_LOOKBACK'])
    
    # 5. Expanding Walls
    u_walls, l_walls = calculate_expanding_walls_numba(
        raw_price, 
        config['P2_WALL_MIN_PERIODS'], 
        config['P2_WALL_STD_MULT'], 
        config['P2_HARD_WALL_UPPER'], 
        config['P2_HARD_WALL_LOWER']
    )

    # 6. Recursive Hawkes
    h_pos, h_neg = calculate_recursive_hawkes_on_eama(
        eama_price, 
        config['P2_HAWKES_ALPHA'], 
        config['P2_HAWKES_BETA'], 
        1.0, 
        config['P2_HAWKES_THRESHOLD_ABS']
    )
    
    # 7. Generate Signals
    signals = generate_hawkes_signals_numba(
        raw_price, h_pos, h_neg, atr, u_walls, l_walls, ss_slope, 
        config['P2_TRADE_COST_ABS'], 
        config['UNIVERSAL_COOLDOWN'], # Use universal or specific cooldown
        config['P2_HAWKES_LOOKBACK'],
        config['P2_HAWKES_SIGNAL_THRESHOLD'],
        config['P2_ATR_MIN_TRADABLE'],
        config['P2_TRAIL_STOP_ABS']
    )
    
    return signals.astype(np.float64)

# ============================================================================
# STRATEGY 3 (VOLATILITY GATED MU SPLIT) - PRIORITY 3 [UPDATED]
# ============================================================================

# --- Numba Core 1: Strategy 1 (Mu < 100) ---
@jit(nopython=True, fastmath=True)
def p3_strategy_1_core(
    time_sec, price, 
    day_open_price,
    start_index,       
    can_trade_long,    
    cooldown_seconds,
    entry_drop_threshold,
    pre_drop_short_sl,
    s1_long_static_sl, s1_long_tp_activation, s1_long_trail_callback,
    s1_reversal_rise_trigger, s1_short_static_sl, s1_short_tp_activation, s1_short_trail_callback
):
    """
    Strategy 1 Logic: Pre-Drop Short -> Flip Long -> Reversal Short
    """
    n = len(price)
    signals = np.zeros(n, dtype=np.int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # Pre-Drop Logic State
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    
    # Trailing Stop State (For Longs)
    trailing_active = False
    trailing_peak_price = 0.0 

    # Trailing Stop State (For Shorts)
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    
    # Reversal State
    waiting_for_short = False
    short_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: 30 MIN TRIGGER ---
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 # Enter Pre-Drop Short
                    pre_drop_short_active = True
            
            # --- A. PRE-DROP SHORT LOGIC ---
            elif pre_drop_short_active:
                if position == -1:
                    short_pnl = entry_price - curr_p
                    # 1. Check STOP LOSS
                    if short_pnl <= -pre_drop_short_sl:
                        signal = 1 # Buy to Close (SL Hit)
                        pre_drop_short_active = False
                    # 2. Check Target
                    elif curr_p <= pre_drop_target:
                        signal = 1 # Close Short (Target Hit)
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING LONG FLIP ---
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1 # Enter Long
                    pending_long_flip = False
                    waiting_for_short = False 
            
            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- LONG POSITION LOGIC ---
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -s1_long_static_sl:
                        exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= s1_long_tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - s1_long_trail_callback):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = -1  # Close Long
                        waiting_for_short = True
                        short_target_price = curr_p + s1_reversal_rise_trigger

                # --- SHORT POSITION LOGIC (REVERSAL SHORTS) ---
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -s1_short_static_sl:
                        signal = 1 # Buy to Close (SL Hit)
                    else:
                        if not short_trailing_active:
                            if pnl >= s1_short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + s1_short_trail_callback):
                                signal = 1 # Buy to Close (Trailing Hit)

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1  # Enter Short
                            waiting_for_short = False 
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1 # Enter Long
                                waiting_for_short = False 
        
        # 4. Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    trailing_active = False
                    trailing_peak_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        
    return signals


# --- Numba Core 2: Strategy 2 (Mu > 100) ---
@jit(nopython=True, fastmath=True)
def p3_strategy_2_core(
    time_sec, price, 
    day_open_price,
    start_index,       
    can_trade_short,   
    cooldown_seconds,
    entry_rise_threshold,
    pre_rise_sl,
    s2_primary_short_sl, s2_primary_tp_activation, s2_primary_trail_callback,
    s2_reversal_drop_trigger, s2_secondary_long_sl, s2_secondary_tp_activation, s2_secondary_trail_callback
):
    """
    Strategy 2 Logic: Pre-Rise Long -> Flip Short -> Reversal Long
    """
    n = len(price)
    signals = np.zeros(n, dtype=np.int64)
    
    # State Variables
    position = 0
    entry_price = 0.0
    
    # --- Pre-Rise (Gap Fill) State ---
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0

    # Execution Flags
    short_trade_executed = False 
    
    # Trailing State (Short - Primary)
    short_trailing_active = False
    short_trailing_valley_price = 0.0

    # Trailing State (Long - Reversal)
    long_trailing_active = False
    long_trailing_peak_price = 0.0
    
    # Reversal Trigger State
    waiting_for_long_reversal = False
    long_reversal_target_price = 0.0
    
    last_signal_time = -cooldown_seconds - 1
    
    for i in range(n):
        if i < start_index: continue

        curr_t = time_sec[i]
        curr_p = price[i]
        is_last_tick = (i == n - 1)
        
        signal = 0
        
        if is_last_tick and position != 0:
            signal = -position 
        else:
            # --- SPECIAL LOGIC: 30 MIN TRIGGER (GAP FILL LONG) ---
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1 # Enter Pre-Rise Long
                    pre_rise_long_active = True

            # --- A. PRE-RISE LONG LOGIC ---
            elif pre_rise_long_active:
                if position == 1:
                    # SL CHECK
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1 # Close Long (Hit SL)
                        pre_rise_long_active = False
                    # TARGET CHECK
                    elif curr_p >= pre_rise_target:
                        signal = -1 # Close Long (Square off)
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            
            # --- B. PENDING SHORT FLIP ---
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1 # Enter Main Short
                    pending_short_flip = False
                    short_trade_executed = True
                    waiting_for_long_reversal = False

            # --- C. STANDARD STRATEGY LOGIC ---
            else:
                # --- PRIMARY POSITION (SHORT) ---
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    
                    if pnl <= -s2_primary_short_sl:
                        exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= s2_primary_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + s2_primary_trail_callback):
                                exit_signal = True
                    
                    if exit_signal:
                        signal = 1  # Buy to Close (Short Exit)
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - s2_reversal_drop_trigger

                # --- SECONDARY POSITION (LONG - REVERSAL) ---
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    
                    if pnl <= -s2_secondary_long_sl:
                        exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= s2_secondary_tp_activation:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - s2_secondary_trail_callback):
                                exit_signal = True

                    if exit_signal:
                        signal = -1 # Sell to Close

                # --- FLAT (ENTRY) LOGIC ---
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1  # Enter Long (Reversal)
                            waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1 # Enter Short
                                short_trade_executed = True
                                waiting_for_long_reversal = False 
        
        # 4. Apply Signal
        if signal != 0:
            if (position == 1 and signal == 1) or (position == -1 and signal == -1):
                signal = 0
            else:
                if position == 0:
                    entry_price = curr_p
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                elif position != 0 and signal == -position:
                    entry_price = 0.0
                    short_trailing_active = False
                    short_trailing_valley_price = 0.0
                    long_trailing_active = False
                    long_trailing_peak_price = 0.0
                position += signal
                last_signal_time = curr_t
        
        signals[i] = signal
        
    return signals

def generate_p3_signals(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Wrapper for Priority 3: Volatility Gated Mu Split Strategy"""
    price_arr = df[config['PRICE_COLUMN']].astype(float).values
    time_sec_arr = df['Time_sec'].values.astype(np.int64)
    
    # Needs at least 10 ticks
    if len(price_arr) < 10:
        return np.zeros(len(price_arr), dtype=np.float64)

    day_open_price = price_arr[0]
    decision_window = config['P3_DECISION_WINDOW_SECONDS']
    
    # Calculate Volatility and Mu for first 30 mins
    mask_30min = time_sec_arr <= decision_window
    
    if not np.any(mask_30min):
        return np.zeros(len(price_arr), dtype=np.float64)
        
    prices_30min = price_arr[mask_30min]
    mu = np.mean(prices_30min)
    sigma = np.std(prices_30min)
    
    # Gate: Sigma Threshold
    if sigma <= config['P3_SIGMA_THRESHOLD']:
        return np.zeros(len(price_arr), dtype=np.float64)

    start_idx = np.searchsorted(time_sec_arr, decision_window, side='right')
    signals = np.zeros(len(price_arr), dtype=np.int64)

    # Branch Logic based on Mu
    if mu < config['P3_MU_THRESHOLD_SPLIT']:
        # Strategy 1
        signals = p3_strategy_1_core(
            time_sec_arr, price_arr, day_open_price, int(start_idx),
            True, # can_trade_long
            config['P3_COOLDOWN_PERIOD_SECONDS'],
            config['P3_S1_ENTRY_DROP_THRESHOLD'],
            config['P3_S1_PRE_DROP_SHORT_SL'],
            config['P3_S1_LONG_STATIC_SL'], config['P3_S1_LONG_TP_ACTIVATION'], config['P3_S1_LONG_TRAIL_CALLBACK'],
            config['P3_S1_REVERSAL_RISE_TRIGGER'], config['P3_S1_SHORT_STATIC_SL'], config['P3_S1_SHORT_TP_ACTIVATION'], config['P3_S1_SHORT_TRAIL_CALLBACK']
        )
    elif mu > config['P3_MU_THRESHOLD_SPLIT']:
        # Strategy 2
        signals = p3_strategy_2_core(
            time_sec_arr, price_arr, day_open_price, int(start_idx),
            True, # can_trade_short
            config['P3_COOLDOWN_PERIOD_SECONDS'],
            config['P3_S2_ENTRY_RISE_THRESHOLD'],
            config['P3_S2_PRE_RISE_SL'],
            config['P3_S2_PRIMARY_SHORT_SL'], config['P3_S2_PRIMARY_TP_ACTIVATION'], config['P3_S2_PRIMARY_TRAIL_CALLBACK'],
            config['P3_S2_REVERSAL_DROP_TRIGGER'], config['P3_S2_SECONDARY_LONG_SL'], config['P3_S2_SECONDARY_TP_ACTIVATION'], config['P3_S2_SECONDARY_TRAIL_CALLBACK']
        )
    
    return signals.astype(np.float64)

# ============================================================================
# MANAGER LOGIC (3 PRIORITIES)
# ============================================================================

@jit(nopython=True, fastmath=True)
def apply_manager_logic(df_time_sec, p1_signals, p2_signals, p3_signals, config_cooldown):
    """Manager Logic for 3 Priorities. Locks are persistent for the day."""
    n = len(df_time_sec)
    final_signals = np.zeros(n, dtype=np.float64)
    position = 0.0
    position_owner = 0
    last_signal_time = -config_cooldown
    
    # Initialize locks to False (reset at start of day only)
    priority_2_locked = False
    priority_3_locked = False
    
    pending_reversal = False
    reversal_owner = 0
    reversal_target_signal = 0.0
    reversal_trigger_time = 0.0
    EPS = 1e-9
    
    for i in range(n):
        current_time = df_time_sec[i]
        p1_sig = p1_signals[i]
        p2_sig = p2_signals[i]
        p3_sig = p3_signals[i]
        
        is_last_bar = (i == n - 1)
        final_sig = 0.0
        
        if is_last_bar:
            if abs(position) > EPS:
                final_sig = -position
                position = 0.0
                position_owner = 0
            final_signals[i] = final_sig
            continue
            
        cooldown_ok = (current_time - last_signal_time) >= config_cooldown
        
        if pending_reversal and abs(position) < EPS:
            if (current_time - reversal_trigger_time) >= config_cooldown:
                final_sig = reversal_target_signal 
                pending_reversal = False
                position_owner = reversal_owner
                last_signal_time = current_time
                reversal_owner = 0
                reversal_target_signal = 0.0
                reversal_trigger_time = 0.0
                
        elif abs(position) < EPS and not pending_reversal:
            position = 0.0 
            # Note: Locks are NOT reset here, forcing daily persistence.
            
            if cooldown_ok:
                if abs(p1_sig) > 0.01:
                    final_sig = p1_sig
                    position_owner = 1
                    priority_2_locked = True
                    priority_3_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01 and not priority_2_locked:
                    final_sig = p2_sig
                    position_owner = 2
                    priority_3_locked = True
                    last_signal_time = current_time
                elif abs(p3_sig) > 0.01 and not priority_3_locked:
                    final_sig = p3_sig
                    position_owner = 3
                    last_signal_time = current_time
                    
        elif abs(position) > EPS and not pending_reversal:
            if position_owner == 1:
                if cooldown_ok and abs(p1_sig) > 0.01:
                    final_sig = p1_sig
                    last_signal_time = current_time
            elif position_owner == 2:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p2_sig) > 0.01:
                    final_sig = p2_sig
                    last_signal_time = current_time
            elif position_owner == 3:
                if abs(p1_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 1
                    reversal_target_signal = p1_sig
                    reversal_trigger_time = current_time
                    priority_2_locked = True
                    priority_3_locked = True
                    last_signal_time = current_time
                elif abs(p2_sig) > 0.01:
                    final_sig = -position
                    pending_reversal = True
                    reversal_owner = 2
                    reversal_target_signal = p2_sig
                    reversal_trigger_time = current_time
                    priority_3_locked = True
                    last_signal_time = current_time
                elif cooldown_ok and abs(p3_sig) > 0.01:
                    final_sig = p3_sig
                    last_signal_time = current_time

        if final_sig != 0:
            proposed_position = position + final_sig
            if proposed_position > 1.0 + EPS: final_sig = 1.0 - position
            elif proposed_position < -1.0 - EPS: final_sig = -1.0 - position
            position += final_sig
            if abs(position) < EPS:
                position = 0.0
                position_owner = 0
        final_signals[i] = final_sig
        
    return final_signals

# ============================================================================
# DAY PROCESSING
# ============================================================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def extract_day_num_csv(filepath):
    match = re.search(r'day(\d+)\.csv', str(filepath))
    return int(match.group(1)) if match else -1

def process_single_day(file_path: str, day_num: int, temp_dir: pathlib.Path, config: dict) -> dict:
    """Process a single day"""
    try:
        temp_dir_path = pathlib.Path(temp_dir)
        required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN'], config['P1_FEATURE_COLUMN']]
        required_cols = list(dict.fromkeys(required_cols))
        
        df = pd.read_parquet(file_path, columns=required_cols)
        if df.empty: return None
        df = df.reset_index(drop=True)
        
        if pd.api.types.is_numeric_dtype(df[config['TIME_COLUMN']]):
            df['Time_sec'] = df[config['TIME_COLUMN']].astype(int)
        else:
            df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
        
        # Generate Signals for all 3 Priorities
        p1_signals, df_with_features = generate_p1_signals(df, config)
        p2_signals = generate_p2_signals(df, config) # New Hawkes
        p3_signals = generate_p3_signals(df, config) # Volatility Gated Mu Split
        
        time_values = df['Time_sec'].values

        # Apply Manager Logic (P2 > P1 > P3) - PRIORITY SWAP APPLIED HERE
        # P2 signals are passed as the first priority argument, P1 signals as the second.
        final_signals = apply_manager_logic(
            time_values,
            p2_signals,  # NEW PRIORITY 1 (was P2)
            p1_signals,  # NEW PRIORITY 2 (was P1)
            p3_signals,
            config['UNIVERSAL_COOLDOWN']
        )
        
        p1_entries = np.sum(np.abs(p1_signals) > 0.5)
        p2_entries = np.sum(np.abs(p2_signals) > 0.5)
        p3_entries = np.sum(np.abs(p3_signals) > 0.5)
        final_entries = np.sum(np.abs(final_signals) > 0.5)
        
        output_df = pd.DataFrame({
            config['TIME_COLUMN']: df[config['TIME_COLUMN']],
            config['PRICE_COLUMN']: df[config['PRICE_COLUMN']],
            'P1_Signal': p1_signals,
            'P2_Signal': p2_signals,
            'P3_Signal': p3_signals,
            'Signal': final_signals
        })
        
        output_path = temp_dir_path / f"day{day_num}.csv"
        output_df.to_csv(output_path, index=False)
        
        del df, df_with_features, p1_signals, p2_signals, p3_signals, final_signals
        return {
            'path': str(output_path),
            'day_num': day_num,
            'p1_entries': p1_entries,
            'p2_entries': p2_entries,
            'p3_entries': p3_entries,
            'final_entries': final_entries
        }
        
    except Exception as e:
        print(f"Error processing day {day_num}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        gc.collect()

def process_wrapper(args):
    return process_single_day(*args)

# ============================================================================
# MAIN
# ============================================================================

def main(config):
    start_time = time.time()
    
    print("="*80)
    print("STRATEGY MANAGER - 3-PRIORITY SIGNAL ROUTING")
    print("="*80)
    # The print statements reflect the effective priority: P2 (Hawkes) > P1 (Kalman) > P3 (Mu Split)
    print(f"Priority 1: Hawkes Process + EAMA Smoothing")
    print(f"Priority 2: Kalman Filter Strategy (Feature: {config['P1_FEATURE_COLUMN']})")
    print(f"Priority 3: Volatility Gated Mu Split Strategy [UPDATED]")
    print(f"\nUniversal Cooldown: {config['UNIVERSAL_COOLDOWN']}s")
    print("="*80)
    
    temp_dir_path = pathlib.Path(config['TEMP_DIR'])
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)
    
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    filtered_files = [f for f in all_files if extract_day_num(f) not in range(60,80)]
    sorted_files = sorted(filtered_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {data_dir}")
        shutil.rmtree(temp_dir_path)
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    print(f"Processing with {config['MAX_WORKERS']} workers...")
    
    processed_files = []
    total_p1 = 0
    total_p2 = 0
    total_p3 = 0
    total_final = 0
    
    tasks = [(f, extract_day_num(f), str(temp_dir_path), config) for f in sorted_files]
    
    try:
        with multiprocessing.Pool(processes=config['MAX_WORKERS']) as pool:
            for result in pool.imap_unordered(process_wrapper, tasks):
                try:
                    if result:
                        processed_files.append(result['path'])
                        total_p1 += result['p1_entries']
                        total_p2 += result['p2_entries']
                        total_p3 += result['p3_entries']
                        total_final += result['final_entries']
                        print(f"✓ Day{result['day_num']:3d} | P1:{result['p1_entries']:3d} P2:{result['p2_entries']:3d} P3:{result['p3_entries']:3d} Final:{result['final_entries']:3d}")
                except Exception as e:
                    print(f"✗ Error collecting result: {e}")
        
        if not processed_files:
            print("\nERROR: No files processed successfully")
            return
        
        print(f"\nCombining {len(processed_files)} day files...")
        sorted_csvs = sorted(processed_files, key=lambda p: extract_day_num_csv(p))
        
        output_path = pathlib.Path(config['OUTPUT_DIR']) / config['OUTPUT_FILE']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as outfile:
            for i, csv_file in enumerate(sorted_csvs):
                with open(csv_file, 'rb') as infile:
                    if i == 0: shutil.copyfileobj(infile, outfile)
                    else: infile.readline(); shutil.copyfileobj(infile, outfile)
        
        final_df = pd.read_csv(output_path)
        num_long = (final_df['Signal'] > 0.5).sum()
        num_short = (final_df['Signal'] < -0.5).sum()
        total_bars = len(final_df)
        
        print(f"\n{'='*80}")
        print("SIGNAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total bars:         {total_bars:,}")
        print(f"Total entries:      {(num_long + num_short):,}")
        print(f"\nPer-Strategy Breakdown:")
        print(f"P1 entries:         {total_p1:,}")
        print(f"P2 entries:         {total_p2:,}")
        print(f"P3 entries:         {total_p3:,}")
        print(f"Final entries:      {total_final:,}")
        print(f"\nAvg per day:        {(num_long + num_short)/len(processed_files):.1f}")
        print(f"{'='*80}")
        
    finally:
        if temp_dir_path.exists():
            shutil.rmtree(temp_dir_path)
            print(f"\nCleaned up temp directory")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Total processing time: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"✓ Output saved to: {output_path}")

if __name__ == "__main__":
    main(CONFIG)