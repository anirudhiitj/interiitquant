import os
import glob
import re
import shutil
import sys
import argparse
import pathlib
import pandas as pd
import numpy as np
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

# --- Strategy 1 Imports ---
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter

# --- Strategy 2 Imports ---
from numba import njit, int64, float64

# --- Configuration ---
TEMP_DIR = "/data/quant14/signals/merged_priority_temp_FIXED"
OUTPUT_FILE = "/data/quant14/signals/final_priority_merged_signals.csv"
REQUIRED_COLUMNS = ['Time', 'Price']
COOLDOWN_PERIOD_SECONDS = 5

# Blacklist (Updated as requested: 60 to 79)
BLACKLIST_DAYS = {60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                  70, 71, 72, 73, 74, 75, 76, 77, 78, 79}

# Strategy Params Mapping
STRATEGY_PARAMS_COLS = {
    "pb9t1": "PB9_T1",
}

# ==============================================================================
#                               PRIORITY 2 PARAMETERS
# ==============================================================================

DECISION_WINDOW_SECONDS = 30 * 60
MU_SPLIT_THRESHOLD = 100.0    

# --- Low Mu (< 100) Settings ---
REVERSAL_RISE_TRIGGER_LOW_MU = 0.2

# Strategy A: High Sigma (> 0.13)
PARAMS_STRATEGY_A = {
    "ENTRY_DROP_THRESHOLD": 0.8,
    "PRE_DROP_SHORT_SL": 0.4,
    "STATIC_SL": 0.7,
    "TP_ACTIVATION": 1.0,
    "TRAIL_CALLBACK": 0.1,
    "SHORT_STATIC_SL": 4.4,
    "SHORT_TP_ACTIVATION": 6.2,
    "SHORT_TRAIL_CALLBACK": 0.04
}

# Strategy B: Mid Sigma (0.07 to 0.1)
PARAMS_STRATEGY_B = {
    "ENTRY_DROP_THRESHOLD": 0.5,
    "PRE_DROP_SHORT_SL": 0.4,
    "STATIC_SL": 0.6,
    "TP_ACTIVATION": 1.2,
    "TRAIL_CALLBACK": 0.1,
    "SHORT_STATIC_SL": 0.1,
    "SHORT_TP_ACTIVATION": 999.0,
    "SHORT_TRAIL_CALLBACK": 0.04
}

# --- High Mu (> 100) Settings ---
HM_SIGMA_THRESHOLD = 0.12  
HM_ENTRY_RISE_THRESHOLD = 0.9  
HM_PRIMARY_STATIC_SL = 0.6
HM_PRIMARY_TP_ACTIVATION = 1.0
HM_PRIMARY_TRAIL_CALLBACK = 0.1
HM_REVERSAL_DROP_TRIGGER = 0.2  
HM_SECONDARY_STATIC_SL = 0.15
HM_SECONDARY_TP_ACTIVATION = 0.8
HM_SECONDARY_TRAIL_CALLBACK = 0.1
HM_PRE_RISE_SL = 0.9

# ==============================================================================
#                           PRIORITY 1: FEATURE ENG & LOGIC
# ==============================================================================

def prepare_derived_features(df: pd.DataFrame, strategy_params: dict) -> pd.DataFrame:
    # --- Kalman Filter ---
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.24
    observation_covariance = 0.7
    initial_state_covariance = 100
    initial_state_mean = df["Price"].iloc[0]
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(df["PB9_T1"].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df["PB9_T1"].dropna().index)
    df["KFTrendFast"] = smoothed_prices.shift(1)
    df["KFTrendFast_Lagged"] = df["KFTrendFast"].shift(1)

    df["Take_Profit"] = 0.4

    # --- KAMA ---
    price = df["PB9_T1"].to_numpy(dtype=float)
    window = 30
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (60 + 1)
    sc_slow = 2 / (300 + 1)
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
    df["KAMA_Slope"] = df["KAMA"].diff(2)
    df["KAMA_Slope_abs"] = df["KAMA_Slope"].abs()
    df["KAMA_Slope_abs_Lagged_2"] = df["KAMA_Slope_abs"].shift(2)
    df["KAMA_Sum"] = df["KAMA_Slope_abs"].rolling(window=5, min_periods=1).sum()

    # --- UCM ---
    series = df["PB9_T1"]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is not None:
        alpha = 0.8
        beta = 0.08
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
        df["UCM_Lagged"] = df["UCM"].shift(1)

    # --- ATR ---
    tr = df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=30).mean()
    df["ATR"] = atr
    df["ATR_High"] = df["ATR"].expanding(min_periods=1).max()
    
    return df

def generate_signal_p1(row, position, cooldown_over, strategy_params, state):
    position = round(position, 2)
    signal = 0
    
    if cooldown_over:
        if position == 0:
            if row["KFTrendFast"] >= row["UCM"] and row["KAMA_Slope_abs"] > 0.001 and row["ATR_High"] > 0.01:
                state["applyTP"] = True
                signal = -1
            elif row["KFTrendFast"] <= row["UCM"] and row["KAMA_Slope_abs"] > 0.001 and row["ATR_High"] > 0.01:
                state["applyTP"] = True
                signal = 1
    return signal, state

# ==============================================================================
#                           PRIORITY 2: NUMBA CORES
# ==============================================================================

@njit(cache=True, nogil=True)
def backtest_core_low_mu(time_sec, price, day_open_price, start_index, can_trade_long, cooldown_seconds,
                         entry_drop_threshold, pre_drop_short_sl, static_sl, tp_activation, trail_callback,
                         reversal_rise_trigger, short_static_sl, short_tp_activation, short_trail_callback):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    position = 0
    entry_price = 0.0
    pre_drop_target = day_open_price - entry_drop_threshold
    pre_drop_short_active = False
    pending_long_flip = False
    flip_wait_start_time = 0
    trailing_active = False
    trailing_peak_price = 0.0 
    short_trailing_active = False
    short_trailing_valley_price = 0.0
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
            if i == start_index and can_trade_long and position == 0:
                if curr_p > pre_drop_target:
                    signal = -1 
                    pre_drop_short_active = True
            elif pre_drop_short_active:
                if position == -1:
                    short_pnl = entry_price - curr_p
                    if short_pnl <= -pre_drop_short_sl:
                        signal = 1 
                        pre_drop_short_active = False
                    elif curr_p <= pre_drop_target:
                        signal = 1 
                        pre_drop_short_active = False
                        pending_long_flip = True
                        flip_wait_start_time = curr_t
            elif pending_long_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = 1 
                    pending_long_flip = False
                    waiting_for_short = False 
            else:
                if position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -static_sl:
                        exit_signal = True
                    else:
                        if not trailing_active:
                            if pnl >= tp_activation:
                                trailing_active = True
                                trailing_peak_price = curr_p
                        if trailing_active:
                            if curr_p > trailing_peak_price:
                                trailing_peak_price = curr_p
                            elif curr_p <= (trailing_peak_price - trail_callback):
                                exit_signal = True
                    if exit_signal:
                        signal = -1 
                        waiting_for_short = True
                        short_target_price = curr_p + reversal_rise_trigger
                elif position == -1:
                    pnl = entry_price - curr_p
                    if pnl <= -short_static_sl:
                        signal = 1 
                    else:
                        if not short_trailing_active:
                            if pnl >= short_tp_activation:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p 
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + short_trail_callback):
                                signal = 1 
                elif position == 0:
                    if waiting_for_short:
                        if curr_p >= short_target_price:
                            signal = -1 
                            waiting_for_short = False 
                    elif can_trade_long:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p <= pre_drop_target:
                                signal = 1 
                                waiting_for_short = False 
        
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
        positions[i] = position
    return signals, positions

@njit(cache=True, nogil=True)
def backtest_core_high_mu(time_sec, price, day_open_price, start_index, can_trade_short, cooldown_seconds,
                          entry_rise_threshold, pre_rise_sl):
    n = len(price)
    signals = np.zeros(n, dtype=int64)
    positions = np.zeros(n, dtype=int64)
    position = 0
    entry_price = 0.0
    pre_rise_target = day_open_price + entry_rise_threshold
    pre_rise_long_active = False
    pending_short_flip = False
    flip_wait_start_time = 0
    short_trade_executed = False  
    short_trailing_active = False
    short_trailing_valley_price = 0.0
    long_trailing_active = False
    long_trailing_peak_price = 0.0
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
            if i == start_index and can_trade_short and position == 0:
                if curr_p < pre_rise_target:
                    signal = 1 
                    pre_rise_long_active = True
            elif pre_rise_long_active:
                if position == 1:
                    if (curr_p - entry_price) <= -pre_rise_sl:
                        signal = -1 
                        pre_rise_long_active = False
                    elif curr_p >= pre_rise_target:
                        signal = -1 
                        pre_rise_long_active = False
                        pending_short_flip = True
                        flip_wait_start_time = curr_t
            elif pending_short_flip:
                if (curr_t - flip_wait_start_time) >= cooldown_seconds:
                    signal = -1 
                    pending_short_flip = False
                    short_trade_executed = True 
                    waiting_for_long_reversal = False
            else:
                if position == -1:
                    pnl = entry_price - curr_p
                    exit_signal = False
                    if pnl <= -HM_PRIMARY_STATIC_SL:
                        exit_signal = True
                    else:
                        if not short_trailing_active:
                            if pnl >= HM_PRIMARY_TP_ACTIVATION:
                                short_trailing_active = True
                                short_trailing_valley_price = curr_p
                        if short_trailing_active:
                            if curr_p < short_trailing_valley_price:
                                short_trailing_valley_price = curr_p
                            elif curr_p >= (short_trailing_valley_price + HM_PRIMARY_TRAIL_CALLBACK):
                                exit_signal = True
                    if exit_signal:
                        signal = 1 
                        waiting_for_long_reversal = True
                        long_reversal_target_price = curr_p - HM_REVERSAL_DROP_TRIGGER
                elif position == 1:
                    pnl = curr_p - entry_price
                    exit_signal = False
                    if pnl <= -HM_SECONDARY_STATIC_SL:
                        exit_signal = True
                    else:
                        if not long_trailing_active:
                            if pnl >= HM_SECONDARY_TP_ACTIVATION:
                                long_trailing_active = True
                                long_trailing_peak_price = curr_p
                        if long_trailing_active:
                            if curr_p > long_trailing_peak_price:
                                long_trailing_peak_price = curr_p
                            elif curr_p <= (long_trailing_peak_price - HM_SECONDARY_TRAIL_CALLBACK):
                                exit_signal = True
                    if exit_signal:
                        signal = -1 
                elif position == 0:
                    if waiting_for_long_reversal:
                        if curr_p <= long_reversal_target_price:
                            signal = 1 
                            waiting_for_long_reversal = False 
                    elif can_trade_short and not short_trade_executed:
                        cooldown_over = (curr_t - last_signal_time) >= cooldown_seconds
                        if cooldown_over:
                            if curr_p >= pre_rise_target:
                                signal = -1 
                                short_trade_executed = True 
                                waiting_for_long_reversal = False 
        
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
        positions[i] = position
    return signals, positions

# ==============================================================================
#                               MERGED PROCESSING
# ==============================================================================

def extract_day_num(filepath):
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1

def process_day(file_path: str, day_num: int, temp_dir: pathlib.Path, strategy_params_cols) -> str:
    try:
        columns = REQUIRED_COLUMNS + list(strategy_params_cols.values())
        df = pd.read_parquet(file_path, columns=columns)
        
        if df.empty:
            return None

        # --- Common Prep ---
        df = df.reset_index(drop=True)
        df = df.sort_values("Time").reset_index(drop=True)
        df["Time_sec"] = pd.to_timedelta(df["Time"].astype(str)).dt.total_seconds().astype(int)
        df["Day"] = day_num

        # --- Blacklist Check ---
        if day_num in BLACKLIST_DAYS:
            df["Signal"] = 0
            df["Position"] = 0
            df["Regime"] = "Blacklisted"
            output_path = temp_dir / f"day{day_num}.csv"
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Regime"]
            df[final_columns].to_csv(output_path, index=False)
            return str(output_path)
            
        # -------------------------------------------------------------
        # STEP 1: EXECUTE PRIORITY 1 (COMPLEX KF STRATEGY)
        # -------------------------------------------------------------
        
        # We must perform feature engineering for P1 first
        df_p1 = prepare_derived_features(df.copy(), strategy_params_cols)
        
        state = {"applyTP": True}
        position = 0
        entry_price = None
        last_signal_time = -COOLDOWN_PERIOD_SECONDS
        signals_p1 = [0] * len(df_p1)
        positions_p1 = [0] * len(df_p1)
        
        # P1 Iteration Loop
        for i in range(len(df_p1)):
            position = round(position, 2)
            row = df_p1.iloc[i]
            current_time = row['Time_sec']
            price = row['Price']
            is_last_tick = (i == len(df_p1) - 1)
            signal = 0

            # EOD Square Off
            if is_last_tick and position != 0:
                signal = -position
            else:
                cooldown_over = (current_time - last_signal_time) >= COOLDOWN_PERIOD_SECONDS
                signal, state = generate_signal_p1(row, position, cooldown_over, strategy_params_cols, state)

                # Entry Logic
                if cooldown_over and position == 0 and signal != 0:
                    entry_price = price
                    tp_stage = 0
                    applySL = False
                    sl = 0.5
                    
                # Exit Logic
                elif cooldown_over and position != 0 and signal == 0:
                    if state["applyTP"]:
                        tp1 = row["Take_Profit"]*1
                        tp2 = row["Take_Profit"]*1.75
                        tp3 = row["Take_Profit"]*2
                        
                        if position > 0:
                            price_diff = price - entry_price
                            tp1pos, tp2pos = 0.7, 0.5
                        elif position < 0:
                            price_diff = entry_price - price
                            tp1pos, tp2pos = -0.7, -0.5

                        if price_diff >= tp3 and tp_stage < 3:
                            tp_stage = 3
                            applySL = False
                            signal = -position
                        elif price_diff >= tp2 and tp_stage < 2:
                            tp_stage = 2
                            applySL = True
                            if row["KAMA_Slope_abs"] < 0.001:
                                signal = -position + tp2pos
                        elif price_diff >= tp1 and tp_stage < 1:
                            tp_stage = 1
                            applySL = True
                            if row["KAMA_Slope_abs"] < 0.001:
                                signal = -position + tp1pos

                        if applySL:
                            if tp_stage == 1:
                                if price_diff - tp1 < -sl:
                                    signal = -position
                            elif tp_stage == 2:
                                if price_diff - tp2 < -sl:
                                    signal = -position

            # Apply Signal
            if signal != 0:
                if (position + signal > 1):
                    signal = 1 - position
                    position = 1
                elif (position + signal < -1):
                    signal = -1 - position
                    position = -1
                else:
                    position += signal
                    last_signal_time = current_time
                    if round(position, 2) == 0:
                        entry_price = None

            signals_p1[i] = signal
            positions_p1[i] = position

        # -------------------------------------------------------------
        # STEP 2: CHECK PRIORITY 1 ACTIVITY
        # -------------------------------------------------------------
        
        # Check if Priority 1 took ANY trade or held a position
        p1_active = np.any(np.array(positions_p1) != 0)
        
        if p1_active:
            # PRIORITY 1 WINS -> SAVE AND RETURN
            df["Signal"] = signals_p1
            df["Position"] = positions_p1
            # Note: We do NOT save the P1 extra feature columns (KFTrend, UCM, etc)
            # to match the P2 schema and prevent parser errors during merge.
            df["Regime"] = "Priority1_KF"
            
            output_path = temp_dir / f"day{day_num}.csv"
            # FIX: Ensure we only save the standard columns matching P2
            final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Regime"]
            df[final_columns].to_csv(output_path, index=False)
            del df_p1
            return str(output_path)
            
        # -------------------------------------------------------------
        # STEP 3: IF P1 INACTIVE -> EXECUTE PRIORITY 2
        # -------------------------------------------------------------
        
        # If we are here, P1 did nothing. We run P2.
        del df_p1 # clear memory
        
        time_sec_arr = df["Time_sec"].values.astype(np.int64)
        price_arr = df["Price"].values.astype(np.float64)
        day_open_price = price_arr[0]
        
        mask_30min = time_sec_arr <= DECISION_WINDOW_SECONDS
        
        can_trade_low_mu = False
        can_trade_high_mu = False
        active_params_low_mu = None
        regime_name = "NoTrade_P2"
        start_idx = 0

        if np.any(mask_30min):
            prices_30min = price_arr[mask_30min]
            mu = np.mean(prices_30min)
            sigma = np.std(prices_30min)
            
            # --- LOW MU (< 100) ---
            if mu < MU_SPLIT_THRESHOLD:
                if sigma > 0.13:
                    active_params_low_mu = PARAMS_STRATEGY_A
                    can_trade_low_mu = True
                    regime_name = "LowMu_StratA_HighSig"
                elif 0.07 <= sigma <= 0.1:
                    active_params_low_mu = PARAMS_STRATEGY_B
                    can_trade_low_mu = True
                    regime_name = "LowMu_StratB_MidSig"
                    
            # --- HIGH MU (> 100) ---
            else: 
                if sigma > HM_SIGMA_THRESHOLD:
                    can_trade_high_mu = True
                    regime_name = "HighMu_Volatile"

            start_idx = np.searchsorted(time_sec_arr, DECISION_WINDOW_SECONDS, side='right')

        # Execution Logic P2
        if can_trade_low_mu and active_params_low_mu is not None:
            signals, positions = backtest_core_low_mu(
                time_sec_arr, price_arr, day_open_price, int(start_idx), bool(can_trade_low_mu), COOLDOWN_PERIOD_SECONDS,
                active_params_low_mu["ENTRY_DROP_THRESHOLD"], active_params_low_mu["PRE_DROP_SHORT_SL"],
                active_params_low_mu["STATIC_SL"], active_params_low_mu["TP_ACTIVATION"], active_params_low_mu["TRAIL_CALLBACK"],
                REVERSAL_RISE_TRIGGER_LOW_MU, active_params_low_mu["SHORT_STATIC_SL"],
                active_params_low_mu["SHORT_TP_ACTIVATION"], active_params_low_mu["SHORT_TRAIL_CALLBACK"]
            )
        elif can_trade_high_mu:
            signals, positions = backtest_core_high_mu(
                time_sec_arr, price_arr, day_open_price, int(start_idx), bool(can_trade_high_mu), COOLDOWN_PERIOD_SECONDS,
                HM_ENTRY_RISE_THRESHOLD, HM_PRE_RISE_SL
            )
        else:
            signals = np.zeros(len(price_arr), dtype=np.int64)
            positions = np.zeros(len(price_arr), dtype=np.int64)

        df["Signal"] = signals
        df["Position"] = positions
        df["Regime"] = regime_name
        
        output_path = temp_dir / f"day{day_num}.csv"
        final_columns = REQUIRED_COLUMNS + ["Signal", "Position", "Day", "Regime"]
        df[final_columns].to_csv(output_path, index=False)
        
        return str(output_path)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(directory: str, max_workers: int, strategy_params_cols):
    # Setup multiprocessing context
    if sys.platform != 'win32':
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass
    
    # Use spawn context if issues arise with GPU libraries, but standard fork is usually fine for this mix
    ctx = get_context("spawn") 

    start_time = time.time()
    temp_dir_path = pathlib.Path(TEMP_DIR)
    
    if temp_dir_path.exists():
        shutil.rmtree(temp_dir_path)
    os.makedirs(temp_dir_path)

    files = sorted(glob.glob(os.path.join(directory, "day*.parquet")), key=extract_day_num)
    
    if not files:
        print("No parquet files found.")
        return

    processed_files = []
    
    # Using Pool from Priority 1 style (more robust for mixed logic)
    print(f"Starting Pool with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_day, f, extract_day_num(f), temp_dir_path, strategy_params_cols): f
            for f in files
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    processed_files.append(res)
            except Exception as e:
                print(f"Error in worker: {e}")

    if not processed_files:
        print("No files processed.")
        return

    processed_sorted = sorted(processed_files, key=extract_day_num)
    print(f"Merging {len(processed_sorted)} files...")
    
    with open(OUTPUT_FILE, "wb") as out:
        for i, csv_file in enumerate(processed_sorted):
            with open(csv_file, "rb") as inp:
                if i == 0:
                    shutil.copyfileobj(inp, out)
                else:
                    inp.readline() 
                    shutil.copyfileobj(inp, out)

    print("Generating trade reports...")
    generate_trade_reports_csv(OUTPUT_FILE)

    shutil.rmtree(temp_dir_path)
    print(f"✅ Output saved: {OUTPUT_FILE}")
    print(f"⏱ Total time: {time.time() - start_time:.2f}s")

def generate_trade_reports_csv(output_file):
    SAVE_DIR = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/"
    SAVE_PATH = os.path.join(SAVE_DIR, "trade_reports_MERGED_PRIORITY.csv")

    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        return

    if "Signal" not in df.columns or "Day" not in df.columns:
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found.")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]
        if pd.isna(signal): continue

        if position is None:
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            if signal == -position or (abs(position + signal) < abs(position)): # Close or partial
                direction = "LONG" if position > 0 else "SHORT"
                pnl = (float(row["Price"]) - float(entry["Price"])) * (1 if position > 0 else -1)
                
                trades.append({
                    "day": entry["Day"],
                    "direction": direction,
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": pnl,
                    "regime": entry["Regime"] if "Regime" in entry else "Unknown"
                })
                position = None
                entry = None

    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(SAVE_PATH, index=False)
        print(f"✓ trade_reports.csv saved at: {SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("--max-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    main(args.directory, args.max_workers, STRATEGY_PARAMS_COLS)