import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
import math

# ===============================================================
# CONFIGURATION
# ===============================================================
DATA_DIR = '/data/quant14/EBX'
OUTPUT_FILE = '/home/raid/Quant14/V_Feature_Analysis/first_strat/trading_signals1.csv'

FIRST_HOUR_MINUTES = 40
HOLD_TIME = 15
PRICE_COL = 'Price'
TIME_COL = 'Time'

# =======================
# FEATURE / WEIGHT CONFIG
# =======================
BB_FEATURES = {
    'BB1_T1': 0.22, 'BB1_T2': 0.22, 'BB1_T3': 0.18,
    'BB4_T1': 0.18, 'BB4_T2': 0.15, 'BB4_T3': 0.03,
    'BB5_T1': 0.01, 'BB5_T2': 0.005, 'BB5_T3': 0.005,
    'PB10_T2': 0.1, 'PB11_T2': -0.1
}

BB_Z_WINDOW = 60
BB_Z_THRESHOLD = 0.5

PB_FEATURES = {
    'PB2_T2': -0.05, 'PB2_T3': -0.05, 'PB2_T4': -0.05,
    'PB5_T2': -0.07, 'PB5_T3': -0.07, 'PB5_T4': -0.07,
    'PB6_T3': -0.08, 'PB6_T4': -0.08,
    'PB7_T3': -0.06, 'PB7_T4': -0.06,
    'PB3_T3': -0.05, 'PB3_T2': -0.05, 'PB3_T4': -0.05
}
PB_Z_WINDOW = 60
PB_Z_THRESHOLD = 0.5

# ATR/trading
ATR_WINDOW = 30
ATR_MULTIPLIER = 0.2
UNIT_POSITION = 1.0

# concurrency / days
NUM_DAYS = 510
MAX_WORKERS = 8  # adjust to your environment

# ===============================================================
# INTRADAY SELF-EVAL KNOBS (30-min adaptive pause)
# ===============================================================
EVAL_WINDOW_MINS = 8          # evaluate every 30 minutes
SUCCESS_WINDOW = 8                # how many past intervals to include in EW success
SUCCESS_ALPHA = 0.3                  # EWMA alpha
SUCCESS_THRESHOLD = 0.40     # if below, pause next interval
DAY_SKIP_PNL_THRESHOLD = -0.075      # if first-hour chosen-strat pnl < this -> skip trading whole day

# ===============================================================
# HELPERS
# ===============================================================
def robust_time_parse(df):
    df = df.copy()
    df[TIME_COL] = pd.to_timedelta(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
    return df

def compute_atr(df, price_col=PRICE_COL, atr_window=ATR_WINDOW):
    df['High'] = df[price_col].rolling(3, center=True, min_periods=1).max()
    df['Low'] = df[price_col].rolling(3, center=True, min_periods=1).min()
    df['TR'] = (df['High'] - df['Low']).abs()
    df['ATR'] = df['TR'].rolling(atr_window, min_periods=1).mean().bfill().fillna(1e-6)
    return df

def exp_weighted_success_rate(success_list, alpha=SUCCESS_ALPHA):
    """
    success_list: list-like of 0/1 values (most recent last).
    returns EW success rate in [0,1].
    """
    if len(success_list) == 0:
        return 1.0
    ew = 0.0
    w = 0.0
    for s in reversed(success_list):  # most recent first for alpha weighting
        ew = alpha * s + (1 - alpha) * ew
        # This computation yields the EWMA itself; we can return ew directly.
    return float(ew)

# -------------------------
# Tier-based signal builders
# -------------------------
def weighted_z_signal(df, feature_dict, z_window, z_thresh):
    valid_feats = [f for f in feature_dict if f in df.columns]
    if len(valid_feats) == 0:
        return np.zeros(len(df), dtype=int), pd.Series(np.zeros(len(df)), index=df.index)

    # normalize by sum(abs(weights)) to avoid cancellation
    total_weight = sum(abs(feature_dict[f]) for f in valid_feats)
    if total_weight == 0:
        return np.zeros(len(df), dtype=int), pd.Series(np.zeros(len(df)), index=df.index)

    weighted = sum(feature_dict[f] * df[f].astype(float) for f in valid_feats) / total_weight
    mean = weighted.rolling(z_window, min_periods=1).mean()
    std = weighted.rolling(z_window, min_periods=1).std(ddof=0).replace(0, 1e-9)
    z_val = (weighted - mean) / std
    sig = np.where(z_val > z_thresh, 1, np.where(z_val < -z_thresh, -1, 0))
    return sig.astype(int), z_val

def generate_bb_z_signal(df):
    return weighted_z_signal(df, BB_FEATURES, BB_Z_WINDOW, BB_Z_THRESHOLD)

def generate_pb_z_signal(df):
    return weighted_z_signal(df, PB_FEATURES, PB_Z_WINDOW, PB_Z_THRESHOLD)

def generate_bands_signal(df):
    keys = list(BB_FEATURES.keys())
    if len(keys) < 2:
        return np.zeros(len(df), dtype=int)
    short_col = keys[0]
    long_col = keys[-1]
    if short_col not in df.columns or long_col not in df.columns:
        return np.zeros(len(df), dtype=int)
    sig = np.where(df[short_col] > df[long_col], 1,
                   np.where(df[short_col] < df[long_col], -1, 0))
    return sig.astype(int)

# -------------------------
# Mini backtester
# -------------------------
def mini_backtest(df, sig, atr_col='ATR'):
    pos, entry_price, pnl, trades = 0, 0.0, 0.0, 0
    for i in range(len(df)):
        price, atr = float(df[PRICE_COL].iloc[i]), float(df[atr_col].iloc[i])
        s = int(sig[i])
        if pos == 0 and s != 0:
            pos, entry_price = s, price
            trades += 1
        elif pos != 0:
            pnl_now = (price - entry_price) * pos
            stop_hit = (
                (pos == 1 and price < entry_price - ATR_MULTIPLIER * atr)
                or (pos == -1 and price > entry_price + ATR_MULTIPLIER * atr)
            )
            if s == -pos or stop_hit:
                pnl += pnl_now
                pos = 0
    if pos != 0:
        pnl += (float(df[PRICE_COL].iloc[-1]) - entry_price) * pos
    return float(pnl), trades

# -------------------------
# Intraday runner with adaptive 30-min pause
# -------------------------
def run_intraday(df, raw_sig):
    """
    raw_sig: array-like per-bar signals (1/-1/0) from chosen strategy for the full day.
    This function enforces:
      - HOLD_TIME between trades,
      - ATR-based trailing exits,
      - intra-day evaluation every EVAL_WINDOW_MINS: compute EW success on recent intervals,
        and if below threshold, pause trading for next EVAL_WINDOW_MINS.
    """
    sig_out = np.zeros(len(df), dtype=int)
    pos, entry_price, entry_atr = 0, 0.0, 1e-6
    tsecs = (df[TIME_COL] - df[TIME_COL].iloc[0]).dt.total_seconds().values
    last_trade_t = -1e9

    # state for interval evaluation
    next_eval_t = EVAL_WINDOW_MINS * 60.0
    cooldown_until = -1.0
    paused = False

    # keep last SUCCESS_WINDOW interval outcomes (1 if interval pnl>0 else 0)
    recent_interval_success = deque(maxlen=SUCCESS_WINDOW)
    # track cumulative pnl to compute interval realized PnL
    cumulative_pnl = 0.0
    interval_start_pnl = 0.0

    for i in range(len(df)):
        price = float(df[PRICE_COL].iloc[i])
        atr = float(df['ATR'].iloc[i])
        s = int(raw_sig[i])
        t = float(tsecs[i])

        # If evaluation time reached -> conclude last interval
        if t >= next_eval_t:
            interval_realized = cumulative_pnl - interval_start_pnl
            interval_start_pnl = cumulative_pnl
            # record success if interval pnl > 0
            recent_interval_success.append(1 if interval_realized > 0 else 0)
            ew_success = exp_weighted_success_rate(list(recent_interval_success), alpha=SUCCESS_ALPHA)
            # if poor, pause
            if ew_success < SUCCESS_THRESHOLD:
                paused = True
                cooldown_until = t + EVAL_WINDOW_MINS * 60.0
                # inform
                # print(f"  → Pausing trading at {t/60:.2f} min (EW success={ew_success:.3f}) until {(cooldown_until)/60:.2f} min")
            next_eval_t += EVAL_WINDOW_MINS * 60.0

        # if in cooldown, skip trading until cooldown_until
        if paused and t < cooldown_until:
            sig_out[i] = pos
            # we still should update current unrealized pnl only when closing; no trade action
            continue
        elif paused and t >= cooldown_until:
            paused = False
            # reset interval timing so next evaluation happens after a full EVAL_WINDOW_MINS
            next_eval_t = t + EVAL_WINDOW_MINS * 60.0
            # continue trading from this bar

        # enforce hold time
        if (t - last_trade_t) < HOLD_TIME:
            sig_out[i] = pos
            continue

        # trading logic with ATR trailing stop
        if pos == 0:
            if s != 0 and not paused:
                pos, entry_price, entry_atr = s, price, atr
                last_trade_t = t
        else:
            stop_hit = (
                (pos == 1 and price < entry_price - ATR_MULTIPLIER * entry_atr)
                or (pos == -1 and price > entry_price + ATR_MULTIPLIER * entry_atr)
            )
            if s == -pos or stop_hit:
                # close and record pnl
                realized = (price - entry_price) * pos
                cumulative_pnl += realized
                pos = 0
                last_trade_t = t

        sig_out[i] = pos

    # if a position left open at EOD, realize at last price
    if pos != 0:
        final_price = float(df[PRICE_COL].iloc[-1])
        cumulative_pnl += (final_price - entry_price) * pos

    return sig_out

# ===============================================================
# PROCESS ONE DAY
# ===============================================================
def process_day(day_num):
    path = os.path.join(DATA_DIR, f'day{day_num}.parquet')
    if not os.path.exists(path):
        return None
    try:
        table = pq.read_table(path)
        df = table.to_pandas(strings_to_categorical=False)
        df = robust_time_parse(df)
        df = compute_atr(df)
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        if df.empty:
            return None

        start_t = df[TIME_COL].iloc[0]
        cutoff = start_t + pd.Timedelta(minutes=FIRST_HOUR_MINUTES)
        df_first = df[df[TIME_COL] <= cutoff].reset_index(drop=True)
        if df_first.empty:
            return None

        # Backtest on first hour
        bb_sig_first, _ = generate_bb_z_signal(df_first)
        pb_sig_first, _ = generate_pb_z_signal(df_first)
        bands_sig_first = generate_bands_signal(df_first)

        pnl_bb, _ = mini_backtest(df_first, bb_sig_first)
        pnl_pb, _ = mini_backtest(df_first, pb_sig_first)
        pnl_bands, _ = mini_backtest(df_first, bands_sig_first)

        best = max(
            {'bb': pnl_bb, 'pb': pnl_pb, 'bands': pnl_bands},
            key=lambda k: {'bb': pnl_bb, 'pb': pnl_pb, 'bands': pnl_bands}[k],
        )

        print(f"Day {day_num:3d} PnL → BB:{pnl_bb:.3f} | PB:{pnl_pb:.3f} | BANDS:{pnl_bands:.3f} → 🏆 {best}")

        # Skip day entirely if chosen strategy's first-hour pnl is too bad
        chosen_pnl = {'bb': pnl_bb, 'pb': pnl_pb, 'bands': pnl_bands}[best]
        if chosen_pnl < DAY_SKIP_PNL_THRESHOLD:
            print(f"  → SKIPPING DAY {day_num} because chosen-strat pnl ({chosen_pnl:.3f}) < {DAY_SKIP_PNL_THRESHOLD}")
            df['Signal'] = 0
            df['Day'] = day_num
            return df[[TIME_COL, PRICE_COL, 'Signal', 'Day']]

        # Generate full-day signals using winning method
        if best == 'bb':
            full_sig, _ = generate_bb_z_signal(df)
        elif best == 'pb':
            full_sig, _ = generate_pb_z_signal(df)
        else:
            full_sig = generate_bands_signal(df)

        # Block first-hour trades
        full_sig = np.array(full_sig, dtype=int)
        full_sig[df[TIME_COL] <= cutoff] = 0

        # Run intraday with adaptive pause
        sig_out = run_intraday(df, full_sig)

        df['Signal'] = sig_out
        df['Day'] = day_num
        return df[[TIME_COL, PRICE_COL, 'Signal', 'Day']]

    except Exception as e:
        print(f"⚠️ Error day {day_num}: {e}")
        return None

# ===============================================================
# MAIN EXECUTION — concurrent.futures version
# ===============================================================
if __name__ == "__main__":
    print("🚀 Running Two-Tier Signal Strategy (BB, PB, Bands) with concurrent futures & intraday adaptive pause")

    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_day, d): d for d in range(NUM_DAYS)}
        for future in as_completed(futures):
            day = futures[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"⚠️ Error in day {day}: {e}")
                errors.append(day)

    if not results:
        raise RuntimeError("❌ No valid data processed!")

    combined = pd.concat(results, ignore_index=True)
    combined.sort_values(by=['Day', TIME_COL], inplace=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(combined):,} rows | {combined['Day'].nunique()} days | File: {OUTPUT_FILE}")
    if errors:
        print(f"⚠️ Skipped {len(errors)} days: {errors}")
