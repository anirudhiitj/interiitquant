import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# FEATURE CONFIG
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

ATR_WINDOW = 30
ATR_MULTIPLIER = 0.2
UNIT_POSITION = 1.0
NUM_DAYS = 510
MAX_WORKERS = 8
PNL_THRESHOLD = -0.075   # 🚫 New cutoff: if below this, skip trading the day

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


# -------------------------
# Tier-based signal builders
# -------------------------
def weighted_z_signal(df, feature_dict, z_window, z_thresh):
    valid_feats = [f for f in feature_dict if f in df.columns]
    if len(valid_feats) == 0:
        return np.zeros(len(df)), pd.Series(np.zeros(len(df)), index=df.index)

    total_weight = sum(abs(feature_dict[f]) for f in valid_feats)
    if total_weight == 0:
        return np.zeros(len(df)), pd.Series(np.zeros(len(df)), index=df.index)

    weighted = sum(feature_dict[f] * df[f].astype(float) for f in valid_feats) / total_weight
    mean = weighted.rolling(z_window, min_periods=1).mean()
    std = weighted.rolling(z_window, min_periods=1).std(ddof=0).replace(0, 1e-9)
    z_val = (weighted - mean) / std
    sig = np.where(z_val > z_thresh, 1, np.where(z_val < -z_thresh, -1, 0))
    return sig, z_val


def generate_bb_z_signal(df):
    return weighted_z_signal(df, BB_FEATURES, BB_Z_WINDOW, BB_Z_THRESHOLD)


def generate_pb_z_signal(df):
    return weighted_z_signal(df, PB_FEATURES, PB_Z_WINDOW, PB_Z_THRESHOLD)


def generate_bands_signal(df):
    keys = list(BB_FEATURES.keys())
    if len(keys) < 2:
        return np.zeros(len(df))
    short_col = keys[0]
    long_col = keys[-1]
    if short_col not in df.columns or long_col not in df.columns:
        return np.zeros(len(df))
    sig = np.where(df[short_col] > df[long_col], 1,
                   np.where(df[short_col] < df[long_col], -1, 0))
    return sig


# -------------------------
# Mini backtester
# -------------------------
def mini_backtest(df, sig, atr_col='ATR'):
    pos, entry_price, pnl, trades = 0, 0, 0, 0
    for i in range(len(df)):
        price, atr = df[PRICE_COL].iloc[i], df[atr_col].iloc[i]
        s = sig[i]
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
        pnl += (df[PRICE_COL].iloc[-1] - entry_price) * pos
    return pnl, trades


# -------------------------
# Intraday runner
# -------------------------
def run_intraday(df, raw_sig):
    sig_out = np.zeros(len(df), dtype=int)
    pos, entry_price, entry_atr = 0, 0, 1e-6
    tsecs = (df[TIME_COL] - df[TIME_COL].iloc[0]).dt.total_seconds().values
    last_trade_t = 0.0

    for i in range(len(df)):
        price, atr = float(df[PRICE_COL].iloc[i]), float(df['ATR'].iloc[i])
        s, t = int(raw_sig[i]), float(tsecs[i])
        if (t - last_trade_t) < HOLD_TIME:
            sig_out[i] = pos
            continue
        if pos == 0 and s != 0:
            pos, entry_price, entry_atr = s, price, atr
            last_trade_t = t
        elif pos != 0:
            stop_hit = (
                (pos == 1 and price < entry_price - ATR_MULTIPLIER * entry_atr)
                or (pos == -1 and price > entry_price + ATR_MULTIPLIER * entry_atr)
            )
            if s == -pos or stop_hit:
                pos = 0
                last_trade_t = t
        sig_out[i] = pos
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
        df_first = df[df[TIME_COL] <= cutoff]

        # Backtest on first hour
        bb_sig_first, _ = generate_bb_z_signal(df_first)
        pb_sig_first, _ = generate_pb_z_signal(df_first)
        bands_sig_first = generate_bands_signal(df_first)

        pnl_bb, _ = mini_backtest(df_first, bb_sig_first)
        pnl_pb, _ = mini_backtest(df_first, pb_sig_first)
        pnl_bands, _ = mini_backtest(df_first, bands_sig_first)

        # Pick the best-performing strategy
        strat_pnls = {'bb': pnl_bb, 'pb': pnl_pb, 'bands': pnl_bands}
        best = max(strat_pnls, key=strat_pnls.get)
        best_pnl = strat_pnls[best]

        # 🚫 Skip this day if best strategy’s first-hour PnL < threshold
        if best_pnl < PNL_THRESHOLD:
            print(f"Day {day_num:3d} → Skipped (Best {best} PnL={best_pnl:.3f} < {PNL_THRESHOLD})")
            df['Signal'] = 0
            df['Day'] = day_num
            return df[[TIME_COL, PRICE_COL, 'Signal', 'Day']]

        print(f"Day {day_num:3d} PnL → BB:{pnl_bb:.3f} | PB:{pnl_pb:.3f} | BANDS:{pnl_bands:.3f} → 🏆 {best}")

        # Generate full-day signals
        if best == 'bb':
            full_sig, _ = generate_bb_z_signal(df)
        elif best == 'pb':
            full_sig, _ = generate_pb_z_signal(df)
        else:
            full_sig = generate_bands_signal(df)

        # Block first-hour trades
        full_sig[df[TIME_COL] <= cutoff] = 0
        sig_out = run_intraday(df, full_sig)

        df['Signal'] = sig_out
        df['Day'] = day_num
        return df[[TIME_COL, PRICE_COL, 'Signal', 'Day']]

    except Exception as e:
        print(f"⚠️ Error day {day_num}: {e}")
        return None


# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    print("🚀 Running Two-Tier Signal Strategy (BB, PB, Bands) with daily PnL cutoff")

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
