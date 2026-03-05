# file: intraday_one_trade_bb_model.py
import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import timedelta

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = '/data/quant14/EBX'
OUT_SIGNALS = '/home/raid/Quant14/V_Feature_Analysis/one_trade_bb_signals.csv'
OUT_DAILY = '/home/raid/Quant14/V_Feature_Analysis/one_trade_bb_daily.csv'

# Trading rules
PRICE_COL = 'PB9_T1'      # <-- use this as price / execution column (user requested)
TIME_COL = 'Time'
FIRST_HOUR_MINUTES = 60   # used only if you later want first-hour logic
HOLD_TIME_SECONDS = 15    # minimum hold
ATR_WINDOW = 30
ATR_MULTIPLIER = 0.20     # stop distance = entry +/- ATR_MULTIPLIER * ATR
Z_WINDOW = 60
Z_THRESHOLD = 0.5

# Capital & sizing
START_CAPITAL = 100000.0  # currency units
# "no fractional trades" -> buy integer units = floor(capital / price)

# BB features (equal-weight)
BB_FEATURES = [
    'BB5_T1','BB13_T1','BB4_T1','BB6_T1','BB13_T2',
    'BB5_T2','BB11_T1','BB14_T1','BB12_T1','BB15_T1',
    'BB4_T2','BB6_T2','BB14_T2'
]

# Days to run (0..N-1). Change range() as needed or build list.
NUM_DAYS = 510

# ----------------------------
# HELPERS
# ----------------------------
def robust_time_parse(df):
    df = df.copy()
    df[TIME_COL] = pd.to_timedelta(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
    return df

def compute_atr(df, price_col=PRICE_COL, atr_window=ATR_WINDOW):
    # small rolling high/low then ATR
    df['High'] = df[price_col].rolling(3, center=True, min_periods=1).max()
    df['Low']  = df[price_col].rolling(3, center=True, min_periods=1).min()
    df['TR']   = (df['High'] - df['Low']).abs()
    df['ATR']  = df['TR'].rolling(atr_window, min_periods=1).mean().bfill().fillna(1e-9)
    return df

def bb_weighted_z(df, feature_list, z_window=Z_WINDOW, z_thresh=Z_THRESHOLD):
    # equal weights across available features
    valid = [f for f in feature_list if f in df.columns]
    if not valid:
        return np.zeros(len(df), dtype=int), pd.Series(np.zeros(len(df)), index=df.index)
    # compute simple average (equal weight)
    weighted = sum(df[f].astype(float) for f in valid) / len(valid)
    mean = weighted.rolling(z_window, min_periods=1).mean()
    std  = weighted.rolling(z_window, min_periods=1).std(ddof=0).replace(0,1e-9)
    zval = (weighted - mean) / std
    sig = np.where(zval > z_thresh, 1, np.where(zval < -z_thresh, -1, 0))
    return sig.astype(int), zval

def bb_crossover_signal(df, short_col='BB4_T2', long_col='BB4_T6'):
    # fallback: if cols not present, return zeros
    if short_col not in df.columns or long_col not in df.columns:
        return np.zeros(len(df), dtype=int)
    return np.where(df[short_col] > df[long_col], 1, np.where(df[short_col] < df[long_col], -1, 0)).astype(int)

# ----------------------------
# SINGLE DAY PROCESS
# ----------------------------
def process_day(day_num, capital=START_CAPITAL):
    path = os.path.join(DATA_DIR, f'day{day_num}.parquet')
    if not os.path.exists(path):
        print(f" missing {path}")
        return None, None

    table = pq.read_table(path)
    df = table.to_pandas(strings_to_categorical=False)
    if PRICE_COL not in df.columns:
        print(f" day {day_num}: price column {PRICE_COL} missing -> skip")
        return None, None

    df = robust_time_parse(df)
    df = compute_atr(df)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if df.empty:
        return None, None

    # generate signals
    z_sig, z_val = bb_weighted_z(df, BB_FEATURES)
    # use a simple BB crossover using two representative cols if exist (we try BB4_T2 and BB4_T6):
    band_sig = bb_crossover_signal(df, short_col='BB4_T2', long_col='BB4_T6')

    # combine: require both z_sig and band_sig to agree and be non-zero to open
    # pick first index in the day where they agree and we are not in min-hold (we haven't traded yet)
    entry_idx = None
    entry_side = 0
    for i in range(len(df)):
        if z_sig[i] != 0 and band_sig[i] != 0 and z_sig[i] == band_sig[i]:
            entry_idx = i
            entry_side = int(z_sig[i])
            break

    signals = np.zeros(len(df), dtype=int)   # per-timestep position
    daily_pnl = 0.0
    trades = 0

    if entry_idx is None:
        # no trade this day
        return pd.DataFrame({TIME_COL: df[TIME_COL], PRICE_COL: df[PRICE_COL], 'Signal': signals, 'Day': day_num}), (day_num, 0.0, 0)
    
    entry_price = float(df[PRICE_COL].iloc[entry_idx])
    qty = int(capital // entry_price)   # integer units, "no fractional trades"
    if qty <= 0:
        # can't afford a single unit -> skip
        return pd.DataFrame({TIME_COL: df[TIME_COL], PRICE_COL: df[PRICE_COL], 'Signal': signals, 'Day': day_num}), (day_num, 0.0, 0)

    # Execute trade at entry_idx: hold at least HOLD_TIME_SECONDS, exit either on ATR stop or EOD
    entry_atr = float(df['ATR'].iloc[entry_idx] if 'ATR' in df.columns else 0.0)
    stop_price = None
    if entry_side == 1:
        stop_price = entry_price - ATR_MULTIPLIER * entry_atr
    else:
        stop_price = entry_price + ATR_MULTIPLIER * entry_atr

    # Set positions array to entry_side from entry_idx onward until exit
    pos = entry_side
    signals[entry_idx] = pos
    trades = 1
    last_trade_time = df[TIME_COL].iloc[entry_idx]
    exit_idx = entry_idx

    # enforce min hold and check stop each subsequent row
    for j in range(entry_idx+1, len(df)):
        signals[j] = pos
        now = df[TIME_COL].iloc[j]
        price = float(df[PRICE_COL].iloc[j])
        # if hold time not reached, continue
        if (now - last_trade_time).total_seconds() < HOLD_TIME_SECONDS:
            continue
        # check stop
        if pos == 1 and price < stop_price:
            exit_idx = j
            break
        if pos == -1 and price > stop_price:
            exit_idx = j
            break
        # otherwise continue until end-of-day

    # if not exited earlier, close at last bar (EOD)
    if exit_idx == entry_idx:
        # didn't find earlier exit, use last index as exit
        exit_idx = len(df) - 1

    # compute pnl: (exit_price - entry_price) * side * qty
    exit_price = float(df[PRICE_COL].iloc[exit_idx])
    pnl = (exit_price - entry_price) * entry_side * qty
    daily_pnl = pnl
    # fill signals beyond entry until exit_idx inclusive
    signals[entry_idx:exit_idx+1] = entry_side

    out_df = pd.DataFrame({
        TIME_COL: df[TIME_COL],
        PRICE_COL: df[PRICE_COL].astype(float),
        'Signal': signals,
        'Day': day_num
    })
    return out_df, (day_num, daily_pnl, trades)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == '__main__':
    all_days = []
    daily_rows = []
    for d in range(NUM_DAYS):
        signals_df, summary = process_day(d, capital=START_CAPITAL)
        if signals_df is not None:
            all_days.append(signals_df)
        if summary is not None:
            daynum, pnl, ntr = summary
            daily_rows.append({'Day': daynum, 'PnL': pnl, 'Trades': ntr})

    if not all_days:
        raise RuntimeError("No valid days processed")

    all_df = pd.concat(all_days, ignore_index=True).sort_values(['Day', TIME_COL])
    all_df.to_csv(OUT_SIGNALS, index=False)
    pd.DataFrame(daily_rows).sort_values('Day').to_csv(OUT_DAILY, index=False)

    print("Saved signals ->", OUT_SIGNALS)
    print("Saved daily summary ->", OUT_DAILY)
