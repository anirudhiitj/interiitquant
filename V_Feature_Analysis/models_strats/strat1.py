# file: intraday_ou_model.py
import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ====================================================
# CONFIG
# ====================================================
DATA_DIR = '/data/quant14/EBX'
OUTPUT_FILE = '/home/raid/Quant14/V_Feature_Analysis/ou_model_signals.csv'
PRICE_COL = 'PB9_T1'
TIME_COL = 'Time'

CAPITAL = 100000
MIN_HOLD_SEC = 15
ATR_MULT = 0.2
WINDOW = 240          # rolling regression window (e.g., ~4 min for 1s bars)
Z_THRESHOLD = 1.0
NUM_DAYS = 510

# ====================================================
# HELPERS
# ====================================================
def robust_time_parse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_timedelta(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
    return df

def compute_atr(df: pd.DataFrame, col: str = PRICE_COL, window: int = 30) -> pd.DataFrame:
    # Simple intraday ATR proxy from a centered 3-bar hi/lo envelope
    hi = df[col].rolling(3, center=True, min_periods=1).max()
    lo = df[col].rolling(3, center=True, min_periods=1).min()
    tr = (hi - lo).abs()
    atr = tr.rolling(window, min_periods=1).mean()
    out = df.copy()
    out['ATR'] = atr.fillna(method='bfill').fillna(1e-6)
    return out

# ====================================================
# OU MEAN REVERSION MODEL (ΔP_t = α + ρ P_{t-1} + ε_t)
# ====================================================
def run_day(day_num: int):
    path = os.path.join(DATA_DIR, f'day{day_num}.parquet')
    if not os.path.exists(path):
        return None, None

    table = pq.read_table(path)
    df = table.to_pandas(strings_to_categorical=False)

    # Ensure price column exists
    if PRICE_COL not in df.columns:
        return None, None

    # Prep
    df = robust_time_parse(df)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df = compute_atr(df, col=PRICE_COL, window=30)
    if df.empty:
        return None, None

    # Build regression inputs
    df['dP'] = df[PRICE_COL].diff()
    df['lagP'] = df[PRICE_COL].shift(1)
    df = df.dropna(subset=['dP', 'lagP']).reset_index(drop=True)

    # Not enough data to run rolling regression
    if len(df) <= WINDOW:
        return None, (day_num, 0.0, 0)

    # Rolling OLS: ΔP_t = α + ρ * P_{t-1} + ε_t
    alphas, rhos, residuals = [], [], []
    # Fit on windows [i-WINDOW, i)
    for i in range(WINDOW, len(df)):
        y = df['dP'].iloc[i - WINDOW:i].values
        x = df['lagP'].iloc[i - WINDOW:i].values
        X = add_constant(x, has_constant='add')
        model = OLS(y, X).fit()
        # collect params and most recent residual
        alphas.append(float(model.params[0]))     # const
        rhos.append(float(model.params[1]))       # rho on lagP
        residuals.append(float(model.resid.iloc[-1]))

    # Align with tail portion
    df = df.iloc[WINDOW:].copy()
    df['alpha'] = alphas
    df['rho'] = rhos
    df['resid'] = residuals

    # Guard: avoid division by ~0 rho
    small_rho = (df['rho'].abs() < 1e-8)
    df.loc[small_rho, 'rho'] = np.nan

    # Equilibrium mean μ = -α/ρ
    df['mu_eq'] = -df['alpha'] / df['rho']
    # Z-score vs ATR (scale-free)
    df['z'] = (df[PRICE_COL] - df['mu_eq']) / df['ATR'].replace(0, 1e-9)

    # Signal (one-shot intent): short if above mean by Z, long if below by Z
    df['signal_raw'] = np.where(
        df['rho'].notna(),  # only if rho is valid
        np.where(df['z'] > Z_THRESHOLD, -1,
                 np.where(df['z'] < -Z_THRESHOLD, 1, 0)),
        0
    )

    # Pick the first qualified signal (only one trade per day)
    idx = df.index[df['signal_raw'] != 0]
    if len(idx) == 0:
        return None, (day_num, 0.0, 0)

    entry_idx = idx[0]                    # this is a LABEL
    entry_pos = df.index.get_loc(entry_idx)  # convert to POSITION (fix for iloc)
    side = int(df.loc[entry_idx, 'signal_raw'])
    entry_price = float(df.loc[entry_idx, PRICE_COL])

    # All capital, integer units
    qty = int(CAPITAL // entry_price)
    if qty == 0:
        return None, (day_num, 0.0, 0)

    # Build signal (flat → position until exit/eod)
    sig = np.zeros(len(df), dtype=int)
    sig[entry_pos:] = side

    # ATR stop based on entry-time ATR
    atr_entry = float(df['ATR'].iloc[entry_pos])
    stop = entry_price - side * ATR_MULT * atr_entry

    # Enforce 15s minimum hold
    t_entry = df[TIME_COL].iloc[entry_pos]
    exit_pos = len(df) - 1  # default EOD

    for j in range(entry_pos + 1, len(df)):
        t = df[TIME_COL].iloc[j]
        if (t - t_entry).total_seconds() < MIN_HOLD_SEC:
            continue
        p = float(df[PRICE_COL].iloc[j])
        # stop check
        if (side == 1 and p < stop) or (side == -1 and p > stop):
            exit_pos = j
            break

    exit_price = float(df[PRICE_COL].iloc[exit_pos])
    pnl = (exit_price - entry_price) * side * qty

    # Output per-bar signals for this day
    df_out = pd.DataFrame({
        TIME_COL: df[TIME_COL].values,
        PRICE_COL: df[PRICE_COL].values,
        'Signal': sig,
        'Day': day_num
    })
    return df_out, (day_num, pnl, 1)

# ====================================================
# MAIN
# ====================================================
if __name__ == '__main__':
    results, daily = [], []

    for d in range(NUM_DAYS):
        df_day, summary = run_day(d)
        if df_day is not None:
            results.append(df_day)
        if summary is not None:
            daily.append({'Day': summary[0], 'PnL': summary[1], 'Trades': summary[2]})

    if results:
        all_df = pd.concat(results, ignore_index=True)
        all_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved signals to {OUTPUT_FILE}")

    if daily:
        daily_df = pd.DataFrame(daily).sort_values('Day')
        daily_path = OUTPUT_FILE.replace('.csv', '_daily.csv')
        daily_df.to_csv(daily_path, index=False)
        print("Daily summary:")
        print(daily_df.describe())
        print(f"✅ Saved daily summary to {daily_path}")
