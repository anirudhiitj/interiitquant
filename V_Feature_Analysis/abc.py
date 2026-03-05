import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq

# =========================
# DEBUG CONFIG
# =========================
DATA_DIR = '/data/quant14/EBX'
DEBUG_DAY = 70   # change to a failing day (70..100 from your tests)
PRICE_COL = 'Price'
TIME_COL = 'Time'

# FEATURES (same as your strategy)
BB_FEATURES = {
    'BB1_T2': 0.22, 'BB1_T3': 0.22, 'BB1_T4': 0.18,
    'BB4_T2': 0.18, 'BB4_T3': 0.15, 'BB4_T4': 0.03,
    'BB5_T2': 0.01, 'BB5_T3': 0.005, 'BB5_T4': 0.005,
    'PB10_T3': 0.1, 'PB11_T3': -0.1
}
PB_FEATURES = {
    'PB2_T2': -0.05, 'PB2_T3': -0.05, 'PB2_T4': -0.05,
    'PB5_T2': -0.07, 'PB5_T3': -0.07, 'PB5_T4': -0.07,
    'PB6_T3': -0.08, 'PB6_T4': -0.08,
    'PB7_T3': -0.06, 'PB7_T4': -0.06,
    'PB3_T3': -0.05, 'PB3_T2': -0.05, 'PB3_T4': -0.05
}

Z_WINDOW = 30
Z_THRESHOLD = 0.3

ATR_WINDOW = 30
ATR_MULTIPLIER = 0.2
FIRST_HOUR_MINUTES = 60

# =========================
# HELPERS
# =========================
def safe_read_day(day):
    path = os.path.join(DATA_DIR, f'day{day}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    df = table.to_pandas(strings_to_categorical=False)
    return df

def prep_df(df):
    # keep a copy and coerce numeric where possible
    df = df.copy()
    # show columns
    print("\nColumns present (first 80):", df.columns.tolist()[:80])
    # coerce numeric for relevant columns
    for c in df.columns:
        try:
            # try convert non-time cols to numeric
            if c != TIME_COL:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
    # parse time
    df[TIME_COL] = pd.to_timedelta(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL]).reset_index(drop=True)
    return df

def compute_atr(df, price_col=PRICE_COL, atr_window=ATR_WINDOW):
    df = df.copy()
    df['High'] = df[price_col].rolling(3, center=True, min_periods=1).max()
    df['Low']  = df[price_col].rolling(3, center=True, min_periods=1).min()
    df['TR']   = (df['High'] - df['Low']).abs()
    df['ATR']  = df['TR'].rolling(atr_window, min_periods=1).mean().bfill().fillna(1e-6)
    return df

def debug_weighted_z(df, feature_dict, name="BB/PB"):
    print(f"\n----- DEBUG {name} -----")
    valid_feats = [f for f in feature_dict if f in df.columns]
    missing_feats = [f for f in feature_dict if f not in df.columns]
    print("valid_feats:", valid_feats)
    print("missing_feats:", missing_feats)
    if len(valid_feats) == 0:
        print(">>> No valid features found. This will produce all-zero signals.")
        return None

    # weight sums
    sum_signed = sum(feature_dict[f] for f in valid_feats)
    sum_abs = sum(abs(feature_dict[f]) for f in valid_feats)
    print(f"sum_signed_weights = {sum_signed:.6f}, sum_abs_weights = {sum_abs:.6f}")

    # compute weighted composite (use abs normalization as in Fix A)
    if sum_abs == 0:
        print(">>> sum_abs == 0 -> can't normalize")
        return None

    weighted = sum(feature_dict[f] * df[f].astype(float) for f in valid_feats) / sum_abs
    weighted.name = f"weighted_{name}"
    # basic stats
    print("weighted stats (all rows):", weighted.describe().to_dict())

    # rolling z
    mean = weighted.rolling(Z_WINDOW, min_periods=1).mean()
    std  = weighted.rolling(Z_WINDOW, min_periods=1).std(ddof=0).replace(0, 1e-9)
    zval = (weighted - mean) / std
    zname = f"z_{name}"
    zval.name = zname
    print("zval stats (all rows):", zval.describe().to_dict())
    nonzero_signals = ((zval > Z_THRESHOLD) | (zval < -Z_THRESHOLD)).sum()
    print(f"nonzero signals count (all rows): {int(nonzero_signals)} / {len(zval)}")

    # show first-hour subset stats
    start_t = df[TIME_COL].iloc[0]
    cutoff = start_t + pd.Timedelta(minutes=FIRST_HOUR_MINUTES)
    df_first = df[df[TIME_COL] <= cutoff].copy()
    weighted_first = weighted.loc[df_first.index]
    z_first = zval.loc[df_first.index]
    print("\nfirst-hour weighted stats:", weighted_first.describe().to_dict())
    print("first-hour z stats:", z_first.describe().to_dict())
    print("first-hour nonzero signals:", ((z_first > Z_THRESHOLD) | (z_first < -Z_THRESHOLD)).sum(), "/", len(z_first))

    # print a small sample
    sample = pd.DataFrame({
        'Time': df.loc[df_first.index, TIME_COL].astype(str).values,
        'Price': df.loc[df_first.index, PRICE_COL].values,
        'weighted': weighted_first.values,
        'zval': z_first.values
    }).head(20)
    print("\nSample first-hour rows (head 20):")
    print(sample.to_string(index=False))

    return weighted, zval

# =========================
# RUN DEBUG FOR A DAY
# =========================
if __name__ == "__main__":
    print(f"DEBUG run for day {DEBUG_DAY}")
    df_raw = safe_read_day(DEBUG_DAY)
    df = prep_df(df_raw)
    df = compute_atr(df)

    # quick checks
    if PRICE_COL not in df.columns:
        print("ERROR: Price column missing!")
    else:
        print(f"Rows: {len(df)}, Price NaNs: {df[PRICE_COL].isna().sum()}")

    # Debug BB
    debug_bb = debug_weighted_z(df, BB_FEATURES, name="BB")
    # Debug PB
    debug_pb = debug_weighted_z(df, PB_FEATURES, name="PB")

    # Also show bands simple crossover stats for first hour
    print("\n----- DEBUG BANDS -----")
    # choose short/long proxies (as your code did)
    keys = list(BB_FEATURES.keys())
    if len(keys) >= 2 and keys[0] in df.columns and keys[-1] in df.columns:
        short = keys[0]; long = keys[-1]
        band_sig = np.where(df[short] > df[long], 1, np.where(df[short] < df[long], -1, 0))
        first_mask = df[TIME_COL] <= (df[TIME_COL].iloc[0] + pd.Timedelta(minutes=FIRST_HOUR_MINUTES))
        print("bands short/long:", short, long)
        print("bands first-hour nonzero signals:", int(((band_sig[first_mask] != 0).sum())), "/", int(first_mask.sum()))
        print("bands sample (first 30):")
        print(pd.DataFrame({'Time': df.loc[first_mask, TIME_COL].astype(str).values[:30],
                            short: df.loc[first_mask, short].values[:30],
                            long: df.loc[first_mask, long].values[:30],
                            'sig': band_sig[first_mask][:30]}).to_string(index=False))
    else:
        print("bands columns missing for short/long check:", keys[:3], "...", keys[-3:])
