import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
DATA_FILE = "/data/quant14/EBY/combined.csv"
OUTPUT_DIR = "."
FEATURES = [
    "PB9_T12","PB4_T12","PB5_T12","PB8_T12","PB12_T12","PB13_T12",
    "PB16_T12","PB17_T12","PB18_T12","BB11_T12","BB12_T12","BB24",
    "BB10_T12","BB7_T12","BB8_T12"
]

# Granger hyperparameters
GRANGER_MAX_LAG = 5          # maximum lags to test
GRANGER_SIGNIF = 0.05        # p-value threshold for "Yes"
GRANGER_SAMPLE = 50000       # rows used for granger (to keep it fast)

# Optional: downsample for Pearson too (None = use all rows)
PEARSON_SAMPLE = None  # e.g., 2_000_000 for speed if needed


# =========================
# Utils
# =========================
def to_float_array(s: pd.Series) -> np.ndarray:
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, copy=False)
    a[np.isinf(a)] = np.nan
    return a

def pearson_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return (np.nan, int(mask.sum()))
    xx = x[mask]
    yy = y[mask]
    # numerically stable pearson
    x_mean = xx.mean()
    y_mean = yy.mean()
    xd = xx - x_mean
    yd = yy - y_mean
    denom = np.sqrt((xd * xd).sum()) * np.sqrt((yd * yd).sum())
    if denom == 0.0 or np.isnan(denom):
        return (np.nan, int(mask.sum()))
    return float((xd * yd).sum() / denom), int(mask.sum())

def granger_yes_no(feature: np.ndarray, price: np.ndarray,
                   max_lag=GRANGER_MAX_LAG, alpha=GRANGER_SIGNIF, sample=GRANGER_SAMPLE) -> str:
    # align + dropna
    mask = ~np.isnan(feature) & ~np.isnan(price)
    f = feature[mask]
    p = price[mask]
    if len(f) < 50:
        return "No"

    # sample head in time order (keeps chronology)
    n = len(f)
    if n > sample:
        f = f[:sample]
        p = p[:sample]

    # statsmodels expects a DataFrame with columns [target, cause] in that order
    df = pd.DataFrame({"Price": p, "Feat": f})
    # maxlag cannot exceed len//4
    max_lag = min(max_lag, max(1, len(df)//4))
    try:
        res = grangercausalitytests(df[["Price", "Feat"]], maxlag=max_lag, verbose=False)
        pvals = []
        for lag in range(1, max_lag+1):
            # use SSR F-test p-value
            pvals.append(res[lag][0]["ssr_ftest"][1])
        min_p = np.nanmin(pvals) if len(pvals) else 1.0
        return "Yes" if (min_p < alpha) else "No"
    except Exception:
        return "No"


# =========================
# Main
# =========================
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(DATA_FILE)

    # Load only needed columns
    usecols = FEATURES + ["Price"]
    print(f"Loading columns ({len(usecols)}): {usecols}")
    df = pd.read_csv(DATA_FILE, usecols=usecols)
    print(f"✓ Loaded {len(df):,} rows")

    # Convert to float arrays
    price = to_float_array(df["Price"])
    feature_arrays = {name: to_float_array(df[name]) for name in FEATURES}

    # Optional downsample for Pearson (still compute engineered series on subset)
    if PEARSON_SAMPLE and len(df) > PEARSON_SAMPLE:
        idx = np.arange(PEARSON_SAMPLE)  # keep first N to preserve time-order
        price_p = price[idx]
    else:
        idx = None
        price_p = price

    results = []

    # Build all ordered pairs (A,B), A != B
    print("Computing correlations for all ordered pairs (A-B and A/B) against Price...")
    for i, A in enumerate(FEATURES):
        a = feature_arrays[A]
        for j, B in enumerate(FEATURES):
            if i == j:
                continue
            b = feature_arrays[B]

            # --- subtraction: A - B
            sub = a - b
            sub[np.isinf(sub)] = np.nan

            if idx is not None:
                sub_p = sub[idx]
            else:
                sub_p = sub

            pearson_sub, n_sub = pearson_corr(sub_p, price_p)
            granger_sub = granger_yes_no(sub, price)

            results.append({
                "Op": "SUB",
                "Left": A,
                "Right": B,
                "Pearson(price, A-B)": pearson_sub,
                "Valid_Points": n_sub,
                "Granger(A-B → Price)": granger_sub
            })

            # --- division: A / B  (safe division)
            div = np.divide(a, b, out=np.full_like(a, np.nan), where=~np.isnan(a) & ~np.isnan(b) & (b != 0))
            div[np.isinf(div)] = np.nan

            if idx is not None:
                div_p = div[idx]
            else:
                div_p = div

            pearson_div, n_div = pearson_corr(div_p, price_p)
            granger_div = granger_yes_no(div, price)

            results.append({
                "Op": "DIV",
                "Left": A,
                "Right": B,
                "Pearson(price, A/B)": pearson_div,
                "Valid_Points": n_div,
                "Granger(A/B → Price)": granger_div
            })

    out = pd.DataFrame(results)
    # Sort (optional): strongest Pearson first within each Op
    out_sorted = (out.sort_values(
        by=["Op", "Pearson(price, A-B)"], ascending=[True, False], kind="mergesort")
        .reset_index(drop=True))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, f"pairwise_sub_div_vs_price_{ts}.csv")
    out_sorted.to_csv(out_csv, index=False)

    print("\nTop 10 results by operation:")
    for op in ["SUB", "DIV"]:
        print(f"\n=== {op} ===")
        print(out_sorted[out_sorted["Op"] == op].head(10).to_string(index=False))

    print(f"\n✓ Saved: {out_csv}")


if __name__ == "__main__":
    main()
