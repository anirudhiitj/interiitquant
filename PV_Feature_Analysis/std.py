import pandas as pd
import numpy as np
import cudf
import dask_cudf
from filter import get_pairs_for_day


def compute_3h_to_5h_stats(days_list):
    """
    Loads each day using get_pairs_for_day(),
    extracts tick prices BETWEEN 03:00:01 and 05:00:00,
    combines them across days, and computes:
        - min
        - max
        - mean
        - std
        - count
    """
    all_prices = []

    lower = pd.to_timedelta("00:00:00")
    upper = pd.to_timedelta("00:30:00")

    for day in days_list:
        print(f"\nLoading Day {day}...")

        success, pairs, df_pandas = get_pairs_for_day(day)

        if not success:
            print(f"❌ Failed to load Day {day}, skipping.")
            continue

        # Convert dask/cuDF → pandas
        if isinstance(df_pandas, dask_cudf.DataFrame):
            df_pandas = df_pandas.compute().to_pandas()
        elif isinstance(df_pandas, cudf.DataFrame):
            df_pandas = df_pandas.to_pandas()

        # Clean data
        df_pandas = df_pandas.dropna(subset=["Price"]).reset_index(drop=True)
        df_pandas["Time"] = pd.to_timedelta(df_pandas["Time"])

        # Extract window 03:00:01 → 05:00:00
        window_df = df_pandas[(df_pandas["Time"] >= lower) & (df_pandas["Time"] <= upper)]

        if len(window_df) == 0:
            print(f"⚠ No ticks in 03:00:01 → 05:00:00 for Day {day}.")
            continue

        prices = window_df["Price"].tolist()
        all_prices.extend(prices)

        print(f"✔ Day {day}: {len(prices)} ticks added in window.")

    if len(all_prices) == 0:
        print("\n❌ No tick data found across all days in this 3h→5h window!")
        return None

    arr = np.array(all_prices)

    stats = {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "count": len(arr)
    }

    print("\n=======================================================")
    print(" 📊 STATS FOR 03:00:01 → 05:00:00 ACROSS ALL DAYS")
    print("=======================================================")
    print(f" Min   : {stats['min']:.6f}")
    print(f" Max   : {stats['max']:.6f}")
    print(f" Mean  : {stats['mean']:.6f}")
    print(f" Std   : {stats['std']:.6f}")
    print(f" Count : {stats['count']}")
    print("=======================================================\n")

    return stats



if __name__ == "__main__":
    # Days 0 → 278
    days_to_process = [261]

    compute_3h_to_5h_stats(days_to_process)
