import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_CSV = "./trade_report.csv"
OUTPUT_PLOT = "./histogram/daily_pnl_histogram.png"

def plot_daily_pnl_histogram(trade_report_path=INPUT_CSV, save_path=OUTPUT_PLOT):
    """
    Reads trade_reports.csv with columns:
    day, direction, entry_time, entry_price, exit_time, exit_price, pnl

    Computes daily total PnL for day0–day509 (missing → 0)
    Creates a histogram-style bar plot:
        X-axis = day number
        Y-axis = daily PnL
    """

    try:
        df = pd.read_csv(trade_report_path)
    except FileNotFoundError:
        print(f"❌ ERROR: trade report not found at {trade_report_path}")
        return

    # Required columns
    if "day" not in df.columns or "raw_pnl" not in df.columns:
        print("❌ CSV must contain 'day' and 'pnl'")
        return

    df["day"] = df["day"].astype(int)

    # Total PnL per day
    daily_pnl = df.groupby("day")["raw_pnl"].sum()

    # Create full list of 510 days (day0–day509)
    pnl_series = pd.Series(0.0, index=range(279))
    pnl_series.update(daily_pnl)

    # BAR PLOT (Histogram style)
    plt.figure(figsize=(22, 7))
    plt.bar(pnl_series.index, pnl_series.values, width=0.9, color='skyblue', edgecolor='black')

    plt.title("Daily PnL Histogram (Day 0 to Day 509)")
    plt.xlabel("Day Number")
    plt.ylabel("Total PnL")
    plt.grid(True, linestyle="--", alpha=0.4, axis='y')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Daily PnL histogram saved to: {save_path}")

    return pnl_series


if __name__ == "__main__":
    plot_daily_pnl_histogram()
