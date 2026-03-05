import numpy as np
import pandas as pd
from pathlib import Path
from numba import jit

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'PRICE_COLUMN': 'Price',
    'VOLATILITY_FEATURE': 'PB9_T1',
    'PRICE_JUMP_THRESHOLD': 0.3,   # price fluctuation threshold
    'ATR_WINDOW': 600,             # 600 seconds (10-minute ATR window)
    'ATR_THRESHOLD': 0.01,         # defines regime shift
    'TIMESTAMP_COLUMN_NAMES': ['time', 'timestamp', 'datetime', 'date']
}

# =====================================================================
# NUMBA ACCELERATED UTILITIES
# =====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs(prices, jump_threshold):
    """
    Find all pairs of indices (i, j) where absolute price difference >= jump_threshold.
    """
    n = len(prices)
    pairs = []
    i = 0
    while i < n:
        start_price = prices[i]
        for j in range(i + 1, n):
            if abs(prices[j] - start_price) >= jump_threshold:
                pairs.append((i, j))
                i = j + 1
                break
        else:
            i += 1
    return pairs

# =====================================================================
# DAILY PROCESSING
# =====================================================================

def process_single_day(day_num, config):
    """
    For each day:
      - Compute ATR proxy (rolling mean abs diff of PB9_T1)
      - Find first ATR >= threshold
      - Classify 0.3 price moves as missed (before) or considered (after)
    """
    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        price_col = config['PRICE_COLUMN']
        vol_col = config['VOLATILITY_FEATURE']

        if price_col not in df.columns or vol_col not in df.columns:
            return {'day': day_num, 'success': False, 'reason': 'missing_columns'}

        # Drop NaNs
        df = df[[price_col, vol_col]].dropna().copy()
        if len(df) < config['ATR_WINDOW']:
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        prices = df[price_col].astype(float).values
        pb9_t1 = df[vol_col].astype(float).values

        # --- Compute ATR proxy (600-second rolling window) ---
        df['atr_proxy'] = (
            pd.Series(pb9_t1)
            .diff()
            .abs()
            .rolling(config['ATR_WINDOW'], min_periods=1)
            .mean()
            .fillna(0)
        )

        atr_values = df['atr_proxy'].values
        atr_hit_index = np.argmax(atr_values >= config['ATR_THRESHOLD'])
        if atr_values[atr_hit_index] < config['ATR_THRESHOLD']:
            atr_hit_index = None  # ATR never hit threshold

        # --- Find price jump pairs (≥ 0.3) ---
        pairs = find_price_jump_pairs(prices, config['PRICE_JUMP_THRESHOLD'])

        missed, considered = 0, 0
        for (i, j) in pairs:
            if atr_hit_index is None or j < atr_hit_index:
                missed += 1
            else:
                considered += 1

        return {
            'day': day_num,
            'success': True,
            'atr_hit_index': int(atr_hit_index) if atr_hit_index is not None else -1,
            'atr_hit_value': float(atr_values[atr_hit_index]) if atr_hit_index is not None else 0.0,
            'considerable_pairs': considered,
            'missed_pairs': missed,
            'total_pairs': len(pairs),
            'price_min': float(prices.min()),
            'price_max': float(prices.max())
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}

# =====================================================================
# SUMMARY REPORT
# =====================================================================

def save_summary_to_file(all_results, config, output_path='price_jump_segmented_summary_600s.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    total_missed = sum(r['missed_pairs'] for r in valid)
    total_considered = sum(r['considerable_pairs'] for r in valid)
    total_pairs = sum(r['total_pairs'] for r in valid)
    days_with_hit = sum(1 for r in valid if r['atr_hit_index'] >= 0)
    days_without_hit = len(valid) - days_with_hit

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("PRICE FLUCTUATION (≥ 0.3) — SEGMENTED BY ATR(600s) THRESHOLD CROSS (0.01)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Data Directory: {config['DATA_DIR']}\n")
        f.write(f"ATR Window: {config['ATR_WINDOW']} seconds | ATR Threshold: {config['ATR_THRESHOLD']}\n")
        f.write(f"Days Processed: {len(valid)} / {config['NUM_DAYS']}\n\n")

        # Section A: ATR crossed threshold
        f.write("=" * 100 + "\n")
        f.write("SECTION A — Days where ATR(600s) ≥ 0.01 (ATR Hit Detected)\n")
        f.write("=" * 100 + "\n")
        f.write("Day | ATR Hit @Index | ATR Value | Considered | Missed | Total | Price Min | Price Max\n")
        f.write("----|----------------|------------|-------------|--------|--------|-----------|----------\n")
        for r in valid:
            if r['atr_hit_index'] >= 0:
                f.write(f"{r['day']:3d} | {r['atr_hit_index']:14d} | {r['atr_hit_value']:10.5f} | "
                        f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                        f"{r['price_min']:9.2f} | {r['price_max']:9.2f}\n")

        f.write(f"\nDays with ATR Cross: {days_with_hit}\n")
        f.write(f"Total Considered Pairs: {total_considered}\n")
        f.write(f"Average Considered/Day: {total_considered / max(days_with_hit, 1):.2f}\n")
        f.write("\n" + "=" * 100 + "\n\n")

        # Section B: ATR never crossed threshold
        f.write("=" * 100 + "\n")
        f.write("SECTION B — Days where ATR(600s) never reached 0.01 (No Hit)\n")
        f.write("=" * 100 + "\n")
        f.write("Day | ATR Hit @Index | ATR Value | Considered | Missed | Total | Price Min | Price Max\n")
        f.write("----|----------------|------------|-------------|--------|--------|-----------|----------\n")
        for r in valid:
            if r['atr_hit_index'] == -1:
                f.write(f"{r['day']:3d} | {'N/A':>14} | {r['atr_hit_value']:10.5f} | "
                        f"{r['considerable_pairs']:11d} | {r['missed_pairs']:6d} | {r['total_pairs']:6d} | "
                        f"{r['price_min']:9.2f} | {r['price_max']:9.2f}\n")

        f.write(f"\nDays without ATR Cross: {days_without_hit}\n")
        f.write(f"Total Missed Pairs: {total_missed}\n")
        f.write(f"Average Missed/Day: {total_missed / max(days_without_hit, 1):.2f}\n")

        # Overall Summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total Pairs Detected: {total_pairs}\n")
        f.write(f"Total Missed Pairs (before ATR 0.01): {total_missed}\n")
        f.write(f"Total Considered Pairs (after ATR 0.01): {total_considered}\n")
        if len(valid) > 0:
            f.write(f"\nAverage Total Pairs/Day: {total_pairs / len(valid):.2f}\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

    print(f"\n✓ Segmented Summary saved to: {output_path}")

# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("PRICE FLUCTUATION (≥ 0.3) — SEGMENTED BY ATR(600s) THRESHOLD CROSS (0.01)")
    print("=" * 100)
    print(f"Data Directory: {config['DATA_DIR']}")
    print(f"Days to Process: {config['NUM_DAYS']}")
    print(f"ATR Window: {config['ATR_WINDOW']} seconds")
    print(f"ATR Threshold: {config['ATR_THRESHOLD']}")
    print(f"Price Jump Threshold: {config['PRICE_JUMP_THRESHOLD']}")
    print("=" * 100)

    all_results = []
    for day_num in range(config['NUM_DAYS']):
        res = process_single_day(day_num, config)
        if res['success']:
            all_results.append(res)

        if (day_num + 1) % 50 == 0:
            print(f"  Processed {day_num + 1}/{config['NUM_DAYS']} days...")

    print(f"\n✓ Successfully processed {len(all_results)} days.")
    save_summary_to_file(all_results, config)
    print("\n✓ Analysis Complete!")

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    main()
