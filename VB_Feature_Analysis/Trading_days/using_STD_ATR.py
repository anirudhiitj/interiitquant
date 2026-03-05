import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import jit

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBX',
    'NUM_DAYS': 510,
    'PRICE_COLUMN': 'Price',
    'VOLATILITY_FEATURE': 'PB9_T1',
    'TIME_COLUMN': 'Time',
    'PRICE_JUMP_THRESHOLD': 0.3,
    'MIN_TRADE_DURATION': 15,     # seconds
    'TRADE_COOLDOWN': 15,         # seconds
    'ATR_WINDOW': 300,
    'ATR_THRESHOLD': 0.01,        # Max ATR must be > 0.01
    'VOL_WINDOW': 20,
    'VOL_THRESHOLD': 0.08,        # Max STD must be < 0.08
    'MAX_WORKERS': 25,
}

# =====================================================================
# NUMBA UTILITIES
# =====================================================================

@jit(nopython=True, fastmath=True)
def find_price_jump_pairs_time(prices, times, jump_threshold, min_duration, cooldown):
    n = len(prices)
    pairs = []
    i = 0

    while i < n - 1:
        start_price = prices[i]
        start_time = times[i]
        found = False

        for j in range(i + 1, n):
            if abs(prices[j] - start_price) >= jump_threshold:
                trade_duration = times[j] - start_time
                if trade_duration >= min_duration:
                    pairs.append((i, j))
                    cooldown_target = times[j] + cooldown
                    k = j + 1
                    while k < n and times[k] < cooldown_target:
                        k += 1
                    i = k
                    found = True
                    break
        if not found:
            i += 1

    return pairs


# =====================================================================
# PROCESS SINGLE DAY
# =====================================================================

def process_single_day(day_num, config):
    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        df = pd.read_parquet(file_path)
        price_col = config['PRICE_COLUMN']
        vol_col = config['VOLATILITY_FEATURE']
        time_col = config['TIME_COLUMN']

        # Basic validation
        for col in [price_col, vol_col, time_col]:
            if col not in df.columns:
                return {'day': day_num, 'success': False, 'reason': f'missing_{col}'}

        df = df[[price_col, vol_col, time_col]].dropna().copy()

        if len(df) < max(config['ATR_WINDOW'], config['VOL_WINDOW']):
            return {'day': day_num, 'success': False, 'reason': 'insufficient_data'}

        # Convert time to seconds
        df[time_col] = pd.to_timedelta(df[time_col]).dt.total_seconds()

        prices = df[price_col].astype(float).values
        pb9_t1 = df[vol_col].astype(float).values
        times = df[time_col].astype(float).values

        # ============================================================
        # CALCULATE MAX ATR
        # ============================================================
        df['atr_proxy'] = (
            pd.Series(pb9_t1)
            .diff()
            .abs()
            .rolling(config['ATR_WINDOW'], min_periods=1)
            .mean()
            .fillna(0)
        )
        max_atr = df['atr_proxy'].max()

        # ============================================================
        # CALCULATE MAX STD
        # ============================================================
        df['std_proxy'] = (
            pd.Series(pb9_t1)
            .rolling(config['VOL_WINDOW'], min_periods=1)
            .std()
            .fillna(0)
        )
        max_std = df['std_proxy'].max()

        # ============================================================
        # CHECK DAY ELIGIBILITY
        # ============================================================
        # Ignore day if ATR <= threshold OR STD >= threshold
        if max_atr <= config['ATR_THRESHOLD']:
            return {
                'day': day_num,
                'success': False,
                'reason': 'atr_below_threshold',
                'max_atr': float(max_atr),
                'max_std': float(max_std)
            }

        if max_std >= config['VOL_THRESHOLD']:
            return {
                'day': day_num,
                'success': False,
                'reason': 'std_above_threshold',
                'max_atr': float(max_atr),
                'max_std': float(max_std)
            }

        # ============================================================
        # TIME-AWARE PRICE JUMPS (≥ 0.3 fluctuation)
        # ============================================================
        pairs = find_price_jump_pairs_time(
            prices,
            times,
            config['PRICE_JUMP_THRESHOLD'],
            config['MIN_TRADE_DURATION'],
            config['TRADE_COOLDOWN']
        )

        # All pairs are considered since day passed filters
        considered_pairs = len(pairs)

        return {
            'day': day_num,
            'success': True,
            'max_atr': float(max_atr),
            'max_std': float(max_std),
            'considered_pairs': considered_pairs,
            'total_pairs': len(pairs),
            'price_min': float(prices.min()),
            'price_max': float(prices.max()),
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}


# =====================================================================
# SUMMARY REPORT
# =====================================================================

def save_summary_to_file(all_results, config, output_path='STD_ATR.txt'):
    valid = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    total_considered = sum(r['considered_pairs'] for r in valid)
    total_pairs = sum(r['total_pairs'] for r in valid)
    
    # Separate failed reasons
    atr_failed = [r for r in failed if r.get('reason') == 'atr_below_threshold']
    std_failed = [r for r in failed if r.get('reason') == 'std_above_threshold']

    with open(output_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("PRICE FLUCTUATION (≥ 0.3) — ATR & STD FILTERED — TIME-BASED\n")
        f.write("=" * 120 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  ATR Window: {config['ATR_WINDOW']} | ATR Threshold: > {config['ATR_THRESHOLD']}\n")
        f.write(f"  STD Window: {config['VOL_WINDOW']} | STD Threshold: < {config['VOL_THRESHOLD']}\n")
        f.write(f"  Min Trade Duration: {config['MIN_TRADE_DURATION']} s | Cooldown: {config['TRADE_COOLDOWN']} s\n")
        f.write(f"  Days Processed: {config['NUM_DAYS']}\n\n")

        f.write("=" * 120 + "\n")
        f.write(f"QUALIFYING DAYS: Max ATR > {config['ATR_THRESHOLD']} AND Max STD < {config['VOL_THRESHOLD']}\n")
        f.write("=" * 120 + "\n")
        f.write("Day | Max ATR | Max STD | Considered Pairs | Total Pairs | Price Min | Price Max\n")
        f.write("----|---------|---------|------------------|-------------|-----------|----------\n")

        for r in sorted(valid, key=lambda x: x['day']):
            f.write(
                f"{r['day']:3d} | "
                f"{r['max_atr']:7.5f} | "
                f"{r['max_std']:7.5f} | "
                f"{r['considered_pairs']:16d} | "
                f"{r['total_pairs']:11d} | "
                f"{r['price_min']:9.2f} | "
                f"{r['price_max']:9.2f}\n"
            )

        f.write("\n" + "-" * 120 + "\n")
        f.write(f"Total Qualifying Days: {len(valid)}\n")
        f.write(f"Total Considered Pairs (0.3+ fluctuation): {total_considered}\n")
        f.write(f"Total Pairs: {total_pairs}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("REJECTED DAYS\n")
        f.write("=" * 120 + "\n\n")

        f.write(f"Days Rejected due to ATR ≤ {config['ATR_THRESHOLD']}: {len(atr_failed)}\n")
        if len(atr_failed) > 0:
            f.write("Day | Max ATR | Max STD\n")
            f.write("----|---------|--------\n")
            for r in sorted(atr_failed, key=lambda x: x['day'])[:20]:  # Show first 20
                f.write(f"{r['day']:3d} | {r.get('max_atr', 0):7.5f} | {r.get('max_std', 0):7.5f}\n")
            if len(atr_failed) > 20:
                f.write(f"... and {len(atr_failed) - 20} more\n")

        f.write(f"\nDays Rejected due to STD ≥ {config['VOL_THRESHOLD']}: {len(std_failed)}\n")
        if len(std_failed) > 0:
            f.write("Day | Max ATR | Max STD\n")
            f.write("----|---------|--------\n")
            for r in sorted(std_failed, key=lambda x: x['day'])[:20]:  # Show first 20
                f.write(f"{r['day']:3d} | {r.get('max_atr', 0):7.5f} | {r.get('max_std', 0):7.5f}\n")
            if len(std_failed) > 20:
                f.write(f"... and {len(std_failed) - 20} more\n")

        f.write("\n" + "=" * 120 + "\n")
        f.write("END OF REPORT\n")

    print(f"\n✓ Summary saved to: {output_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    config = CONFIG
    print("=" * 120)
    print("TIME-AWARE PRICE FLUCTUATION DETECTION — ATR > 0.01 + STD < 0.08")
    print("=" * 120)
    print(f"\nFilters:")
    print(f"  - Max ATR must be > {config['ATR_THRESHOLD']}")
    print(f"  - Max STD must be < {config['VOL_THRESHOLD']}")
    print(f"  - Price fluctuation must be ≥ {config['PRICE_JUMP_THRESHOLD']}")
    print(f"  - Min trade duration: {config['MIN_TRADE_DURATION']}s, Cooldown: {config['TRADE_COOLDOWN']}s\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
        futures = {
            executor.submit(process_single_day, day, config): day
            for day in range(config['NUM_DAYS'])
        }

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                print(f"Error on day {futures[future]}: {e}")

            if i % 25 == 0:
                print(f"  Processed {i}/{config['NUM_DAYS']} days...")

    valid_days = [r for r in all_results if r.get('success', False)]
    print(f"\n✓ Successfully processed {len(all_results)} days.")
    print(f"✓ Qualifying days (ATR > {config['ATR_THRESHOLD']} AND STD < {config['VOL_THRESHOLD']}): {len(valid_days)}")
    
    save_summary_to_file(all_results, config)
    print("\n✓ ATR + STD Filtered Trade Analysis Complete!")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()