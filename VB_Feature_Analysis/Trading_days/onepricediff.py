import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'DATA_DIR': '/data/quant14/EBY',
    'NUM_DAYS': 279,
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',          # must exist in parquet
    'WINDOW_SECONDS': 1200,          # Rolling window size
    'SPREAD_THRESHOLD': 1.0,        # Max-Min difference threshold
    'MAX_WORKERS': 25,              # number of CPU workers
    
    # Days to exclude from analysis
    'BLACKLIST_DAYS': [6, 104, 110, 115, 135, 165, 209]
}

# =====================================================================
# PROCESS SINGLE DAY
# =====================================================================

def process_single_day(day_num, config):
    # --- Check Blacklist First ---
    if day_num in config['BLACKLIST_DAYS']:
        return {'day': day_num, 'success': False, 'reason': 'blacklisted'}

    try:
        file_path = Path(config['DATA_DIR']) / f"day{day_num}.parquet"
        if not file_path.exists():
            return {'day': day_num, 'success': False, 'reason': 'file_missing'}

        # Read only necessary columns
        columns = [config['PRICE_COLUMN'], config['TIME_COLUMN']]
        df = pd.read_parquet(file_path, columns=columns)
        
        # Validation
        if df.empty:
            return {'day': day_num, 'success': False, 'reason': 'empty_data'}
        
        # --- Time Conversion for Rolling Window ---
        # Convert Time column to Timedelta or Datetime
        df[config['TIME_COLUMN']] = pd.to_timedelta(df[config['TIME_COLUMN']])
        
        # Set Time as index and sort
        df = df.set_index(config['TIME_COLUMN']).sort_index()

        # --- Vectorized Rolling Calculation ---
        # Calculate Max - Min for every point looking back 300 seconds
        indexer = df[config['PRICE_COLUMN']].rolling(f"{config['WINDOW_SECONDS']}s")
        
        # Calculate spread directly
        rolling_spread = indexer.max() - indexer.min()

        # --- Extract Statistics ---
        max_spread_val = rolling_spread.max()
        
        if pd.isna(max_spread_val):
            max_spread_val = 0.0

        return {
            'day': day_num,
            'success': True,
            'max_spread': float(max_spread_val),
            'threshold_crossed': bool(max_spread_val >= config['SPREAD_THRESHOLD']),
            'price_min': float(df[config['PRICE_COLUMN']].min()),
            'price_max': float(df[config['PRICE_COLUMN']].max()),
            'data_points': len(df)
        }

    except Exception as e:
        return {'day': day_num, 'success': False, 'error': str(e)}

# =====================================================================
# SUMMARY REPORT
# =====================================================================

def save_summary_to_file(all_results, config, output_path='Rolling_Spread_Report.txt'):
    # Separate successful, blacklisted, and failed runs
    valid = [r for r in all_results if r.get('success', False)]
    blacklisted = [r for r in all_results if r.get('reason') == 'blacklisted']
    
    # Sort by Day
    valid.sort(key=lambda x: x['day'])
    
    days_crossed = sum(1 for r in valid if r['threshold_crossed'])
    total_valid_days = len(valid)

    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ROLLING WINDOW VOLATILITY REPORT (Window: {config['WINDOW_SECONDS']}s)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Config:\n")
        f.write(f"  - Goal: Spread (Max-Min) >= {config['SPREAD_THRESHOLD']}\n")
        f.write(f"  - Blacklisted Days: {sorted(config['BLACKLIST_DAYS'])}\n\n")
        
        f.write(f"Summary:\n")
        f.write(f"  - Analyzed Days: {total_valid_days}\n")
        f.write(f"  - Skipped (Blacklist): {len(blacklisted)}\n")
        f.write(f"  - Days Meeting Criteria: {days_crossed} ({(days_crossed/total_valid_days)*100:.1f}% of analyzed)\n\n")

        # Table Header
        f.write(f"{'Day':<5} | {'Max Spread (300s)':<20} | {'Status':<10} | {'Day Min':<10} | {'Day Max':<10} | {'Points':<8}\n")
        f.write("-" * 80 + "\n")

        # Rows
        for r in valid:
            status = "[HIT]" if r['threshold_crossed'] else " ."
            f.write(f"{r['day']:<5} | {r['max_spread']:<20.4f} | {status:<10} | {r['price_min']:<10.2f} | {r['price_max']:<10.2f} | {r['data_points']:<8}\n")

        # High Level Stats
        f.write("\n" + "=" * 100 + "\n")
        f.write("STATISTICS\n")
        if valid:
            avg_spread = np.mean([r['max_spread'] for r in valid])
            global_max = max([r['max_spread'] for r in valid])
            best_day = max(valid, key=lambda x: x['max_spread'])['day']
            f.write(f"Average Max Spread across all days: {avg_spread:.4f}\n")
            f.write(f"Highest Spread found in any single window: {global_max:.4f} (Day {best_day})\n")
        f.write("=" * 100 + "\n")

    print(f"\n✓ Report saved to: {output_path}")

# =====================================================================
# MAIN
# =====================================================================

def main():
    config = CONFIG
    print("=" * 100)
    print("ROLLING WINDOW ANALYSIS (Pandas Time-Aware)")
    print("=" * 100)
    print(f"Data Directory: {config['DATA_DIR']}")
    print(f"Blacklisted Days: {config['BLACKLIST_DAYS']}")
    print("=" * 100)

    all_results = []

    with ProcessPoolExecutor(max_workers=config['MAX_WORKERS']) as executor:
        futures = {executor.submit(process_single_day, day, config): day for day in range(config['NUM_DAYS'])}
        
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                # Capture unhandled errors
                all_results.append({'day': futures[future], 'success': False, 'error': str(e)})
            
            if i % 25 == 0:
                print(f"  Processed {i}/{config['NUM_DAYS']} days...")

    print(f"\n✓ Completed {len(all_results)} days.")
    save_summary_to_file(all_results, config)
    print("\n✓ Analysis Complete!")

if __name__ == "__main__":
    main()