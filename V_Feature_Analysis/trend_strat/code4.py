import os
import glob
import re
import pandas as pd
import numpy as np
import dask_cudf
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def analyze_single_day_quick(file_path: str, day_num: int, 
                             time_col: str = 'Time', 
                             price_col: str = 'Price',
                             decision_window: int = 1800) -> dict:
    """Quick analysis for sigma only"""
    try:
        required_cols = [time_col, price_col]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            return None
        
        df = df.reset_index(drop=True)
        df['Time_sec'] = pd.to_timedelta(df[time_col].astype(str)).dt.total_seconds().astype(int)
        
        first_30_data = df[df['Time_sec'] <= decision_window]
        
        if len(first_30_data) < 10:
            return None
        
        first_30_sigma = first_30_data[price_col].std()
        full_day_sigma = df[price_col].std()
        
        result = {
            'Day': day_num,
            'First_30min_Sigma': round(first_30_sigma, 6),
            'Full_Day_Sigma': round(full_day_sigma, 6),
        }
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing day {day_num}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Diagnose sigma percentile changes")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    print("="*100)
    print("SIGMA PERCENTILE DIAGNOSTIC TOOL")
    print("="*100)
    print(f"\nData directory: {args.data_dir}")
    
    # Days you excluded
    excluded_days = {311, 421, 446, 87, 273}
    
    # Find all files
    files_pattern = os.path.join(args.data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    # Filter out 600-790 range for both analyses
    filtered_files = [
        f for f in sorted_files 
        if not (600 <= extract_day_num(f) <= 790)
    ]
    
    if not filtered_files:
        print(f"\nERROR: No parquet files found")
        return
    
    print(f"\nTotal files found: {len(sorted_files)}")
    print(f"After removing 600-790 range: {len(filtered_files)}")
    print(f"\nProcessing with {args.max_workers} workers...\n")
    
    # Process all days
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(analyze_single_day_quick, f, extract_day_num(f)): f
            for f in filtered_files
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"✗ Error: {e}")
    
    if not results:
        print("\nERROR: No files processed successfully")
        return
    
    # Create DataFrame
    df_all = pd.DataFrame(results)
    df_all = df_all.sort_values('Day').reset_index(drop=True)
    
    # Analysis WITH excluded days (before exclusion)
    print(f"\n{'='*100}")
    print("ANALYSIS #1: ALL DAYS (Before Exclusion)")
    print(f"{'='*100}")
    print(f"Total days: {len(df_all)}")
    
    stats_before = {
        'mean': df_all['First_30min_Sigma'].mean(),
        'median': df_all['First_30min_Sigma'].median(),
        'std': df_all['First_30min_Sigma'].std(),
        'min': df_all['First_30min_Sigma'].min(),
        'max': df_all['First_30min_Sigma'].max(),
        'q25': df_all['First_30min_Sigma'].quantile(0.25),
        'q50': df_all['First_30min_Sigma'].quantile(0.50),
        'q75': df_all['First_30min_Sigma'].quantile(0.75),
        'q90': df_all['First_30min_Sigma'].quantile(0.90),
    }
    
    print(f"\nFirst 30min Sigma Statistics:")
    print(f"  Mean:              {stats_before['mean']:.6f}")
    print(f"  Median:            {stats_before['median']:.6f}")
    print(f"  Std Dev:           {stats_before['std']:.6f}")
    print(f"  Min:               {stats_before['min']:.6f}")
    print(f"  Max:               {stats_before['max']:.6f}")
    print(f"  25th Percentile:   {stats_before['q25']:.6f}")
    print(f"  50th Percentile:   {stats_before['q50']:.6f}")
    print(f"  75th Percentile:   {stats_before['q75']:.6f}")
    print(f"  90th Percentile:   {stats_before['q90']:.6f}")
    
    # Check if excluded days exist in dataset
    print(f"\n{'='*100}")
    print("EXCLUDED DAYS ANALYSIS")
    print(f"{'='*100}")
    print(f"Days to exclude: {sorted(excluded_days)}")
    
    excluded_data = df_all[df_all['Day'].isin(excluded_days)]
    
    if len(excluded_data) > 0:
        print(f"\nFound {len(excluded_data)} excluded days in dataset:")
        print(f"\n{'Day':>5} | {'30min Sigma':>15} | {'Full Day Sigma':>15}")
        print(f"{'-'*42}")
        for _, row in excluded_data.iterrows():
            print(f"{int(row['Day']):5d} | {row['First_30min_Sigma']:15.6f} | {row['Full_Day_Sigma']:15.6f}")
        
        print(f"\nExcluded Days Statistics:")
        print(f"  Mean Sigma:        {excluded_data['First_30min_Sigma'].mean():.6f}")
        print(f"  Median Sigma:      {excluded_data['First_30min_Sigma'].median():.6f}")
        print(f"  Min Sigma:         {excluded_data['First_30min_Sigma'].min():.6f}")
        print(f"  Max Sigma:         {excluded_data['First_30min_Sigma'].max():.6f}")
    else:
        print(f"\n⚠ WARNING: None of the excluded days were found in the dataset!")
        print(f"  This means the exclusion had NO EFFECT.")
    
    # Analysis AFTER excluding days
    df_filtered = df_all[~df_all['Day'].isin(excluded_days)]
    
    print(f"\n{'='*100}")
    print("ANALYSIS #2: AFTER EXCLUSION")
    print(f"{'='*100}")
    print(f"Total days: {len(df_filtered)} (removed {len(df_all) - len(df_filtered)} days)")
    
    stats_after = {
        'mean': df_filtered['First_30min_Sigma'].mean(),
        'median': df_filtered['First_30min_Sigma'].median(),
        'std': df_filtered['First_30min_Sigma'].std(),
        'min': df_filtered['First_30min_Sigma'].min(),
        'max': df_filtered['First_30min_Sigma'].max(),
        'q25': df_filtered['First_30min_Sigma'].quantile(0.25),
        'q50': df_filtered['First_30min_Sigma'].quantile(0.50),
        'q75': df_filtered['First_30min_Sigma'].quantile(0.75),
        'q90': df_filtered['First_30min_Sigma'].quantile(0.90),
    }
    
    print(f"\nFirst 30min Sigma Statistics:")
    print(f"  Mean:              {stats_after['mean']:.6f}")
    print(f"  Median:            {stats_after['median']:.6f}")
    print(f"  Std Dev:           {stats_after['std']:.6f}")
    print(f"  Min:               {stats_after['min']:.6f}")
    print(f"  Max:               {stats_after['max']:.6f}")
    print(f"  25th Percentile:   {stats_after['q25']:.6f}")
    print(f"  50th Percentile:   {stats_after['q50']:.6f}")
    print(f"  75th Percentile:   {stats_after['q75']:.6f}")
    print(f"  90th Percentile:   {stats_after['q90']:.6f}")
    
    # Compare
    print(f"\n{'='*100}")
    print("COMPARISON: BEFORE vs AFTER EXCLUSION")
    print(f"{'='*100}")
    
    print(f"\n{'Metric':<20} | {'Before':>15} | {'After':>15} | {'Change':>15} | {'% Change':>10}")
    print(f"{'-'*85}")
    
    for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'q90']:
        before = stats_before[key]
        after = stats_after[key]
        change = after - before
        pct_change = (change / before * 100) if before != 0 else 0
        
        label_map = {
            'mean': 'Mean',
            'median': 'Median',
            'std': 'Std Dev',
            'min': 'Min',
            'max': 'Max',
            'q25': '25th Percentile',
            'q50': '50th Percentile',
            'q75': '75th Percentile',
            'q90': '90th Percentile',
        }
        
        print(f"{label_map[key]:<20} | {before:15.6f} | {after:15.6f} | {change:+15.6f} | {pct_change:+9.2f}%")
    
    # Find days near the 75th percentile boundary
    print(f"\n{'='*100}")
    print("DAYS NEAR 75TH PERCENTILE BOUNDARIES")
    print(f"{'='*100}")
    
    before_75th = stats_before['q75']
    after_75th = stats_after['q75']
    
    # Days between the old and new 75th percentile
    boundary_days = df_all[
        (df_all['First_30min_Sigma'] >= min(before_75th, after_75th) - 0.01) &
        (df_all['First_30min_Sigma'] <= max(before_75th, after_75th) + 0.01)
    ].sort_values('First_30min_Sigma')
    
    print(f"\nDays with sigma between {min(before_75th, after_75th) - 0.01:.6f} and {max(before_75th, after_75th) + 0.01:.6f}:")
    print(f"(This shows which days are affecting the percentile shift)")
    print(f"\n{'Day':>5} | {'30min Sigma':>15} | {'Excluded?':>10}")
    print(f"{'-'*42}")
    
    for _, row in boundary_days.iterrows():
        excluded = 'YES' if row['Day'] in excluded_days else 'NO'
        print(f"{int(row['Day']):5d} | {row['First_30min_Sigma']:15.6f} | {excluded:>10}")
    
    # Distribution analysis
    print(f"\n{'='*100}")
    print("DISTRIBUTION ANALYSIS")
    print(f"{'='*100}")
    
    # Count days in different sigma ranges
    ranges = [
        (0, 0.05, '0.00-0.05'),
        (0.05, 0.10, '0.05-0.10'),
        (0.10, 0.15, '0.10-0.15'),
        (0.15, 0.20, '0.15-0.20'),
        (0.20, 0.30, '0.20-0.30'),
        (0.30, 1.00, '0.30+'),
    ]
    
    print(f"\n{'Sigma Range':<15} | {'Before':>10} | {'After':>10} | {'Excluded':>10}")
    print(f"{'-'*55}")
    
    for low, high, label in ranges:
        before_count = ((df_all['First_30min_Sigma'] >= low) & (df_all['First_30min_Sigma'] < high)).sum()
        after_count = ((df_filtered['First_30min_Sigma'] >= low) & (df_filtered['First_30min_Sigma'] < high)).sum()
        excluded_count = before_count - after_count
        print(f"{label:<15} | {before_count:10d} | {after_count:10d} | {excluded_count:10d}")
    
    print(f"\n{'='*100}")
    print("✓ Diagnostic complete!")
    print(f"{'='*100}\n")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  • 75th percentile changed from {before_75th:.6f} to {after_75th:.6f}")
    print(f"  • Change: {after_75th - before_75th:+.6f} ({(after_75th - before_75th)/before_75th*100:+.2f}%)")
    print(f"  • Days excluded: {len(excluded_data)} out of {len(df_all)}")
    
    if len(excluded_data) > 0:
        avg_excluded_sigma = excluded_data['First_30min_Sigma'].mean()
        print(f"  • Average sigma of excluded days: {avg_excluded_sigma:.6f}")
        print(f"  • This is {'BELOW' if avg_excluded_sigma < before_75th else 'ABOVE'} the original 75th percentile")


if __name__ == "__main__":
    main()