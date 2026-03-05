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


def analyze_time_windows(file_path: str, day_num: int, 
                         time_windows: list = [2, 5, 10, 15, 20, 30],
                         time_col: str = 'Time', 
                         price_col: str = 'Price') -> list:
    """Analyze mu and sigma for multiple time windows from start of day"""
    try:
        # Read the parquet file
        required_cols = [time_col, price_col]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return []
        
        df = df.reset_index(drop=True)
        
        # Convert time to seconds
        df['Time_sec'] = pd.to_timedelta(df[time_col].astype(str)).dt.total_seconds().astype(int)
        
        if len(df) < 10:
            print(f"Warning: Day {day_num} insufficient data. Skipping.")
            return []
        
        # Get start time
        start_time = df['Time_sec'].iloc[0]
        
        results = []
        
        # Calculate stats for each time window
        for window_min in time_windows:
            cutoff_time = start_time + (window_min * 60)  # Convert minutes to seconds
            df_window = df[df['Time_sec'] <= cutoff_time].copy()
            
            if len(df_window) < 2:
                print(f"Warning: Day {day_num}, window {window_min}min insufficient data.")
                continue
            
            prices = df_window[price_col].values
            mu = np.mean(prices)
            sigma = np.std(prices, ddof=1)  # Sample std
            
            results.append({
                'Day': day_num,
                'Window_Minutes': window_min,
                'Mu': mu,
                'Sigma': sigma,
                'Data_Points': len(df_window)
            })
        
        if results:
            print(f"✓ Day{day_num:3d} | Windows processed: {len(results)}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_percentile_stats(df, percentiles=[5, 10, 50, 75, 90, 99]):
    """Calculate mean and percentiles for mu and sigma by time window"""
    summary = []
    
    for window_min in sorted(df['Window_Minutes'].unique()):
        window_data = df[df['Window_Minutes'] == window_min]
        
        mu_values = window_data['Mu'].dropna().values
        sigma_values = window_data['Sigma'].dropna().values
        
        if len(mu_values) == 0 or len(sigma_values) == 0:
            continue
        
        row = {
            'Window_Minutes': window_min,
            'Mu_Mean': np.mean(mu_values),
            'Sigma_Mean': np.mean(sigma_values),
        }
        
        # Add percentiles for Mu
        for p in percentiles:
            row[f'Mu_P{p}'] = np.percentile(mu_values, p)
        
        # Add percentiles for Sigma
        for p in percentiles:
            row[f'Sigma_P{p}'] = np.percentile(sigma_values, p)
        
        row['Days_Count'] = len(window_data)
        
        summary.append(row)
    
    return pd.DataFrame(summary)


def print_summary_table(summary_df, title):
    """Print formatted summary table"""
    
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}")
    
    # Print Mu statistics
    print(f"\n{'MU (Mean Price) STATISTICS':^120}")
    print("-" * 120)
    header = f"{'Window':<10} {'Mean':<12} {'P5':<12} {'P10':<12} {'P50':<12} {'P75':<12} {'P90':<12} {'P99':<12}"
    print(header)
    print("-" * 120)
    
    for _, row in summary_df.iterrows():
        window = f"{int(row['Window_Minutes'])} min"
        print(f"{window:<10} {row['Mu_Mean']:<12.4f} {row['Mu_P5']:<12.4f} {row['Mu_P10']:<12.4f} "
              f"{row['Mu_P50']:<12.4f} {row['Mu_P75']:<12.4f} {row['Mu_P90']:<12.4f} {row['Mu_P99']:<12.4f}")
    
    # Print Sigma statistics
    print(f"\n{'SIGMA (Std Dev) STATISTICS':^120}")
    print("-" * 120)
    header = f"{'Window':<10} {'Mean':<12} {'P5':<12} {'P10':<12} {'P50':<12} {'P75':<12} {'P90':<12} {'P99':<12}"
    print(header)
    print("-" * 120)
    
    for _, row in summary_df.iterrows():
        window = f"{int(row['Window_Minutes'])} min"
        print(f"{window:<10} {row['Sigma_Mean']:<12.4f} {row['Sigma_P5']:<12.4f} {row['Sigma_P10']:<12.4f} "
              f"{row['Sigma_P50']:<12.4f} {row['Sigma_P75']:<12.4f} {row['Sigma_P90']:<12.4f} {row['Sigma_P99']:<12.4f}")
    
    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze mu and sigma over time windows from raw data")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/raid/Quant14/V_Feature_Analysis/trend_strat/ebx',
        help="Output directory for CSV files"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help="Number of parallel workers"
    )
    parser.add_argument(
        '--time-windows',
        type=int,
        nargs='+',
        default=[2, 5, 10, 15, 20, 30],
        help="Time windows in minutes (default: 2 5 10 15 20 30)"
    )
    parser.add_argument(
        '--target-range-start',
        type=int,
        default=106,
        help="Start day of target range (default: 106)"
    )
    parser.add_argument(
        '--target-range-end',
        type=int,
        default=146,
        help="End day of target range (default: 146)"
    )
    args = parser.parse_args()
    
    target_range = list(range(args.target_range_start, args.target_range_end + 1))
    
    print("="*120)
    print("MU AND SIGMA ANALYSIS - TIME WINDOWS FROM DAY START")
    print("="*120)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max workers: {args.max_workers}")
    print(f"Time windows: {args.time_windows} minutes")
    print(f"Target day range: {args.target_range_start} to {args.target_range_end} ({len(target_range)} days)")
    print(f"Percentiles: [5, 10, 50, 75, 90, 99]")
    
    # Find input files
    files_pattern = os.path.join(args.data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {args.data_dir}")
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    print(f"\nProcessing with {args.max_workers} workers...\n")
    
    # Process all days in parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                analyze_time_windows, 
                f, 
                extract_day_num(f),
                args.time_windows
            ): f
            for f in sorted_files
        }
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"✗ Error in {file_path}: {e}")
    
    if not all_results:
        print("\nERROR: No files processed successfully")
        return
    
    # Create DataFrame
    df_all = pd.DataFrame(all_results)
    df_all = df_all.sort_values(['Day', 'Window_Minutes']).reset_index(drop=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed stats for all days
    detailed_csv = os.path.join(args.output_dir, 'detailed_stats_all_days.csv')
    df_all.to_csv(detailed_csv, index=False)
    print(f"\n✓ Saved detailed stats: {detailed_csv}")
    
    # ========================================================================
    # ANALYSIS 1: ALL DAYS
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("CALCULATING STATISTICS FOR ALL DAYS")
    print(f"{'='*120}")
    
    percentiles = [5, 10, 50, 75, 90, 99]
    all_days_summary = calculate_percentile_stats(df_all, percentiles)
    
    # Print ALL days summary
    print_summary_table(all_days_summary, "ALL DAYS STATISTICS")
    
    # Save ALL days summary
    all_days_csv = os.path.join(args.output_dir, 'summary_all_days.csv')
    all_days_summary.to_csv(all_days_csv, index=False)
    print(f"✓ Saved all days summary: {all_days_csv}")
    
    # ========================================================================
    # ANALYSIS 2: TARGET DAY RANGE (106-146)
    # ========================================================================
    
    print(f"\n{'='*120}")
    print(f"CALCULATING STATISTICS FOR TARGET RANGE: Days {args.target_range_start}-{args.target_range_end}")
    print(f"{'='*120}")
    
    df_target_range = df_all[df_all['Day'].isin(target_range)]
    
    if len(df_target_range) == 0:
        print("\nWARNING: No data found for target day range")
    else:
        target_range_summary = calculate_percentile_stats(df_target_range, percentiles)
        
        # Print TARGET RANGE summary
        print_summary_table(target_range_summary, f"TARGET RANGE (Days {args.target_range_start}-{args.target_range_end}) STATISTICS")
        
        # Save TARGET RANGE summary
        target_range_csv = os.path.join(args.output_dir, f'summary_target_range_{args.target_range_start}_{args.target_range_end}.csv')
        target_range_summary.to_csv(target_range_csv, index=False)
        print(f"✓ Saved target range summary: {target_range_csv}")
        
        # Save detailed stats for target range
        detailed_target_csv = os.path.join(args.output_dir, f'detailed_stats_target_range_{args.target_range_start}_{args.target_range_end}.csv')
        df_target_range.to_csv(detailed_target_csv, index=False)
        print(f"✓ Saved detailed target range stats: {detailed_target_csv}")
        
        # ========================================================================
        # COMPARISON TABLE
        # ========================================================================
        
        print(f"\n{'='*120}")
        print(f"COMPARISON: ALL DAYS vs TARGET RANGE (Days {args.target_range_start}-{args.target_range_end})")
        print(f"{'='*120}")
        
        print(f"\n{'SIGMA MEAN COMPARISON':^120}")
        print("-" * 120)
        print(f"{'Window':<12} {'All Days':<15} {'Target Range':<15} {'Difference':<15} {'% Change':<15}")
        print("-" * 120)
        
        for _, all_row in all_days_summary.iterrows():
            window_min = int(all_row['Window_Minutes'])
            target_row = target_range_summary[target_range_summary['Window_Minutes'] == window_min]
            
            if len(target_row) > 0:
                all_sigma = all_row['Sigma_Mean']
                target_sigma = target_row.iloc[0]['Sigma_Mean']
                diff = target_sigma - all_sigma
                pct_change = (diff / all_sigma * 100) if all_sigma != 0 else 0
                
                window = f"{window_min} min"
                print(f"{window:<12} {all_sigma:<15.6f} {target_sigma:<15.6f} {diff:<15.6f} {pct_change:<15.2f}%")
        
        print(f"\n{'MU MEAN COMPARISON':^120}")
        print("-" * 120)
        print(f"{'Window':<12} {'All Days':<15} {'Target Range':<15} {'Difference':<15} {'% Change':<15}")
        print("-" * 120)
        
        for _, all_row in all_days_summary.iterrows():
            window_min = int(all_row['Window_Minutes'])
            target_row = target_range_summary[target_range_summary['Window_Minutes'] == window_min]
            
            if len(target_row) > 0:
                all_mu = all_row['Mu_Mean']
                target_mu = target_row.iloc[0]['Mu_Mean']
                diff = target_mu - all_mu
                pct_change = (diff / all_mu * 100) if all_mu != 0 else 0
                
                window = f"{window_min} min"
                print(f"{window:<12} {all_mu:<15.4f} {target_mu:<15.4f} {diff:<15.4f} {pct_change:<15.2f}%")
        
        print(f"{'='*120}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*120}")
    print(f"\nTotal days processed: {len(df_all['Day'].unique())}")
    print(f"Target range days processed: {len(df_target_range['Day'].unique()) if len(df_target_range) > 0 else 0}")
    print(f"Total data points: {len(df_all)}")
    print(f"\nOutput files saved to: {args.output_dir}")
    print(f"  1. detailed_stats_all_days.csv - All days, all windows")
    print(f"  2. summary_all_days.csv - Percentile summary for all days")
    if len(df_target_range) > 0:
        print(f"  3. summary_target_range_{args.target_range_start}_{args.target_range_end}.csv - Percentile summary for target range")
        print(f"  4. detailed_stats_target_range_{args.target_range_start}_{args.target_range_end}.csv - Detailed stats for target range")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()