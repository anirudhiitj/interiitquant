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


def analyze_single_day_with_30min(file_path: str, day_num: int, 
                                   time_col: str = 'Time', 
                                   price_col: str = 'Price') -> dict:
    """Analyze full day with 30-minute window analysis"""
    try:
        # Read the parquet file
        required_cols = [time_col, price_col]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty:
            print(f"Warning: Day {day_num} file is empty. Skipping.")
            return None
        
        df = df.reset_index(drop=True)
        
        # Convert time to seconds
        df['Time_sec'] = pd.to_timedelta(df[time_col].astype(str)).dt.total_seconds().astype(int)
        
        if len(df) < 10:
            print(f"Warning: Day {day_num} insufficient data. Skipping.")
            return None
        
        # Get start time
        start_time = df['Time_sec'].iloc[0]
        cutoff_time = start_time + 1800  # 30 minutes = 1800 seconds
        
        # Split into first 30 minutes and rest of day
        df_30min = df[df['Time_sec'] <= cutoff_time].copy()
        df_rest = df[df['Time_sec'] > cutoff_time].copy()
        
        if len(df_30min) < 2:
            print(f"Warning: Day {day_num} insufficient 30-min data. Skipping.")
            return None
        
        # Calculate 30-minute statistics
        mean_30min = df_30min[price_col].mean()
        sigma_30min = df_30min[price_col].std()
        price_at_30min = df_30min[price_col].iloc[-1]
        
        # Use FULL DAY high and low (not just 30-min window)
        full_day_high = df[price_col].max()
        full_day_low = df[price_col].min()
        
        high_minus_price30 = full_day_high - price_at_30min
        price30_minus_low = price_at_30min - full_day_low
        peak_to_peak_30min = high_minus_price30 + price30_minus_low
        
        # Calculate % area above/below 100 in first 30 min
        pct_area_above_100_30min = (df_30min[price_col] > 100).sum() / len(df_30min) * 100
        pct_area_below_100_30min = (df_30min[price_col] < 100).sum() / len(df_30min) * 100
        
        # Calculate full-day statistics
        full_day_mean = df[price_col].mean()
        full_day_sigma = df[price_col].std()
        opening_price = df.iloc[0][price_col]
        closing_price = df.iloc[-1][price_col]
        
        # Rest of day statistics
        if len(df_rest) > 0:
            pct_rest_above_100 = (df_rest[price_col] > 100).sum() / len(df_rest) * 100
            pct_rest_below_100 = (df_rest[price_col] < 100).sum() / len(df_rest) * 100
        else:
            pct_rest_above_100 = 0
            pct_rest_below_100 = 0
        
        # Check if price never crosses 100 in first 30 min
        never_crosses_100_30min = not ((df_30min[price_col] > 100).any() and (df_30min[price_col] < 100).any())
        
        # Check if >95% of 30min area is above/below 100
        area_above_100_gt_95 = pct_area_above_100_30min > 95
        area_below_100_gt_95 = pct_area_below_100_30min > 95
        
        result = {
            'Day': day_num,
            'Full_Day_Mean': round(full_day_mean, 4),
            'Full_Day_Sigma': round(full_day_sigma, 4),
            'Mean_30min': round(mean_30min, 4),
            'Sigma_30min': round(sigma_30min, 4),
            'Price_at_30min': round(price_at_30min, 4),
            'Full_Day_High': round(full_day_high, 4),
            'Full_Day_Low': round(full_day_low, 4),
            'High_minus_Price30': round(high_minus_price30, 4),
            'Price30_minus_Low': round(price30_minus_low, 4),
            'Peak_to_Peak_30min': round(peak_to_peak_30min, 4),
            'Pct_Area_Above_100_30min': round(pct_area_above_100_30min, 2),
            'Pct_Area_Below_100_30min': round(pct_area_below_100_30min, 2),
            'Pct_Rest_Above_100': round(pct_rest_above_100, 2),
            'Pct_Rest_Below_100': round(pct_rest_below_100, 2),
            'Opening_Price': round(opening_price, 4),
            'Closing_Price': round(closing_price, 4),
            'Close_Above_100': closing_price > 100,
            'Never_Crosses_100_30min': never_crosses_100_30min,
            'Area_Above_100_gt_95': area_above_100_gt_95,
            'Area_Below_100_gt_95': area_below_100_gt_95,
            'Num_Points_30min': len(df_30min),
            'Num_Points_Rest': len(df_rest),
        }
        
        print(f"✓ Day{day_num:3d} | μ_full: {full_day_mean:7.2f} | σ_full: {full_day_sigma:6.2f} | μ_30min: {mean_30min:7.2f} | σ_30min: {sigma_30min:6.2f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze full-day and 30-minute window mean/sigma")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/raid/Quant14/V_Feature_Analysis/trend_strat/ebx/50pct_full_day_30min_analysis.csv',
        help="Output CSV file path"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("FULL DAY + 30-MINUTE WINDOW ANALYSIS")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Max workers: {args.max_workers}")
    
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
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                analyze_single_day_with_30min, 
                f, 
                extract_day_num(f)
            ): f
            for f in sorted_files
        }
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"✗ Error in {file_path}: {e}")
    
    if not results:
        print("\nERROR: No files processed successfully")
        return
    
    # Create DataFrame and sort by Day
    df_all = pd.DataFrame(results)
    df_all = df_all.sort_values('Day').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_all.to_csv(args.output_file, index=False)
    
    # ========================================================================
    # SAVE DETAILED ANALYSIS TO TEXT FILE
    # ========================================================================
    
    summary_file = args.output_file.replace('.csv', '_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FULL DAY + 30-MINUTE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total days analyzed: {len(df_all)}\n\n")
        
        # Sigma statistics (30-min)
        f.write("="*80 + "\n")
        f.write("SIGMA STATISTICS (30-MINUTE WINDOW)\n")
        f.write("="*80 + "\n\n")
        f.write(f"25th Percentile of σ (30-min): {df_all['Sigma_30min'].quantile(0.25):.2f}\n")
        f.write(f"50th Percentile of σ (30-min): {df_all['Sigma_30min'].quantile(0.50):.2f}\n")
        f.write(f"75th Percentile of σ (30-min): {df_all['Sigma_30min'].quantile(0.75):.2f}\n")
        f.write(f"Average σ (30-min) across all {len(df_all)} days: {df_all['Sigma_30min'].mean():.2f}\n")
        f.write(f"Min σ (30-min): {df_all['Sigma_30min'].min():.2f}\n")
        f.write(f"Max σ (30-min): {df_all['Sigma_30min'].max():.2f}\n")
        f.write(f"Median σ (30-min): {df_all['Sigma_30min'].median():.2f}\n\n")
        
        # Sigma distribution (30-min)
        f.write("="*80 + "\n")
        f.write("SIGMA DISTRIBUTION (30-MINUTE WINDOW)\n")
        f.write("="*80 + "\n")
        q1 = df_all['Sigma_30min'].quantile(0.25)
        q2 = df_all['Sigma_30min'].quantile(0.50)
        q3 = df_all['Sigma_30min'].quantile(0.75)
        
        q1_count = (df_all['Sigma_30min'] < q1).sum()
        q2_count = ((df_all['Sigma_30min'] >= q1) & (df_all['Sigma_30min'] < q2)).sum()
        q3_count = ((df_all['Sigma_30min'] >= q2) & (df_all['Sigma_30min'] < q3)).sum()
        q4_count = (df_all['Sigma_30min'] >= q3).sum()
        
        f.write(f"Days with σ < 25th percentile ({q1:.2f}):    {q1_count:4d} ({q1_count/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with σ between 25-50th percentile:      {q2_count:4d} ({q2_count/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with σ between 50-75th percentile:      {q3_count:4d} ({q3_count/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with σ > 75th percentile ({q3:.2f}):    {q4_count:4d} ({q4_count/len(df_all)*100:5.1f}%)\n\n")
        
        # Mean distribution (30-min)
        f.write("="*80 + "\n")
        f.write("MEAN DISTRIBUTION (30-MINUTE WINDOW)\n")
        f.write("="*80 + "\n")
        days_above = (df_all['Mean_30min'] > 100).sum()
        days_below = (df_all['Mean_30min'] < 100).sum()
        f.write(f"Days with μ > 100:  {days_above:4d} ({days_above/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with μ < 100:  {days_below:4d} ({days_below/len(df_all)*100:5.1f}%)\n\n")
        
        # Analysis when μ > 100 (30-min)
        f.write("="*80 + "\n")
        f.write("SIGMA ANALYSIS - WHEN μ > 100 (30-MIN)\n")
        f.write("="*80 + "\n")
        mu_above = df_all[df_all['Mean_30min'] > 100]
        if len(mu_above) > 0:
            f.write(f"Average σ (30-min):   {mu_above['Sigma_30min'].mean():.2f}\n")
            f.write(f"Days with σ > 75th pct:    {(mu_above['Sigma_30min'] > q3).sum():4d} / {len(mu_above):4d} ({(mu_above['Sigma_30min'] > q3).sum()/len(mu_above)*100:5.1f}%)\n")
            f.write(f"Avg Price@30min:          {mu_above['Price_at_30min'].mean():.2f}\n")
            f.write(f"Avg (High - Price@30min):   {mu_above['High_minus_Price30'].mean():.2f}\n")
            f.write(f"Avg (Price@30min - Low):    {mu_above['Price30_minus_Low'].mean():.2f}\n\n")
        
        # Analysis when μ < 100 (30-min)
        f.write("="*80 + "\n")
        f.write("SIGMA ANALYSIS - WHEN μ < 100 (30-MIN)\n")
        f.write("="*80 + "\n")
        mu_below = df_all[df_all['Mean_30min'] < 100]
        if len(mu_below) > 0:
            f.write(f"Average σ (30-min):   {mu_below['Sigma_30min'].mean():.2f}\n")
            f.write(f"Days with σ > 75th pct:    {(mu_below['Sigma_30min'] > q3).sum():4d} / {len(mu_below):4d} ({(mu_below['Sigma_30min'] > q3).sum()/len(mu_below)*100:5.1f}%)\n")
            f.write(f"Avg Price@30min:           {mu_below['Price_at_30min'].mean():.2f}\n")
            f.write(f"Avg (High - Price@30min):   {mu_below['High_minus_Price30'].mean():.2f}\n")
            f.write(f"Avg (Price@30min - Low):    {mu_below['Price30_minus_Low'].mean():.2f}\n\n")
        
        # ========================================================================
        # TARGETED ANALYSIS: σ between 0.07 and 0.11 (30-MIN)
        # ========================================================================
        
        f.write("="*80 + "\n")
        f.write("TARGETED SIGMA ANALYSIS: σ BETWEEN 0.07 AND 0.11 (30-MIN)\n")
        f.write("="*80 + "\n\n")
        
        sigma_filtered = df_all[(df_all['Sigma_30min'] >= 0.04) & (df_all['Sigma_30min'] <= 0.07)]
        mu_above_100 = sigma_filtered[sigma_filtered['Mean_30min'] > 100]
        mu_below_100 = sigma_filtered[sigma_filtered['Mean_30min'] < 100]
        
        f.write("-"*80 + "\n")
        f.write("DIVISION: μ>100, σ ∈ [0.07, 0.11] (30-MIN)\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of days:                            {len(mu_above_100):4d} ({len(mu_above_100)/len(df_all)*100:5.1f}% of total)\n")
        
        if len(mu_above_100) > 0:
            f.write(f"Days that close above 100:                 {mu_above_100['Close_Above_100'].sum():4d} / {len(mu_above_100):4d} ({mu_above_100['Close_Above_100'].sum()/len(mu_above_100)*100:5.1f}%)\n")
            f.write(f"Days with first 30min area above 100 > 95%:        {mu_above_100['Area_Above_100_gt_95'].sum():4d} / {len(mu_above_100):4d} ({mu_above_100['Area_Above_100_gt_95'].sum()/len(mu_above_100)*100:5.1f}%)\n")
            f.write(f"Days where price never goes below 100 (first 30min):     {mu_above_100['Never_Crosses_100_30min'].sum():4d} / {len(mu_above_100):4d} ({mu_above_100['Never_Crosses_100_30min'].sum()/len(mu_above_100)*100:5.1f}%)\n")
            f.write(f"Avg μ (first 30-min):                    {mu_above_100['Mean_30min'].mean():.2f}\n")
            f.write(f"Avg σ (first 30-min):                      {mu_above_100['Sigma_30min'].mean():.2f}\n")
            f.write(f"Avg Price@30min:                         {mu_above_100['Price_at_30min'].mean():.2f}\n")
            f.write(f"Avg (High - Price@30min):                  {mu_above_100['High_minus_Price30'].mean():.2f}\n")
            f.write(f"Avg (Price@30min - Low):                   {mu_above_100['Price30_minus_Low'].mean():.2f}\n")
            f.write(f"Avg Peak-to-Peak from Price@30min:         {mu_above_100['Peak_to_Peak_30min'].mean():.2f}\n")
            f.write(f"Avg % area above 100 (first 30min):      {mu_above_100['Pct_Area_Above_100_30min'].mean():.2f}%\n")
            f.write(f"Avg % area below 100 (first 30min):      {mu_above_100['Pct_Area_Below_100_30min'].mean():.2f}%\n")
            f.write(f"Avg % of rest-of-day price > 100:        {mu_above_100['Pct_Rest_Above_100'].mean():.2f}%\n")
            f.write(f"Avg % of rest-of-day price < 100:        {mu_above_100['Pct_Rest_Below_100'].mean():.2f}%\n")
            f.write(f"Avg closing price:                       {mu_above_100['Closing_Price'].mean():.2f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DIVISION: μ<100, σ ∈ [0.07, 0.11] (30-MIN)\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of days:                            {len(mu_below_100):4d} ({len(mu_below_100)/len(df_all)*100:5.1f}% of total)\n")
        
        if len(mu_below_100) > 0:
            f.write(f"Days that close above 100:                 {mu_below_100['Close_Above_100'].sum():4d} / {len(mu_below_100):4d} ({mu_below_100['Close_Above_100'].sum()/len(mu_below_100)*100:5.1f}%)\n")
            f.write(f"Days with first 30min area below 100 > 95%:        {mu_below_100['Area_Below_100_gt_95'].sum():4d} / {len(mu_below_100):4d} ({mu_below_100['Area_Below_100_gt_95'].sum()/len(mu_below_100)*100:5.1f}%)\n")
            f.write(f"Days where price never goes above 100 (first 30min):     {mu_below_100['Never_Crosses_100_30min'].sum():4d} / {len(mu_below_100):4d} ({mu_below_100['Never_Crosses_100_30min'].sum()/len(mu_below_100)*100:5.1f}%)\n")
            f.write(f"Avg μ (first 30-min):                    {mu_below_100['Mean_30min'].mean():.2f}\n")
            f.write(f"Avg σ (first 30-min):                      {mu_below_100['Sigma_30min'].mean():.2f}\n")
            f.write(f"Avg Price@30min:                         {mu_below_100['Price_at_30min'].mean():.2f}\n")
            f.write(f"Avg (High - Price@30min):                  {mu_below_100['High_minus_Price30'].mean():.2f}\n")
            f.write(f"Avg (Price@30min - Low):                   {mu_below_100['Price30_minus_Low'].mean():.2f}\n")
            f.write(f"Avg Peak-to-Peak from Price@30min:         {mu_below_100['Peak_to_Peak_30min'].mean():.2f}\n")
            f.write(f"Avg % area above 100 (first 30min):      {mu_below_100['Pct_Area_Above_100_30min'].mean():.2f}%\n")
            f.write(f"Avg % area below 100 (first 30min):      {mu_below_100['Pct_Area_Below_100_30min'].mean():.2f}%\n")
            f.write(f"Avg % of rest-of-day price > 100:        {mu_below_100['Pct_Rest_Above_100'].mean():.2f}%\n")
            f.write(f"Avg % of rest-of-day price < 100:        {mu_below_100['Pct_Rest_Below_100'].mean():.2f}%\n")
            f.write(f"Avg closing price:                       {mu_below_100['Closing_Price'].mean():.2f}\n\n")
        
        # Overall statistics
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total days closing above 100:                       {df_all['Close_Above_100'].sum():4d} / {len(df_all):4d} ({df_all['Close_Above_100'].sum()/len(df_all)*100:5.1f}%)\n")
        f.write(f"Total days first 30min area above 100 > 95%:          {df_all['Area_Above_100_gt_95'].sum():4d} / {len(df_all):4d} ({df_all['Area_Above_100_gt_95'].sum()/len(df_all)*100:5.1f}%)\n")
        f.write(f"Total days first 30min area below 100 > 95%:          {df_all['Area_Below_100_gt_95'].sum():4d} / {len(df_all):4d} ({df_all['Area_Below_100_gt_95'].sum()/len(df_all)*100:5.1f}%)\n")
        f.write(f"Total days price never crosses 100 (first 30min):     {df_all['Never_Crosses_100_30min'].sum():4d} / {len(df_all):4d} ({df_all['Never_Crosses_100_30min'].sum()/len(df_all)*100:5.1f}%)\n")
        f.write(f"Avg μ (first 30-min, all days):                      {df_all['Mean_30min'].mean():.2f}\n")
        f.write(f"Avg σ (first 30-min, all days):                       {df_all['Sigma_30min'].mean():.2f}\n")
        f.write(f"Avg Price@30min (all days):                          {df_all['Price_at_30min'].mean():.2f}\n")
        f.write(f"Avg (High - Price@30min, all days):                   {df_all['High_minus_Price30'].mean():.2f}\n")
        f.write(f"Avg (Price@30min - Low, all days):                    {df_all['Price30_minus_Low'].mean():.2f}\n")
        f.write(f"Avg Peak-to-Peak from Price@30min (all days):         {df_all['Peak_to_Peak_30min'].mean():.2f}\n")
        f.write(f"Avg % area above 100 (first 30min):                 {df_all['Pct_Area_Above_100_30min'].mean():.2f}%\n")
        f.write(f"Avg % area below 100 (first 30min):                 {df_all['Pct_Area_Below_100_30min'].mean():.2f}%\n")
        f.write(f"Avg % of rest-of-day price > 100:                   {df_all['Pct_Rest_Above_100'].mean():.2f}%\n")
        f.write(f"Avg % of rest-of-day price < 100:                   {df_all['Pct_Rest_Below_100'].mean():.2f}%\n")
        f.write(f"Avg closing price (all days):                       {df_all['Closing_Price'].mean():.2f}\n")
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"  CSV output:     {args.output_file}")
    print(f"  Summary output: {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()