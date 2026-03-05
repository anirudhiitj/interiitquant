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


def analyze_single_day_15min(file_path: str, day_num: int, 
                              time_col: str = 'Time', 
                              price_col: str = 'Price',
                              blacklist: set = None) -> dict:
    """Analyze first 15 minutes and calculate metrics if sigma > 0.1"""
    try:
        # Skip blacklisted days
        if blacklist and day_num in blacklist:
            print(f"⊗ Day{day_num:3d} | BLACKLISTED - Skipping")
            return None
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
        cutoff_time = start_time + 900  # 15 minutes = 900 seconds
        
        # Split into first 15 minutes and full day
        df_15min = df[df['Time_sec'] <= cutoff_time].copy()
        
        if len(df_15min) < 2:
            print(f"Warning: Day {day_num} insufficient 15-min data. Skipping.")
            return None
        
        # Calculate 15-minute sigma
        sigma_15min = df_15min[price_col].std()
        
        # Only proceed if sigma is between 0.1 and 0.2
        if sigma_15min <= 0.1 or sigma_15min >= 0.2:
            return None
        
        # Calculate requested metrics
        price_at_15min = df_15min[price_col].iloc[-1]
        full_day_high = df[price_col].max()
        full_day_low = df[price_col].min()
        peak_to_peak = full_day_high - full_day_low
        closing_price = df.iloc[-1][price_col]
        opening_price = df.iloc[0][price_col]
        mean_15min = df_15min[price_col].mean()
        
        # Additional useful metrics
        high_minus_price15 = full_day_high - price_at_15min
        price15_minus_low = price_at_15min - full_day_low
        
        result = {
            'Day': day_num,
            'Sigma_15min': round(sigma_15min, 4),
            'Mean_15min': round(mean_15min, 4),
            'Price_at_15min': round(price_at_15min, 4),
            'Full_Day_High': round(full_day_high, 4),
            'Full_Day_Low': round(full_day_low, 4),
            'Peak_to_Peak': round(peak_to_peak, 4),
            'High_minus_Price15': round(high_minus_price15, 4),
            'Price15_minus_Low': round(price15_minus_low, 4),
            'Closing_Price': round(closing_price, 4),
            'Opening_Price': round(opening_price, 4),
            'Close_Above_100': closing_price > 100,
            'Price15_Above_100': price_at_15min > 100,
            'Num_Points_15min': len(df_15min),
        }
        
        print(f"✓ Day{day_num:3d} | σ_15min: {sigma_15min:6.4f} | Price@15min: {price_at_15min:7.2f} | High: {full_day_high:7.2f} | Low: {full_day_low:7.2f} | Close: {closing_price:7.2f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze days where 15-minute sigma is between 0.1 and 0.2")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/raid/Quant14/V_Feature_Analysis/trend_strat/ebx/15min_sigma_0.1_to_0.2_analysis.csv',
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
    print("15-MINUTE SIGMA BETWEEN 0.1 AND 0.2 ANALYSIS")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Max workers: {args.max_workers}")
    print(f"\nFiltering for days where 0.1 < σ (first 15 min) < 0.2")
    
    # Define blacklisted days
    BLACKLIST = {87, 311, 273, 446}
    print(f"Blacklisted days: {sorted(BLACKLIST)}\n")
    
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
                analyze_single_day_15min, 
                f, 
                extract_day_num(f),
                'Time',
                'Price',
                BLACKLIST
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
        print("\nERROR: No days found with 0.1 < σ (15-min) < 0.2")
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
        f.write("15-MINUTE SIGMA BETWEEN 0.1 AND 0.2 ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total days with 0.1 < σ (15-min) < 0.2: {len(df_all)}\n")
        f.write(f"Blacklisted days excluded: {sorted(BLACKLIST)}\n\n")
        
        # Basic statistics
        f.write("="*80 + "\n")
        f.write("BASIC STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average σ (15-min):              {df_all['Sigma_15min'].mean():.4f}\n")
        f.write(f"Min σ (15-min):                  {df_all['Sigma_15min'].min():.4f}\n")
        f.write(f"Max σ (15-min):                  {df_all['Sigma_15min'].max():.4f}\n")
        f.write(f"Median σ (15-min):               {df_all['Sigma_15min'].median():.4f}\n\n")
        
        f.write(f"Average Price @ 15min:           {df_all['Price_at_15min'].mean():.2f}\n")
        f.write(f"Average Full Day High:           {df_all['Full_Day_High'].mean():.2f}\n")
        f.write(f"Average Full Day Low:            {df_all['Full_Day_Low'].mean():.2f}\n")
        f.write(f"Average Peak-to-Peak:            {df_all['Peak_to_Peak'].mean():.2f}\n")
        f.write(f"Average Closing Price:           {df_all['Closing_Price'].mean():.2f}\n\n")
        
        # Distance metrics
        f.write("="*80 + "\n")
        f.write("DISTANCE METRICS FROM PRICE @ 15MIN\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average (High - Price@15min):    {df_all['High_minus_Price15'].mean():.2f}\n")
        f.write(f"Average (Price@15min - Low):     {df_all['Price15_minus_Low'].mean():.2f}\n\n")
        
        # Closing analysis
        f.write("="*80 + "\n")
        f.write("CLOSING ANALYSIS\n")
        f.write("="*80 + "\n\n")
        days_close_above = df_all['Close_Above_100'].sum()
        days_close_below = len(df_all) - days_close_above
        f.write(f"Days closing above 100:          {days_close_above:4d} ({days_close_above/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days closing below 100:          {days_close_below:4d} ({days_close_below/len(df_all)*100:5.1f}%)\n\n")
        
        # 15-min position analysis
        f.write("="*80 + "\n")
        f.write("PRICE @ 15MIN POSITION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        days_price15_above = df_all['Price15_Above_100'].sum()
        days_price15_below = len(df_all) - days_price15_above
        f.write(f"Days with Price@15min > 100:     {days_price15_above:4d} ({days_price15_above/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with Price@15min < 100:     {days_price15_below:4d} ({days_price15_below/len(df_all)*100:5.1f}%)\n\n")
        
        # Conditional analysis
        f.write("="*80 + "\n")
        f.write("CONDITIONAL ANALYSIS: PRICE@15MIN > 100\n")
        f.write("="*80 + "\n\n")
        df_price15_above = df_all[df_all['Price15_Above_100']]
        if len(df_price15_above) > 0:
            f.write(f"Number of days:                  {len(df_price15_above):4d}\n")
            f.write(f"Days closing above 100:          {df_price15_above['Close_Above_100'].sum():4d} ({df_price15_above['Close_Above_100'].sum()/len(df_price15_above)*100:5.1f}%)\n")
            f.write(f"Average σ (15-min):              {df_price15_above['Sigma_15min'].mean():.4f}\n")
            f.write(f"Average Price@15min:             {df_price15_above['Price_at_15min'].mean():.2f}\n")
            f.write(f"Average Peak-to-Peak:            {df_price15_above['Peak_to_Peak'].mean():.2f}\n")
            f.write(f"Average (High - Price@15min):    {df_price15_above['High_minus_Price15'].mean():.2f}\n")
            f.write(f"Average (Price@15min - Low):     {df_price15_above['Price15_minus_Low'].mean():.2f}\n")
            f.write(f"Average Closing Price:           {df_price15_above['Closing_Price'].mean():.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONDITIONAL ANALYSIS: PRICE@15MIN < 100\n")
        f.write("="*80 + "\n\n")
        df_price15_below = df_all[~df_all['Price15_Above_100']]
        if len(df_price15_below) > 0:
            f.write(f"Number of days:                  {len(df_price15_below):4d}\n")
            f.write(f"Days closing above 100:          {df_price15_below['Close_Above_100'].sum():4d} ({df_price15_below['Close_Above_100'].sum()/len(df_price15_below)*100:5.1f}%)\n")
            f.write(f"Average σ (15-min):              {df_price15_below['Sigma_15min'].mean():.4f}\n")
            f.write(f"Average Price@15min:             {df_price15_below['Price_at_15min'].mean():.2f}\n")
            f.write(f"Average Peak-to-Peak:            {df_price15_below['Peak_to_Peak'].mean():.2f}\n")
            f.write(f"Average (High - Price@15min):    {df_price15_below['High_minus_Price15'].mean():.2f}\n")
            f.write(f"Average (Price@15min - Low):     {df_price15_below['Price15_minus_Low'].mean():.2f}\n")
            f.write(f"Average Closing Price:           {df_price15_below['Closing_Price'].mean():.2f}\n\n")
        
        # List of days
        f.write("="*80 + "\n")
        f.write("LIST OF ALL QUALIFYING DAYS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Day':<8} {'Sigma_15min':<12} {'Price@15min':<12} {'High':<10} {'Low':<10} {'P2P':<10} {'Close':<10}\n")
        f.write("-"*80 + "\n")
        for _, row in df_all.iterrows():
            f.write(f"{row['Day']:<8} {row['Sigma_15min']:<12.4f} {row['Price_at_15min']:<12.2f} {row['Full_Day_High']:<10.2f} {row['Full_Day_Low']:<10.2f} {row['Peak_to_Peak']:<10.2f} {row['Closing_Price']:<10.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"  Total qualifying days: {len(df_all)}")
    print(f"  CSV output:           {args.output_file}")
    print(f"  Summary output:       {summary_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()