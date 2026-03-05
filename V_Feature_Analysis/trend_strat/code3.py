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


def analyze_single_day(file_path: str, day_num: int, 
                       time_col: str = 'Time', 
                       price_col: str = 'Price',
                       decision_window: int = 1800) -> dict:
    """Analyze first 30-min and full-day mean and sigma"""
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
        
        # Split data: first 30 minutes vs rest of day
        first_30_data = df[df['Time_sec'] <= decision_window]
        rest_of_day = df[df['Time_sec'] > decision_window]
        
        if len(first_30_data) < 10 or len(rest_of_day) < 10:
            print(f"Warning: Day {day_num} insufficient data. Skipping.")
            return None
        
        # Calculate first 30-min mean and sigma
        first_30_mean = first_30_data[price_col].mean()
        first_30_sigma = first_30_data[price_col].std()
        first_30_mean_above_100 = first_30_mean > 100
        
        # Calculate full-day mean and sigma
        full_day_mean = df[price_col].mean()
        full_day_sigma = df[price_col].std()
        
        # Get price at end of 30 min mark
        price_at_30min = first_30_data.iloc[-1][price_col]
        
        # Get highest high and lowest low of the entire day
        day_high = df[price_col].max()
        day_low = df[price_col].min()
        
        # Calculate differences
        high_minus_30min = day_high - price_at_30min
        low_minus_30min = price_at_30min - day_low
        
        # Get closing price (last price of the day)
        closing_price = df.iloc[-1][price_col]
        closes_above_100 = closing_price > 100
        
        # Calculate percentage of time price was above 100 for rest of day
        rest_above_100_count = (rest_of_day[price_col] > 100).sum()
        rest_total_count = len(rest_of_day)
        pct_above_100_rest = (rest_above_100_count / rest_total_count) * 100
        
        # Calculate percentage of time price was below 100 for rest of day
        rest_below_100_count = (rest_of_day[price_col] < 100).sum()
        pct_below_100_rest = (rest_below_100_count / rest_total_count) * 100
        
        # Calculate area above and below 100 for FIRST 30 MINUTES only
        first_30_area_above_100 = first_30_data[first_30_data[price_col] > 100][price_col].apply(lambda x: x - 100).sum()
        first_30_area_below_100 = first_30_data[first_30_data[price_col] < 100][price_col].apply(lambda x: 100 - x).sum()
        
        # Calculate area percentage for first 30 minutes
        first_30_total_area = first_30_area_above_100 + first_30_area_below_100
        first_30_pct_area_above_100 = (first_30_area_above_100 / first_30_total_area * 100) if first_30_total_area > 0 else 0
        first_30_pct_area_below_100 = (first_30_area_below_100 / first_30_total_area * 100) if first_30_total_area > 0 else 0
        
        # Check if area above 100 is greater than 95% of total area (first 30 min)
        first_30_area_above_gt_95pct = first_30_pct_area_above_100 > 95
        
        # Check if area below 100 is greater than 95% of total area (first 30 min)
        first_30_area_below_gt_95pct = first_30_pct_area_below_100 > 95
        
        # Check if price never crosses 100 in first 30 minutes
        first_30_price_never_crosses_100 = False
        if first_30_mean_above_100:
            # μ > 100: price should never go below 100
            first_30_price_never_crosses_100 = (first_30_data[price_col] >= 100).all()
        else:
            # μ < 100: price should never go above 100
            first_30_price_never_crosses_100 = (first_30_data[price_col] <= 100).all()
        
        result = {
            'Day': day_num,
            'First_30min_Mean': round(first_30_mean, 2),
            'First_30min_Sigma': round(first_30_sigma, 4),
            'Full_Day_Mean': round(full_day_mean, 2),
            'Full_Day_Sigma': round(full_day_sigma, 4),
            'First_30min_Mean_Above_100': 'Yes' if first_30_mean_above_100 else 'No',
            'Price_at_30min': round(price_at_30min, 2),
            'Day_High': round(day_high, 2),
            'Day_Low': round(day_low, 2),
            'High_minus_30min': round(high_minus_30min, 2),
            'Low_minus_30min': round(low_minus_30min, 2),
            'Closing_Price': round(closing_price, 2),
            'Closes_Above_100': 'Yes' if closes_above_100 else 'No',
            'Pct_Above_100_RestOfDay': round(pct_above_100_rest, 2),
            'Pct_Below_100_RestOfDay': round(pct_below_100_rest, 2),
            'First_30min_Area_Above_100': round(first_30_area_above_100, 2),
            'First_30min_Area_Below_100': round(first_30_area_below_100, 2),
            'First_30min_Pct_Area_Above_100': round(first_30_pct_area_above_100, 2),
            'First_30min_Pct_Area_Below_100': round(first_30_pct_area_below_100, 2),
            'First_30min_Area_Above_GT_95pct': 'Yes' if first_30_area_above_gt_95pct else 'No',
            'First_30min_Area_Below_GT_95pct': 'Yes' if first_30_area_below_gt_95pct else 'No',
            'First_30min_Price_Never_Crosses_100': 'Yes' if first_30_price_never_crosses_100 else 'No',
        }
        
        status = "ABOVE" if first_30_mean_above_100 else "BELOW"
        print(f"✓ Day{day_num:3d} | 30min-μ: {first_30_mean:7.2f} | 30min-σ: {first_30_sigma:6.4f} | Day-σ: {full_day_sigma:6.4f} | {status}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze days with sigma > 75th percentile")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/raid/Quant14/V_Feature_Analysis/trend_strat/ebx/sigma_75th_percentile_analysis.csv',
        help="Output CSV file path"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help="Number of parallel workers"
    )
    parser.add_argument(
        '--decision-window',
        type=int,
        default=1800,
        help="Decision window in seconds (default: 1800 = 30 minutes)"
    )
    args = parser.parse_args()
    
    print("="*80)
    print("SIGMA > 75TH PERCENTILE ANALYSIS")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Max workers: {args.max_workers}")
    print(f"Decision window: {args.decision_window}s ({args.decision_window/60:.0f} min)")
    
    # Days to exclude from final analysis (but include in threshold calculation)
    excluded_days = {311, 421, 446, 87, 273}
    
    # Find input files
    files_pattern = os.path.join(args.data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    # STEP 1: Filter out ONLY the 600-790 range (NOT the excluded days yet)
    filtered_files = [
        f for f in sorted_files 
        if not (600 <= extract_day_num(f) <= 790)
    ]
    
    if not filtered_files:
        print(f"\nERROR: No parquet files found in {args.data_dir}")
        return
    
    print(f"\nFound {len(filtered_files)} day files to process (excluding 600-790 range)")
    print(f"Days to exclude from final analysis: {sorted(excluded_days)}")
    print(f"\nProcessing with {args.max_workers} workers...\n")
    
    # STEP 2: Process ALL days (including the ones we'll exclude later)
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                analyze_single_day, 
                f, 
                extract_day_num(f),
                decision_window=args.decision_window
            ): f
            for f in filtered_files
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
    
    # Create DataFrame with ALL days
    df_all = pd.DataFrame(results)
    df_all = df_all.sort_values('Day').reset_index(drop=True)
    
    # STEP 3: Calculate overall statistics from ALL days (INCLUDING excluded days)
    print(f"\n{'='*80}")
    print("STEP 1: CALCULATING THRESHOLD FROM ALL DAYS")
    print(f"{'='*80}")
    print(f"\nTotal days analyzed: {len(df_all)}")
    
    overall_first_30_sigma_mean = df_all['First_30min_Sigma'].mean()
    overall_first_30_sigma_25th = df_all['First_30min_Sigma'].quantile(0.25)
    overall_first_30_sigma_75th = df_all['First_30min_Sigma'].quantile(0.75)
    overall_full_day_sigma_mean = df_all['Full_Day_Sigma'].mean()
    overall_full_day_sigma_25th = df_all['Full_Day_Sigma'].quantile(0.25)
    overall_full_day_sigma_75th = df_all['Full_Day_Sigma'].quantile(0.75)
    
    overall_first_30_mean_mean = df_all['First_30min_Mean'].mean()
    overall_first_30_mean_25th = df_all['First_30min_Mean'].quantile(0.25)
    overall_full_day_mean_mean = df_all['Full_Day_Mean'].mean()
    overall_full_day_mean_25th = df_all['Full_Day_Mean'].quantile(0.25)
    
    print(f"\n--- FIRST 30 MINUTES ---")
    print(f"Average σ (30-min):      {overall_first_30_sigma_mean:.4f}")
    print(f"25th Percentile σ:       {overall_first_30_sigma_25th:.4f}")
    print(f"75th Percentile σ:       {overall_first_30_sigma_75th:.4f} ← THRESHOLD")
    print(f"Median σ:                {df_all['First_30min_Sigma'].median():.4f}")
    print(f"Min σ:                   {df_all['First_30min_Sigma'].min():.4f}")
    print(f"Max σ:                   {df_all['First_30min_Sigma'].max():.4f}")
    print(f"\nAverage μ (30-min):      {overall_first_30_mean_mean:.2f}")
    print(f"25th Percentile μ:       {overall_first_30_mean_25th:.2f}")
    
    print(f"\n--- FULL DAY ---")
    print(f"Average σ (full-day):    {overall_full_day_sigma_mean:.4f}")
    print(f"25th Percentile σ:       {overall_full_day_sigma_25th:.4f}")
    print(f"75th Percentile σ:       {overall_full_day_sigma_75th:.4f}")
    print(f"Median σ:                {df_all['Full_Day_Sigma'].median():.4f}")
    print(f"Min σ:                   {df_all['Full_Day_Sigma'].min():.4f}")
    print(f"Max σ:                   {df_all['Full_Day_Sigma'].max():.4f}")
    print(f"\nAverage μ (full-day):    {overall_full_day_mean_mean:.2f}")
    print(f"25th Percentile μ:       {overall_full_day_mean_25th:.2f}")
    
    # STEP 4: Filter days with sigma > threshold
    sigma_threshold = overall_first_30_sigma_75th
    df_above_threshold = df_all[df_all['First_30min_Sigma'] > sigma_threshold].copy()
    
    print(f"\n{'='*80}")
    print(f"STEP 2: FILTERING BY THRESHOLD (σ > {sigma_threshold:.4f})")
    print(f"{'='*80}")
    print(f"Days with σ > {sigma_threshold:.4f}: {len(df_above_threshold):4d} / {len(df_all):4d} ({len(df_above_threshold)/len(df_all)*100:5.2f}%)")
    
    # STEP 5: NOW exclude the specified days from final analysis
    df_filtered = df_above_threshold[~df_above_threshold['Day'].isin(excluded_days)].copy()
    
    print(f"\n{'='*80}")
    print(f"STEP 3: EXCLUDING SPECIFIED DAYS FROM FINAL ANALYSIS")
    print(f"{'='*80}")
    print(f"Days excluded: {sorted(excluded_days)}")
    print(f"Days after exclusion: {len(df_filtered):4d} / {len(df_above_threshold):4d}")
    
    # Show which excluded days were above threshold
    excluded_and_above = df_above_threshold[df_above_threshold['Day'].isin(excluded_days)]
    if len(excluded_and_above) > 0:
        print(f"\nExcluded days that WERE above threshold:")
        for _, row in excluded_and_above.iterrows():
            print(f"  Day {row['Day']:3d}: σ = {row['First_30min_Sigma']:.4f}")
    
    if len(df_filtered) == 0:
        print(f"\nNo days found after filtering and exclusion")
        return
    
    # Add classification columns
    df_filtered['Sigma_Category'] = f'σ>{sigma_threshold:.4f}'
    
    # Create 2 divisions based on mean
    df_filtered['Division'] = df_filtered['First_30min_Mean_Above_100'].apply(
        lambda x: f'μ>100_σ>{sigma_threshold:.4f}' if x == 'Yes' else f'μ<100_σ>{sigma_threshold:.4f}'
    )
    
    # Add benchmark columns
    df_filtered['Overall_First_30_Sigma_Mean'] = overall_first_30_sigma_mean
    df_filtered['Overall_First_30_Sigma_25th'] = overall_first_30_sigma_25th
    df_filtered['Overall_First_30_Sigma_75th'] = overall_first_30_sigma_75th
    df_filtered['Overall_Full_Day_Sigma_Mean'] = overall_full_day_sigma_mean
    df_filtered['Overall_Full_Day_Sigma_25th'] = overall_full_day_sigma_25th
    df_filtered['Overall_Full_Day_Sigma_75th'] = overall_full_day_sigma_75th
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_filtered.to_csv(args.output_file, index=False)
    
    # Split by mean
    mu_above_100 = df_filtered[df_filtered['First_30min_Mean_Above_100'] == 'Yes']
    mu_below_100 = df_filtered[df_filtered['First_30min_Mean_Above_100'] == 'No']
    
    print(f"\n{'='*80}")
    print("MEAN DISTRIBUTION (FINAL FILTERED DAYS)")
    print(f"{'='*80}")
    print(f"Days with μ > 100: {len(mu_above_100):4d} ({len(mu_above_100)/len(df_filtered)*100:5.1f}%)")
    print(f"Days with μ < 100: {len(mu_below_100):4d} ({len(mu_below_100)/len(df_filtered)*100:5.1f}%)")
    
    # Detailed analysis for μ > 100
    if len(mu_above_100) > 0:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: μ > 100, σ > {sigma_threshold:.4f}")
        print(f"{'='*80}")
        print(f"\nNumber of days: {len(mu_above_100)}")
        
        closes_above = (mu_above_100['Closes_Above_100'] == 'Yes').sum()
        never_crosses = (mu_above_100['First_30min_Price_Never_Crosses_100'] == 'Yes').sum()
        area_above_gt_95 = (mu_above_100['First_30min_Area_Above_GT_95pct'] == 'Yes').sum()
        
        print(f"\n--- BEHAVIORAL METRICS ---")
        print(f"Days that close above 100:                    {closes_above:4d} / {len(mu_above_100):4d} ({closes_above/len(mu_above_100)*100:5.2f}%)")
        print(f"Days with 30min area above 100 > 95%:         {area_above_gt_95:4d} / {len(mu_above_100):4d} ({area_above_gt_95/len(mu_above_100)*100:5.2f}%)")
        print(f"Days where price never goes below 100:        {never_crosses:4d} / {len(mu_above_100):4d} ({never_crosses/len(mu_above_100)*100:5.2f}%)")
        
        print(f"\n--- FIRST 30 MINUTES ---")
        print(f"Avg σ (30-min):                               {mu_above_100['First_30min_Sigma'].mean():.4f} (Overall: {overall_first_30_sigma_mean:.4f}, Diff: {mu_above_100['First_30min_Sigma'].mean() - overall_first_30_sigma_mean:+.4f})")
        print(f"25th Percentile σ:                            {mu_above_100['First_30min_Sigma'].quantile(0.25):.4f} (Overall: {overall_first_30_sigma_25th:.4f})")
        print(f"75th Percentile σ:                            {mu_above_100['First_30min_Sigma'].quantile(0.75):.4f} (Overall: {overall_first_30_sigma_75th:.4f})")
        print(f"Min σ:                                        {mu_above_100['First_30min_Sigma'].min():.4f}")
        print(f"Max σ:                                        {mu_above_100['First_30min_Sigma'].max():.4f}")
        print(f"Avg μ (30-min):                               {mu_above_100['First_30min_Mean'].mean():.2f} (Overall: {overall_first_30_mean_mean:.2f}, Diff: {mu_above_100['First_30min_Mean'].mean() - overall_first_30_mean_mean:+.2f})")
        print(f"25th Percentile μ:                            {mu_above_100['First_30min_Mean'].quantile(0.25):.2f} (Overall: {overall_first_30_mean_25th:.2f})")
        
        print(f"\n--- FULL DAY ---")
        print(f"Avg σ (full-day):                             {mu_above_100['Full_Day_Sigma'].mean():.4f} (Overall: {overall_full_day_sigma_mean:.4f}, Diff: {mu_above_100['Full_Day_Sigma'].mean() - overall_full_day_sigma_mean:+.4f})")
        print(f"25th Percentile σ:                            {mu_above_100['Full_Day_Sigma'].quantile(0.25):.4f} (Overall: {overall_full_day_sigma_25th:.4f})")
        print(f"75th Percentile σ:                            {mu_above_100['Full_Day_Sigma'].quantile(0.75):.4f} (Overall: {overall_full_day_sigma_75th:.4f})")
        print(f"Avg μ (full-day):                             {mu_above_100['Full_Day_Mean'].mean():.2f} (Overall: {overall_full_day_mean_mean:.2f}, Diff: {mu_above_100['Full_Day_Mean'].mean() - overall_full_day_mean_mean:+.2f})")
        print(f"25th Percentile μ:                            {mu_above_100['Full_Day_Mean'].quantile(0.25):.2f} (Overall: {overall_full_day_mean_25th:.2f})")
        
        print(f"\n--- PRICE METRICS ---")
        print(f"Avg Price@30min:                              {mu_above_100['Price_at_30min'].mean():.2f}")
        print(f"Avg Day High:                                 {mu_above_100['Day_High'].mean():.2f}")
        print(f"Avg Day Low:                                  {mu_above_100['Day_Low'].mean():.2f}")
        print(f"Avg (High - Price@30min):                     {mu_above_100['High_minus_30min'].mean():.2f}")
        print(f"Avg (Price@30min - Low):                      {mu_above_100['Low_minus_30min'].mean():.2f}")
        print(f"Avg Closing Price:                            {mu_above_100['Closing_Price'].mean():.2f}")
        
        print(f"\n--- AREA & TIME ANALYSIS ---")
        print(f"Avg % area above 100 (first 30min):           {mu_above_100['First_30min_Pct_Area_Above_100'].mean():.2f}%")
        print(f"Avg % area below 100 (first 30min):           {mu_above_100['First_30min_Pct_Area_Below_100'].mean():.2f}%")
        print(f"Avg % of rest-of-day price > 100:             {mu_above_100['Pct_Above_100_RestOfDay'].mean():.2f}%")
        print(f"Avg % of rest-of-day price < 100:             {mu_above_100['Pct_Below_100_RestOfDay'].mean():.2f}%")
        
        print(f"\n{'-'*80}")
        print(f"INDIVIDUAL DAY BREAKDOWN: μ > 100")
        print(f"{'-'*80}")
        print(f"{'Day':>5} | {'30m-μ':>7} | {'30m-σ':>7} | {'Day-σ':>7} | {'P@30m':>7} | {'High':>7} | {'Low':>7} | {'H-P30':>7} | {'P30-L':>7} | {'Close':>7} | {'C>100':>6}")
        print(f"{'-'*80}")
        for _, row in mu_above_100.iterrows():
            print(f"{row['Day']:5d} | {row['First_30min_Mean']:7.2f} | {row['First_30min_Sigma']:7.4f} | {row['Full_Day_Sigma']:7.4f} | "
                  f"{row['Price_at_30min']:7.2f} | {row['Day_High']:7.2f} | {row['Day_Low']:7.2f} | "
                  f"{row['High_minus_30min']:7.2f} | {row['Low_minus_30min']:7.2f} | {row['Closing_Price']:7.2f} | {row['Closes_Above_100']:>6}")
    
    # Detailed analysis for μ < 100
    if len(mu_below_100) > 0:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS: μ < 100, σ > {sigma_threshold:.4f}")
        print(f"{'='*80}")
        print(f"\nNumber of days: {len(mu_below_100)}")
        
        closes_above = (mu_below_100['Closes_Above_100'] == 'Yes').sum()
        never_crosses = (mu_below_100['First_30min_Price_Never_Crosses_100'] == 'Yes').sum()
        area_below_gt_95 = (mu_below_100['First_30min_Area_Below_GT_95pct'] == 'Yes').sum()
        
        print(f"\n--- BEHAVIORAL METRICS ---")
        print(f"Days that close above 100:                    {closes_above:4d} / {len(mu_below_100):4d} ({closes_above/len(mu_below_100)*100:5.2f}%)")
        print(f"Days with 30min area below 100 > 95%:         {area_below_gt_95:4d} / {len(mu_below_100):4d} ({area_below_gt_95/len(mu_below_100)*100:5.2f}%)")
        print(f"Days where price never goes above 100:        {never_crosses:4d} / {len(mu_below_100):4d} ({never_crosses/len(mu_below_100)*100:5.2f}%)")
        
        print(f"\n--- FIRST 30 MINUTES ---")
        print(f"Avg σ (30-min):                               {mu_below_100['First_30min_Sigma'].mean():.4f} (Overall: {overall_first_30_sigma_mean:.4f}, Diff: {mu_below_100['First_30min_Sigma'].mean() - overall_first_30_sigma_mean:+.4f})")
        print(f"25th Percentile σ:                            {mu_below_100['First_30min_Sigma'].quantile(0.25):.4f} (Overall: {overall_first_30_sigma_25th:.4f})")
        print(f"75th Percentile σ:                            {mu_below_100['First_30min_Sigma'].quantile(0.75):.4f} (Overall: {overall_first_30_sigma_75th:.4f})")
        print(f"Min σ:                                        {mu_below_100['First_30min_Sigma'].min():.4f}")
        print(f"Max σ:                                        {mu_below_100['First_30min_Sigma'].max():.4f}")
        print(f"Avg μ (30-min):                               {mu_below_100['First_30min_Mean'].mean():.2f} (Overall: {overall_first_30_mean_mean:.2f}, Diff: {mu_below_100['First_30min_Mean'].mean() - overall_first_30_mean_mean:+.2f})")
        print(f"25th Percentile μ:                            {mu_below_100['First_30min_Mean'].quantile(0.25):.2f} (Overall: {overall_first_30_mean_25th:.2f})")
        
        print(f"\n--- FULL DAY ---")
        print(f"Avg σ (full-day):                             {mu_below_100['Full_Day_Sigma'].mean():.4f} (Overall: {overall_full_day_sigma_mean:.4f}, Diff: {mu_below_100['Full_Day_Sigma'].mean() - overall_full_day_sigma_mean:+.4f})")
        print(f"25th Percentile σ:                            {mu_below_100['Full_Day_Sigma'].quantile(0.25):.4f} (Overall: {overall_full_day_sigma_25th:.4f})")
        print(f"75th Percentile σ:                            {mu_below_100['Full_Day_Sigma'].quantile(0.75):.4f} (Overall: {overall_full_day_sigma_75th:.4f})")
        print(f"Avg μ (full-day):                             {mu_below_100['Full_Day_Mean'].mean():.2f} (Overall: {overall_full_day_mean_mean:.2f}, Diff: {mu_below_100['Full_Day_Mean'].mean() - overall_full_day_mean_mean:+.2f})")
        print(f"25th Percentile μ:                            {mu_below_100['Full_Day_Mean'].quantile(0.25):.2f} (Overall: {overall_full_day_mean_25th:.2f})")
        
        print(f"\n--- PRICE METRICS ---")
        print(f"Avg Price@30min:                              {mu_below_100['Price_at_30min'].mean():.2f}")
        print(f"Avg Day High:                                 {mu_below_100['Day_High'].mean():.2f}")
        print(f"Avg Day Low:                                  {mu_below_100['Day_Low'].mean():.2f}")
        print(f"Avg (High - Price@30min):                     {mu_below_100['High_minus_30min'].mean():.2f}")
        print(f"Avg (Price@30min - Low):                      {mu_below_100['Low_minus_30min'].mean():.2f}")
        print(f"Avg Closing Price:                            {mu_below_100['Closing_Price'].mean():.2f}")
        
        print(f"\n--- AREA & TIME ANALYSIS ---")
        print(f"Avg % area above 100 (first 30min):           {mu_below_100['First_30min_Pct_Area_Above_100'].mean():.2f}%")
        print(f"Avg % area below 100 (first 30min):           {mu_below_100['First_30min_Pct_Area_Below_100'].mean():.2f}%")
        print(f"Avg % of rest-of-day price > 100:             {mu_below_100['Pct_Above_100_RestOfDay'].mean():.2f}%")
        print(f"Avg % of rest-of-day price < 100:             {mu_below_100['Pct_Below_100_RestOfDay'].mean():.2f}%")
        
        print(f"\n{'-'*80}")
        print(f"INDIVIDUAL DAY BREAKDOWN: μ < 100")
        print(f"{'-'*80}")
        print(f"{'Day':>5} | {'30m-μ':>7} | {'30m-σ':>7} | {'Day-σ':>7} | {'P@30m':>7} | {'High':>7} | {'Low':>7} | {'H-P30':>7} | {'P30-L':>7} | {'Close':>7} | {'C>100':>6}")
        print(f"{'-'*80}")
        for _, row in mu_below_100.iterrows():
            print(f"{row['Day']:5d} | {row['First_30min_Mean']:7.2f} | {row['First_30min_Sigma']:7.4f} | {row['Full_Day_Sigma']:7.4f} | "
                  f"{row['Price_at_30min']:7.2f} | {row['Day_High']:7.2f} | {row['Day_Low']:7.2f} | "
                  f"{row['High_minus_30min']:7.2f} | {row['Low_minus_30min']:7.2f} | {row['Closing_Price']:7.2f} | {row['Closes_Above_100']:>6}")
    
    # Comparative analysis
    if len(mu_above_100) > 0 and len(mu_below_100) > 0:
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS: μ > 100 vs μ < 100")
        print(f"{'='*80}")
        
        print(f"\n{'Metric':<50} | {'μ>100':>12} | {'μ<100':>12} | {'Difference':>12}")
        print(f"{'-'*85}")
        print(f"{'Number of days':<50} | {len(mu_above_100):12d} | {len(mu_below_100):12d} | {len(mu_above_100)-len(mu_below_100):+12d}")
        print(f"{'Avg σ (first 30min)':<50} | {mu_above_100['First_30min_Sigma'].mean():12.4f} | {mu_below_100['First_30min_Sigma'].mean():12.4f} | {mu_above_100['First_30min_Sigma'].mean()-mu_below_100['First_30min_Sigma'].mean():+12.4f}")
        print(f"{'Avg μ (first 30min)':<50} | {mu_above_100['First_30min_Mean'].mean():12.2f} | {mu_below_100['First_30min_Mean'].mean():12.2f} | {mu_above_100['First_30min_Mean'].mean()-mu_below_100['First_30min_Mean'].mean():+12.2f}")
        print(f"{'Avg σ (full-day)':<50} | {mu_above_100['Full_Day_Sigma'].mean():12.4f} | {mu_below_100['Full_Day_Sigma'].mean():12.4f} | {mu_above_100['Full_Day_Sigma'].mean()-mu_below_100['Full_Day_Sigma'].mean():+12.4f}")
        print(f"{'Avg μ (full-day)':<50} | {mu_above_100['Full_Day_Mean'].mean():12.2f} | {mu_below_100['Full_Day_Mean'].mean():12.2f} | {mu_above_100['Full_Day_Mean'].mean()-mu_below_100['Full_Day_Mean'].mean():+12.2f}")
        print(f"{'Avg Price@30min':<50} | {mu_above_100['Price_at_30min'].mean():12.2f} | {mu_below_100['Price_at_30min'].mean():12.2f} | {mu_above_100['Price_at_30min'].mean()-mu_below_100['Price_at_30min'].mean():+12.2f}")
        print(f"{'Avg (High - Price@30min)':<50} | {mu_above_100['High_minus_30min'].mean():12.2f} | {mu_below_100['High_minus_30min'].mean():12.2f} | {mu_above_100['High_minus_30min'].mean()-mu_below_100['High_minus_30min'].mean():+12.2f}")
        print(f"{'Avg (Price@30min - Low)':<50} | {mu_above_100['Low_minus_30min'].mean():12.2f} | {mu_below_100['Low_minus_30min'].mean():12.2f} | {mu_above_100['Low_minus_30min'].mean()-mu_below_100['Low_minus_30min'].mean():+12.2f}")
        print(f"{'Avg Closing Price':<50} | {mu_above_100['Closing_Price'].mean():12.2f} | {mu_below_100['Closing_Price'].mean():12.2f} | {mu_above_100['Closing_Price'].mean()-mu_below_100['Closing_Price'].mean():+12.2f}")
        
        closes_above_g1 = (mu_above_100['Closes_Above_100'] == 'Yes').sum()
        closes_above_g2 = (mu_below_100['Closes_Above_100'] == 'Yes').sum()
        print(f"{'% days closing above 100':<50} | {closes_above_g1/len(mu_above_100)*100:11.2f}% | {closes_above_g2/len(mu_below_100)*100:11.2f}% | {(closes_above_g1/len(mu_above_100)-closes_above_g2/len(mu_below_100))*100:+11.2f}%")
        
        print(f"{'Avg % area above 100 (first 30min)':<50} | {mu_above_100['First_30min_Pct_Area_Above_100'].mean():11.2f}% | {mu_below_100['First_30min_Pct_Area_Above_100'].mean():11.2f}% | {mu_above_100['First_30min_Pct_Area_Above_100'].mean()-mu_below_100['First_30min_Pct_Area_Above_100'].mean():+11.2f}%")
        print(f"{'Avg % rest-of-day above 100':<50} | {mu_above_100['Pct_Above_100_RestOfDay'].mean():11.2f}% | {mu_below_100['Pct_Above_100_RestOfDay'].mean():11.2f}% | {mu_above_100['Pct_Above_100_RestOfDay'].mean()-mu_below_100['Pct_Above_100_RestOfDay'].mean():+11.2f}%")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! Output saved to: {args.output_file}")
    print(f"✓ Threshold used: {sigma_threshold:.4f} (calculated from ALL {len(df_all)} days)")
    print(f"✓ Final days in output: {len(df_filtered)} (after exclusion)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()