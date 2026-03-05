import os
import glob
import re
import pandas as pd
import numpy as np
import dask_cudf
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import argparse

def extract_day_num(filepath):
    """Extract day number from filename"""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def analyze_single_day_touch_100_09(file_path: str, day_num: int, 
                                     time_col: str = 'Time', 
                                     price_col: str = 'Price',
                                     target_price: float = 100.09,
                                     min_window_sec: int = 60,
                                     max_window_sec: int = 300) -> dict:
    """
    Analyze days when price touches target_price (default 100.09) 
    within the first 1-5 minute window
    """
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
        
        # Define the 1-5 minute window
        window_start = start_time + min_window_sec  # 1 minute after start
        window_end = start_time + max_window_sec    # 5 minutes after start
        
        # Filter data for the 1-5 minute window
        df_window = df[(df['Time_sec'] >= window_start) & (df['Time_sec'] <= window_end)].copy()
        
        if len(df_window) < 2:
            print(f"Warning: Day {day_num} insufficient data in 1-5 min window. Skipping.")
            return None
        
        # Check if price touched target_price in the 1-5 minute window
        # "Touched" means price crossed or reached target_price
        touched_target = (
            (df_window[price_col] >= target_price).any() and 
            (df_window[price_col].min() < target_price)
        ) or (df_window[price_col] == target_price).any()
        
        if not touched_target:
            # Day doesn't meet criteria, skip it
            return None
        
        # If we reach here, price touched target_price in 1-5 min window
        # Now calculate full day statistics
        
        # Full day statistics
        full_day_mean = df[price_col].mean()
        full_day_sigma = df[price_col].std()
        full_day_high = df[price_col].max()
        full_day_low = df[price_col].min()
        opening_price = df.iloc[0][price_col]
        closing_price = df.iloc[-1][price_col]
        
        # Time when target was first touched
        touch_mask = (df['Time_sec'] >= window_start) & (df['Time_sec'] <= window_end)
        touch_idx = df[touch_mask & (df[price_col] >= target_price)].index
        if len(touch_idx) > 0:
            first_touch_idx = touch_idx[0]
            time_to_touch = df.loc[first_touch_idx, 'Time_sec'] - start_time
            price_at_touch = df.loc[first_touch_idx, price_col]
        else:
            time_to_touch = None
            price_at_touch = None
        
        # 5-minute window statistics (for comparison)
        df_5min = df[df['Time_sec'] <= (start_time + 300)].copy()
        mean_5min = df_5min[price_col].mean()
        sigma_5min = df_5min[price_col].std()
        price_at_5min = df_5min[price_col].iloc[-1]
        
        # Direction analysis
        close_above_100 = closing_price > 100
        close_above_target = closing_price > target_price
        
        # Volatility analysis - percentage of time spent above/below 100
        pct_time_above_100 = (df[price_col] > 100).sum() / len(df) * 100
        pct_time_below_100 = (df[price_col] < 100).sum() / len(df) * 100
        
        # Peak-to-peak analysis
        peak_to_peak = full_day_high - full_day_low
        
        # Post-touch behavior (rest of day after 5 min mark)
        df_post = df[df['Time_sec'] > window_end].copy()
        if len(df_post) > 0:
            post_mean = df_post[price_col].mean()
            post_sigma = df_post[price_col].std()
            post_high = df_post[price_col].max()
            post_low = df_post[price_col].min()
        else:
            post_mean = None
            post_sigma = None
            post_high = None
            post_low = None
        
        result = {
            'Day': day_num,
            'Touched_Target_in_1to5min': True,
            'Time_to_Touch_sec': time_to_touch if time_to_touch else np.nan,
            'Price_at_Touch': price_at_touch if price_at_touch else np.nan,
            
            # Full day statistics
            'Full_Day_Mean': round(full_day_mean, 4),
            'Full_Day_Sigma': round(full_day_sigma, 4),
            'Full_Day_High': round(full_day_high, 4),
            'Full_Day_Low': round(full_day_low, 4),
            'Peak_to_Peak': round(peak_to_peak, 4),
            'Opening_Price': round(opening_price, 4),
            'Closing_Price': round(closing_price, 4),
            
            # 5-minute window statistics
            'Mean_5min': round(mean_5min, 4),
            'Sigma_5min': round(sigma_5min, 4),
            'Price_at_5min': round(price_at_5min, 4),
            
            # Direction indicators
            'Close_Above_100': close_above_100,
            'Close_Above_Target': close_above_target,
            
            # Time distribution
            'Pct_Time_Above_100': round(pct_time_above_100, 2),
            'Pct_Time_Below_100': round(pct_time_below_100, 2),
            
            # Post-touch statistics
            'Post_Touch_Mean': round(post_mean, 4) if post_mean else np.nan,
            'Post_Touch_Sigma': round(post_sigma, 4) if post_sigma else np.nan,
            'Post_Touch_High': round(post_high, 4) if post_high else np.nan,
            'Post_Touch_Low': round(post_low, 4) if post_low else np.nan,
            
            # Data points
            'Num_Points_Total': len(df),
            'Num_Points_5min': len(df_5min),
            'Num_Points_Post': len(df_post) if len(df_post) > 0 else 0,
        }
        
        print(f"✓ Day{day_num:3d} | Touched @{time_to_touch}s | μ_full: {full_day_mean:7.2f} | σ_full: {full_day_sigma:6.2f} | Close: {closing_price:7.2f}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing day {day_num} ({file_path}): {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_all_days_statistics(file_path: str, day_num: int,
                                  time_col: str = 'Time',
                                  price_col: str = 'Price') -> dict:
    """Calculate basic statistics for all days (for comparison)"""
    try:
        required_cols = [time_col, price_col]
        ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
        gdf = ddf.compute()
        df = gdf.to_pandas()
        
        if df.empty or len(df) < 10:
            return None
        
        return {
            'Day': day_num,
            'Full_Day_Mean': df[price_col].mean(),
            'Full_Day_Sigma': df[price_col].std(),
            'Closing_Price': df.iloc[-1][price_col]
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze market behavior when price touches 100.09 in first 1-5 minutes")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/data/quant14/EBX/',
        help="Directory containing the 'day*.parquet' files"
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/raid/Quant14/V_Feature_Analysis/trend_strat/ebx/touch_100_09_analysis.csv',
        help="Output CSV file path"
    )
    parser.add_argument(
        '--target-price',
        type=float,
        default=100.09,
        help="Target price to check for touch (default: 100.09)"
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    print("="*80)
    print(f"PRICE TOUCH {args.target_price} ANALYSIS (1-5 MINUTE WINDOW)")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Target price: {args.target_price}")
    print(f"Max workers: {args.max_workers}")
    
    # Find input files
    files_pattern = os.path.join(args.data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    if not sorted_files:
        print(f"\nERROR: No parquet files found in {args.data_dir}")
        return
    
    print(f"\nFound {len(sorted_files)} day files to process")
    print(f"\nStep 1: Calculating statistics for ALL days (for comparison)...\n")
    
    # First pass: Get statistics for ALL days
    all_days_results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                calculate_all_days_statistics,
                f,
                extract_day_num(f)
            ): f
            for f in sorted_files
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    all_days_results.append(result)
            except Exception as e:
                pass
    
    # Calculate overall statistics
    if all_days_results:
        df_all_days = pd.DataFrame(all_days_results)
        overall_mean_mu = df_all_days['Full_Day_Mean'].mean()
        overall_median_mu = df_all_days['Full_Day_Mean'].median()
        overall_mean_sigma = df_all_days['Full_Day_Sigma'].mean()
        overall_median_sigma = df_all_days['Full_Day_Sigma'].median()
        overall_mean_close = df_all_days['Closing_Price'].mean()
        total_days_analyzed = len(df_all_days)
        
        print(f"✓ All days statistics calculated (n={total_days_analyzed}):")
        print(f"  Overall μ (mean):    {overall_mean_mu:.4f}")
        print(f"  Overall μ (median):  {overall_median_mu:.4f}")
        print(f"  Overall σ (mean):    {overall_mean_sigma:.4f}")
        print(f"  Overall σ (median):  {overall_median_sigma:.4f}")
        print(f"  Overall close (mean):{overall_mean_close:.4f}\n")
    else:
        print("WARNING: Could not calculate overall statistics")
        overall_mean_mu = None
        overall_median_mu = None
        overall_mean_sigma = None
        overall_median_sigma = None
        overall_mean_close = None
        total_days_analyzed = 0
    
    print(f"Step 2: Analyzing days where price touched {args.target_price} in 1-5 min window...\n")
    
    # Process all days in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                analyze_single_day_touch_100_09, 
                f, 
                extract_day_num(f),
                target_price=args.target_price
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
        print(f"\nNo days found where price touched {args.target_price} in the 1-5 minute window")
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
        f.write(f"PRICE TOUCH {args.target_price} ANALYSIS SUMMARY (1-5 MINUTE WINDOW)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total days where price touched {args.target_price} in 1-5 min window: {len(df_all)}\n")
        f.write(f"Total days in dataset: {total_days_analyzed}\n")
        f.write(f"Percentage of days with touch: {len(df_all)/total_days_analyzed*100:.2f}%\n\n")
        
        # ========================================================================
        # COMPARISON WITH ALL DAYS
        # ========================================================================
        f.write("="*80 + "\n")
        f.write("COMPARISON: TOUCH DAYS vs ALL DAYS\n")
        f.write("="*80 + "\n\n")
        
        if overall_mean_mu is not None:
            touch_mean_mu = df_all['Full_Day_Mean'].mean()
            touch_median_mu = df_all['Full_Day_Mean'].median()
            touch_mean_sigma = df_all['Full_Day_Sigma'].mean()
            touch_median_sigma = df_all['Full_Day_Sigma'].median()
            touch_mean_close = df_all['Closing_Price'].mean()
            
            f.write("MEAN (μ) COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Touch days μ (mean):      {touch_mean_mu:.4f}\n")
            f.write(f"  All days μ (mean):        {overall_mean_mu:.4f}\n")
            f.write(f"  Difference:               {touch_mean_mu - overall_mean_mu:+.4f} ({(touch_mean_mu - overall_mean_mu)/overall_mean_mu*100:+.2f}%)\n\n")
            
            f.write(f"  Touch days μ (median):    {touch_median_mu:.4f}\n")
            f.write(f"  All days μ (median):      {overall_median_mu:.4f}\n")
            f.write(f"  Difference:               {touch_median_mu - overall_median_mu:+.4f} ({(touch_median_mu - overall_median_mu)/overall_median_mu*100:+.2f}%)\n\n")
            
            f.write("VOLATILITY (σ) COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Touch days σ (mean):      {touch_mean_sigma:.4f}\n")
            f.write(f"  All days σ (mean):        {overall_mean_sigma:.4f}\n")
            f.write(f"  Difference:               {touch_mean_sigma - overall_mean_sigma:+.4f} ({(touch_mean_sigma - overall_mean_sigma)/overall_mean_sigma*100:+.2f}%)\n\n")
            
            f.write(f"  Touch days σ (median):    {touch_median_sigma:.4f}\n")
            f.write(f"  All days σ (median):      {overall_median_sigma:.4f}\n")
            f.write(f"  Difference:               {touch_median_sigma - overall_median_sigma:+.4f} ({(touch_median_sigma - overall_median_sigma)/overall_median_sigma*100:+.2f}%)\n\n")
            
            f.write("CLOSING PRICE COMPARISON:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Touch days close (mean):  {touch_mean_close:.4f}\n")
            f.write(f"  All days close (mean):    {overall_mean_close:.4f}\n")
            f.write(f"  Difference:               {touch_mean_close - overall_mean_close:+.4f} ({(touch_mean_close - overall_mean_close)/overall_mean_close*100:+.2f}%)\n\n")
            
            # Statistical significance note
            from scipy import stats
            
            # T-test for mean
            all_days_mu_list = df_all_days['Full_Day_Mean'].tolist()
            touch_days_mu_list = df_all['Full_Day_Mean'].tolist()
            t_stat_mu, p_val_mu = stats.ttest_ind(touch_days_mu_list, all_days_mu_list)
            
            # T-test for sigma
            all_days_sigma_list = df_all_days['Full_Day_Sigma'].tolist()
            touch_days_sigma_list = df_all['Full_Day_Sigma'].tolist()
            t_stat_sigma, p_val_sigma = stats.ttest_ind(touch_days_sigma_list, all_days_sigma_list)
            
            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  μ difference t-test p-value:  {p_val_mu:.6f} {'(SIGNIFICANT)' if p_val_mu < 0.05 else '(not significant)'}\n")
            f.write(f"  σ difference t-test p-value:  {p_val_sigma:.6f} {'(SIGNIFICANT)' if p_val_sigma < 0.05 else '(not significant)'}\n\n")
        
        # Touch timing statistics
        f.write("="*80 + "\n")
        f.write("TOUCH TIMING STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average time to touch:     {df_all['Time_to_Touch_sec'].mean():.1f} seconds\n")
        f.write(f"Median time to touch:      {df_all['Time_to_Touch_sec'].median():.1f} seconds\n")
        f.write(f"Min time to touch:         {df_all['Time_to_Touch_sec'].min():.1f} seconds\n")
        f.write(f"Max time to touch:         {df_all['Time_to_Touch_sec'].max():.1f} seconds\n\n")
        
        # Volatility statistics (Full Day)
        f.write("="*80 + "\n")
        f.write("VOLATILITY STATISTICS (FULL DAY)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average σ (full day):      {df_all['Full_Day_Sigma'].mean():.4f}\n")
        f.write(f"Median σ (full day):       {df_all['Full_Day_Sigma'].median():.4f}\n")
        f.write(f"Min σ (full day):          {df_all['Full_Day_Sigma'].min():.4f}\n")
        f.write(f"Max σ (full day):          {df_all['Full_Day_Sigma'].max():.4f}\n")
        f.write(f"25th Percentile σ:         {df_all['Full_Day_Sigma'].quantile(0.25):.4f}\n")
        f.write(f"75th Percentile σ:         {df_all['Full_Day_Sigma'].quantile(0.75):.4f}\n\n")
        
        f.write(f"Average peak-to-peak:      {df_all['Peak_to_Peak'].mean():.4f}\n")
        f.write(f"Median peak-to-peak:       {df_all['Peak_to_Peak'].median():.4f}\n\n")
        
        # Mean statistics (Full Day)
        f.write("="*80 + "\n")
        f.write("MEAN PRICE STATISTICS (FULL DAY)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average μ (full day):      {df_all['Full_Day_Mean'].mean():.4f}\n")
        f.write(f"Median μ (full day):       {df_all['Full_Day_Mean'].median():.4f}\n")
        f.write(f"Min μ (full day):          {df_all['Full_Day_Mean'].min():.4f}\n")
        f.write(f"Max μ (full day):          {df_all['Full_Day_Mean'].max():.4f}\n\n")
        
        days_mean_above_100 = (df_all['Full_Day_Mean'] > 100).sum()
        days_mean_below_100 = (df_all['Full_Day_Mean'] < 100).sum()
        f.write(f"Days with μ > 100:         {days_mean_above_100:4d} ({days_mean_above_100/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days with μ < 100:         {days_mean_below_100:4d} ({days_mean_below_100/len(df_all)*100:5.1f}%)\n\n")
        
        # Closing price statistics
        f.write("="*80 + "\n")
        f.write("CLOSING PRICE STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average closing price:     {df_all['Closing_Price'].mean():.4f}\n")
        f.write(f"Median closing price:      {df_all['Closing_Price'].median():.4f}\n")
        f.write(f"Min closing price:         {df_all['Closing_Price'].min():.4f}\n")
        f.write(f"Max closing price:         {df_all['Closing_Price'].max():.4f}\n\n")
        
        days_close_above_100 = df_all['Close_Above_100'].sum()
        days_close_above_target = df_all['Close_Above_Target'].sum()
        f.write(f"Days closing above 100:    {days_close_above_100:4d} ({days_close_above_100/len(df_all)*100:5.1f}%)\n")
        f.write(f"Days closing above {args.target_price}:  {days_close_above_target:4d} ({days_close_above_target/len(df_all)*100:5.1f}%)\n\n")
        
        # 5-minute window statistics
        f.write("="*80 + "\n")
        f.write("5-MINUTE WINDOW STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average μ (5-min):         {df_all['Mean_5min'].mean():.4f}\n")
        f.write(f"Average σ (5-min):         {df_all['Sigma_5min'].mean():.4f}\n")
        f.write(f"Average price@5min:        {df_all['Price_at_5min'].mean():.4f}\n\n")
        
        # Post-touch behavior
        f.write("="*80 + "\n")
        f.write("POST-TOUCH BEHAVIOR (After 5 minutes)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average μ (post-touch):    {df_all['Post_Touch_Mean'].mean():.4f}\n")
        f.write(f"Average σ (post-touch):    {df_all['Post_Touch_Sigma'].mean():.4f}\n")
        f.write(f"Average high (post-touch): {df_all['Post_Touch_High'].mean():.4f}\n")
        f.write(f"Average low (post-touch):  {df_all['Post_Touch_Low'].mean():.4f}\n\n")
        
        # Time distribution
        f.write("="*80 + "\n")
        f.write("TIME DISTRIBUTION (FULL DAY)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average % time above 100:  {df_all['Pct_Time_Above_100'].mean():.2f}%\n")
        f.write(f"Average % time below 100:  {df_all['Pct_Time_Below_100'].mean():.2f}%\n\n")
        
        # Volatility categories
        f.write("="*80 + "\n")
        f.write("VOLATILITY DISTRIBUTION\n")
        f.write("="*80 + "\n\n")
        q1 = df_all['Full_Day_Sigma'].quantile(0.25)
        q2 = df_all['Full_Day_Sigma'].quantile(0.50)
        q3 = df_all['Full_Day_Sigma'].quantile(0.75)
        
        low_vol = (df_all['Full_Day_Sigma'] < q1).sum()
        med_low_vol = ((df_all['Full_Day_Sigma'] >= q1) & (df_all['Full_Day_Sigma'] < q2)).sum()
        med_high_vol = ((df_all['Full_Day_Sigma'] >= q2) & (df_all['Full_Day_Sigma'] < q3)).sum()
        high_vol = (df_all['Full_Day_Sigma'] >= q3).sum()
        
        f.write(f"Low volatility (σ < {q1:.4f}):       {low_vol:4d} ({low_vol/len(df_all)*100:5.1f}%)\n")
        f.write(f"Med-low volatility:                   {med_low_vol:4d} ({med_low_vol/len(df_all)*100:5.1f}%)\n")
        f.write(f"Med-high volatility:                  {med_high_vol:4d} ({med_high_vol/len(df_all)*100:5.1f}%)\n")
        f.write(f"High volatility (σ >= {q3:.4f}):     {high_vol:4d} ({high_vol/len(df_all)*100:5.1f}%)\n\n")
        
        # Correlation analysis
        f.write("="*80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        corr_time_vs_close = df_all['Time_to_Touch_sec'].corr(df_all['Closing_Price'])
        corr_time_vs_vol = df_all['Time_to_Touch_sec'].corr(df_all['Full_Day_Sigma'])
        corr_vol_vs_close = df_all['Full_Day_Sigma'].corr(df_all['Closing_Price'])
        
        f.write(f"Correlation (touch time vs closing price):  {corr_time_vs_close:.4f}\n")
        f.write(f"Correlation (touch time vs volatility):     {corr_time_vs_vol:.4f}\n")
        f.write(f"Correlation (volatility vs closing price):  {corr_vol_vs_close:.4f}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"  CSV output:     {args.output_file}")
    print(f"  Summary output: {summary_file}")
    print(f"  Days found:     {len(df_all)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()