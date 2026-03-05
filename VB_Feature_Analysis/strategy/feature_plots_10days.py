import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# --- Configuration ---
BASE_DATA_PATH = '/data/quant14/EBX/'
START_DAY = 100
NUM_DAYS = 10 # This will run for days 100, 101, ..., 109

# Get current script directory for saving files
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd() # Fallback to current working directory

print(f"Saving output files to: {script_dir}")

# Define VB buckets
vb_buckets = ['VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6']

# --- Load and Combine Data for All Days ---
all_dfs = []
total_time_offset = timedelta(seconds=0)
first_day_start_time = None

for i in range(NUM_DAYS):
    day_num = START_DAY + i
    data_path = os.path.join(BASE_DATA_PATH, f'day{day_num}.csv')

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"\nFile not found: {data_path}. Skipping day {day_num}.")
        continue

    print(f"Loading data for Day {day_num}...")
    df_day = pd.read_csv(data_path)

    # Convert Time to datetime
    if df_day['Time'].dtype == 'object':
        df_day['Time'] = pd.to_datetime(df_day['Time'])
    
    # Calculate time offset for concatenation
    if first_day_start_time is None:
        first_day_start_time = df_day['Time'].iloc[0]

    # Adjust time to be continuous
    # We add a cumulative timedelta to each day's time column
    # The actual date part will change, but the relative time will be continuous
    df_day['Time_Continuous'] = df_day['Time'] + total_time_offset
    total_time_offset += (df_day['Time'].iloc[-1] - df_day['Time'].iloc[0]) + timedelta(seconds=1) # Add 1 second gap between days

    all_dfs.append(df_day)

if not all_dfs:
    print("No data loaded for any day. Exiting.")
    exit()

df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined['Time'] = df_combined['Time_Continuous'] # Use the continuous time for plotting
print(f"Combined data for {len(all_dfs)} days. Total rows: {len(df_combined)}")

# Extract VB columns (VB1_T1 through VB6_T12) from one of the daily DFs (assuming they are consistent)
vb_columns = [col for col in all_dfs[0].columns if col.startswith('VB') and '_T' in col]

if not vb_columns:
    print("No VB features found in the combined data. Exiting.")
    exit()
    
# Normalize VB features according to price on the combined DataFrame
df_normalized = df_combined.copy()
for col in vb_columns:
    df_normalized[f'{col}_norm'] = df_normalized[col] / df_normalized['Price']

# --- Generate Combined Plots ---

# Create individual bucket plots for the combined data
for bucket in vb_buckets:
    bucket_cols = [col for col in vb_columns if col.startswith(f'{bucket}_')]
    
    if not bucket_cols:
        continue
    
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_normalized['Time'],
            y=df_normalized['Price'],
            name='Price',
            line=dict(color='black', width=2),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    for col in bucket_cols:
        fig.add_trace(
            go.Scatter(
                x=df_normalized['Time'],
                y=df_normalized[f'{col}_norm'],
                name=f'{col} (normalized)',
                mode='lines',
                opacity=0.7
            ),
            secondary_y=False
        )
    
    fig.update_layout(
        title=f'{bucket} Features Normalized by Price - Days {START_DAY}-{START_DAY + NUM_DAYS - 1}',
        xaxis_title='Time (Continuous)',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    
    fig.update_yaxes(title_text="Normalized Feature Value", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True)
    
    output_path = os.path.join(script_dir, f'{bucket}_features_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.html')
    fig.write_html(output_path)
    print(f"Saved combined {bucket} plot to: {output_path}")

# Create a comprehensive plot with all VB features for the combined data
fig_all = make_subplots(
    rows=1, cols=1,
    specs=[[{"secondary_y": True}]]
)

fig_all.add_trace(
    go.Scatter(
        x=df_normalized['Time'],
        y=df_normalized['Price'],
        name='Price',
        line=dict(color='black', width=3),
        yaxis='y2'
    ),
    secondary_y=True
)

for col in vb_columns:
    fig_all.add_trace(
        go.Scatter(
            x=df_normalized['Time'],
            y=df_normalized[f'{col}_norm'],
            name=f'{col}',
            mode='lines',
            opacity=0.3
        ),
        secondary_y=False
    )

fig_all.update_layout(
    title=f'All VB Features Normalized by Price - Days {START_DAY}-{START_DAY + NUM_DAYS - 1}',
    xaxis_title='Time (Continuous)',
    hovermode='x unified',
    height=700,
    showlegend=True
)

fig_all.update_yaxes(title_text="Normalized Feature Value", secondary_y=False)
fig_all.update_yaxes(title_text="Price", secondary_y=True)

output_path_all = os.path.join(script_dir, f'all_VB_features_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.html')
fig_all.write_html(output_path_all)
print(f"Saved comprehensive combined plot to: {output_path_all}")

print("\n--- All Analysis Complete ---")

# --- Generate Trading Strategy Document (Runs Once) ---
strategy_text = f"""
# VB Features Trading Strategy - Days {START_DAY}-{START_DAY + NUM_DAYS - 1} Analysis

## Strategy Overview
Based on the Volume Bucket (VB) features normalized by price, this strategy identifies trading opportunities 
by analyzing volume distribution patterns across different price buckets. This analysis covers a continuous period 
from day {START_DAY} to day {START_DAY + NUM_DAYS - 1}.

## Key Insights from VB Features

### 1. Volume Bucket Structure
- VB1-VB6: Represent volume buckets at different price levels
- T1-T12: Time-based aggregations within each bucket
- Normalized values show relative volume intensity compared to current price

### 2. Trading Signals

#### BUY Signals:
1. **Volume Accumulation**: When normalized VB features in lower buckets (VB1-VB3) show increasing 
   trends while price is stable or declining, it indicates accumulation.
   
2. **Cross-Bucket Divergence**: When lower bucket volumes increase while upper bucket volumes 
   (VB4-VB6) decrease, suggesting buying pressure building.

3. **Time-Based Patterns**: Look for consistent increases across T1-T6 timeframes in lower buckets 
   indicating sustained buying interest.

#### SELL Signals:
1. **Distribution Pattern**: Rising normalized volumes in upper buckets (VB4-VB6) with declining 
   lower bucket volumes suggests distribution.
   
2. **Volume Exhaustion**: Sharp spikes in normalized VB features followed by rapid decline often 
   precede price reversals.

3. **Divergence Warning**: When price rises but normalized volume features across all buckets 
   decline, indicating weakening momentum.

### 3. Risk Management

- **Stop Loss**: Set at 0.5% below entry for buy trades, 0.5% above for sell trades
- **Position Sizing**: Use volume intensity (sum of normalized VB features) to determine position size
  - High intensity (>threshold): Reduce position size
  - Low intensity: Increase position size

- **Time-Based Exit**: Close positions if no significant price movement within timeframes indicated 
  by T1-T12 patterns

### 4. Implementation Rules

1. Calculate moving averages of normalized VB features across T1-T6 and T7-T12 separately
2. Generate signal when short-term (T1-T6) MA crosses long-term (T7-T12) MA
3. Confirm signal with price action alignment
4. Enter trade only when multiple VB buckets confirm the signal

### 5. Backtesting Recommendations

- Test on multiple days to validate consistency
- Optimize bucket weights based on historical performance
- Adjust normalization factor if price ranges vary significantly
- Monitor correlation between VB features and price changes

## Files Generated
- Individual bucket plots: VB1_features_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.html through VB6_features_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.html
- Comprehensive plot: all_VB_features_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.html

## Next Steps
1. Review the generated HTML plots to identify visual patterns
2. Calculate correlation between VB features and future price movements
3. Backtest the strategy rules on historical data
4. Optimize parameters based on performance metrics
"""

# Save strategy document
strategy_path = os.path.join(script_dir, f'VB_trading_strategy_days_{START_DAY}_to_{START_DAY + NUM_DAYS - 1}.txt')
with open(strategy_path, 'w') as f:
    f.write(strategy_text)

print(f"\nStrategy document saved to: {strategy_path}")
print(f"\nAnalysis complete! Generated {len(vb_buckets) + 1} combined HTML plots and 1 strategy document.")
print(f"All files saved in: {script_dir}")
