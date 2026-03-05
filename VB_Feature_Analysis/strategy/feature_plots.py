import dask_cudf
import plotly.graph_objects as go
import cudf
import numpy as np
import pandas as pd

# Read the CSV file using dask_cudf
print("Reading CSV file with GPU acceleration...")
ddf = dask_cudf.read_csv('portfolio_weights.csv')
df = ddf.compute()  # Convert to cudf DataFrame
print(f"Loaded {len(df):,} rows")

# Process time and detect day transitions
print("Processing time data...")
# Convert time column to pandas for processing
time_pd = df["Time"].to_pandas()
time_pd = pd.to_timedelta(time_pd)
time_diff = time_pd.diff()
# [Keep all your code from line 1 to 18]
# ...
# time_diff = time_pd.diff()
day_transitions = (time_diff < pd.Timedelta(0)).astype(int).cumsum()

# ===================================================================
# FIX 1: Assign the NumPy array directly to bypass index alignment
# This fixes the "Cannot align indices" traceback
# ===================================================================
df["Day"] = day_transitions.values

# ===================================================================
# FIX 2: DELETE the entire "Calculating positions..." block.
# The 'Signal' column from your strategy *IS* the position.
# We will just use the 'Signal' column directly.
# ===================================================================
print("Using 'Signal' column as position data.")


# Get total number of days
total_days = int(df['Day'].max()) + 1
print(f"Total days: {total_days}")
print(f"Available days: 0 to {total_days - 1}")

# ============================================
# SPECIFY WHICH DAYS TO PLOT HERE
# ============================================
days_to_plot = [5, 14, 17, 117, 208, 358, 273, 39, 87, 130, 242, 290]  # Change this list to plot different days
# ============================================

print(f"\nPlotting {len(days_to_plot)} days: {days_to_plot}")

# Create a separate plot for each specified day
for day_num in days_to_plot:
    if day_num > df['Day'].max():
        print(f"\nDay {day_num} does not exist (max day is {df['Day'].max()}). Skipping...")
        continue
        
    print(f"\nCreating plot for Day {day_num}...")
    
    # Filter data for this day
    day_data = df[df['Day'] == day_num].reset_index(drop=True)
    
    if len(day_data) == 0:
        print(f"  No data for Day {day_num}. Skipping...")
        continue
    
    # Convert to pandas for plotting
    day_data_pd = day_data.to_pandas()
    
    print(f"  Day {day_num}: {len(day_data_pd):,} data points")
    
    # Create the plot
    fig = go.Figure()
    
    # Plot black price line
    fig.add_trace(
        go.Scatter(
            x=day_data_pd.index,
            y=day_data_pd['Price'],
            name='Price',
            line=dict(color='black', width=2.5),
            mode='lines',
            hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
        )
    )

    # ===================================================================
    # FIX 3: Find *trade entries* (where signal *changes*), not just state
    # ===================================================================
    
    # Get the previous signal, fill start-of-day with 0 (flat)
    day_data_pd['Signal_prev'] = day_data_pd['Signal'].shift(1).fillna(0)
    
    # Add green dots for +1 signals (long entry)
    # This is where Signal is 1 AND the previous signal was 0 or -1
    enter_long = day_data_pd[
        (day_data_pd['Signal'] == 1) & 
        (day_data_pd['Signal_prev'] <= 0)
    ]
    if len(enter_long) > 0:
        fig.add_trace(
            go.Scatter(
                x=enter_long.index,
                y=enter_long['Price'],
                name='Long Signal (+1)',
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                hovertemplate='<b>Long Entry</b><br>Price: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
            )
        )
    
    # Add red dots for -1 signals (short entry)
    # This is where Signal is -1 AND the previous signal was 0 or 1
    enter_short = day_data_pd[
        (day_data_pd['Signal'] == -1) & 
        (day_data_pd['Signal_prev'] >= 0)
    ]
    if len(enter_short) > 0:
        fig.add_trace(
            go.Scatter(
                x=enter_short.index,
                y=enter_short['Price'],
                name='Short Signal (-1)',
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                hovertemplate='<b>Short Entry</b><br>Price: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
            )
        )
    
    # [Your layout code from here down is fine, but I'm including it for completeness]
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f"Day {day_num} - Price vs Trading Signals ({len(day_data_pd):,} points)",
        xaxis_title="Index",
        yaxis=dict(
            title="Price ($)",
            side='left'
        ),
        # You weren't plotting position, but you can add it if you want
        # yaxis2=dict(
        #     title="Position",
        #     overlaying='y',
        #     side='right',
        #     range=[-1.5, 1.5]
        # ),
        height=800,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Show the plot
    fig.show()
    
    # Print day statistics
    long_signals = len(enter_long)
    short_signals = len(enter_short)
    
    print(f"  Long Entries: {long_signals}")
    print(f"  Short Entries: {short_signals}")
    print(f"  Price range: ${day_data_pd['Price'].min():.2f} - ${day_data_pd['Price'].max():.2f}")

print("\n=== OVERALL SUMMARY ===")
print(f"Total data points: {len(df):,}")
print(f"Total days: {total_days}")