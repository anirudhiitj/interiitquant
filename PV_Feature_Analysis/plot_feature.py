import dask.dataframe as dd
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = '/data/quant14/EBX/combined.csv'
OUTPUT_DIR = './Feature_Plots' # Changed output dir to avoid overwriting
PLOT_WIDTH = 1600
PLOT_HEIGHT = 800
MAX_POINTS_PER_TRACE = 40000  # Maximum points per line for interactive HTML

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def lttb_downsample(x, y, threshold):
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.
    Keeps the most visually significant points while reducing data size.
    
    Parameters:
    -----------
    x : array-like
        X-axis values
    y : array-like
        Y-axis values   
    threshold : int
        Target number of points to keep
        
    Returns:
    --------
    x_sampled, y_sampled : arrays
        Downsampled data
    """
    if len(x) <= threshold:
        return x, y
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Always keep first and last points
    sampled_x = [x[0]]
    sampled_y = [y[0]]
    
    # Bucket size
    every = (len(x) - 2) / (threshold - 2)
    
    a = 0  # Initially a is the first point
    
    for i in range(threshold - 2):
        # Calculate point average for next bucket
        avg_x = 0
        avg_y = 0
        avg_range_start = int(np.floor((i + 1) * every) + 1)
        avg_range_end = int(np.floor((i + 2) * every) + 1)
        avg_range_end = min(avg_range_end, len(x))
        
        avg_range_length = avg_range_end - avg_range_start
        
        while avg_range_start < avg_range_end:
            avg_x += x[avg_range_start]
            avg_y += y[avg_range_start]
            avg_range_start += 1
        
        avg_x /= avg_range_length
        avg_y /= avg_range_length
        
        # Get the range for this bucket
        range_offs = int(np.floor((i + 0) * every) + 1)
        range_to = int(np.floor((i + 1) * every) + 1)
        
        # Point a
        point_a_x = x[a]
        point_a_y = y[a]
        
        max_area = -1
        next_a = range_offs
        
        for j in range(range_offs, range_to):
            # Calculate triangle area
            area = abs((point_a_x - avg_x) * (y[j] - point_a_y) - 
                       (point_a_x - x[j]) * (avg_y - point_a_y))
            
            if area > max_area:
                max_area = area
                next_a = j
        
        sampled_x.append(x[next_a])
        sampled_y.append(y[next_a])
        a = next_a
    
    # Always add last point
    sampled_x.append(x[-1])
    sampled_y.append(y[-1])
    
    return np.array(sampled_x), np.array(sampled_y)

# Determine columns to load
print("Determining columns to load...")
sample_df = pd.read_csv(CSV_FILE_PATH, nrows=1)
all_columns = sample_df.columns.tolist()
columns_to_load = ['Time', 'Price'] + [col for col in all_columns if col.startswith('PV')]
print(f"Loading {len(columns_to_load)} columns: Time, Price, and {len(columns_to_load)-2} PV features")

# Load with Dask for parallel processing
print("\nLoading CSV file with Dask...")
ddf = dd.read_csv(
    CSV_FILE_PATH,
    usecols=columns_to_load,
    assume_missing=True,
    dtype={col: 'float64' for col in columns_to_load if col != 'Time'}
)

print(f"CSV loaded lazily. Partitions: {ddf.npartitions}")

# Drop null Price values
print("Filtering null Price values...")
ddf = ddf.dropna(subset=['Price'])

# Convert Time to timedelta
print("Converting Time to timedelta...")
ddf['Time'] = dd.to_timedelta(ddf['Time'])

# Convert all PV columns to numeric (coerce errors)
print("Converting PV columns to numeric...")
pv_columns = [col for col in columns_to_load if col.startswith('PV')]
for col in pv_columns:
    ddf[col] = dd.to_numeric(ddf[col], errors='coerce')

# Persist in memory for faster access
print("Persisting dataframe in distributed memory...")
ddf = ddf.persist()

# Compute to pandas for sequential operations (day calculation)
print("Computing to pandas for day rollover detection...")
df = ddf.compute()

print(f"Loaded {len(df):,} rows")

# Calculate day rollovers - when time hits 00:00:00 (or resets to a small value)
print("Calculating day numbers (new day when time hits 00:00:00)...")
df['TimeSeconds'] = df['Time'].dt.total_seconds()

# Detect day rollover: when current time is near 0 (within 5 seconds) 
# AND previous time was much larger (> 60 seconds)
# This catches the transition like 06:13:xx -> 00:00:00
time_near_zero = df['TimeSeconds'] < 5.0  # Current time is at or near midnight
prev_time_large = df['TimeSeconds'].shift(1) > 60.0  # Previous time was not near midnight
is_new_day = time_near_zero & prev_time_large

# Cumulative sum to get day numbers (starts at 0)
df['Day'] = is_new_day.cumsum()

# Create fractional day for continuous x-axis
# Day 0 goes from 0.0 to 0.999, Day 1 from 1.0 to 1.999, etc.
# Find max time per day to normalize properly
day_max_times = df.groupby('Day')['TimeSeconds'].max()
df['DayDuration'] = df['Day'].map(day_max_times)

# Create fractional day: Day number + (current_time / day_duration)
df['FractionalDay'] = df['Day'] + (df['TimeSeconds'] / df['DayDuration'])

num_rows = len(df)
num_days = int(df['Day'].max()) + 1
print(f"\nDataset Statistics:")
print(f"  Total rows: {num_rows:,}")
print(f"  Number of days detected: {num_days}")
print(f"  Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
print(f"  Target points per trace: {MAX_POINTS_PER_TRACE:,}")

# Show day boundaries for verification
print(f"\nDay boundary verification:")
for day in range(min(5, num_days)):  # Show first 5 days
    day_data = df[df['Day'] == day]
    if len(day_data) > 0:
        start_time = day_data['Time'].iloc[0]
        end_time = day_data['Time'].iloc[-1]
        print(f"  Day {day}: {start_time} to {end_time} ({len(day_data):,} rows)")

# Calculate downsampling ratio
if num_rows > MAX_POINTS_PER_TRACE:
    downsample_ratio = num_rows / MAX_POINTS_PER_TRACE
    print(f"\nDownsampling ratio: {downsample_ratio:.1f}x (from {num_rows:,} to ~{MAX_POINTS_PER_TRACE:,} points)")
else:
    print(f"\nNo downsampling needed")
print()

# *** NORMALIZATION FUNCTION REMOVED ***

def create_plotly_chart(df, price_col, feature_cols, title, filename):
    """Create an interactive Plotly line chart with smart downsampling"""
    print(f"Creating: {title}")
    
    # Filter valid features
    valid_features = []
    for col in feature_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            valid_features.append(col)
        else:
            print(f"  Skipping {col} - no valid data")
    
    if not valid_features:
        print(f"  No valid features to plot!")
        return
    
    print(f"  Plotting {len(valid_features)} features")
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    
    # Add features first (so price is on top)
    for i, col in enumerate(valid_features):
        print(f"  Processing {col}...")
        
        # Get clean data (no NaNs)
        mask = df[col].notna() & df['FractionalDay'].notna()
        x_data = df.loc[mask, 'FractionalDay'].values
        
        # *** CHANGED: Use raw feature value, no normalization ***
        y_data = df.loc[mask, col].values
        
        if len(x_data) == 0:
            print(f"    Skipping {col} - no valid data after cleaning")
            continue
        
        # Downsample if needed
        if len(x_data) > MAX_POINTS_PER_TRACE:
            print(f"    Downsampling {len(x_data):,} -> {MAX_POINTS_PER_TRACE:,} points using LTTB...")
            x_data, y_data = lttb_downsample(x_data, y_data, MAX_POINTS_PER_TRACE)
        
        fig.add_trace(go.Scattergl(
            x=x_data,
            y=y_data,
            mode='lines',
            name=col,
            line=dict(
                color=colors[i % len(colors)], 
                width=2,
                shape='linear'
            ),
            opacity=0.75,
            hovertemplate=f'<b>{col}</b><br>Day: %{{x:.3f}}<br>Value: %{{y:.4f}}<extra></extra>',
            # *** CHANGED: Assign to secondary Y-axis ***
            yaxis="y2" 
        ))
    
    # Add price on top (thicker, black line)
    print(f"  Processing Price...")
    mask = df[price_col].notna() & df['FractionalDay'].notna()
    x_data = df.loc[mask, 'FractionalDay'].values
    y_data = df.loc[mask, price_col].values
    
    if len(x_data) > MAX_POINTS_PER_TRACE:
        print(f"    Downsampling {len(x_data):,} -> {MAX_POINTS_PER_TRACE:,} points using LTTB...")
        x_data, y_data = lttb_downsample(x_data, y_data, MAX_POINTS_PER_TRACE)
    
    fig.add_trace(go.Scattergl(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Price',
        line=dict(
            color='black', 
            width=2,
            shape='linear'
        ),
        opacity=0.95,
        hovertemplate='<b>Price</b><br>Day: %{x:.3f}<br>Value: %{y:.2f}<extra></extra>',
        # *** This trace uses the default yaxis (y1) ***
    ))
    
    # No day boundary markers - removed per user request
    
    # Update layout for better line chart appearance
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Price (Left Axis) vs. Features (Right Axis) | LTTB Downsampled</sub>", # Updated subtitle
            font=dict(size=22, color='#1f1f1f', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(
                text='Day',
                font=dict(size=16, family='Arial')
            ),
            gridcolor='rgba(200,200,200,0.4)',
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            tickmode='linear',
            tick0=0,
            dtick=20  # Show tick every 20 days
        ),
        # *** CHANGED: This is now the LEFT axis for Price ***
        yaxis=dict(
            title=dict(
                text='Price', # Updated title
                font=dict(size=16, family='Arial', color='black') # Match line color
            ),
            side='left', # Explicitly set side
            gridcolor='rgba(200,200,200,0.4)',
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        # *** CHANGED: Added new RIGHT axis for Features ***
        yaxis2=dict(
            title=dict(
                text='Feature Value', # New title
                font=dict(size=16, family='Arial')
            ),
            side='right', # Set side
            overlaying='y', # Overlay on top of the 'y' axis
            showgrid=False, # Hide secondary grid to avoid clutter
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#666',
            borderwidth=2,
            font=dict(size=12)
        ),
        margin=dict(l=100, r=150, t=120, b=80) # Keep large right margin for axis + legend
    )
    
    # Save as HTML with config for better interaction
    output_path = Path(OUTPUT_DIR) / filename
    print(f"  Saving to {output_path}...")
    fig.write_html(
        output_path, 
        config={
            'displayModeBar': True, 
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': filename.replace('.html', ''),
                'height': PLOT_HEIGHT,
                'width': PLOT_WIDTH,
                'scale': 2
            }
        }
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved successfully ({file_size_mb:.1f} MB)\n")

# Generate all plots
print("="*70)
print("GENERATING PLOTS (Dual Y-Axis)")
print("="*70)

# Plot 1: Price vs PV1_T1 through PV1_T12
print("\n=== Plot 1: Price vs PV1_T Lookbacks ===")
pv1_features = [f'PV1_T{i}' for i in range(1, 13)]
create_plotly_chart(df, 'Price', pv1_features, 
                    'Price (Left) vs PV1_T Lookbacks (Right)', 
                    'pv1_plot_full_dual_axis.html')

# Plot 2: Price vs PV2
print("=== Plot 2: Price vs PV2 ===")
pv2_features = ['PV2']
create_plotly_chart(df, 'Price', pv2_features, 
                    'Price (Left) vs PV2 (Right)', 
                    'pv2_plot_full_dual_axis.html')

# Plots 3-9: Price vs PV3 Branches B1 through B7
for branch_num in range(1, 8):
    print(f"=== Plot {branch_num + 2}: Price vs PV3 B{branch_num} ===")
    pv3_branch_features = [f'PV3_B{branch_num}_T{i}' for i in range(1, 13)]
    create_plotly_chart(df, 'Price', pv3_branch_features,
                        f'Price (Left) vs PV3 B{branch_num} (Right)',
                        f'pv3_b{branch_num}_plot_full_dual_axis.html')

print("="*70)
print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*70)
print(f"\nPlots saved to: {OUTPUT_DIR}/")
print(f"Total plots created: 9")
print(f"\nPlot info:")
print(f"  • Price is plotted on the LEFT Y-axis")
print(f"  • Features are plotted on the RIGHT Y-axis")
print(f"  • Each trace reduced to ~{MAX_POINTS_PER_TRACE:,} points using LTTB")
print(f"\nInteractive features:")
print("  • Zoom: Click and drag on chart")
print("  • Pan: Hold shift + click and drag")
print("  • Reset view: Double click anywhere")
print("  • Toggle traces: Click legend items")
print("  • Compare values: Hover over lines")
print("  • X-axis shows fractional days (0.0 = start of Day 0, 1.0 = start of Day 1, etc.)")
print("="*70)