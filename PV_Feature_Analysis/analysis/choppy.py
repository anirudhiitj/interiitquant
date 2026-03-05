import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Try to import acceleration libraries
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(nopython=True, fastmath=True):
        def decorator(func):
            return func
        return decorator

print(f"Acceleration: {'Numba JIT' if NUMBA_AVAILABLE else 'Standard Python'}")


# ============================================================================
# CHOPPINESS INDEX CALCULATION
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_choppiness_index(close_prices, highs, lows, period=10):
    """
    Calculate Choppiness Index using 60-second window
    
    Args:
        close_prices: Array of close prices (PB9_T4)
        highs: Array of high prices (PB10_T4)
        lows: Array of low prices (PB11_T4)
        period: Lookback period (default 60)
    
    Returns:
        Array of CI values
    """
    n = len(close_prices)
    ci = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        max_high = -np.inf
        min_low = np.inf
        
        # Calculate over the period
        for j in range(i - period + 1, i + 1):
            # True Range
            high_low = highs[j] - lows[j]
            
            if j > 0:
                high_close = abs(highs[j] - close_prices[j-1])
                low_close = abs(lows[j] - close_prices[j-1])
                tr = max(high_low, high_close, low_close)
            else:
                tr = high_low
            
            tr_sum += tr
            
            # Track max/min
            if highs[j] > max_high:
                max_high = highs[j]
            if lows[j] < min_low:
                min_low = lows[j]
        
        # Calculate CI
        range_hl = max_high - min_low
        
        if range_hl > 1e-10 and tr_sum > 1e-10:
            ci[i] = 100.0 * np.log10(tr_sum / range_hl) / np.log10(float(period))
        else:
            ci[i] = 50.0
    
    # Fill initial values
    for i in range(period):
        ci[i] = 50.0
    
    return ci


# ============================================================================
# COLOR ASSIGNMENT BASED ON CI REGIME
# ============================================================================

def assign_regime_colors(ci_values, lower_threshold=40, upper_threshold=60):
    """
    Assign colors based on CI regime
    
    Returns:
        List of colors for each data point
    """
    colors = []
    for ci in ci_values:
        if ci < lower_threshold:
            colors.append('green')  # Trending
        elif ci > upper_threshold:
            colors.append('red')    # Choppy
        else:
            colors.append('cyan')   # Transition
    return colors


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_process_day(day_num, data_dir):
    """
    Load a single day and calculate CI
    
    Args:
        day_num: Day number to load
        data_dir: Directory containing parquet files
    
    Returns:
        DataFrame with Time, Price, CI columns, or None if error
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            print(f"  ✗ Day {day_num}: File not found")
            return None
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Check required columns
        required_cols = ['Price', 'PB9_T4', 'PB10_T4', 'PB11_T4']
        if not all(col in df.columns for col in required_cols):
            print(f"  ✗ Day {day_num}: Missing required columns")
            return None
        
        # Get timestamp column
        timestamp_col = next((c for c in df.columns 
                            if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col is None:
            print(f"  ✗ Day {day_num}: No timestamp column found")
            return None
        
        # Extract required data
        df_clean = df[[timestamp_col, 'Price', 'PB9_T4', 'PB10_T4', 'PB11_T4']].copy()
        df_clean.columns = ['Time', 'Price', 'Close', 'High', 'Low']
        df_clean = df_clean.dropna()
        
        if len(df_clean) < 31:
            print(f"  ✗ Day {day_num}: Insufficient data ({len(df_clean)} rows)")
            return None
        
        # Calculate Choppiness Index
        close_prices = df_clean['Close'].values.astype(np.float64)
        highs = df_clean['High'].values.astype(np.float64)
        lows = df_clean['Low'].values.astype(np.float64)
        
        ci_values = calculate_choppiness_index(close_prices, highs, lows, period=60)
        df_clean['CI'] = ci_values
        
        # Calculate 5-second moving average
        df_clean['Price_MA5'] = df_clean['Price'].rolling(window=5, min_periods=1).mean()
        
        # Convert time to datetime for proper formatting
        try:
            df_clean['Time'] = pd.to_datetime(df_clean['Time'], format='mixed', errors='coerce')
        except:
            print(f"  ✗ Day {day_num}: Could not parse timestamps")
            return None
        
        print(f"  ✓ Day {day_num}: Loaded {len(df_clean):,} data points")
        return df_clean
        
    except Exception as e:
        print(f"  ✗ Day {day_num}: Error - {str(e)}")
        return None


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_ci_regime_plot(df, day_num, ci_lower=40, ci_upper=60):
    """
    Create dual subplot visualization:
    - Top (60%): Price with CI-based color coding
    - Bottom (40%): CI indicator with threshold lines
    
    Args:
        df: DataFrame with Time, Price, CI columns
        day_num: Day number for title
        ci_lower: Lower CI threshold (trending)
        ci_upper: Upper CI threshold (choppy)
    """
    # Create subplots with 60/40 height ratio
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=(f'Day {day_num} - Price (Color-coded by CI Regime)', 
                       'Choppiness Index'),
        vertical_spacing=0.1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    # Get regime colors
    colors = assign_regime_colors(df['CI'].values, ci_lower, ci_upper)
    
    # Calculate regime statistics
    trending_pct = (df['CI'] < ci_lower).sum() / len(df) * 100
    transition_pct = ((df['CI'] >= ci_lower) & (df['CI'] <= ci_upper)).sum() / len(df) * 100
    choppy_pct = (df['CI'] > ci_upper).sum() / len(df) * 100
    
    # ========================================================================
    # SUBPLOT 1 (60%): PRICE WITH COLOR-CODED SEGMENTS (OVERLAY APPROACH)
    # ========================================================================
    
    # LAYER 1: Add continuous baseline (smooth, always visible)
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Price'],
            mode='lines',
            line=dict(color='black', width=1.5),
            name='Price Baseline',
            showlegend=False,
            hovertemplate='<b>Price</b>: $%{y:,.2f}<br>' +
                         '<extra></extra>',
        ),
        row=1, col=1,
        secondary_y=False
    )
    
    # LAYER 2: Add 5-second moving average on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['Price_MA5'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='5s Moving Average',
            showlegend=True,
            hoverinfo='skip',
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    # LAYER 3: Add colored segments on top
    current_color = colors[0]
    segment_start = 0
    
    for i in range(1, len(df)):
        if colors[i] != current_color or i == len(df) - 1:
            # End of segment - plot it
            segment_end = i if i < len(df) - 1 else i + 1
            segment_df = df.iloc[segment_start:segment_end]
            
            # Determine label for legend (only first occurrence)
            if current_color == 'green':
                label = f'Trending (CI<{ci_lower}): {trending_pct:.1f}%'
            elif current_color == 'red':
                label = f'Choppy (CI>{ci_upper}): {choppy_pct:.1f}%'
            else:
                label = f'Transition ({ci_lower}-{ci_upper}): {transition_pct:.1f}%'
            
            # Check if this color already in legend
            showlegend = not any(trace.name == label for trace in fig.data)
            
            fig.add_trace(
                go.Scatter(
                    x=segment_df['Time'],
                    y=segment_df['Price'],
                    mode='lines',
                    line=dict(color=current_color, width=3),
                    name=label,
                    showlegend=showlegend,
                    hovertemplate='<b>Price</b>: $%{y:,.2f}<br>' +
                                 '<extra></extra>',
                    legendgroup=current_color
                ),
                row=1, col=1,
                secondary_y=False
            )
            
            # Start new segment
            segment_start = i
            current_color = colors[i]
    
    # ========================================================================
    # SUBPLOT 2 (40%): CHOPPINESS INDEX INDICATOR
    # ========================================================================
    
    # Background shading for regimes
    fig.add_hrect(
        y0=0, y1=ci_lower,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        row=2, col=1
    )
    
    fig.add_hrect(
        y0=ci_upper, y1=100,
        fillcolor="red", opacity=0.1,
        layer="below", line_width=0,
        row=2, col=1
    )
    
    # CI line
    fig.add_trace(
        go.Scatter(
            x=df['Time'],
            y=df['CI'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Choppiness Index',
            hovertemplate='<b>Time</b>: %{x|%H:%M:%S}<br>' +
                         '<b>CI</b>: %{y:.2f}<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Threshold lines
    fig.add_hline(
        y=ci_lower, line_dash="dash", line_color="green", 
        line_width=2, opacity=0.7,
        annotation_text=f"Trending Threshold ({ci_lower})",
        annotation_position="right",
        row=2, col=1
    )
    
    fig.add_hline(
        y=ci_upper, line_dash="dash", line_color="red",
        line_width=2, opacity=0.7,
        annotation_text=f"Choppy Threshold ({ci_upper})",
        annotation_position="right",
        row=2, col=1
    )
    
    # ========================================================================
    # LAYOUT CONFIGURATION
    # ========================================================================
    
    fig.update_xaxes(
        title_text="Time (HH:MM:SS)",
        tickformat='%H:%M:%S',
        row=2, col=1
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="CI Value", range=[0, 100], row=2, col=1)
    
    fig.update_layout(
        height=900,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title_text=f"<b>Day {day_num} - CI Regime Analysis</b><br>" +
                   f"<sub>{len(df):,} data points | " +
                   f"Trending: {trending_pct:.1f}% | " +
                   f"Transition: {transition_pct:.1f}% | " +
                   f"Choppy: {choppy_pct:.1f}%</sub>",
        title_x=0.5
    )
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    data_dir = '/data/quant14/EBY'
    days_to_plot = [104, 275]  # ← CHANGE THIS LIST TO PLOT DIFFERENT DAYS
    
    CI_LOWER = 75
    CI_UPPER = 91.5
    
    print("="*80)
    print("CHOPPINESS INDEX REGIME VISUALIZER")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Days to plot: {days_to_plot}")
    print(f"\nCI Thresholds:")
    print(f"  Trending: CI < {CI_LOWER} (GREEN)")
    print(f"  Transition: {CI_LOWER} ≤ CI ≤ {CI_UPPER} (GRAY)")
    print(f"  Choppy: CI > {CI_UPPER} (RED)")
    print("="*80)
    
    # Check data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n✗ ERROR: Directory not found: {data_dir}")
        return
    
    print(f"\n✓ Data directory found")
    
    # Process each day
    print(f"\nLoading and processing {len(days_to_plot)} days...\n")
    
    for day_num in days_to_plot:
        print(f"Processing Day {day_num}:")
        
        # Load and process data
        df = load_and_process_day(day_num, data_dir)
        
        if df is None:
            print(f"  Skipping Day {day_num}\n")
            continue
        
        # Create visualization
        print(f"  Creating visualization...")
        fig = create_ci_regime_plot(df, day_num, CI_LOWER, CI_UPPER)
        
        # Display plot
        fig.show()
        
        # Print statistics
        price_min = df['Price'].min()
        price_max = df['Price'].max()
        ci_mean = df['CI'].mean()
        ci_std = df['CI'].std()
        
        print(f"  Statistics:")
        print(f"    Price range: ${price_min:,.2f} - ${price_max:,.2f}")
        print(f"    CI mean: {ci_mean:.2f} ± {ci_std:.2f}")
        print(f"  ✓ Day {day_num} complete\n")
    
    print("="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()