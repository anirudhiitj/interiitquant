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
# DIRECTIONAL CONSISTENCY CUSUM
# ============================================================================

@jit(nopython=True, fastmath=True)
def calculate_directional_cusum(changes, drift=0.02, decay=0.9):
    """
    Calculate directional CUSUM on a pre-calculated 'changes' series.
    The opposing CUSUM now DECAYS instead of instantly resetting.
    
    *** MODIFIED: This function now accepts a 'changes' array directly ***
    *** instead of calculating P[i] - P[i-1] internally. ***
    """
    n = len(changes)
    
    # Initialize CUSUM tracking
    cusum_up = 0.0
    cusum_down = 0.0
    
    cusum_up_values = np.zeros(n, dtype=np.float64)
    cusum_down_values = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        change = changes[i] # Use the pre-calculated change
        
        # Upward move
        if change > drift:
            cusum_up += change
            cusum_down = max(0.0, cusum_down * decay) # Opposing side now decays
        
        # Downward move
        elif change < -drift:
            cusum_down += abs(change)
            cusum_up = max(0.0, cusum_up * decay) # Opposing side now decays
        
        # Small move (noise/consolidation) - both decay
        else:
            cusum_up = max(0.0, cusum_up * decay)
            cusum_down = max(0.0, cusum_down * decay)
        
        # Store values
        cusum_up_values[i] = cusum_up
        cusum_down_values[i] = cusum_down
    
    # Return tuple, removing the internally calculated 'PB9_T1_changes'
    return cusum_up_values, cusum_down_values


# ============================================================================
# NEW REGIME CLASSIFICATION & SEGMENTATION
# ============================================================================

def classify_regime_by_difference(cusum_up, cusum_down, 
                                  choppy_up_thresh, choppy_down_thresh,
                                  trending_up_thresh, trending_down_thresh):
    """
    Classifies every time step based on the CUSUM difference.
    This function supports threshold inputs as either floats or np.ndarrays.
    (This function is unchanged)
    """
    n = len(cusum_up)
    cusum_diff = cusum_up - cusum_down
    
    # Default to 'Transition'
    regimes = np.full(n, 'Transition', dtype='<U15')
    colors = np.full(n, 'yellow', dtype='<U10')

    # Prioritized Classification:
    
    # 1. Trending (Highest priority)
    is_trending_up = cusum_diff > trending_up_thresh
    is_trending_down = cusum_diff < trending_down_thresh
    
    regimes[is_trending_up] = 'Trending Up'
    colors[is_trending_up] = 'green'
    
    regimes[is_trending_down] = 'Trending Down'
    colors[is_trending_down] = 'red'

    # 2. Choppy (Second priority)
    is_choppy = (cusum_diff <= choppy_up_thresh) & (cusum_diff >= choppy_down_thresh)
    is_not_trending = ~is_trending_up & ~is_trending_down
    
    regimes[is_choppy & is_not_trending] = 'Choppy'
    colors[is_choppy & is_not_trending] = 'magenta'

    return regimes, colors, cusum_diff


def create_regime_segments(df, regimes, colors, cusum_diff):
    """
    Groups contiguous points of the same regime for plotting.
    (This function is unchanged but correct)
    """
    segments = []
    if len(regimes) == 0:
        return segments
        
    current_regime = regimes[0]
    current_color = colors[0]
    segment_start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime:
            segments.append({
                'start_idx': segment_start_idx,
                'end_idx': i - 1,
                'regime': current_regime,
                'color': current_color,
                'duration': (i - 1) - segment_start_idx + 1
            })
            current_regime = regimes[i]
            current_color = colors[i]
            segment_start_idx = i
            
    segments.append({
        'start_idx': segment_start_idx,
        'end_idx': len(regimes) - 1,
        'regime': current_regime,
        'color': current_color,
        'duration': (len(regimes) - 1) - segment_start_idx + 1
    })
    
    return segments

# *** NEW FUNCTION ***
def filter_and_merge_segments(segments, min_duration):
    """
    Filters out short-lived segments by merging them into the previous valid segment.
    Also merges consecutive segments that have the same regime.
    (This function is unchanged)
    """
    if not segments:
        return []

    filtered_segments = [segments[0].copy()]

    for i in range(1, len(segments)):
        current_segment = segments[i]
        last_valid_segment = filtered_segments[-1]

        # Check if current segment is too short (noise)
        if current_segment['duration'] < min_duration:
            # Absorb this short segment into the last valid one
            last_valid_segment['end_idx'] = current_segment['end_idx']
            last_valid_segment['duration'] += current_segment['duration']
        
        # Check if current segment (even if long) is the same as the last valid one
        elif current_segment['regime'] == last_valid_segment['regime']:
            # Merge consecutive segments of the same type
            last_valid_segment['end_idx'] = current_segment['end_idx']
            last_valid_segment['duration'] += current_segment['duration']
        
        # Otherwise, this is a new, valid, different segment
        else:
            filtered_segments.append(current_segment.copy())

    return filtered_segments
    

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_and_process_day(day_num, data_dir):
    """
    Load a single day's data
    (This function is unchanged)
    """
    try:
        file_path = Path(data_dir) / f'day{day_num}.parquet'
        if not file_path.exists():
            print(f"   ✗ Day {day_num}: File not found")
            return None
        
        df = pd.read_parquet(file_path)
        
        if 'PB9_T1' not in df.columns:
            print(f"   ✗ Day {day_num}: Missing PB9_T1 column")
            return None
        
        timestamp_col = next((c for c in df.columns 
                              if c.lower() in ['time', 'timestamp', 'datetime', 'date']), None)
        
        if timestamp_col is None:
            print(f"   ✗ Day {day_num}: No timestamp column found")
            return None
        
        df_clean = df[[timestamp_col, 'PB9_T1']].copy()
        df_clean.columns = ['Time', 'PB9_T1']
        df_clean = df_clean.dropna()
        
        # Need at least 6 rows for a 5-period SMA + 1 shift
        if len(df_clean) < 31: 
            print(f"   ✗ Day {day_num}: Insufficient data ({len(df_clean)} rows)")
            return None
        
        try:
            df_clean['Time'] = pd.to_datetime(df_clean['Time'], format='mixed', errors='coerce')
        except:
            print(f"   ✗ Day {day_num}: Could not parse timestamps")
            return None

        print(f"   ✓ Day {day_num}: Loaded {len(df_clean):,} data points")
        return df_clean
        
    except Exception as e:
        print(f"   ✗ Day {day_num}: Error - {str(e)}")
        return None


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_PB9_T1_cusum_plot(df, segments, day_num, cusum_diff,
                            choppy_up_thresh, choppy_down_thresh,
                            trending_up_thresh, trending_down_thresh,
                            ewm_span):
    """
    Create visualization with directional CUSUM regime detection
    (This function is unchanged)
    """
    # Calculate regime statistics
    total_points = len(df)
    trending_up_points = sum(s['duration'] for s in segments if s['regime'] == 'Trending Up')
    trending_down_points = sum(s['duration'] for s in segments if s['regime'] == 'Trending Down')
    transition_points = sum(s['duration'] for s in segments if s['regime'] == 'Transition')
    choppy_points = sum(s['duration'] for s in segments if s['regime'] == 'Choppy')
    
    trending_pct = (trending_up_points + trending_down_points) / total_points * 100
    transition_pct = transition_points / total_points * 100
    choppy_pct = choppy_points / total_points * 100
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f'Day {day_num} - PB9_T1 with Dynamic EWM Regimes',
            f'CUSUM Difference and EWM Thresholds (Span={ewm_span})'
        ),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # ========================================================================
    # SUBPLOT 1: PB9_T1 WITH REGIME-COLORED LINE
    # (Unchanged, stitch logic is correct)
    # ========================================================================
    for seg in segments:
        start_time = df.iloc[seg['start_idx']]['Time']
        end_time = df.iloc[seg['end_idx']]['Time']
        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=seg['color'], opacity=0.2,
            layer="below", line_width=0, row=1, col=1
        )
        
    for seg in segments:
        start_idx = seg['start_idx']
        end_idx = seg['end_idx'] + 1
        if start_idx > 0:
            start_idx -= 1
        segment_df = df.iloc[start_idx : end_idx]
        
        fig.add_trace(
            go.Scatter(
                x=segment_df['Time'], y=segment_df['PB9_T1'],
                mode='lines', line=dict(color=seg['color'], width=2.5),
                name=seg['regime'], showlegend=False,
                # Hover template updated for individual point hovering
            ), row=1, col=1
        )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', width=4),
                      name=f'Trending ({trending_pct:.1f}%)', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='yellow', width=4),
                      name=f'Transition ({transition_pct:.1f}%)', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='magenta', width=4),
                      name=f'Choppy ({choppy_pct:.1f}%)', showlegend=True), row=1, col=1)
    
    # ========================================================================
    # SUBPLOT 2: CUSUM DIFFERENCE
    # (Unchanged)
    # ========================================================================
    
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=cusum_diff,
            mode='lines', line=dict(color='blue', width=2),
            name='CUSUM Diff',
            showlegend=False,
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=trending_up_thresh,
            mode='lines', line=dict(color='green', width=2, dash='solid'),
            name='Trending Up Threshold',
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=trending_down_thresh,
            mode='lines', line=dict(color='red', width=2, dash='solid'),
            name='Trending Down Threshold',
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=choppy_up_thresh,
            mode='lines', line=dict(color='magenta', width=1.5, dash='dash'),
            name='Choppy Upper Threshold',
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['Time'], y=choppy_down_thresh,
            mode='lines', line=dict(color='magenta', width=1.5, dash='dash'),
            name='Choppy Lower Threshold',
        ), row=2, col=1
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    
    # ========================================================================
    # LAYOUT CONFIGURATION
    # (Unchanged)
    # ========================================================================
    
    fig.update_xaxes(title_text="Time (HH:MM:SS)", tickformat='%H:%M:%S', row=2, col=1) 
    fig.update_yaxes(title_text="PB9_T1 ($)", row=1, col=1)
    fig.update_yaxes(title_text="CUSUM Diff ($)", row=2, col=1)
    
    fig.update_layout(
        height=900,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        title_text=f"<b>Day {day_num} - Dynamic CUSUM EWM Regimes (Filtered)</b><br>" + # Title updated
                     f"<sub>{len(df):,} points | {len(segments)} segments | " +
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
    days_to_plot = [195, 196]  # ← CHANGE THIS LIST TO PLOT DIFFERENT DAYS
    
    # *** NEW: SMA window for ci calculation ***
    SMA_WINDOW_ci = 1
    
    # CUSUM Parameters
    DRIFT = 0.005    # Minimum change (PB9_T1 - SMA) to count as directional
    DECAY = 0.98     # Decay factor (e.g., 0.9 to 0.999)
    
    # EWM Dynamic Threshold Parameters (for the cusum_diff)
    EWM_SPAN = 180           # EWM Span (similar to a lookback period)
    
    TRENDING_UP_STD_MULT = 1.0   # Multiplier for Trending Up threshold (e.g., 1.0)
    TRENDING_DOWN_STD_MULT = 1.0 # Multiplier for Trending Down threshold (e.g., 0.8)
    
    CHOPPY_STD_MULT = 1.0       # Multiplier for Choppy threshold (e.g., 0.5)
    
    # Regime Filter Parameter
    MIN_REGIME_DURATION = 15 # Min data points for a regime to be "valid"
    
    print("="*80)
    print("CUSUM DIFFERENCE REGIME DETECTOR (DYNAMIC EWM LOGIC + FILTERING)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Days to plot: {days_to_plot}")
    
    # *** MODIFIED: Added SMA printout ***
    print(f"\nInput 'ci' Calculation:")
    print(f"   ci = PB9_T1[t] - SMA(PB9_T1, window={SMA_WINDOW_ci})[t-1]")
    
    print(f"\nCUSUM Parameters (applied to 'ci'):")
    print(f"   Drift: ${DRIFT} (min 'ci' move to count)")
    print(f"   Decay: {DECAY} (applied on noise or counter-move)")
    
    print(f"\nDynamic Classification (EWM):")
    print(f"   EWM Span: {EWM_SPAN} data points (applied to cusum_diff)")
    
    print(f"   Trending Up:    |Diff - Center| > (Vol * {TRENDING_UP_STD_MULT})")
    print(f"   Trending Down:  |Diff - Center| < -(Vol * {TRENDING_DOWN_STD_MULT})")
    print(f"   Choppy:         |Diff - Center| < (Vol * {CHOPPY_STD_MULT})")
    
    print(f"\nRegime Filtering:")
    print(f"   Min Duration: {MIN_REGIME_DURATION} points (short segments will be absorbed)")
    print("="*80)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n✗ ERROR: Directory not found: {data_dir}")
        return
    
    print(f"\n✓ Data directory found")
    print(f"\nProcessing {len(days_to_plot)} days...\n")
    
    for day_num in days_to_plot:
        print(f"Day {day_num}:")
        
        df = load_and_process_day(day_num, data_dir)
        
        if df is None:
            print(f"   Skipping Day {day_num}\n")
            continue
            
        # 1. *** NEW: Calculate the 'ci' series ***
        print(f"   Calculating {SMA_WINDOW_ci}-period SMA and 'ci' series...")
        # Calculate 5-day SMA
        sma_5 = df['PB9_T1'].rolling(window=SMA_WINDOW_ci, min_periods=SMA_WINDOW_ci).mean()
        # Shift SMA to get SMA of t-1 to t-5
        sma_5_shifted = sma_5.shift(1)
        # Calculate ci = P[t] - SMA[t-1]
        # Use bfill() to fill the initial NaNs from the rolling/shift operations
        ci_series = (df['PB9_T1'] - sma_5_shifted).bfill()
        
        # 2. Calculate CUSUM on the new 'ci' series
        print(f"   Calculating directional CUSUM (on 'ci' series)...")
        changes_array = ci_series.values.astype(np.float64)
        
        # *** MODIFIED: Updated function call ***
        cusum_up, cusum_down = calculate_directional_cusum(
            changes_array, drift=DRIFT, decay=DECAY
        )
        cusum_diff = cusum_up - cusum_down
        
        # 3. Calculate Dynamic EWM Thresholds (on the cusum_diff)
        print(f"   Calculating dynamic EWM thresholds (Span={EWM_SPAN})...")
        cusum_diff_series = pd.Series(cusum_diff)
        ewm_center = cusum_diff_series.ewm(span=EWM_SPAN).mean()
        ewm_std = cusum_diff_series.ewm(span=EWM_SPAN).std()
        
        trending_up_thresh = (ewm_center + (TRENDING_UP_STD_MULT * ewm_std)).bfill().values
        trending_down_thresh = (ewm_center - (TRENDING_DOWN_STD_MULT * ewm_std)).bfill().values
        choppy_up_thresh = (ewm_center + (CHOPPY_STD_MULT * ewm_std)).bfill().values
        choppy_down_thresh = (ewm_center - (CHOPPY_STD_MULT * ewm_std)).bfill().values
        
        # 4. Classify Regimes
        print(f"   Classifying regimes by dynamic thresholds...")
        regimes, colors, _ = classify_regime_by_difference(
            cusum_up,
            cusum_down,
            choppy_up_thresh=choppy_up_thresh,
            choppy_down_thresh=choppy_down_thresh,
            trending_up_thresh=trending_up_thresh,
            trending_down_thresh=trending_down_thresh
        )
        
        # 5. Group for plotting
        print(f"   Grouping regimes for visualization...")
        raw_segments = create_regime_segments(df, regimes, colors, cusum_diff)
        print(f"   Generated {len(raw_segments)} raw segments.")

        # 6. Filter and Merge Segments
        print(f"   Filtering segments smaller than {MIN_REGIME_DURATION} points...")
        segments = filter_and_merge_segments(raw_segments, MIN_REGIME_DURATION)
        print(f"   Generated {len(segments)} final filtered segments.")
        
        # Print summary of filtered segments
        print(f"   Regime breakdown (by filtered segments):")
        regime_counts = {'Trending Up': 0, 'Trending Down': 0, 'Transition': 0, 'Choppy': 0}
        for seg in segments:
            if seg['regime'] in regime_counts:
                regime_counts[seg['regime']] += seg['duration']
            
        total_points = len(df)
        print(f"   - Trending Up:    {regime_counts['Trending Up']:6} points ({regime_counts['Trending Up']/total_points*100:5.1f}%)")
        print(f"   - Trending Down:  {regime_counts['Trending Down']:6} points ({regime_counts['Trending Down']/total_points*100:5.1f}%)")
        print(f"   - Transition:     {regime_counts['Transition']:6} points ({regime_counts['Transition']/total_points*100:5.1f}%)")
        print(f"   - Choppy:         {regime_counts['Choppy']:6} points ({regime_counts['Choppy']/total_points*100:5.1f}%)")
        
        # 7. Create Visualization
        print(f"   Creating visualization...")
        fig = create_PB9_T1_cusum_plot(
            df, segments, day_num,
            cusum_diff,
            choppy_up_thresh=choppy_up_thresh,
            choppy_down_thresh=choppy_down_thresh,
            trending_up_thresh=trending_up_thresh,
            trending_down_thresh=trending_down_thresh,
            ewm_span=EWM_SPAN
        )
        
        fig.show()
        print(f"   ✓ Displaying visualization...")
        
        PB9_T1_min = df['PB9_T1'].min()
        PB9_T1_max = df['PB9_T1'].max()
        PB9_T1_range = PB9_T1_max - PB9_T1_min
        
        print(f"   Statistics:")
        print(f"         PB9_T1 range: ${PB9_T1_min:,.2f} - ${PB9_T1_max:,.2f} (${PB9_T1_range:.2f})")
        print(f"         Max CUSUM Diff: ${cusum_diff.max():.3f}")
        print(f"         Min CUSUM Diff: ${cusum_diff.min():.3f}")
        print(f"   ✓ Day {day_num} complete\n")
    
    print("="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()