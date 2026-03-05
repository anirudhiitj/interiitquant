import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from pathlib import Path
from scipy import stats
import warnings
import os

def day_time_series_plot(data_df):
    # data_df = pd.read_csv(data_csv_path, usecols=["Time","Price"])
    print(data_df)
    # For time-only data (hh:mm:ss), anchor to a fixed date
    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    anchor_date = pd.Timestamp("1970-01-01")  
    data_df["Time_dt"] = anchor_date + data_df["Time"]  
    data_df.set_index("Time_dt", inplace=True)

    warnings.filterwarnings('ignore')
    
    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price
    base_price = data_df["Price"].iloc[0]
    
    print(data_df)
    # Create figure
    fig = go.Figure()
    
    # Add cumulative returns line
    fig.add_trace(go.Scatter(
        x=data_df.index,
        y=data_df["Price"],
        mode='lines',
        name='Price',
        line=dict(color='#FF6B6B', width=0.5),
        hovertemplate='<b>Time</b>: %{x|%H:%M:%S}<br>' +
                      '<b>Price</b>: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout with better scaling
    fig.update_layout(
        title={
            'text': 'Price Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'black'}
        },
        xaxis_title='Time (HH:MM:SS)',
        yaxis_title='Price',
        hovermode='x unified',
        height=600,
        width=1400,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Set y-axis range with padding
    y_padding = (max_price - min_price) * 0.1
    fig.update_yaxes(
        range=[min_price - y_padding, max_price + y_padding],
        gridcolor='lightgray',
        tickformat='.4f',
        title_standoff=10
    )
    
    # Format x-axis with time formatting
    fig.update_xaxes(
        tickformat='%H:%M:%S',
        gridcolor='lightgray',
        tickangle=-45,
        title_standoff=10
    )
    
    # Add annotation with statistics
    stats_text = (
        f'Initial Price: ${base_price:.2f}<br>'
        f'Final Price: ${data_df["Price"].iloc[-1]:.2f}<br>'
        f'Max Price: ${max_price:.2f}<br>'
        f'Min Price: ${min_price:.2f}<br>'
        f'Range: ${price_range:.2f}<br>'
    )
    
    fig.add_annotation(
        text="annotations",
        hovertext=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="black",
        borderwidth=2,
        borderpad=10,
        align="left",
        font=dict(size=11, family="monospace")
    )
    
    # Save as HTML
    out_path = "day_time_series_plot.html"
    fig.write_html(out_path)
    
    print(f"Plot saved to: {out_path}")
    
    # Show plot
    fig.show()
    
    return data_df

def combined_time_series_plot(data_df):
    warnings.filterwarnings('ignore')

    # Convert Time to timedelta
    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    
    # Detect when time repeats/resets (new day starts)
    # When current time < previous time, it's a new day
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()

    data_df = data_df.dropna(subset=["Price"])
    
    # Create sequential index for continuous plotting
    data_df = data_df.reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))
    
    # Calculate statistics
    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price

    # Create single plot figure
    fig = go.Figure()
    
    # Plot: Price vs Time (using sequential index)
    for day, df_day in data_df.groupby("Day"):
        # Extract time strings for hover
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"],
            y=df_day["Price"],
            mode='lines',
            name=f"Day {day}",
            line=dict(width=2),
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Price</b>: %{y:.4f}<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Combined Multi-Day Price Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'black'}
        },
        xaxis_title="Time (HH:MM:SS)",
        yaxis_title="Price",
        hovermode='x unified',
        height=600,
        width=1600,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    num_days = int(data_df["Day"].max()) + 1
    
    # Create tick positions at the start of each day
    tick_positions = []
    tick_labels = []
    for day in range(num_days):
        day_data = data_df[data_df["Day"] == day]
        if len(day_data) > 0:
            tick_positions.append(day_data["PlotIndex"].iloc[0])
            tick_labels.append(f"Day {day}")
    
    # Update axes
    fig.update_xaxes(
        tickvals=tick_positions,
        ticktext=tick_labels,
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        tickformat='.4f', 
        gridcolor='lightgray'
    )
    
    # Set y-axis range with padding
    y_padding = (max_price - min_price) * 0.1
    fig.update_yaxes(
        range=[min_price - y_padding, max_price + y_padding]
    )
    
    # Add annotation with statistics
    stats_text = (
        f'Number of Days: {num_days}<br>'
        f'Initial Price: ${data_df["Price"].iloc[0]:.2f}<br>'
        f'Final Price: ${data_df["Price"].iloc[-1]:.2f}<br>'
        f'Max Price: ${max_price:.2f}<br>'
        f'Min Price: ${min_price:.2f}<br>'
        f'Range: ${price_range:.2f}<br>'
    )
    
    fig.add_annotation(
        text="annotations",
        hovertext=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="black",
        borderwidth=2,
        borderpad=10,
        align="left",
        font=dict(size=11, family="monospace")
    )
    
    # Save as HTML
    out_path = "combined_time_series_plot.html"
    fig.write_html(out_path)
    
    print(f"\nPlot saved to: {out_path}")
    print(f"Number of Days: {num_days}")
    print(f"Price Range: ${min_price:.2f} - ${max_price:.2f}")
    
    # Show plot
    fig.show()
    
    return data_df

def perform_adf_test(data_df):
    """
    Performs the ADF test on a PROCESSED DataFrame.
    It assumes the DataFrame has already been cleaned and has a 'Price' column.
    
    --- CORRECTED to work on processed data (no more 'Time' column) ---
    """
    warnings.filterwarnings('ignore')

    # Create a copy to be safe
    test_df = data_df.copy()

    # --- 1. No longer need to process 'Time' column ---
    # The input DataFrame is already processed.

    # --- 2. Calculate Log Returns ---
    test_df['Log_Return'] = np.log(test_df['Price']).diff()
    testable_returns = test_df['Log_Return'].dropna()

    print("--- Augmented Dickey-Fuller Test ---")

    # --- 3. Perform ADF Test on Raw Price ---
    print("\n1. Testing Raw Price Series...")
    adf_result_price = adfuller(test_df['Price'])
    
    print(f'ADF Statistic: {adf_result_price[0]:.4f}')
    print(f'p-value: {adf_result_price[1]:.4f}')
    
    if adf_result_price[1] <= 0.05:
        print("Conclusion: (p-value <= 0.05) The raw price series is likely stationary.")
    else:
        print("Conclusion: (p-value > 0.05) The raw price series is likely non-stationary.")
        
    print("-" * 40)

    # --- 4. Perform ADF Test on Log Returns ---
    print("\n2. Testing Log Returns Series...")
    adf_result_returns = adfuller(testable_returns)

    print(f'ADF Statistic: {adf_result_returns[0]:.4f}')
    print(f'p-value: {adf_result_returns[1]:.4f}')

    if adf_result_returns[1] <= 0.05:
        print("Conclusion: (p-value <= 0.05) The log returns series is likely stationary.")
    else:
        print("Conclusion: (p-value > 0.05) The log returns series is likely non-stationary.")
        
    return test_df

def analyze_feature(data_df):
    try:
        warnings.filterwarnings('ignore')
        
        # Make a copy to avoid modifying the original DataFrame
        data_df = data_df.copy()
        
        # Convert Time to timedelta
        data_df["Time"] = pd.to_timedelta(data_df["Time"])
        
        # Detect when time repeats/resets (new day starts)
        # When current time < previous time, it's a new day
        data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()

        data_df = data_df.dropna(subset=["Price"])
        
        # Create sequential index for continuous plotting
        data_df = data_df.reset_index(drop=True)
        data_df["PlotIndex"] = range(len(data_df))
        
        # Calculate statistics
        min_price = data_df["Price"].min()
        max_price = data_df["Price"].max()
        price_range = max_price - min_price
        
        # Get all columns except Time, Price, Day, PlotIndex
        feature_columns = [col for col in data_df.columns 
                          if col not in ['Time', 'Price', 'Day', 'PlotIndex']]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Plot: Price vs Time (using sequential index) - EXACTLY like combined_time_series_plot
        for day, df_day in data_df.groupby("Day"):
            # Extract time strings for hover
            time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
            
            fig.add_trace(go.Scatter(
                x=df_day["PlotIndex"],
                y=df_day["Price"],
                mode='lines',
                name=f"Day {day}",
                line=dict(width=2),
                hovertemplate=(
                    "<b>Day</b>: %{customdata[0]}<br>"
                    "<b>Time</b>: %{customdata[1]}<br>"
                    "<b>Price</b>: %{y:.4f}<extra></extra>"
                ),
                customdata=[[day, time_str] for time_str in time_strings],
                connectgaps=False
            ), secondary_y=False)
        
        # Define colors for features
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
        
        # Plot all other features (secondary y-axis) - toggleable, disconnected per day
        for i, feature in enumerate(feature_columns):
            color = colors[i % len(colors)]
            
            # Plot each day separately for this feature (disconnected)
            for day, df_day in data_df.groupby("Day"):
                # Get non-null feature data for this day
                feature_mask = df_day[feature].notna()
                feature_data = df_day.loc[feature_mask, feature]  # Use .loc to get Series
                
                if len(feature_data) > 0:
                    # Get corresponding time strings and plot indices for non-null feature values
                    corresponding_times = df_day.loc[feature_mask, "Time"].apply(lambda x: str(x).split()[-1]).tolist()
                    corresponding_plot_indices = df_day.loc[feature_mask, "PlotIndex"].tolist();
                    
                    fig.add_trace(
                        go.Scatter(
                            x=corresponding_plot_indices,
                            y=feature_data.values.tolist(),  # Use .values.tolist() for Series
                            mode='lines',
                            name=f"{feature}",  # Same name for all days - single toggle
                            legendgroup=f"{feature}",  # Group by feature for single toggle
                            line=dict(color=color, width=1.5),
                            visible='legendonly',  # Start hidden, can be toggled on
                            hovertemplate=(
                                "<b>Day</b>: %{customdata[0]}<br>"
                                "<b>Time</b>: %{customdata[1]}<br>"
                                f"<b>{feature}</b>: %{{y:.4f}}<br>"
                                "<extra></extra>"
                            ),
                            customdata=[[day, time_str] for time_str in corresponding_times],
                            connectgaps=False,
                            showlegend=(day == 0)  # Only show legend for first day
                        ),
                        secondary_y=True
                    )
        
        num_days = int(data_df["Day"].max()) + 1
        
        # Create tick positions at the start of each day - EXACTLY like combined_time_series_plot
        tick_positions = []
        tick_labels = []
        for day in range(num_days):
            day_data = data_df[data_df["Day"] == day]
            if len(day_data) > 0:
                tick_positions.append(day_data["PlotIndex"].iloc[0])
                tick_labels.append(f"Day {day}")
        
        # Update layout - EXACTLY like combined_time_series_plot
        fig.update_layout(
            title={
                'text': 'Combined Multi-Day Price Analysis with Features',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            hovermode='x unified',
            height=600,
            width=1600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        # Update axes - REMOVED secondary_y parameter from update_xaxes
        fig.update_xaxes(
            title_text="Time (HH:MM:SS)",
            tickvals=tick_positions,
            ticktext=tick_labels,
            gridcolor='lightgray'
        )
        
        # Update primary y-axis (Price)
        fig.update_yaxes(
            title_text="Price",
            tickformat='.4f', 
            gridcolor='lightgray',
            secondary_y=False
        )
        
        # Update secondary y-axis (Features)
        fig.update_yaxes(
            title_text="Feature Values",
            tickformat='.4f',
            secondary_y=True
        )
        
        # Set y-axis range with padding - EXACTLY like combined_time_series_plot
        y_padding = (max_price - min_price) * 0.1
        fig.update_yaxes(
            range=[min_price - y_padding, max_price + y_padding],
            secondary_y=False
        )
        
        # Add annotation with statistics - EXACTLY like combined_time_series_plot
        stats_text = (
            f'Number of Days: {num_days}<br>'
            f'Features Available: {len(feature_columns)}<br>'
            f'Initial Price: ${data_df["Price"].iloc[0]:.2f}<br>'
            f'Final Price: ${data_df["Price"].iloc[-1]:.2f}<br>'
            f'Max Price: ${max_price:.2f}<br>'
            f'Min Price: ${min_price:.2f}<br>'
            f'Range: ${price_range:.2f}<br>'
        )
        
        fig.add_annotation(
            text="annotations",
            hovertext=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            bgcolor="rgba(255, 250, 205, 0.9)",
            bordercolor="black",
            borderwidth=2,
            borderpad=10,
            align="left",
            font=dict(size=11, family="monospace")
        )
        
        # Save as HTML
        out_path = "feature_analysis.html"
        os.makedirs("feature_plots", exist_ok=True)
        fig.write_html(os.path.join("feature_plots",out_path))
        
        print(f"\nFeature analysis plot saved to: feature_plots/{out_path}")
        print(f"Number of Days: {num_days}")
        print(f"Price Range: ${min_price:.2f} - ${max_price:.2f}")
        print(f"Number of features: {len(feature_columns)}")
        print("Note: Features start hidden - click legend to show/hide them")
        
        # Show plot
        fig.show()
        
        return data_df
        
    except Exception as e:
        print(f"✗ Error in feature analysis: {str(e)}")
        return None

def plot_returns_distribution(daily_returns):
    """
    Plots a histogram and (if possible) a Kernel Density Estimation (KDE)
    of the provided daily returns.
    
    --- UPDATED TO HANDLE < 2 DATA POINTS ---
    """
    if daily_returns.empty:
        print("\nNo daily returns to plot.")
        return

    daily_returns = daily_returns.dropna()
    data_count = len(daily_returns)
    
    if data_count == 0:
        print("\nNo daily returns left after dropping NaN. Cannot plot distribution.")
        return

    mean_ret = daily_returns.mean()
    # std() returns NaN for a single value, which is handled in the text output
    std_ret = daily_returns.std() 
    
    # --- Check if KDE is possible ---
    if data_count < 2:
        print(f"\nWarning: Only {data_count} day(s) of data. Cannot plot Kernel Density Estimation (KDE).")
        print("Plotting histogram only.")
        
        # Create a simple histogram plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=daily_returns,
            name='Daily Returns',
            showlegend=True
        ))
        
        fig.update_layout(
            title={
                'text': 'Distribution of Daily Returns (Histogram Only)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            xaxis_title="Daily Return",
            yaxis_title="Count",
        )
        
        # Stats text for histogram
        stats_text = (
            f'Mean: {mean_ret:.4f}<br>'
            f'Std Dev: N/A<br>'
            f'Count: {data_count}'
        )
        print(f"\nReturns distribution plot saved to: returns_distribution_plot.html")
        print(f"Daily Returns Mean: {mean_ret:.4f}, Std: N/A")

    else:
        # --- We have enough data for KDE ---
        print("\nGenerating returns distribution plot (Histogram + KDE)...")
        
        # Set a reasonable bin size
        bin_size = std_ret / 10
        
        # Create the distribution plot (Histogram + KDE)
        fig = ff.create_distplot(
            [daily_returns],
            ['Daily Returns'],
            show_hist=True,
            show_rug=False,
            bin_size=bin_size
        )
        
        fig.update_layout(
            title={
                'text': 'Distribution of Daily Returns (Histogram + KDE)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': 'black'}
            },
            xaxis_title="Daily Return",
            yaxis_title="Density",
        )
        
        # Stats text for KDE plot
        stats_text = (
            f'Mean: {mean_ret:.4f}<br>'
            f'Std Dev: {std_ret:.4f}<br>'
            f'Count: {data_count}'
        )
        print(f"\nReturns distribution plot saved to: returns_distribution_plot.html")
        print(f"Daily Returns Mean: {mean_ret:.4f}, Std: {std_ret:.4f}")

    # --- Common layout updates for both plot types ---
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        hovermode='x unified',
    )
    
    fig.add_annotation(
        text="<b>Statistics</b><br>" + stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor='right', yanchor='top',
        showarrow=False,
        bgcolor="rgba(230, 240, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        align="left",
    )

    # Save as HTML
    out_path = "returns_distribution_plot.html"
    fig.write_html(out_path)
    
    # Show plot
    fig.show()

def plot_returns_combined(data_df):
    """
    Plots a 2-part analysis:
    1. A continuous time series of tick-by-tick returns across all days.
    2. A distribution plot (Histogram + KDE) of all tick-by-tick returns.
    """
    warnings.filterwarnings('ignore')
    
    # --- 1. Data Processing (similar to combined_time_series_plot) ---
    
    # Convert Time to timedelta
    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    
    # Detect when time repeats/resets (new day starts)
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()

    data_df = data_df.dropna(subset=["Price"])
    
    # Create sequential index for continuous plotting
    data_df = data_df.reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    # --- 2. Calculate Returns & Prep Data ---
    
    all_returns_list = []
    
    # Calculate tick-by-tick returns for each day
    # We'll also store them in the main df for plotting
    data_df['Returns'] = np.nan
    for day, df_day in data_df.groupby("Day"):
        day_indices = df_day.index
        day_returns = (df_day["Price"].pct_change() * 100) # as percentage
        
        # Store returns in the main dataframe
        data_df.loc[day_indices, 'Returns'] = day_returns
        
        # Add non-null returns to our list for the distribution plot
        all_returns_list.append(day_returns.dropna())

    # Combine all returns into one Series for distribution analysis
    all_returns = pd.concat(all_returns_list).dropna()

    # --- 3. Create 2-Row Subplot Figure ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False, # X-axes are different (Index vs Return Value)
        vertical_spacing=0.15,
        subplot_titles=(
            'Continuous Tick-by-Tick Returns (by Day)',
            'Distribution of All Tick-by-Tick Returns'
        ),
        specs=[[{}], # Row 1
               [{"secondary_y": True}]] # Row 2 has secondary Y
    )

    # --- 4. Top Plot (Continuous Returns) ---
    
    num_days = int(data_df["Day"].max()) + 1
    tick_positions = []
    tick_labels = []

    for day, df_day in data_df.groupby("Day"):
        # Get day start for x-axis ticks
        if len(df_day) > 0:
            tick_positions.append(df_day["PlotIndex"].iloc[0])
            tick_labels.append(f"Day {day}")
            
        # Extract time strings for hover
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        
        # Add return trace for the day
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"],
            y=df_day["Returns"],
            mode='lines',
            name=f"Day {day}",
            line=dict(width=1.5),
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Return</b>: %{y:.4f}%<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False
        ), row=1, col=1)

    # Configure Top Plot Axes
    fig.update_xaxes(
        title_text="Time",
        tickvals=tick_positions,
        ticktext=tick_labels,
        gridcolor='lightgray',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Returns (%)",
        tickformat='.4f', 
        gridcolor='lightgray',
        row=1, col=1
    )

    # --- 5. Bottom Plot (Distribution) ---
    
    if all_returns.empty:
        print("Warning: No tick-by-tick returns found. Skipping distribution plot.")
    else:
        # Add Histogram (Primary Y-axis)
        fig.add_trace(go.Histogram(
            x=all_returns,
            name="Histogram (Count)",
            marker_color='#1f77b4',
            opacity=0.7
        ), row=2, col=1, secondary_y=False)

        # Add KDE (Secondary Y-axis)
        data_count = len(all_returns)
        if data_count < 2:
            print("Warning: Not enough data for KDE plot (need >= 2). Plotting histogram only.")
        else:
            try:
                kde = stats.gaussian_kde(all_returns)
                x_range = np.linspace(all_returns.min(), all_returns.max(), 500)
                kde_y = kde(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_y,
                    mode='lines',
                    name="KDE (Density)",
                    line=dict(color='firebrick', width=2)
                ), row=2, col=1, secondary_y=True)
            except Exception as e:
                print(f"Warning: Could not generate KDE plot. {e}")

        # Configure Bottom Plot Axes
        fig.update_xaxes(title_text="Return Value (%)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Density", row=2, col=1, secondary_y=True)

    # --- 6. Final Layout & Save ---
    
    fig.update_layout(
        title={
            'text': 'Tick-by-Tick Returns Analysis',
            'x': 0.5, 'xanchor': 'center',
            'font': {'size': 20, 'color': 'black'}
        },
        height=900,
        width=1600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    out_path = "returns_combined_plot.html"
    fig.write_html(out_path)
    print(f"\nCombined returns plot saved to: {out_path}")
    
    # Show plot
    fig.show()

    return data_df

def process_multi_day_time_series(df, start_date='2025-01-01'):
    """
    Converts a DataFrame with a resetting 'Time' column (hhmmss format)
    into a time series with a continuous datetime index.
    
    --- CORRECTED to use 'Time' (uppercase T) ---
    """
    processed_df = df.copy()

    # --- Step 1: Convert time string to a timedelta object ---
    # CORRECTED: Changed 'time' to 'Time'
    processed_df['time_delta'] = pd.to_timedelta(
        processed_df['Time'].astype(str).str.zfill(6).str[:2] + ':' +
        processed_df['Time'].astype(str).str.zfill(6).str[2:4] + ':' +
        processed_df['Time'].astype(str).str.zfill(6).str[4:6],
        errors='coerce' # Handle potential malformed time strings
    )

    # --- Step 2: Detect where the day rolls over ---
    time_diff = processed_df['time_delta'].diff()
    day_change_marker = time_diff < pd.Timedelta(0)

    # --- Step 3: Create a cumulative day counter ---
    processed_df['day_counter'] = day_change_marker.cumsum()

    # --- Step 4: Construct the final datetime index ---
    base_date = pd.to_datetime(start_date)
    processed_df['datetime'] = base_date + pd.to_timedelta(processed_df['day_counter'], unit='D') + processed_df['time_delta']

    # --- Step 5: Final Cleanup ---
    processed_df.set_index('datetime', inplace=True)
    # CORRECTED: Changed 'time' to 'Time'
    processed_df.drop(columns=['Time', 'time_delta', 'day_counter'], inplace=True)
    
    # Drop any rows that couldn't be parsed
    processed_df.dropna(subset=['Price'], inplace=True)

    return processed_df

def main():
    """
    Main function to load data and run analysis.
    """
    # Define the input file
    input_file = Path("/data/quant14/EBY/day0.csv")
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found.")
        print("Please run `parallel_signal_generator.py` first to generate signals.")
        return

    print(f"Loading data from '{input_file}'...")
    try:
        data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    if "Price" not in data_df.columns or "Time" not in data_df.columns:
        print("Error: CSV must contain 'Time' and 'Price' columns.")
        return

    # --- MODIFIED: Load multiple dataframes for combined analysis ---
    print("Loading multiple CSVs for combined analysis...")
    try:
        # Re-create the combined df from your example
        df1 = pd.read_csv("/data/quant14/EBY/day1.csv", usecols=["Time","Price"])
        # Simulating more days by re-using the same file
        # In a real case, you'd load day0.csv, day1.csv, etc.
        df2 = pd.read_csv("/data/quant14/EBY/day2.csv", usecols=["Time","Price"])
        df3 = pd.read_csv("/data/quant14/EBY/day3.csv", usecols=["Time","Price"])
        data_df = pd.concat([df1, df2, df3], ignore_index=True)
        print(f"Loaded and combined data, total rows: {len(data_df)}")
    except FileNotFoundError:
        print("Warning: 'trading_signals.csv' not found. Falling back to single load.")
        data_df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading combined CSVs: {e}")
        return

    # --- Call the new combined returns plot ---
    plot_returns_combined(data_df)

    # --- Original plot functions (can be un-commented if needed) ---
    
    # 1. Generate the time series plot
    # This function is modified to return the calculated daily_returns
    # processed_df, daily_returns = combined_time_series_plot(data_df)
    
    # 2. Generate the returns distribution plot
    # if daily_returns is not None and not daily_returns.empty:
    #    plot_returns_distribution(daily_returns)
    # else:
    #    print("Could not generate returns plot: No daily returns were calculated.")

if __name__=="__main__":
    # # for i in range(0,6,1):
    # #     data_csv_path=f"/data/quant14/EBY/day{i}.csv"
    # #     day_time_series_plot(data_csv_path)
    
    data_csv_path=f"/data/quant14/EBY/combined.csv"
    combined_df=pd.read_csv(data_csv_path,usecols=["Time","Price"])
    # df1 = pd.read_csv("/data/quant14/EBX/day0.csv",usecols=["Time","Price","PB1_T1","PB1_T2"])
    # df2 = pd.read_csv("/data/quant14/EBX/day1.csv",usecols=["Time","Price","PB1_T1","PB1_T2"])
    # df3 = pd.read_csv("/data/quant14/EBX/day2.csv",usecols=["Time","Price","PB1_T1","PB1_T2"])
    # df = pd.concat([df1, df2, df3], ignore_index=True)
    # plot_returns_combined(df)
    #main()
    cdf=process_multi_day_time_series(combined_df)
    perform_adf_test(cdf)