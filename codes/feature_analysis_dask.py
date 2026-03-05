import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pandas as pd

def analyze_feature(data_df):
    warnings.filterwarnings('ignore')

    if isinstance(data_df, (dask_cudf.DataFrame, cudf.DataFrame)):
        data_df = data_df.compute().to_pandas()

    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price
    
    feature_columns = [c for c in data_df.columns if c not in ['Time', 'Price', 'Day', 'PlotIndex']]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
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
            connectgaps=False,
            showlegend=False
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
                corresponding_plot_indices = df_day.loc[feature_mask, "PlotIndex"].tolist()
                
                fig.add_trace(
                    go.Scatter(
                        x=corresponding_plot_indices,
                        y=feature_data.values.tolist(),
                        mode='lines',
                        name=f"{feature}",  # Same name for all days - single toggle
                        legendgroup=f"{feature}",  # Group by feature for single toggle
<<<<<<< HEAD
                        line=dict(color=color, width=3),
=======
                        line=dict(color=color, width=1.5),
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
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
    
    os.makedirs("feature_plots", exist_ok=True)
    fig.write_html("feature_plots/feature_analysis.html")
    fig.show()

if __name__ == "__main__":
    #csv_path = "/data/quant14/EBY/combined.csv"

    # Load large data on GPU
    #combined_df = dask_cudf.read_csv(csv_path, usecols=["Time", "Price"])
    
    # Example: also concatenate two large CSVs directly on GPU
    df1 = dask_cudf.read_csv("/data/quant14/EBX/day0.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    df2 = dask_cudf.read_csv("/data/quant14/EBX/day1.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    df3 = dask_cudf.read_csv("/data/quant14/EBX/day2.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    gpu_concat_df = dask_cudf.concat([df1, df2, df3])

    # Convert back to pandas only when plotting
    analyze_feature(gpu_concat_df)
