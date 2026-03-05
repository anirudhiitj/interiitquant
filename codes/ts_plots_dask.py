import dask_cudf
import cudf
import dask
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import timedelta
import pandas as pd

def day_time_series_plot(data_df):
    warnings.filterwarnings('ignore')

    # If it's a dask_cudf or cudf DataFrame, convert to pandas for plotting
    if isinstance(data_df, (dask_cudf.DataFrame, cudf.DataFrame)):
        data_df = data_df.compute().to_pandas()

    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    anchor_date = pd.Timestamp("1970-01-01")
    data_df["Time_dt"] = anchor_date + data_df["Time"]
    data_df.set_index("Time_dt", inplace=True)

    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price
    base_price = data_df["Price"].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_df.index,
        y=data_df["Price"],
        mode='lines',
        name='Price',
        line=dict(color='#FF6B6B', width=0.5)
    ))

    fig.update_layout(
        title='Price Analysis',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified',
        height=600, width=1400
    )

    out_path = "time_series_plots/day_time_series_plot.html"
    fig.write_html(out_path)
    print(f"Plot saved to: {out_path}")
    fig.show()

def combined_time_series_plot(data_df):
    warnings.filterwarnings('ignore')

    # Convert GPU DataFrame → pandas
    if isinstance(data_df, (dask_cudf.DataFrame, cudf.DataFrame)):
        data_df = data_df.compute().to_pandas()

    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price

    fig = go.Figure()
    for day, df_day in data_df.groupby("Day"):
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

    out_path = "time_series_plots/combined_time_series_plot.html"
    fig.write_html(out_path)
    print(f"Plot saved to: {out_path}")
    fig.show()

if __name__ == "__main__":
    csv_path = "/data/quant14/EBY/combined.csv"

    # Load large data on GPU
    #combined_df = dask_cudf.read_csv(csv_path, usecols=["Time", "Price"])
    
    # Example: also concatenate two large CSVs directly on GPU
    df1 = dask_cudf.read_csv("/data/quant14/EBX/day0.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    df2 = dask_cudf.read_csv("/data/quant14/EBX/day1.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    df3 = dask_cudf.read_csv("/data/quant14/EBX/day2.csv",usecols=["Time", "Price","PB1_T1","PB1_T2"])
    gpu_concat_df = dask_cudf.concat([df1, df2, df3])

    # Convert back to pandas only when plotting
    combined_time_series_plot(gpu_concat_df)
