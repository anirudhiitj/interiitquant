import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pandas as pd

# Feature lists remain available for reference
EBX_Price_Proxies = [
    'PB6_T11', 'PB6_T12', 'PB9_T1', 'PB9_T2', 'PB9_T3', 'PB9_T4', 'PB9_T5',
    'PB9_T6', 'PB9_T7', 'PB9_T8', 'PB9_T9', 'PB9_T10', 'PB9_T11', 'PB9_T12',
    'PB10_T1', 'PB10_T2', 'PB10_T3', 'PB10_T4', 'PB10_T5', 'PB10_T6', 'PB10_T7',
    'PB10_T8', 'PB10_T9', 'PB10_T10', 'PB10_T11', 'PB10_T12', 'PB11_T1', 'PB11_T2',
    'PB11_T3', 'PB11_T4', 'PB11_T5', 'PB11_T6', 'PB11_T7', 'PB11_T8', 'PB11_T9',
    'PB11_T10', 'PB11_T11', 'PB11_T12', 'PB13_T12'
]
EBY_Crossover=['PB6_T6','PB6_T11','PB10_T4','PB11_T4']
# ... (Other feature lists can be kept here)

def analyze_feature(data_df, primary_features, secondary_features):
    """
    Analyzes and plots price data with two sets of features:
    - primary_features: Plotted on the primary Y-axis with the price.
    - secondary_features: Plotted on the secondary Y-axis.
    """
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
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot Price (Primary Y-axis)
    for day, df_day in data_df.groupby("Day"):
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"], y=df_day["Price"],
            mode='lines', name=f"Day {day}", line=dict(width=2, color='black'),
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Price</b>: %{y:.4f}<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False, showlegend=False
        ), secondary_y=False)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # --- NEW: Plot PRIMARY axis features ---
    for i, feature in enumerate(primary_features):
        color = colors[i % len(colors)]
        for day, df_day in data_df.groupby("Day"):
            feature_mask = df_day[feature].notna()
            if feature_mask.any():
                df_subset = df_day[feature_mask]
                time_subset = df_subset["Time"].apply(lambda x: str(x).split()[-1]).tolist()
                fig.add_trace(go.Scatter(
                    x=df_subset["PlotIndex"], y=df_subset[feature],
                    mode='lines', name=f"{feature} (Price Axis)", legendgroup=f"{feature}",
                    line=dict(color=color, width=1.5),
                    hovertemplate=(
                        f"<b>{feature}</b>: %{{y:.4f}}<br>"
                        "<b>Time</b>: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=[[day, time_str] for time_str in time_subset],
                    connectgaps=False, showlegend=(day == 0)
                ), secondary_y=False) # Plot on primary axis

    # --- MODIFIED: Plot SECONDARY axis features ---
    for i, feature in enumerate(secondary_features):
        color_idx = i + len(primary_features) # Offset color index
        color = colors[color_idx % len(colors)]
        for day, df_day in data_df.groupby("Day"):
            feature_mask = df_day[feature].notna()
            if feature_mask.any():
                df_subset = df_day[feature_mask]
                time_subset = df_subset["Time"].apply(lambda x: str(x).split()[-1]).tolist()
                fig.add_trace(go.Scatter(
                    x=df_subset["PlotIndex"], y=df_subset[feature],
                    mode='lines', name=f"{feature}", legendgroup=f"{feature}",
                    line=dict(color=color, width=1.5), visible='legendonly',
                    hovertemplate=(
                        f"<b>{feature}</b>: %{{y:.10f}}<br>"
                        "<b>Time</b>: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=[[day, time_str] for time_str in time_subset],
                    connectgaps=False, showlegend=(day == 0)
                ), secondary_y=True) # Plot on secondary axis

    num_days = int(data_df["Day"].max()) + 1
    tick_positions = [data_df[data_df["Day"] == d]["PlotIndex"].iloc[0] for d in range(num_days) if not data_df[data_df["Day"] == d].empty]
    tick_labels = [f"Day {d}" for d in range(num_days) if not data_df[data_df["Day"] == d].empty]
    
    fig.update_layout(
        title={'text': 'Multi-Day Price Analysis with Primary & Secondary Features', 'x': 0.5},
        hovermode='x unified', height=700, width=1600, showlegend=True,
        plot_bgcolor='white', paper_bgcolor='white', font=dict(size=12)
    )
    
    fig.update_xaxes(title_text="Time", tickvals=tick_positions, ticktext=tick_labels, gridcolor='lightgray')
    fig.update_yaxes(title_text="Price & Primary Features", tickformat='.4f', gridcolor='lightgray', secondary_y=False)
    fig.update_yaxes(title_text="Secondary Feature Values", tickformat='.10f', secondary_y=True, showgrid=False)
    
    y_padding = (max_price - min_price) * 0.1
    fig.update_yaxes(range=[min_price - y_padding, max_price + y_padding], secondary_y=False)
    
    stats_text = (
        f'Number of Days: {num_days}<br>'
        f'Features: {len(primary_features) + len(secondary_features)}<br>'
        f'Max Price: ${max_price:.2f}<br>'
        f'Min Price: ${min_price:.2f}<br>'
        f'Range: ${price_range:.2f}<br>'
    )
    
    fig.add_annotation(
        text="annotations", hovertext=stats_text, xref="paper", yref="paper",
        x=0.02, y=0.98, xanchor='left', yanchor='top', showarrow=False,
        bgcolor="rgba(255, 250, 205, 0.9)", bordercolor="black", borderwidth=2,
        borderpad=10, align="left", font=dict(size=11, family="monospace")
    )
    
    fig.write_html("PB_Feature_Analysis/EBY_day0_Useful_DualAxis.html")
    fig.show()

if __name__ == "__main__":
    
    # --- DEFINE YOUR FEATURES FOR EACH AXIS HERE ---
    PRIMARY_AXIS_FEATURES = ['PB11_T4', 'PB10_T4']  # e.g., features that are price-like (bands, stops)
    SECONDARY_AXIS_FEATURES = ['PB6_T6', 'PB6_T11'] # e.g., indicators with different scales (oscillators)

    all_features_to_load = PRIMARY_AXIS_FEATURES + SECONDARY_AXIS_FEATURES
    
    # Determine columns to load
    cols = cudf.read_csv("/data/quant14/EBY/day0.csv", nrows=0).columns
    usecols = [c for c in cols if c in ["Time", "Price"] or c in all_features_to_load]
    
    # Load data for the specified days
    # Note: The original loop was range(196, 197), which only loads day 196. 
    # The first day loaded was day 7. I'm keeping this logic.
    gpu_df_list = []
    gpu_df_list.append(dask_cudf.read_csv("/data/quant14/EBY/day7.csv", usecols=usecols))
    for i in range(196, 197):
        df = dask_cudf.read_csv(f"/data/quant14/EBY/day{i}.csv", usecols=usecols)
        gpu_df_list.append(df)
    
    gpu_concat_df = dask_cudf.concat(gpu_df_list)

    # Call the modified plotting function with the two feature lists
    analyze_feature(gpu_concat_df, PRIMARY_AXIS_FEATURES, SECONDARY_AXIS_FEATURES)