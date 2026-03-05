import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pykalman import KalmanFilter
from statsmodels.tsa.seasonal import STL

EBX_Price_Proxies = [
    'PB6_T11', 'PB6_T12',
    'PB9_T1', 'PB9_T2', 'PB9_T3', 'PB9_T4', 'PB9_T5', 'PB9_T6', 'PB9_T7', 'PB9_T8', 'PB9_T9', 'PB9_T10', 'PB9_T11', 'PB9_T12',
    'PB10_T1', 'PB10_T2', 'PB10_T3', 'PB10_T4', 'PB10_T5', 'PB10_T6', 'PB10_T7', 'PB10_T8', 'PB10_T9', 'PB10_T10', 'PB10_T11', 'PB10_T12',
    'PB11_T1', 'PB11_T2', 'PB11_T3', 'PB11_T4', 'PB11_T5', 'PB11_T6', 'PB11_T7', 'PB11_T8', 'PB11_T9', 'PB11_T10', 'PB11_T11', 'PB11_T12',
    'PB13_T12'
]

EBX_Gems = [
    'PB1_T1', 'PB1_T2', 'PB1_T3', 'PB1_T4', 'PB1_T5', 'PB1_T6', 'PB1_T7', 'PB1_T8', 
    'PB1_T9', 'PB1_T10', 'PB1_T11', 'PB1_T12', 'PB2_T1', 'PB2_T2', 'PB2_T3', 'PB2_T4', 
    'PB2_T5', 'PB2_T6', 'PB2_T8', 'PB2_T10', 'PB2_T11', 'PB3_T3', 'PB3_T4', 'PB3_T6', 
    'PB3_T9', 'PB4_T1', 'PB4_T2', 'PB4_T3', 'PB4_T4', 'PB4_T5', 'PB4_T6', 'PB4_T7', 
    'PB4_T8', 'PB4_T9', 'PB4_T10', 'PB4_T11', 'PB4_T12', 'PB5_T1', 'PB5_T2', 'PB5_T3', 
    'PB5_T4', 'PB5_T5', 'PB5_T7', 'PB5_T8', 'PB5_T9', 'PB5_T10', 'PB5_T11', 'PB5_T12', 
    'PB6_T1', 'PB6_T2', 'PB6_T3', 'PB6_T4', 'PB6_T5', 'PB6_T6', 'PB6_T7', 'PB6_T8', 
    'PB6_T9', 'PB6_T10', 'PB7_T5', 'PB7_T8', 'PB7_T9', 'PB7_T10', 'PB7_T11', 'PB7_T12', 
    'PB8_T4', 'PB8_T7', 'PB8_T8', 'PB8_T9', 'PB8_T10', 'PB8_T11', 'PB8_T12', 'PB12_T2', 
    'PB12_T3', 'PB12_T4', 'PB12_T5', 'PB12_T6', 'PB12_T7', 'PB12_T8', 'PB12_T10', 
    'PB12_T11', 'PB12_T12', 'PB13_T1', 'PB13_T2', 'PB13_T3', 'PB13_T5', 'PB13_T8', 
    'PB13_T9', 'PB13_T10', 'PB13_T11', 'PB14_T2', 'PB14_T8', 'PB14_T10', 'PB14_T11', 
    'PB14_T12', 'PB15_T3', 'PB15_T5', 'PB15_T7', 'PB15_T9', 'PB15_T10', 'PB15_T11', 
    'PB15_T12', 'PB16_T7', 'PB16_T8', 'PB16_T9', 'PB16_T10', 'PB16_T11', 'PB16_T12', 
    'PB17_T7', 'PB17_T9', 'PB17_T10', 'PB17_T11', 'PB17_T12', 'PB18_T5', 'PB18_T7', 
    'PB18_T9', 'PB18_T10', 'PB18_T11', 'PB18_T12'
]

EBX_Others = [
    'PB2_T7', 'PB2_T9', 'PB2_T12', 'PB3_T1', 'PB3_T2', 'PB3_T5', 'PB3_T7', 'PB3_T8', 
    'PB3_T10', 'PB3_T11', 'PB3_T12', 'PB5_T6', 'PB7_T1', 'PB7_T2', 'PB7_T3', 'PB7_T4', 
    'PB7_T6', 'PB7_T7', 'PB8_T1', 'PB8_T2', 'PB8_T3', 'PB8_T5', 'PB8_T6', 'PB12_T1', 'PB12_T9', 'PB13_T4', 'PB13_T6', 'PB13_T7', 
    'PB14_T1', 'PB14_T3', 'PB14_T4', 'PB14_T5', 'PB14_T6', 'PB14_T7', 'PB14_T9', 
    'PB15_T1', 'PB15_T2', 'PB15_T4', 'PB15_T6', 'PB15_T8', 'PB16_T1', 'PB16_T2', 
    'PB16_T3', 'PB16_T4', 'PB16_T5', 'PB16_T6', 'PB17_T1', 'PB17_T2', 'PB17_T3', 
    'PB17_T4', 'PB17_T5', 'PB17_T6', 'PB17_T8', 'PB18_T1', 'PB18_T2', 'PB18_T3', 
    'PB18_T4', 'PB18_T6', 'PB18_T8'
]

EBX_High_Lookback_Gems = [
    'PB1_T10', 'PB1_T11', 'PB1_T12', 'PB2_T10', 'PB2_T11', 'PB4_T10', 'PB4_T11', 'PB4_T12',
    'PB5_T10', 'PB5_T11', 'PB5_T12', 'PB6_T10', 'PB7_T10', 'PB7_T11', 'PB7_T12', 'PB8_T10',
    'PB8_T11', 'PB8_T12', 'PB12_T10', 'PB12_T11', 'PB12_T12', 'PB13_T10', 'PB13_T11',
    'PB14_T10', 'PB14_T11', 'PB14_T12', 'PB15_T10', 'PB15_T11', 'PB15_T12', 'PB16_T10',
    'PB16_T11', 'PB16_T12', 'PB17_T10', 'PB17_T11', 'PB17_T12', 'PB18_T10', 'PB18_T11',
    'PB18_T12'
]

EBY_Price_Proxies = [
    'PB6_T11', 'PB6_T12',
    'PB9_T10', 'PB9_T11', 'PB9_T12',
    'PB10_T1', 'PB10_T2', 'PB10_T3', 'PB10_T4', 'PB10_T5', 'PB10_T6', 'PB10_T7', 'PB10_T8', 'PB10_T9', 'PB10_T10', 'PB10_T11', 'PB10_T12',
    'PB11_T1', 'PB11_T2', 'PB11_T3', 'PB11_T4', 'PB11_T5', 'PB11_T6', 'PB11_T7', 'PB11_T8', 'PB11_T9', 'PB11_T10', 'PB11_T11', 'PB11_T12',
]

EBY_Gems = [
    'PB1_T1', 'PB1_T2', 'PB1_T3', 'PB1_T4', 'PB1_T5', 'PB1_T6', 'PB1_T7', 'PB1_T8', 
    'PB1_T9', 'PB1_T10', 'PB1_T11', 'PB1_T12', 'PB2_T1', 'PB2_T3', 'PB2_T4', 'PB2_T5', 
    'PB2_T6', 'PB2_T7', 'PB2_T8', 'PB2_T9', 'PB2_T10', 'PB2_T11', 'PB2_T12', 'PB3_T7', 
    'PB3_T8', 'PB3_T10', 'PB4_T1', 'PB4_T2', 'PB4_T3', 'PB4_T4', 'PB4_T5', 'PB4_T7', 
    'PB4_T9', 'PB4_T10', 'PB4_T11', 'PB4_T12', 'PB5_T2', 'PB5_T3', 'PB5_T4', 'PB5_T5', 
    'PB5_T6', 'PB5_T7', 'PB5_T8', 'PB5_T9', 'PB5_T10', 'PB5_T11', 'PB5_T12', 'PB6_T2', 
    'PB6_T3', 'PB6_T4', 'PB6_T5', 'PB6_T6', 'PB6_T7', 'PB6_T8', 'PB6_T9', 'PB6_T10', 
    'PB7_T2', 'PB7_T7', 'PB7_T8', 'PB7_T10', 'PB7_T11', 'PB7_T12', 'PB8_T8', 'PB8_T9', 
    'PB8_T10', 'PB8_T11', 'PB12_T1', 'PB12_T2', 'PB12_T4', 'PB12_T6', 'PB12_T7', 
    'PB12_T8', 'PB12_T9', 'PB12_T10', 'PB12_T11', 'PB12_T12', 'PB13_T1', 'PB13_T2', 
    'PB13_T3', 'PB13_T4', 'PB13_T5', 'PB13_T6', 'PB13_T7', 'PB13_T8', 'PB13_T9', 
    'PB13_T10', 'PB13_T11', 'PB13_T12', 'PB14_T7', 'PB14_T8', 'PB14_T9', 'PB14_T10', 
    'PB14_T11', 'PB14_T12', 'PB15_T4', 'PB15_T7', 'PB15_T8', 'PB15_T11', 'PB16_T8', 
    'PB16_T10', 'PB16_T11', 'PB16_T12', 'PB17_T4', 'PB17_T7', 'PB17_T9', 'PB17_T11', 
    'PB17_T12', 'PB18_T10', 'PB18_T12'
]

EBY_Others = [
    'PB2_T2', 'PB3_T1', 'PB3_T2', 'PB3_T3', 'PB3_T4', 'PB3_T5', 'PB3_T6', 'PB3_T9', 
    'PB3_T11', 'PB3_T12', 'PB4_T6', 'PB4_T8', 'PB5_T1', 'PB6_T1', 'PB7_T1', 'PB7_T3', 
    'PB7_T4', 'PB7_T5', 'PB7_T6', 'PB7_T9', 'PB8_T1', 'PB8_T2', 'PB8_T3', 'PB8_T4', 
    'PB8_T5', 'PB8_T6', 'PB8_T7', 'PB8_T12', 'PB12_T3', 
    'PB12_T5', 'PB14_T1', 'PB14_T2', 'PB14_T3', 'PB14_T4', 'PB14_T5', 'PB14_T6', 
    'PB15_T1', 'PB15_T2', 'PB15_T3', 'PB15_T5', 'PB15_T6', 'PB15_T9', 'PB15_T10', 
    'PB15_T12', 'PB16_T1', 'PB16_T2', 'PB16_T3', 'PB16_T4', 'PB16_T5', 'PB16_T6', 
    'PB16_T7', 'PB16_T9', 'PB17_T1', 'PB17_T2', 'PB17_T3', 'PB17_T5', 'PB17_T6', 
    'PB17_T8', 'PB17_T10', 'PB18_T1', 'PB18_T2', 'PB18_T3', 'PB18_T4', 'PB18_T5', 
    'PB18_T6', 'PB18_T7', 'PB18_T8', 'PB18_T9', 'PB18_T11'
]

EBY_High_Lookback_Gems = [
    'PB1_T10', 'PB1_T11', 'PB1_T12', 'PB2_T10', 'PB2_T11', 'PB2_T12', 'PB3_T10',
    'PB4_T10', 'PB4_T11', 'PB4_T12', 'PB5_T10', 'PB5_T11', 'PB5_T12', 'PB6_T10',
    'PB7_T10', 'PB7_T11', 'PB7_T12', 'PB8_T10', 'PB8_T11', 'PB12_T10', 'PB12_T11',
    'PB12_T12', 'PB13_T10', 'PB13_T11', 'PB13_T12', 'PB14_T10', 'PB14_T11', 'PB14_T12',
    'PB15_T11', 'PB16_T10', 'PB16_T11', 'PB16_T12', 'PB17_T11', 'PB17_T12', 'PB18_T10',
    'PB18_T12'
]

EBY_Crossover=['PB6_T6','PB6_T11','PB10_T4','PB11_T4', 'PB2_T10', 'PB2_T11', 'PB2_T12', 'PB3_T10']

Useful=[ 'PB1_T5', 'PB1_T6', 'PB1_T7', 'PB1_T8', 'PB1_T9', 'PB1_T10', 'PB1_T11', 'PB1_T12',
        'PB2_T5', 'PB2_T6', 'PB2_T7', 'PB2_T8', 'PB2_T9', 'PB2_T10', 'PB2_T11', 'PB2_T12', 'PB3_T7', 'PB3_T8', 'PB3_T10',
        'PB6_T5', 'PB6_T6', 'PB6_T7', 'PB6_T8', 'PB6_T9', 'PB6_T10', 'PB6_T11', 'PB6_T12',
        'PB7_T7', 'PB7_T8', 'PB7_T10', 'PB7_T11', 'PB7_T12',
        'PB9_T1',
        'PB10_T9', 'PB10_T10', 'PB10_T11', 'PB10_T12',
        'PB11_T9', 'PB11_T10', 'PB11_T11', 'PB11_T12'
    ]
PB1=['PB1_T1', 'PB1_T2', 'PB1_T3', 'PB1_T4','PB1_T5', 'PB1_T6', 'PB1_T7', 'PB1_T8', 'PB1_T9', 'PB1_T10', 'PB1_T11', 'PB1_T12']

PB6=['PB6_T2', 'PB6_T3', 'PB6_T4', 'PB6_T5', 'PB6_T6', 'PB6_T7', 'PB6_T8', 'PB6_T9', 'PB6_T10', 'PB6_T11', 'PB6_T12']

# --- axis assignment lists ---
# Features listed here will be plotted on the primary y-axis (Price)
PRIMARY_AXIS_FEATURES = [
    # example: "KFTrend", "SMA"
    "KFTrend",
    "SMA",
    "KAMA",
    "KAMA_MA"
]

# Features listed here will be plotted on the secondary y-axis (features)
SECONDARY_AXIS_FEATURES = [
    # example: "KAMA", "ATR", "STD"
    "ATR",
    "STD",
]
# If a feature is not in either list it will default to the secondary y-axis.

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
    
    print("Calculating STLTrend...")
    stl_result = STL(data_df['PB9_T1'].dropna(), period=30, robust=True).fit()
    data_df['STLTrend'] = stl_result.trend
    print("Calculating KFTrend...")
    transition_matrix = [1] 
    observation_matrix = [1]
    transition_covariance = 0.005
    observation_covariance = 1.0  # Tune this!
    initial_state_covariance = 1.0
    initial_state_mean = 100
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    filtered_state_means, _ = kf.filter(data_df["PB9_T1"].dropna())

    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=data_df["PB9_T1"].dropna().index)
    # print("Calculating VWMA...")
    # window=1800
    # price_series = data_df["PB9_T1"]
    # kama_signal = data_df["PB9_T1"].diff(window).abs()
    # kama_noise = data_df["PB9_T1"].diff(1).abs().rolling(window).sum()
    # er = (kama_signal / kama_noise).fillna(0)
    # price_vals = price_series.to_numpy()
    # weight_vals = er.to_numpy()
    # price_windows = strided_windows(price_vals, window)
    # weight_windows = strided_windows(weight_vals, window)
    # numerator = np.sum(price_windows * weight_windows, axis=1)
    # denominator = np.sum(weight_windows, axis=1)
    # vwma_values = np.full_like(price_vals, np.nan)
    # result = np.divide(numerator, denominator, 
    #                    out=np.full_like(numerator, np.nan), 
    #                    where=denominator!=0)
    
    # vwma_values[window-1:] = result
    
    # data_df[f"VWMA"] = vwma_values
    # print("VWMA_30 calculation completed.")
    print("Calculating KAMA")
    kama_signal = data_df["PB9_T1"].diff(20).abs()
    kama_noise = data_df["PB9_T1"].diff(1).abs().rolling(window=20).sum()
    er = (kama_signal / kama_noise).fillna(0)

    sc_fast = 2 / (45 + 1)
    sc_slow = 2 / (150 + 1)
    
    sc = (er * (sc_fast - sc_slow) + sc_slow).pow(2)

    kama = pd.Series(np.nan, index=data_df["PB9_T1"].index, dtype=float)

    first_valid_index = sc.first_valid_index()
    if first_valid_index is None:
        return kama

    kama.loc[first_valid_index] = data_df["PB9_T1"].loc[first_valid_index]

    sc_values = sc.to_numpy()
    price_values = data_df["PB9_T1"].to_numpy()
    kama_values = kama.to_numpy()

    for i in range(first_valid_index + 1, len(data_df["PB9_T1"])):
        if np.isnan(kama_values[i-1]):
             kama_values[i] = price_values[i]
        else:
             kama_values[i] = kama_values[i-1] + sc_values[i] * (price_values[i] - kama_values[i-1])
    data_df["KAMA"] = pd.Series(kama_values, index=data_df["PB9_T1"].index)
    data_df["KAMA_MA"]=data_df["KAMA"].rolling(window=30, min_periods=1).mean()
    print("Calculating ATR and STD...")
    tr = data_df['PB9_T1'].diff().abs()
    atr = tr.ewm(span=20).mean()
    data_df["ATR"]=atr

    data_df["STD"]=data_df["PB9_T1"].rolling(window=20, min_periods=1).std()

    data_df["SMA"]=data_df["PB9_T1"].rolling(window=150, min_periods=1).mean()

    data_df["KFTrend"] = smoothed_prices
    print(data_df["KFTrend"])

    feature_columns = [c for c in data_df.columns if c not in ['Time', 'Price', 'Day', 'PlotIndex']]

    normalized_feature_columns = {} # Dictionary to store names {original: normalized}

    for col in feature_columns:
        print(f"  Normalizing feature '{col}'...")
        col_min = data_df[col].min()
        col_max = data_df[col].max()
        col_range = col_max - col_min
        norm_col_name = f"{col}_norm"
        normalized_feature_columns[col] = norm_col_name # Store mapping
        data_df[norm_col_name] = data_df[col]
        # if col_range > 1e-9: # Avoid division by zero
        #     data_df[norm_col_name] = (data_df[col] - col_min) / col_range
        #     print(f"    Created normalized column: {norm_col_name} (Min={col_min:.4f}, Max={col_max:.4f})")
        # else: # Constant value column
        #     data_df[norm_col_name] = 0.5 # Assign midpoint
        #     print(f"    Created normalized column: {norm_col_name} (Constant value, set to 0.5)")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Price (Primary Y-axis) - Day by Day (disconnected)
    for day, df_day in data_df.groupby("Day"):
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"],
            y=df_day["Price"],
            mode='lines',
            name=f"Price Day {day}", # Name for hover, not legend
            line=dict(width=2, color='rgba(0,0,0,0.7)'), # Consistent dark color for price
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Price</b>: %{y:.4f}<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False,
            showlegend=False # Hide individual day price lines from legend
        ), secondary_y=False)

    # Define colors for features
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    plotted_legend_items = set() # To show legend only once per feature

    # Plot Normalized Features (Secondary Y-axis) - Toggleable, disconnected per day
    for i, original_feature_name in enumerate(feature_columns):
        norm_feature = normalized_feature_columns[original_feature_name] # Get normalized column name
        color = colors[i % len(colors)]

        # Plot each day separately for this normalized feature
        for day, df_day in data_df.groupby("Day"):
            feature_mask = df_day[norm_feature].notna()
            feature_data = df_day.loc[feature_mask, norm_feature]

            if len(feature_data) > 0:
                corresponding_times = df_day.loc[feature_mask, "Time"].apply(lambda x: str(x).split()[-1]).tolist()
                corresponding_plot_indices = df_day.loc[feature_mask, "PlotIndex"].tolist()

                # Show legend only for the first trace of each feature group
                show_legend_for_this_trace = (original_feature_name not in plotted_legend_items)
                if show_legend_for_this_trace:
                    plotted_legend_items.add(original_feature_name)

                # decide which axis to use: primary if in PRIMARY_AXIS_FEATURES, else secondary
                if original_feature_name in PRIMARY_AXIS_FEATURES:
                    secondary_y_flag = False
                elif original_feature_name in SECONDARY_AXIS_FEATURES:
                    secondary_y_flag = True
                else:
                    # default to secondary axis
                    secondary_y_flag = True

                fig.add_trace(
                    go.Scatter(
                        x=corresponding_plot_indices,
                        y=feature_data.values.tolist(),
                        mode='lines',
                        name=original_feature_name, # Legend uses original name
                        legendgroup=original_feature_name, # Group traces by original name
                        line=dict(color=color, width=1.5),
                        visible='legendonly', # Start hidden
                        hovertemplate=(
                            "<b>Day</b>: %{customdata[0]}<br>"
                            "<b>Time</b>: %{customdata[1]}<br>"
                            f"<b>{original_feature_name} (Norm)</b>: %{{y:.4f}}<br>" # Show normalized value
                            f"<b>{original_feature_name} (Orig)</b>: %{{customdata[2]:.4f}}<br>" # Show original value
                            "<extra></extra>"
                        ),
                        # Include original feature value in customdata for hover
                        customdata=[[day, time_str, orig_val] for day, time_str, orig_val in zip(
                            [day]*len(corresponding_times),
                            corresponding_times,
                            df_day.loc[feature_mask, original_feature_name].tolist() # Get original values
                        )],
                        connectgaps=False,
                        showlegend=show_legend_for_this_trace # Show legend only once
                    ),
                    secondary_y=secondary_y_flag # choose axis based on lists (default secondary)
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
        tickformat='.10f',
        secondary_y=False
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

    fig.write_html("EBY_1day143.html")
    fig.show()

if __name__ == "__main__":

    cols = cudf.read_parquet("/data/quant14/EBY/day0.parquet", nrows=0).columns
    usecols = [c for c in cols if c in ["Time", "Price"] or c=="PB9_T1"]
    gpu_concat_df = dask_cudf.read_parquet("/data/quant14/EBY/day143.parquet", columns=usecols)
    for i in range(180, 180):
        df= dask_cudf.read_parquet(f"/data/quant14/EBY/day{i}.parquet", columns=usecols)
        gpu_concat_df = dask_cudf.concat([gpu_concat_df, df])

    #combined_df = dask_cudf.read_csv("/data/quant14/EBY/combined.csv", usecols=usecols)
    # Convert back to pandas only when plotting
    analyze_feature(gpu_concat_df)
