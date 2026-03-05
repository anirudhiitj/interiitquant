import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import plotly.graph_objects as go # Import Plotly

def smooth_prices_kalman(prices):
    """
    Smooths a time series of prices using a simple Kalman Filter.
    
    This uses a "local level" model, assuming the price is a 
    random walk with observation noise.
    
    Args:
        prices (pd.Series or np.array): A time series of observed prices.

    Returns:
        pd.Series: The smoothed (filtered) price estimates.
    """
    # --- 1. Configure the Kalman Filter ---
    
    # We are modeling a 1D random walk.
    # State: [true_price]
    # Observation: [observed_price]
    
    # transition_matrix (F): How the state evolves. 
    # [true_price_t] = 1 * [true_price_t-1] + noise
    transition_matrix = [1] 
    
    # observation_matrix (H): How the state is observed.
    # [observed_price_t] = 1 * [true_price_t] + noise
    observation_matrix = [1]
    
    # transition_covariance (Q): The noise in the state transition (process noise).
    # How much do we expect the "true" price to jump around?
    # This is a key tuning parameter. A smaller value means a smoother line.
    transition_covariance = 0.01  # Tune this!
    
    # observation_covariance (R): The noise in the observation (measurement noise).
    # How much noise do we think is in our observed close price?
    # We can often set this to 1 and tune Q relative to it.
    observation_covariance = 1.0  # Tune this!
    
    # initial_state_mean (x_0): Our best guess of the starting price.
    initial_state_mean = prices.iloc[0] if isinstance(prices, pd.Series) else prices[0]
    
    # initial_state_covariance (P_0): Our uncertainty about the starting price.
    initial_state_covariance = 1.0

    # --- 2. Create and Run the Filter ---
    
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance
    )
    
    # 'filter' is the standard forward-pass estimation.
    filtered_state_means, _ = kf.filter(prices)
    
    # The result is 1D, so we squeeze it.
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=prices.index)
    
    return smoothed_prices

# --- Example Usage with Plotly ---

# 1. Create some sample data
if __name__ == "__main__":
    
    observed_price = pd.read_parquet("/data/quant14/EBY/day0.parquet",columns=["PB9_T1"]).dropna()["PB9_T1"]
    
    # 2. Run the smoother
    # NOTE: You'll pass your actual close prices here
    smoothed_price = smooth_prices_kalman(observed_price)
    
    # 3. Plot the results with Plotly
    fig = go.Figure()

    # Add Observed Price
    fig.add_trace(go.Scatter(
        x=observed_price.index, 
        y=observed_price,
        mode='lines',
        name='Observed Close Price',
        line=dict(color='blue', width=1, dash='dash'),
        opacity=0.7
    ))

    # Add Kalman Filter Smoothed Price
    fig.add_trace(go.Scatter(
        x=smoothed_price.index, 
        y=smoothed_price,
        mode='lines',
        name='Kalman Filter (Smoothed)',
        line=dict(color='red', width=2.5)
    ))

    # Update layout
    fig.update_layout(
        title='Kalman Filter for Price Smoothing',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        legend_title='Series',
        hovermode="x unified" # Great for time series
    )

    fig.show()