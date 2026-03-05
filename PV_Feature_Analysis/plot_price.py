import plotly.graph_objects as go
import pandas as pd
import sys

# --- Your New, Simple Plotting Function ---

def plot_price_vs_time(df_pandas, day_num):
    """
    Plots a simple Price vs. Time chart for a single day's data.
    """
    print(f"Generating simple price plot for day {day_num}...")

    # 1. Ensure 'Time' is in a plottable format (timedelta)
    # This assumes 'Time' column exists, as per your reference code
    try:
        df_pandas["Time"] = pd.to_timedelta(df_pandas["Time"])
        df_pandas = df_pandas.sort_values(by="Time")
    except KeyError:
        print("Error: 'Time' column not found in the DataFrame.")
        return
    except Exception as e:
        print(f"Error converting 'Time' column: {e}")
        return

    # 2. Check for 'Price' column
    if "Price" not in df_pandas.columns:
        print("Error: 'Price' column not found in the DataFrame.")
        return

    # 3. Create the figure
    fig = go.Figure()

    # 4. Add the price trace
    fig.add_trace(go.Scatter(
        x=df_pandas["Time"],
        y=df_pandas["Price"],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2),
        hovertemplate=(
            "<b>Time</b>: %{x|%H:%M:%S}<br>" +
            "<b>Price</b>: %{y:.4f}<extra></extra>"
        )
    ))

    # 5. Update the layout for a clean look
    fig.update_layout(
        title={
            'text': f"Price vs. Time for Day {day_num}",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_tickformat="%H:%M:%S", # Format x-axis to show HH:MM:SS
        yaxis_tickformat=".4f",      # Format y-axis for price
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # 6. Add gridlines
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray')

    # 7. Show the plot
    fig.show()
    print(f"Plot for day {day_num} complete.")


# --- Main execution block to load data and call the plot ---

if __name__ == "__main__":
    # Add the current directory to the path to find the 'filter' module
    sys.path.append('.') 
    
    try:
        from filter import get_pairs_for_day
    except ImportError:
        print("Error: Could not import 'get_pairs_for_day' from filter.py.")
        print("Please make sure 'filter.py' is in the same directory.")
        sys.exit(1)
    
    # --- *** CHANGE THIS VALUE TO PLOT A DIFFERENT DAY *** ---
    day_to_plot = 104
    # ---------------------------------------------------------
    
    print(f"Attempting to load data for day {day_to_plot}...")
    
    # Get pairs and data from your analysis script
    # We ignore 'pairs' here but still need to accept it
    success, pairs, df_pandas = get_pairs_for_day(day_to_plot)
    
    if success:
        print(f"Successfully loaded {len(df_pandas)} rows for day {day_to_plot}.")
        
        # Call the new, simple plotting function
        plot_price_vs_time(df_pandas, day_to_plot)
        
    else:
        print(f"Failed to load data for day {day_to_plot}.")