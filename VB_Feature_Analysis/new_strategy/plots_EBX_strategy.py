import dask_cudf
import plotly.graph_objects as go
import cudf
import numpy as np
import pandas as pd

def get_signals_for_day(day_num):
    try:
        print("Reading CSV file with GPU acceleration...")
        ddf = dask_cudf.read_csv('/data/quant14/signals/temp_trading_signals_EBX.csv')
        df = ddf.compute()

        # Remove non-unique GPU index
        df = df.reset_index(drop=True)

        print(f"Loaded {len(df):,} rows")

        print("Processing time data...")
        time_pd = df["Time"].to_pandas()
        time_pd = pd.to_timedelta(time_pd)
        time_diff = time_pd.diff()
        day_transitions = (time_diff < pd.Timedelta(0)).astype(int).cumsum()

        # Safe assignment
        df["Day"] = cudf.Series(day_transitions.values)

        total_days = int(df['Day'].max()) + 1
        print(f"Total days: {total_days}")
        print(f"Available days: 0 to {total_days - 1}")

        if day_num > df['Day'].max() or day_num < 0:
            print(f"Day {day_num} does not exist")
            return False, None

        print(f"Filtering data for day {day_num}...")
        day_data = df[df['Day'] == day_num].reset_index(drop=True)

        if len(day_data) == 0:
            return False, None

        df_pandas = day_data.to_pandas()

        print(f"Successfully loaded day {day_num}")
        print(f"Data shape: {len(df_pandas)} rows")

        return True, df_pandas

    except Exception as e:
        print(f"Error loading data for day {day_num}: {e}")
        return False, None

if __name__ == "__main__":
    # ============================================
    # SPECIFY WHICH DAYS TO PLOT HERE
    # ============================================
    days_to_plot = [89]  # Change this list to plot different days
    # ============================================

    print(f"\nPlotting {len(days_to_plot)} days: {days_to_plot}")

    # Create a separate plot for each specified day
    for day_num in days_to_plot:
        # Use the helper function to get data
        success, day_data_pd = get_signals_for_day(day_num)
        
        if not success:
            print(f"Skipping day {day_num}...")
            continue
        
        print(f"\nCreating plot for Day {day_num}...")
        print(f"  Day {day_num}: {len(day_data_pd):,} data points")
        
        # Create the plot
        fig = go.Figure()
        
        # Plot black price line
        fig.add_trace(
            go.Scatter(
                x=day_data_pd.index,
                y=day_data_pd['Price'],
                name='Price',
                line=dict(color='black', width=2.5),
                mode='lines',
                hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
            )
        )

        # ============================================
        # ADD STRATEGY PARAMETERS TO PLOT
        # ============================================

        # KAMA (Kaufman Adaptive Moving Average)
        if 'KAMA' in day_data_pd.columns:
            fig.add_trace(
                go.Scatter(
                    x=day_data_pd.index,
                    y=day_data_pd["KAMA"],
                    name="KAMA",
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>KAMA</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )

        # HMA 9 (Hull Moving Average)
        if 'HMA_9' in day_data_pd.columns:
            fig.add_trace(
                go.Scatter(
                    x=day_data_pd.index,
                    y=day_data_pd["HMA_9"],
                    name="HMA 9",
                    line=dict(color='green', width=2),
                    hovertemplate='<b>HMA 9</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )

        # EMA 130
        if 'EMA_130' in day_data_pd.columns:
            fig.add_trace(
                go.Scatter(
                    x=day_data_pd.index,
                    y=day_data_pd["EMA_130"],
                    name="EMA 130",
                    line=dict(color='purple', width=2, dash="dot"),
                    hovertemplate='<b>EMA 130</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )

        # SMA 240
        if 'SMA_240' in day_data_pd.columns:
            fig.add_trace(
                go.Scatter(
                    x=day_data_pd.index,
                    y=day_data_pd["SMA_240"],
                    name="SMA 240",
                    line=dict(color='orange', width=2, dash="dash"),
                    hovertemplate='<b>SMA 240</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )

        
        # Add green circles for +1 signals (long entry)
        enter_long = day_data_pd[day_data_pd['Signal'] == 1]
        if len(enter_long) > 0:
            fig.add_trace(
                go.Scatter(
                    x=enter_long.index,
                    y=enter_long['Price'],
                    name='Long Signal (+1)',
                    mode='markers',
                    marker=dict(
                        symbol='circle', 
                        size=10, 
                        color='green',
                        line=dict(color='darkgreen', width=2)
                    ),
                    hovertemplate='<b>Long Signal</b><br>Price: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )
        
        # Add red circles for -1 signals (short entry)
        enter_short = day_data_pd[day_data_pd['Signal'] == -1]
        if len(enter_short) > 0:
            fig.add_trace(
                go.Scatter(
                    x=enter_short.index,
                    y=enter_short['Price'],
                    name='Short Signal (-1)',
                    mode='markers',
                    marker=dict(
                        symbol='circle', 
                        size=10, 
                        color='red',
                        line=dict(color='darkred', width=2)
                    ),
                    hovertemplate='<b>Short Signal</b><br>Price: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"Day {day_num} - Price vs Trading Signals ({len(day_data_pd):,} points)",
            xaxis_title="Index",
            yaxis=dict(
                title="Price ($)",
                side='left',
                tickformat='.2f'
            ),
            height=800,
            width=1600,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Show the plot
        fig.show()
        
        # Print day statistics
        long_signals = (day_data_pd['Signal'] == 1).sum()
        short_signals = (day_data_pd['Signal'] == -1).sum()
        
        print(f"  Long Signals (+1): {long_signals}")
        print(f"  Short Signals (-1): {short_signals}")
        print(f"  Price range: ${day_data_pd['Price'].min():.2f} - ${day_data_pd['Price'].max():.2f}")
        
        # Print available columns for verification
        print(f"  Available columns: {', '.join(day_data_pd.columns)}")