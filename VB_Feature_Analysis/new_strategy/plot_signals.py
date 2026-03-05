import dask_cudf
import plotly.graph_objects as go
import cudf
import numpy as np
import pandas as pd

def get_signals_for_day(day_num):
    try:
        print("Reading CSV file with GPU acceleration...")
        ddf = dask_cudf.read_csv('/data/quant14/signals/eama_kama_sma_crossover_signals.csv')
        df = ddf.compute()

        # ============================================
        # CRITICAL FIX: REMOVE NON-UNIQUE GPU INDEX
        # ============================================
        df = df.reset_index(drop=True)

        print(f"Loaded {len(df):,} rows")

        print("Processing time data...")
        time_pd = df["Time"].to_pandas()
        time_pd = pd.to_timedelta(time_pd)
        time_diff = time_pd.diff()
        day_transitions = (time_diff < pd.Timedelta(0)).astype(int).cumsum()

        # Safe assignment
        df["Day"] = cudf.Series(day_transitions.values)

        print("Calculating positions...")
        signals = df['Signal'].to_pandas().values

        position = []
        current_position = 0
        for signal in signals:
            if signal == 1:
                current_position = 1 if current_position <= 0 else 0
            elif signal == -1:
                current_position = -1 if current_position >= 0 else 0
            position.append(current_position)

        # Safe assignment (indexes now match)
        df["Position"] = cudf.Series(position)

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

        # ============================================================
        # FIX: RECOMPUTE BASE FILTERS (they are NOT in the CSV)
        # ============================================================

        prices = df_pandas["Price"].astype(float).to_numpy()

        # ----- 1. Base Kalman -----
        def kalman_smooth_plot(prices, Q=0.0001, R=0.1):
            n = len(prices)
            out = np.zeros(n)
            out[0] = prices[0]
            p = 1.0  # error estimate

            for i in range(1, n):
                # Predict
                p = p + Q

                # Update
                K = p / (p + R)
                out[i] = out[i-1] + K * (prices[i] - out[i-1])
                p = (1 - K) * p

            return out

        df_pandas["Base_Kalman"] = kalman_smooth_plot(prices)


        # ----- 2. Recompute Ehlers Supersmoother -----
        def supersmooth_plot(series, length=20):
            series = np.asarray(series)
            n = len(series)
            out = np.full(n, np.nan)
            
            a1 = np.exp(-1.414 * np.pi / length)
            b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
            c2 = b1
            c3 = -a1 * a1
            c1 = 1 - c2 - c3
            
            out[0] = series[0]
            out[1] = series[1]
            
            for i in range(2, n):
                out[i] = c1 * (series[i] + series[i-1]) / 2 + c2 * out[i-1] + c3 * out[i-2]
            
            return out

        df_pandas["Base_Supersmoother"] = supersmooth_plot(prices)


        # ----- 3. Adaptive EMA -----
        def adaptive_ema_plot(series, fast=2, slow=30):
            series = pd.Series(series)
            change = abs(series.diff(slow))
            vol = series.diff().abs().rolling(slow).sum()
            er = (change / vol.replace(0, np.nan)).fillna(0)

            fast_sc = 2/(fast+1)
            slow_sc = 2/(slow+1)
            sc = (er*(fast_sc - slow_sc) + slow_sc)**2
            
            out = np.zeros(len(series))
            out[0] = series.iloc[0]

            for i in range(1, len(series)):
                out[i] = out[i-1] + sc.iloc[i] * (series.iloc[i] - out[i-1])
            
            return out

        df_pandas["Base_AEMA"] = adaptive_ema_plot(prices)


        # ----- 4. Ehlers Filter -----
        def ehlers_filter_plot(series, length=25):
            series = np.asarray(series)
            alpha = 2 / (length + 1)
            n = len(series)
            out = np.zeros(n)
            out[0] = series[0]
            out[1] = series[1]

            for i in range(2, n):
                out[i] = (alpha - alpha**2/4)*series[i] + \
                        (alpha**2/2)*series[i-1] - \
                        (alpha - 3*alpha**2/4)*series[i-2] + \
                        2*(1-alpha)*out[i-1] - \
                        (1-alpha)**2 * out[i-2]
            return out

        df_pandas["Base_Ehlers"] = ehlers_filter_plot(prices)


        df_pandas = df_pandas.copy()
        df_pandas["Price"] = df_pandas["Price"].astype(float).to_numpy()


        # =====================================================
        # CALCULATE 4 REQUIRED FEATURES FOR PLOTTING
        # EAMA, SMA240, EMA130, HMA9
        # =====================================================

        prices = df_pandas["Price"].to_numpy(dtype=float)

        # --------------------------
        # 1. EAMA (adaptive EMA)
        # --------------------------
        eama_period = 10
        eama_fast = 15
        eama_slow = 40

        ss = pd.Series(prices).ewm(span=30, adjust=False).mean()

        direction_ss = ss.diff(eama_period).abs()
        volatility_ss = ss.diff().abs().rolling(eama_period).sum()
        er_ss = (direction_ss / volatility_ss).fillna(0)

        fast_sc = 2/(eama_fast+1)
        slow_sc = 2/(eama_slow+1)
        sc_ss = ((er_ss*(fast_sc - slow_sc)) + slow_sc)**2

        eama = np.zeros(len(ss))
        eama[0] = ss.iloc[0]
        for i in range(1, len(ss)):
            eama[i] = eama[i-1] + sc_ss.iloc[i] * (ss.iloc[i] - eama[i-1])

        df_pandas["EAMA"] = eama

        # --------------------------
        # 2. SMA240
        # --------------------------
        df_pandas["SMA_240"] = (
            pd.Series(prices)
            .rolling(240, min_periods=1)
            .mean()
            .bfill()
        )


        # --------------------------
        # 3. EMA130
        # --------------------------
        df_pandas["EMA_130"] = pd.Series(prices).ewm(span=130, adjust=False).mean()

        # --------------------------
        # 4. HMA9
        # --------------------------
        # ============================================================
        # ⭐ Correct Hull Moving Average (HMA-9) from Kalman Smoothed Price
        # ============================================================

        def wma_numpy(arr, length):
            """
            Proper Weighted Moving Average (WMA)
            using full NumPy, no pandas, no GPU objects.
            """
            arr = np.asarray(arr, dtype=float)
            out = np.full(len(arr), np.nan)
            weights = np.arange(1, length + 1)

            for i in range(length - 1, len(arr)):
                window = arr[i - length + 1 : i + 1]
                out[i] = np.dot(window, weights) / weights.sum()

            return out


        # -------------------------
        # Use Kalman smoothed data
        # -------------------------
        kal = df_pandas["Base_Kalman"].to_numpy(dtype=float)

        # HMA(9) = WMA( 2*WMA(kal,4) - WMA(kal,9), 3 )
        # ------------------------------------------

        # Step 1: WMA(9)
        wma_9 = wma_numpy(kal, 9)

        # Step 2: WMA(4)  (because 9/2 = 4)
        wma_4 = wma_numpy(kal, 4)

        # Step 3: Intermediate series = 2*WMA(4) − WMA(9)
        hma_intermediate = 2 * wma_4 - wma_9

        # Step 4: Final HMA = WMA( sqrt(9)=3 )
        HMA_9 = wma_numpy(hma_intermediate, 3)

        # Save to dataframe
        df_pandas["HMA_9"] = HMA_9


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
    days_to_plot = [50]  # Change this list to plot different days
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
        # ADD FEATURE PLOTS ON PRICE PANEL
        # ============================================

        # EAMA
        fig.add_trace(
            go.Scatter(
                x=day_data_pd.index,
                y=day_data_pd["EAMA"],
                name="EAMA (10/15/40)",
                line=dict(color='blue', width=2),
            )
        )

        # SMA240
        fig.add_trace(
            go.Scatter(
                x=day_data_pd.index,
                y=day_data_pd["SMA_240"],
                name="SMA 240",
                line=dict(color='orange', width=2, dash="dash"),
            )
        )

        # EMA130
        fig.add_trace(
            go.Scatter(
                x=day_data_pd.index,
                y=day_data_pd["EMA_130"],
                name="EMA 130",
                line=dict(color='purple', width=2, dash="dot"),
            )
        )

        # HMA9
        fig.add_trace(
            go.Scatter(
                x=day_data_pd.index,
                y=day_data_pd["HMA_9"],
                name="HMA 9",
                line=dict(color='green', width=2),
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
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"Day {day_num} - Price vs Trading Signals ({len(day_data_pd):,} points)",
            xaxis_title="Index",
            yaxis=dict(
                title="Price ($)",
                side='left',
                tickformat='.2f'
            ),
            yaxis2=dict(
                title="Position",
                overlaying='y',
                side='right',
                range=[-1.5, 1.5]
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
