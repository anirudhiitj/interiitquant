import dask_cudf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def backtest(df, initial_capital: float=100000, transaction_cost_rate:float=0.02, slippage: float = 0.0):
    # REMOVED: transaction_cost_rate /= 100 
    # We now treat 'transaction_cost_rate' as a FIXED value per signal (e.g., 0.02)
    
    holding_penalty = 1
    interval_minutes = 30
    df["Time"] = pd.to_timedelta(df["Time"])
    df["Day"] = (df["Time"].diff() < pd.Timedelta(0)).cumsum()
    df["Day_Index"] = df.groupby("Day").cumcount()
    df = df.dropna(subset=["Price"])
    
    # Create sequential index for continuous plotting
    df = df.reset_index(drop=True)
    df["PlotIndex"] = range(len(df))

    df["Adjusted_Signal"] = 0
    df["Position"] = 0

    Days=0
    
    Daily_PnL=[]
    # List to track the specific Day ID for the CSV
    Day_Tracker = [] 
    
    # List to track Drawdown for each individual trade
    Trade_Drawdowns = []

    # NEW: List to track PnL details for each individual trade
    Trade_PnL_Details = []
    Global_Trade_Counter = 0

    Total_PnL=0

    Daily_Trade_Count=[]

    Penalty_Counts=0
    All_Trades = []
    Total_Trades = 0
    Winning_Trades = 0
    Losing_Trades = 0
    Winning_Trade_PnLs = []
    Losing_Trade_PnLs = []

    Interval_PnL = np.zeros(15)
    
    # --- Main Loop ---
    for day, day_df in df.groupby("Day"):
        current_position = 0
        Position=[]
        Adjusted_Signal = []
        Entry_Indices = []
        shifted_signals = day_df["Signal"].shift(1).fillna(day_df["Signal"].iloc[0]).tolist()
        shifted_signals[-1] = day_df["Signal"].iloc[-1]
        Factor=initial_capital/day_df["Price"].iloc[0]
        
        # --- Signal Processing ---
        for i,sig in enumerate(day_df["Signal"]):
            if sig != current_position and sig != 0:
                Adjusted_Signal.append(sig)
                current_position += sig
                Entry_Indices.append((i, sig))
            else:
                Adjusted_Signal.append(0)
            
            if i == len(day_df) - 1 and current_position != 0:
                Adjusted_Signal[-1] += -current_position
                current_position = 0
                Penalty_Counts+=1

            Position.append(current_position)
        
        exit_idx = 0
        # Iterate through identified entries
        for entry_idx, entry_direction in Entry_Indices:
            entry_price = day_df["Price"].iloc[entry_idx]
            exit_idx = None

            # Find exit
            for j in range(entry_idx + 1, len(Adjusted_Signal)):
                # We look for a signal that is non-zero and opposite to entry
                if Adjusted_Signal[j] != 0 and np.sign(Adjusted_Signal[j]) == -np.sign(entry_direction):
                    exit_idx = j
                    break

            # If no exit found → forced exit at last bar
            if exit_idx is None:
                exit_idx = len(Adjusted_Signal) - 1

            exit_price = day_df["Price"].iloc[exit_idx]

            # --- Calculate Trade Specific Drawdown (MAE) ---
            prices_during_trade = day_df["Price"].iloc[entry_idx : exit_idx + 1].values
            
            if len(prices_during_trade) > 0:
                if entry_direction > 0: # LONG
                    min_price = np.min(prices_during_trade)
                    trade_dd = (min_price - entry_price) / entry_price
                else: # SHORT
                    max_price = np.max(prices_during_trade)
                    trade_dd = (entry_price - max_price) / entry_price
                
                Trade_Drawdowns.append(min(0.0, trade_dd))
            else:
                Trade_Drawdowns.append(0.0)
            # ----------------------------------------------------

            # HOLD TIME
            hold_time = exit_idx - entry_idx
            All_Trades.append(hold_time)

            # TRADE PnL (Gross)
            if entry_direction > 0:  # LONG
                raw_trade_pnl = exit_price - entry_price
                direction_str = "Long"
            else:                    # SHORT
                raw_trade_pnl = entry_price - exit_price
                direction_str = "Short"

            # --- CORRECTION: Fixed Cost Logic ---
            # Cost = (Entry Cost) + (Exit Cost) + (Slippage * 2)
            # User specified: 0.02 per signal -> 0.04 per trade
            entry_cost = transaction_cost_rate + slippage
            exit_cost = transaction_cost_rate + slippage
            
            total_trade_cost = entry_cost + exit_cost
            
            # Net PnL = Gross PnL - Fixed Costs
            net_trade_pnl = raw_trade_pnl - total_trade_cost
            
            # Convert to Cash (using the Factor)
            realized_pnl_cash = net_trade_pnl * Factor
            # ------------------------------------

            Global_Trade_Counter += 1
            
            Trade_PnL_Details.append({
                "Trade_Number": Global_Trade_Counter,
                "Day": day,
                "Direction": direction_str,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "PnL": realized_pnl_cash
            })
            
            # WIN / LOSS COUNTING
            if net_trade_pnl > 0:
                Winning_Trades += 1
                Winning_Trade_PnLs.append(realized_pnl_cash)
            elif net_trade_pnl <= 0:
                Losing_Trades += 1
                Losing_Trade_PnLs.append(realized_pnl_cash)

        # Count total trades for this day
        Total_Trades += len(Entry_Indices)

        day_df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        day_df.loc[day_df.index, "Position"] = Position

        Trade_Quantity = day_df["Position"].diff() * Factor
        Trade_Quantity.iloc[0] = day_df["Position"].iloc[0] * Factor

        # --- CORRECTION: Global Transaction Cost Logic ---
        # Fixed cost per signal: abs(Signal) * rate (e.g., 1 * 0.02)
        # Old (Percentage): abs(Signal * Price) * rate
        Unit_Transaction_Cost = abs(day_df["Adjusted_Signal"]) * transaction_cost_rate + abs(day_df["Adjusted_Signal"]) * slippage
        Transaction_Cost = Unit_Transaction_Cost * Factor
        df.loc[day_df.index, "Transaction_Cost"] = Transaction_Cost
        # -------------------------------------------------

        Unit_Cash = day_df["Price"].iloc[0] - (Adjusted_Signal * day_df["Price"] + Unit_Transaction_Cost).cumsum()
        
        Cash = Unit_Cash * Factor

        Unit_Holdings = day_df["Position"] * day_df["Price"]
        Holdings = Unit_Holdings * Factor
        NAV = Cash + Holdings
        PnL = NAV - initial_capital

        df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        df.loc[day_df.index, "Position"] = Position
        df.loc[day_df.index, "Trade_Quantity"] = Trade_Quantity
        df.loc[day_df.index, "Transaction_Cost"] = Transaction_Cost
        df.loc[day_df.index, "Unit_Cash"] = Unit_Cash
        df.loc[day_df.index, "Cash"] = Cash
        df.loc[day_df.index, "Holdings"] = Holdings
        df.loc[day_df.index, "NAV"] = NAV
        df.loc[day_df.index, "PnL"] = PnL

        day_df["PnL"]=PnL
        interval = pd.Timedelta(minutes=interval_minutes)
        start_time = day_df["Time"].iloc[0]
        end_time = day_df["Time"].iloc[-1]
        num_intervals = int(np.ceil((end_time - start_time) / interval))

        for i in range(num_intervals):
            t_start = start_time + i * interval
            t_end = start_time + (i + 1) * interval

            # Ensure we stay within data bounds
            df_interval = day_df[(day_df["Time"] >= t_start) & (day_df["Time"] < t_end)]
            if df_interval.empty:
                pnl_in_bin = 0
            else:
                pnl_start = df_interval["PnL"].iloc[0]
                pnl_end = df_interval["PnL"].iloc[-1]
                pnl_in_bin = pnl_end - pnl_start

            if i < len(Interval_PnL):
                Interval_PnL[i] += pnl_in_bin
            else:
                Interval_PnL = np.append(Interval_PnL, pnl_in_bin)

        Days+=1

        Daily_PnL.append(PnL.iloc[-1])
        # Track the day index corresponding to the PnL
        Day_Tracker.append(day)
        
        Total_PnL+=PnL.iloc[-1]

        Daily_Trade_Count.append((day_df["Adjusted_Signal"]!=0).sum())

        print("Day",day,"-> Daily PnL:", PnL.iloc[-1], "| Total PnL:", Total_PnL)
    print("\n")

    # --- Generate and Save Daily PnL CSV ---
    daily_csv_filename = "daily_pnl.csv"
    daily_report_df = pd.DataFrame({
        "Day": Day_Tracker,
        "Net_PnL": Daily_PnL
    })
    daily_report_df.to_csv(daily_csv_filename, index=False)
    print(f"✓ Daily PnL report saved to '{daily_csv_filename}'")
    # ---------------------------------------------

    # --- Generate and Save Trade Drawdown CSV ---
    trade_dd_filename = "trade_drawdowns.csv"
    trade_dd_df = pd.DataFrame({
        "Trade_Number": range(1, len(Trade_Drawdowns) + 1),
        "Drawdown_Pct": Trade_Drawdowns
    })
    trade_dd_df.to_csv(trade_dd_filename, index=False)
    print(f"✓ Trade Drawdown report saved to '{trade_dd_filename}'")
    # -------------------------------------------------

    # --- NEW: Generate and Save Trade PnL CSV ---
    trade_pnl_filename = "trade_pnl.csv"
    if Trade_PnL_Details:
        trade_pnl_df = pd.DataFrame(Trade_PnL_Details)
        # Reorder columns for better readability
        cols = ["Trade_Number", "Day", "Direction", "Entry_Price", "Exit_Price", "PnL"]
        trade_pnl_df = trade_pnl_df[cols]
        trade_pnl_df.to_csv(trade_pnl_filename, index=False)
        print(f"✓ Trade PnL report saved to '{trade_pnl_filename}'")
    else:
        print(f"ℹ No trades executed, skipping {trade_pnl_filename}")
    # -------------------------------------------------

    if len(All_Trades) > 0:
        Avg_Hold_Period_Bars = np.mean(All_Trades)
        Median_Hold_Period_Bars = np.median(All_Trades)
        Min_Hold_Period_Bars = np.min(All_Trades)
        Max_Hold_Period_Bars = np.max(All_Trades)
        # Convert to seconds (assuming 1 bar = 1 second)
        Avg_Hold_Period_Seconds = Avg_Hold_Period_Bars
        Avg_Hold_Period_Minutes = Avg_Hold_Period_Seconds / 60
        
        # Calculate percentiles
        P25_Hold = np.percentile(All_Trades, 25)
        P75_Hold = np.percentile(All_Trades, 75)
        P90_Hold = np.percentile(All_Trades, 90)
        P95_Hold = np.percentile(All_Trades, 95)
    else:
        Avg_Hold_Period_Bars = 0
        Median_Hold_Period_Bars = 0
        Min_Hold_Period_Bars = 0
        Max_Hold_Period_Bars = 0
        Avg_Hold_Period_Seconds = 0
        Avg_Hold_Period_Minutes = 0
        P25_Hold = 0
        P75_Hold = 0
        P90_Hold = 0
        P95_Hold = 0

    Final_Capital = initial_capital + Total_PnL
    Total_Returns = (Final_Capital - initial_capital) * 100 / initial_capital
    Final_Returns = Total_Returns - Penalty_Counts * holding_penalty

    CAGR = 100*((Final_Capital / initial_capital) ** (252 / Days) - 1)

    # Daily_Returns = np.array(Daily_PnL) / initial_capital
    
    Daily_MDD = []
    for day, day_df in df.groupby("Day"):
        NAV = day_df["NAV"].values
        Running_Max = np.maximum.accumulate(NAV)
        DD = (NAV - Running_Max) / Running_Max
        df.loc[day_df.index, "Drawdown"] = DD
        Daily_MDD.append(DD.min())

    Equity = initial_capital+np.cumsum(Daily_PnL)
    Returns=np.diff(Equity)/(np.abs(Equity[:-1])+1e-9)
    Sharpe_Ratio=(np.mean(Returns) / np.std(Returns)) * np.sqrt(len(Returns))
    Total_TCost = np.cumsum(df["Transaction_Cost"])
    
    Maximum_Drawdown = 100 * (min(Daily_MDD))
    Annualized_Returns = 100 * (((Final_Capital-initial_capital) * 252) / (initial_capital * Days))
    Calmar_Ratio = Annualized_Returns / abs(Maximum_Drawdown)
    Best_Day_PnL = max(Daily_PnL)
    Worst_Day_PnL = min(Daily_PnL)
    Best_Day = Daily_PnL.index(Best_Day_PnL)
    Worst_Day = Daily_PnL.index(Worst_Day_PnL)
    Winning_Days = sum(1 for x in Daily_PnL if x > 0)
    Losing_Days = sum(1 for x in Daily_PnL if x < 0)
    Average_Winning_Day_PnL = np.mean([x for x in Daily_PnL if x > 0]) if Winning_Days > 0 else 0
    Average_Losing_Day_PnL = np.mean([x for x in Daily_PnL if x < 0]) if Losing_Days > 0 else 0
    Win_Rate = (Winning_Trades / Total_Trades * 100) if Total_Trades > 0 else 0
    Average_Winning_Trade = np.mean(Winning_Trade_PnLs) if len(Winning_Trade_PnLs) > 0 else 0
    Average_Losing_Trade = np.mean(Losing_Trade_PnLs) if len(Losing_Trade_PnLs) > 0 else 0

    Results = {
        "Initial Capital": initial_capital,
        "Final Capital": Final_Capital,
        "Total PnL": Total_PnL,
        "Total Transaction Cost": Total_TCost.iloc[-1],
        "Penalty Counts": Penalty_Counts,
        "Total Returns": Total_Returns,
        "CAGR": CAGR,
        "Annualized Returns": Annualized_Returns,
        "Sharpe Ratio": Sharpe_Ratio,
        "Calmar Ratio": Calmar_Ratio,
        "Maximum Drawdown": Maximum_Drawdown,
        "No. of Days": Days,
        "Winning Days": Winning_Days,
        "Losing Days": Losing_Days,
        "Best Day": Best_Day,
        "Worst Day": Worst_Day,
        "Best Day PnL": Best_Day_PnL,
        "Worst Day PnL": Worst_Day_PnL,
        "Average Winning Day PnL": Average_Winning_Day_PnL,
        "Average Losing Day PnL": Average_Losing_Day_PnL,
        "Total Trades": Total_Trades,
        "Winning Trades": Winning_Trades,
        "Losing Trades": Losing_Trades,
        "Win Rate (%)": Win_Rate,
        "Average Winning Trade": Average_Winning_Trade,
        "Average Losing Trade": Average_Losing_Trade,
        "Average Hold Period (seconds)": Avg_Hold_Period_Seconds,
    }

    print("\n=== Backtest Summary ===")
    print(f"Initial Capital               : {initial_capital:,.4f}")
    print(f"Final Capital                 : {Final_Capital:,.4f}")
    print(f"Total PnL                     : {Total_PnL:,.4f}")
    print(f"Total Transaction Cost         : {Total_TCost.iloc[-1]:,.4f}")
    print(f"Penalty Counts                : {Penalty_Counts}")
    print(f"Final Returns                 : {Total_Returns:,.4f}%")
    print(f"CAGR                          : {CAGR:,.4f}%")
    print(f"Annualized Returns            : {Annualized_Returns:,.4f}%")
    print(f"Sharpe Ratio                  : {Sharpe_Ratio:,.4f}")
    print(f"Calmar Ratio                  : {Calmar_Ratio:,.4f}")
    print(f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%")
    print(f"No. of Days                   : {Days}")
    print(f"Winning Days                  : {Winning_Days}")
    print(f"Losing Days                   : {Losing_Days}")
    print(f"Best Day                      : {Best_Day}")
    print(f"Worst Day                     : {Worst_Day}")
    print(f"Best Day PnL                  : {Best_Day_PnL:,.4f}")
    print(f"Worst Day PnL                 : {Worst_Day_PnL:,.4f}")
    print(f"Total Trades                  : {Total_Trades}")
    print(f"Winning Trades                : {Winning_Trades}")
    print(f"Losing Trades                 : {Losing_Trades}")
    print(f"Win Rate (%)                  : {Win_Rate:,.4f}%")
    print(f"Average Winning Trade         : {Average_Winning_Trade:,.4f}")
    print(f"Average Losing Trade          : {Average_Losing_Trade:,.4f}")
    print(f"Average Hold Period (seconds) : {Avg_Hold_Period_Seconds:,.4f}")
    print("========================\n")
    print(Interval_PnL)
    
    # --- save summary to a text file ---
    txt_filename = f"backtest_results.txt"
    report_lines = [
        "=== Backtest Summary ===",
        f"Initial Capital               : {initial_capital:,.4f}",
        f"Final Capital                 : {Final_Capital:,.4f}",
        f"Total PnL                     : {Total_PnL:,.4f}",
        f"Total Transaction Cost         : {Total_TCost.iloc[-1]:,.4f}",
        f"Penalty Counts                : {Penalty_Counts}",
        f"Final Returns                 : {Total_Returns:,.4f}%",
        f"CAGR                          : {CAGR:,.4f}%",
        f"Annualized Returns            : {Annualized_Returns:,.4f}%",
        f"Sharpe Ratio                  : {Sharpe_Ratio:,.4f}",
        f"Calmar Ratio                  : {Calmar_Ratio:,.4f}",
        f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%",
        f"No. of Days                   : {Days}",
        f"Winning Days                  : {Winning_Days}",
        f"Losing Days                   : {Losing_Days}",
        f"Best Day                      : {Best_Day}",
        f"Worst Day                     : {Worst_Day}",
        f"Best Day PnL                  : {Best_Day_PnL:,.4f}",
        f"Worst Day PnL                 : {Worst_Day_PnL:,.4f}",
        f"Average Winning Day PnL       : {Average_Winning_Day_PnL:,.4f}",
        f"Average Losing Day PnL        : {Average_Losing_Day_PnL:,.4f}",
        f"Total Trades                  : {Total_Trades}",
        f"Winning Trades                : {Winning_Trades}",
        f"Losing Trades                 : {Losing_Trades}",
        f"Win Rate (%)                  : {Win_Rate:,.4f}",
        f"Average Winning Trade         : {Average_Winning_Trade:,.4f}",
        f"Average Losing Trade          : {Average_Losing_Trade:,.4f}",
        f"Average Hold Period (seconds) : {Avg_Hold_Period_Seconds:,.4f}",
        f"Average Hold Period (minutes) : {Avg_Hold_Period_Minutes:,.4f}",
        f"Median Hold Period (seconds)  : {Median_Hold_Period_Bars:,.4f}",
        f"Min Hold Period (seconds)     : {Min_Hold_Period_Bars:,.4f}",
        f"Max Hold Period (seconds)     : {Max_Hold_Period_Bars:,.4f}",
        f"P25 Hold Period (seconds)     : {P25_Hold:,.4f}",
        f"P75 Hold Period (seconds)     : {P75_Hold:,.4f}",
        f"P90 Hold Period (seconds)     : {P90_Hold:,.4f}",
        f"P95 Hold Period (seconds)     : {P95_Hold:,.4f}",
        "========================",
        "",
    ]
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Text summary saved to '{txt_filename}'")
    
    print(f"✓ Backtest completed successfully!")
    
    return df

# Example usage
if __name__ == "__main__":
    # Ensure you are passing the FIXED COST per signal here (0.02)
    df=pd.read_csv("/data/quant14/signals/combined_signals_EBX2.csv")
    out=backtest(df, 100, 0.02)