import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def backtest(df, initial_capital: float=100000, transaction_cost_rate:float=0.02, slippage: float = 0.0):
    transaction_cost_rate /= 100
    holding_penalty = 1
    df["Time"] = pd.to_timedelta(df["Time"])
    df["Day"] = (df["Time"].diff() < pd.Timedelta(0)).cumsum()
    df = df.dropna(subset=["Price"])
    
    # Create sequential index for continuous plotting
    df = df.reset_index(drop=True)
    df["PlotIndex"] = range(len(df))

    df["Adjusted_Signal"] = 0
    df["Position"] = 0

    Days=0
    
    Daily_PnL=[]
    Total_PnL=0

    Daily_Trade_Count=[]
    Total_Trades=0

    Penalty_Counts=0
    
    # Track holding periods - store tuples of (entry_idx, exit_idx)
    All_Trades = []

    for day, day_df in df.groupby("Day"):
        current_position = 0
        Position=[]
        Adjusted_Signal = []
        entry_indices = []  # Track all entry points
        
        shifted_signals = day_df["Signal"].shift(1).fillna(day_df["Signal"].iloc[0]).tolist()
        shifted_signals[-1] = day_df["Signal"].iloc[-1]
        Factor=initial_capital/day_df["Price"].iloc[0]
        
        for i,sig in enumerate(day_df["Signal"]):
            if sig != current_position and sig != 0:
                Adjusted_Signal.append(sig)
                entry_indices.append((i, sig))  # Store entry index and direction
                current_position += sig
            else:
                Adjusted_Signal.append(0)
            
            if i == len(day_df) - 1 and current_position != 0:
                Adjusted_Signal[-1] += -current_position
                current_position = 0
                Penalty_Counts+=1

            Position.append(current_position)

        # Now process entry_indices to match with exits
        # An exit occurs when Adjusted_Signal reverses position
        exit_idx = 0
        for entry_idx, entry_direction in entry_indices:
            # Find the next exit after this entry
            found_exit = False
            for j in range(entry_idx + 1, len(Adjusted_Signal)):
                # Exit is when we have an opposite signal
                if Adjusted_Signal[j] != 0 and np.sign(Adjusted_Signal[j]) == -np.sign(entry_direction):
                    hold_time = j - entry_idx
                    All_Trades.append(hold_time)
                    found_exit = True
                    break
            
            # If no exit found, it means EOD forced exit
            if not found_exit:
                hold_time = len(Adjusted_Signal) - 1 - entry_idx
                All_Trades.append(hold_time)

        day_df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        day_df.loc[day_df.index, "Position"] = Position

        Trade_Quantity = day_df["Position"].diff() * Factor
        Trade_Quantity.iloc[0] = day_df["Position"].iloc[0] * Factor

        Unit_Transaction_Cost = abs(day_df["Adjusted_Signal"] * day_df["Price"]) * transaction_cost_rate + abs(day_df["Adjusted_Signal"]) * slippage
        Transaction_Cost = Unit_Transaction_Cost * Factor

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

        Days+=1

        Daily_PnL.append(PnL.iloc[-1])
        Total_PnL+=PnL.iloc[-1]

        Daily_Trade_Count.append((day_df["Adjusted_Signal"]!=0).sum())
        Total_Trades+=(day_df["Adjusted_Signal"]!=0).sum()

        print("Day",day,"-> Daily PnL:", PnL.iloc[-1], "| Total PnL:", Total_PnL)
    print("\n")

    # Calculate average holding period
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

    Daily_Returns = np.array(Daily_PnL) / initial_capital
    Sharpe_Ratio = np.sqrt(252) * (Daily_Returns.mean() / Daily_Returns.std())

    Daily_MDD = []
    Intraday_Sharpes = []
    for day, day_df in df.groupby("Day"):
        NAV = day_df["NAV"].values
        Returns = day_df["NAV"].pct_change().dropna()
        if len(Returns) > 1 and Returns.std() != 0:
            intraday_sharpe = Returns.mean() / Returns.std()
        else:
            intraday_sharpe = 0
        Intraday_Sharpes.append(intraday_sharpe)
        Running_Max = np.maximum.accumulate(NAV)
        DD = (NAV - Running_Max) / Running_Max
        df.loc[day_df.index, "Drawdown"] = DD
        Daily_MDD.append(DD.min())
    Overall_Intraday_Sharpe_Ratio = np.mean(Intraday_Sharpes)
    Annualized_Sharpe_Ratio = Sharpe_Ratio
    Maximum_Drawdown = 100 * (min(Daily_MDD))
    Annualized_Returns = 100 * ((Final_Capital / initial_capital) ** (252/Days) - 1)
    Calmar_Ratio = Annualized_Returns / abs(Maximum_Drawdown)
    Best_Day_PnL = max(Daily_PnL)
    Worst_Day_PnL = min(Daily_PnL)
    Best_Day = Daily_PnL.index(Best_Day_PnL)
    Worst_Day = Daily_PnL.index(Worst_Day_PnL)
    Losing_Days = sum(1 for x in Daily_PnL if x < 0)
    Winning_Days = sum(1 for x in Daily_PnL if x > 0)
    Average_Winning_Day_PnL = np.mean([x for x in Daily_PnL if x > 0]) if Winning_Days > 0 else 0
    Average_Losing_Day_PnL = np.mean([x for x in Daily_PnL if x < 0]) if Losing_Days > 0 else 0
    Total_Trades /= 2

    Results = {
        "Initial Capital": initial_capital,
        "Final Capital": Final_Capital,
        "Total PnL": Total_PnL,
        "Penalty Counts": Penalty_Counts,
        "Total Returns": Total_Returns,
        "CAGR": CAGR,
        "Annualized Returns": Annualized_Returns,
        "Overall Intraday Sharpe Ratio": Overall_Intraday_Sharpe_Ratio,
        "Annualized Sharpe Ratio":Annualized_Sharpe_Ratio,
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
        "Total Trades Executed": Total_Trades,
        "Average Hold Period (seconds)": Avg_Hold_Period_Seconds,
        "Average Hold Period (minutes)": Avg_Hold_Period_Minutes,
        "Median Hold Period (seconds)": Median_Hold_Period_Bars,
        "Min Hold Period (seconds)": Min_Hold_Period_Bars,
        "Max Hold Period (seconds)": Max_Hold_Period_Bars,
        "P25 Hold Period (seconds)": P25_Hold,
        "P75 Hold Period (seconds)": P75_Hold,
        "P90 Hold Period (seconds)": P90_Hold,
        "P95 Hold Period (seconds)": P95_Hold,
    }

    print("\n=== Backtest Summary ===")
    print(f"Initial Capital               : {initial_capital:,.2f}")
    print(f"Final Capital                 : {Final_Capital:,.2f}")
    print(f"Total PnL                     : {Total_PnL:,.2f}")
    print(f"Penalty Counts                : {Penalty_Counts}")
    print(f"Final Returns                 : {Total_Returns:,.2f}%")
    print(f"CAGR                          : {CAGR:,.4f}%")
    print(f"Annualized Returns            : {Annualized_Returns:,.4f}%")
    print(f"Overall Intraday Sharpe Ratio : {Overall_Intraday_Sharpe_Ratio:,.4f}")
    print(f"Annualized Sharpe Ratio       : {Annualized_Sharpe_Ratio:,.4f}")
    print(f"Calmar Ratio                  : {Calmar_Ratio:,.4f}")
    print(f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%")
    print(f"No. of Days                   : {Days}")
    print(f"Winning Days                  : {Winning_Days}")
    print(f"Losing Days                   : {Losing_Days}")
    print(f"Best Day                      : {Best_Day}")
    print(f"Worst Day                     : {Worst_Day}")
    print(f"Best Day PnL                  : {Best_Day_PnL:,.2f}")
    print(f"Worst Day PnL                 : {Worst_Day_PnL:,.2f}")
    print(f"Average Winning Day PnL       : {Average_Winning_Day_PnL:,.2f}")
    print(f"Average Losing Day PnL        : {Average_Losing_Day_PnL:,.2f}")
    print(f"Total Trades Executed         : {Total_Trades}")
    print(f"Average Hold Period           : {Avg_Hold_Period_Seconds:,.2f}s ({Avg_Hold_Period_Minutes:,.2f} min)")
    print("========================\n")

    # --- save the same summary to a text file ---
    txt_filename = f"backtest_results.txt"
    report_lines = [
        "=== Backtest Summary ===",
        f"Initial Capital               : {initial_capital:,.2f}",
        f"Final Capital                 : {Final_Capital:,.2f}",
        f"Total PnL                     : {Total_PnL:,.2f}",
        f"Penalty Counts                : {Penalty_Counts}",
        f"Final Returns                 : {Total_Returns:,.2f}%",
        f"CAGR                          : {CAGR:,.4f}%",
        f"Annualized Returns            : {Annualized_Returns:,.4f}%",
        f"Overall Intraday Sharpe Ratio : {Overall_Intraday_Sharpe_Ratio:,.4f}",
        f"Annualized Sharpe Ratio       : {Annualized_Sharpe_Ratio:,.4f}",
        f"Calmar Ratio                  : {Calmar_Ratio:,.4f}",
        f"Maximum Drawdown              : {Maximum_Drawdown:,.4f}%",
        f"No. of Days                   : {Days}",
        f"Winning Days                  : {Winning_Days}",
        f"Losing Days                   : {Losing_Days}",
        f"Best Day                      : {Best_Day}",
        f"Worst Day                     : {Worst_Day}",
        f"Best Day PnL                  : {Best_Day_PnL:,.2f}",
        f"Worst Day PnL                 : {Worst_Day_PnL:,.2f}",
        f"Average Winning Day PnL       : {Average_Winning_Day_PnL:,.2f}",
        f"Average Losing Day PnL        : {Average_Losing_Day_PnL:,.2f}",
        f"Total Trades Executed         : {Total_Trades}",
        "",
        "=== HOLDING PERIOD STATISTICS ===",
        f"Average Hold Period           : {Avg_Hold_Period_Seconds:,.2f}s ({Avg_Hold_Period_Minutes:,.2f} min)",
        f"Median Hold Period            : {Median_Hold_Period_Bars:,.2f}s ({Median_Hold_Period_Bars/60:,.2f} min)",
        f"Min Hold Period               : {Min_Hold_Period_Bars:,.2f}s ({Min_Hold_Period_Bars/60:,.2f} min)",
        f"Max Hold Period               : {Max_Hold_Period_Bars:,.2f}s ({Max_Hold_Period_Bars/60:,.2f} min)",
        "",
        "========================",
        "",
    ]
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Text summary saved to '{txt_filename}'")
    
    print(f"✓ Backtest completed successfully!")
    
    return Results

# Example usage
if __name__ == "__main__":
    df=pd.read_csv("/data/quant14/signals/bb_signals.csv")
    backtest(df, 100, 0.02)