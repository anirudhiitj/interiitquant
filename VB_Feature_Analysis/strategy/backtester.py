import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def backtest(df, initial_capital: float=100000, transaction_cost_rate:float=0.1, slippage: float = 0.0):
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

    for day, day_df in df.groupby("Day"):
        current_position = 0
        Position=[]
        Adjusted_Signal = []
        shifted_signals = day_df["Signal"].shift(1).fillna(day_df["Signal"].iloc[0]).tolist()
        shifted_signals[-1] = day_df["Signal"].iloc[-1]
        Factor=initial_capital/day_df["Price"].iloc[0]
<<<<<<< HEAD
        for i,sig in enumerate(day_df["Signal"]):
=======
        for i,sig in enumerate(shifted_signals):
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
            if sig != current_position and sig != 0:
                Adjusted_Signal.append(sig)
                current_position += sig
            else:
                Adjusted_Signal.append(0)
            
            if i == len(day_df) - 1 and current_position != 0:
                Adjusted_Signal[-1] += -current_position
                current_position = 0
                Penalty_Counts+=1

            Position.append(current_position)

        day_df.loc[day_df.index, "Adjusted_Signal"] = Adjusted_Signal
        day_df.loc[day_df.index, "Position"] = Position

        Trade_Quantity = day_df["Position"].diff() * Factor
        Trade_Quantity.iloc[0] = day_df["Position"].iloc[0] * Factor

        Unit_Transaction_Cost = abs(day_df["Adjusted_Signal"] * day_df["Price"]) * transaction_cost_rate + abs(day_df["Adjusted_Signal"]) * slippage
        Transaction_Cost = Unit_Transaction_Cost * Factor

        Unit_Cash = day_df["Price"].iloc[0] - (Adjusted_Signal * day_df["Price"] + Unit_Transaction_Cost).cumsum()
        # Unit_Cash = pd.Series(np.zeros(len(day_df)), index=day_df.index)
        # for i in range(0, len(day_df)):
        #     if i==0:
        #         Unit_Cash.iloc[i] = day_df["Price"].iloc[0] - (day_df["Adjusted_Signal"].iloc[i] * day_df["Price"].iloc[i]) - Unit_Transaction_Cost.iloc[i]
        #     else:
        #         Unit_Cash.iloc[i] = Unit_Cash.iloc[i-1] - (day_df["Adjusted_Signal"].iloc[i] * day_df["Price"].iloc[i]) - Unit_Transaction_Cost.iloc[i]
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
    Winning_Days = sum(1 for x in Daily_PnL if x > 0)
    Losing_Days = Days - Winning_Days
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
        "Total Trades Executed": Total_Trades
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
    print(f"Total Trades Executed         : {Total_Trades}")
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
        f"Total Trades Executed         : {Total_Trades}",
        "========================",
        "",
    ]
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✓ Text summary saved to '{txt_filename}'")
    # --- end text file save ---

    Best10Days = np.argsort(Daily_PnL)[-10:][::-1]  # descending order
    Worst10Days = np.argsort(Daily_PnL)[:10]       # ascending order
    Worst5Days = np.argsort(Daily_PnL)[:5]

    days_to_plot = sorted(np.concatenate([Worst5Days]))

    if len(days_to_plot) != 0:

        plot_data=df[df["Day"]==days_to_plot[0]]
        for day in days_to_plot[1:]:
            plot_data=pd.concat([plot_data,df[df["Day"]==day]])

        Days=len(days_to_plot)
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Price vs Strategy Portfolio Value',
                'Price vs Signals',
                'Maximum Drawdown vs Time'
            ),
            specs=[[{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}]]
        )
        
        # Plot 1: Portfolio Value vs Price
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['NAV'],
                name='Strategy Portfolio',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Strategy</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['Price'],
                name='Price',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Plot 2: Price vs Signals
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['Price'],
                name='Price',
                line=dict(color='#1f77b4', width=2),
                showlegend=False,
                hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=2, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=[max(p, 0) for p in plot_data['Position']],
                name='Long Position',
                mode='none',  # no line markers
                fill='tozeroy',
                fillcolor='rgba(0, 128, 0, 0.5)',
                line_shape='hv',
            ),
            row=2, col=1, secondary_y=True
        )

        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=[min(p, 0) for p in plot_data['Position']],
                name='Short Position',
                mode='none',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.5)',
                line_shape='hv',
            ),
            row=2, col=1, secondary_y=True
        )
        
        # Plot 3: Drawdown
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['Drawdown'],
                name='Strategy Drawdown',
                fill='tozeroy',
                line=dict(color='#d62728', width=1),
                fillcolor='rgba(214, 39, 40, 0.3)',
                hovertemplate='<b>Drawdown</b><br>Value: %{y:.2f}%<br>Time: %{x}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # tick_format = '%b %d'
        # dtick = 5 * 24 * 60 * 60 * 1000
        
        # # Update x-axes
        # for row in range(1, 4):
        #     fig.update_xaxes(
        #         tickformat=tick_format,
        #         dtick=dtick,
        #         row=row, col=1,
        #         title_text="Time"
        #     )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1,secondary_y=True)
        fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Signal",range=[-5,5], row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Price", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Drawdown", row=3, col=1)
                
        # Update layout
        fig.update_layout(
            title={
                'text': f'<b>Intraday Backtest Report</b>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            height=1400,
            width=1200,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Save and open HTML file
        html_filename = "backtest_report.html"
        fig.write_html(html_filename, auto_open=True)
        
        print(f"✓ Interactive report saved to '{html_filename}' and opened in browser")
    print(f"✓ Backtest completed successfully!")
    
    return Results

# Example usage
if __name__ == "__main__":
    df=pd.read_csv("portfolio_weights.csv")
<<<<<<< HEAD
    backtest(df, 100000,0.02)
=======
    backtest(df, 100000,0)
>>>>>>> 0d551f03a9295744bc32d11e3f1efb6a9c54aadd
