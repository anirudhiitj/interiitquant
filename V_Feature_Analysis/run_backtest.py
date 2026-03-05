from v_backtester import backtest
import pandas as pd
import os

SIGNAL_DIR = "/home/raid/Quant14/V_Feature_Analysis/first_strat/daily_signals"
INITIAL_CAPITAL = 100000

completed = []
failed = []
results_summary = []  # ✅ store all results in one list

for day in range(510):  # change to 510 later
    signal_file = os.path.join(SIGNAL_DIR, f"trading_signals_day{day}.csv")
    
    if os.path.exists(signal_file):
        try:
            print(f"\n📊 Running backtest for Day {day} → {signal_file}")
    
            df = pd.read_csv(signal_file)

            # Capture the returned metrics dictionary
            results = backtest(df, INITIAL_CAPITAL, 0.0)

            # Print a concise summary for this day
            print(f"✅ Day {day} Results → PnL: {results['Total PnL']:.2f}, "
                  f"Sharpe: {results['Annualized Sharpe Ratio']:.3f}, "
                  f"Calmar: {results['Calmar Ratio']:.3f}, "
                  f"Drawdown: {results['Maximum Drawdown']:.3f}\n")

            # Store in list for later aggregation
            results_summary.append({"Day": day, **results})
            completed.append(day)

        except Exception as e:
            print(f"⚠️ Error on Day {day}: {e}")
            failed.append(day)
    else:
        print(f"❌ Missing file: {signal_file}")
        failed.append(day)

# ===============================
# Final Summary
# ===============================
print("\n" + "="*100)
print(f"✅ Completed {len(completed)} backtests.")
if failed:
    print(f"⚠️ Skipped or failed days: {failed}")
else:
    print("🎯 All days processed successfully!")

# Convert to DataFrame for analysis
if results_summary:
    df_results = pd.DataFrame(results_summary)
    print("\n📈 Overall Performance Summary (first few days):")
    print(df_results[[
        "Day", "Total PnL", "Annualized Sharpe Ratio", 
        "Calmar Ratio", "Maximum Drawdown", "Final Capital"
    ]].head())

    # Optionally save results
    df_results.to_csv("all_backtest_results.csv", index=False)
    print("\n💾 Saved summary → all_backtest_results.csv")

print("="*100)
