import pandas as pd
import numpy as np

# ===============================================================
# CONFIGURATION
# ===============================================================
SIGNAL_FILE = '/home/raid/Quant14/V_Feature_Analysis/trading_signals_dynamic.csv'
INITIAL_CAPITAL = 100000  # ₹100K base capital

# ===============================================================
# LOAD SIGNALS
# ===============================================================
df = pd.read_csv(SIGNAL_FILE)

# Basic returns
df['Return'] = df['Price'].pct_change().fillna(0)

# Strategy returns (lagged signal to avoid lookahead bias)
df['StrategyRet'] = df['Signal'].shift(1) * df['Return']

# ===============================================================
# DAILY PERFORMANCE
# ===============================================================
daily_summary = df.groupby('Day').agg(
    Trades=('Signal', lambda x: (x.diff() != 0).sum() // 2),
    TotalReturn=('StrategyRet', lambda x: (x + 1).prod() - 1)
).reset_index()

# Compute PnL in ₹ based on initial capital
daily_summary['DailyPnL'] = daily_summary['TotalReturn'] * INITIAL_CAPITAL

# Add cumulative metrics
daily_summary['CumulativeReturn'] = (1 + daily_summary['TotalReturn']).cumprod() - 1
daily_summary['CumulativePnL'] = daily_summary['CumulativeReturn'] * INITIAL_CAPITAL

# ===============================================================
# OVERALL METRICS
# ===============================================================
total_trades = int(daily_summary['Trades'].sum())
final_value = INITIAL_CAPITAL * (1 + daily_summary['TotalReturn']).prod()
total_pnl = final_value - INITIAL_CAPITAL
overall_return = total_pnl / INITIAL_CAPITAL

# ===============================================================
# OUTPUT
# ===============================================================
print("📈 5-Day Backtest Summary:")
print(daily_summary[['Day', 'Trades', 'TotalReturn', 'DailyPnL', 'CumulativePnL']].round(2))

print("\n💰 Overall Results:")
print(f"   Starting Capital  : ₹{INITIAL_CAPITAL:,.0f}")
print(f"   Ending Capital    : ₹{final_value:,.2f}")
print(f"   Total PnL         : ₹{total_pnl:,.2f}")
print(f"   Total Return      : {overall_return*100:.2f}%")
print(f"   Total Trades      : {total_trades}")
