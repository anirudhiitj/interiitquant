import sys
from datetime import datetime
import pandas as pd

from alpha_research import BacktesterIIT , Side, Ticker

EBX_Path = "EBX_signals.csv"
EBY_Path = "EBY_signals.csv"

def load_signals(path, name):
    try:
        df = pd.read_csv(path,usecols=["Time","Price","Signal"])
        df["FormatTime"] = pd.to_timedelta(df["Time"])
        df["Day"] = (df["FormatTime"].diff() < pd.Timedelta(0)).cumsum()
        df["Index"] = df["Day"].astype(str) + "_" + df["Time"].astype(str)
        df.set_index("Index", inplace=True)
        print(f"{name} loaded successfully with {len(df)} rows")
        return df
    except Exception as e:
        print(f"No {name} signal file found: {e}")
        return None

def Get_Signal(df, day, ts):
    if df is None:
        return 0
    try:
        idx = f"{day}_{ts}"
        return int(df.at[idx, "Signal"]) if idx in df.index else 0
    except Exception:
        return 0
    
def my_broadcast_callback(state, day, ts):
    tickers = [t for t, d in state.items() if d['Price'] != 0]

    if "EBX" in tickers:
        signal = Get_Signal(ebx_df,day,ts)
        ticker="EBX"
        if signal==1:
            backtest.place_order(
                ticker=ticker,
                qty=1,
                side=Side.BUY
            )
        elif signal==-1:
            backtest.place_order(
                ticker=ticker,
                qty=1,
                side=Side.SELL
            )

    if "EBY" in tickers:
        signal = Get_Signal(eby_df,day,ts)
        ticker="EBY"
        if signal==1:
            backtest.place_order(
                ticker=ticker,
                qty=1,
                side=Side.BUY
            )
        elif signal==-1:
            backtest.place_order(
                ticker=ticker,
                qty=1,
                side=Side.SELL
            )
    return

def on_timer(ts):
    print("On timer callback")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)
    try:
        ebx_df = load_signals(EBX_Path, "EBX")
    except Exception as e:
        print(f"[WARN] Failed to load EBX signals: {e}")
        ebx_df = None
    try:
        eby_df = load_signals(EBY_Path, "EBY")
    except Exception as e:
        print(f"[WARN] Failed to load EBY signals: {e}")
        eby_df = None

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss