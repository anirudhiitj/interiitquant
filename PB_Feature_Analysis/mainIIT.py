import sys
import random
from datetime import datetime
import pandas as pd

from alpha_research import BacktesterIIT , Side, Ticker

ebx_df = pd.DataFrame()
ebx_last_ts = -1
eby_df = pd.DataFrame()
eby_last_ts = -1

def my_broadcast_callback(state, ts):
    # print(f"\n[STATIC] Timestamp: {ts}")
    # for ticker, data in state.items():
    #     print(f"{ticker}: {len(data)} timestamp={data['Time']} PRICE={data['Price']}")
    # return
    time=int(ts[0:2]+ts[3:5]+ts[6:8])
    h, m, s = map(int, str(ts).split(":"))
    time = h * 3600 + m * 60 + s
    global ebx_df, ebx_last_ts
    global eby_df, eby_last_ts
    if time>ebx_last_ts:
        new_df=pd.DataFrame([state["EBX"]])
        ebx_df=pd.concat([ebx_df,new_df],ignore_index=True)
        ebx_last_ts=time
        
    elif time<ebx_last_ts:
        new_df = pd.DataFrame([state["EBX"]])
        ebx_df = new_df
        ebx_last_ts = time

    if time>eby_last_ts:
        new_df=pd.DataFrame([state["EBY"]])
        eby_df=pd.concat([eby_df,new_df],ignore_index=True)
        eby_last_ts=time
        
    elif time<eby_last_ts:
        new_df = pd.DataFrame([state["EBY"]])
        eby_df = new_df
        eby_last_ts = time
        print(eby_df[["Time", "Price"]].tail(1))

    # buy_ticker = random.choice(tickers)
    # sell_ticker = random.choice([t for t in tickers if t != buy_ticker])
    # # place BUY
    # trade_buy = backtest.place_order(
    #     ticker=buy_ticker,
    #     qty=1,
    #     side=Side.BUY
    # )
    # trade_sell = backtest.place_order(
    #     ticker=sell_ticker,
    #     qty=1,
    #     side=Side.SELL
    # )


def on_timer(ts):
    print("On timer callback")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss
    backtest.run(broadcast_callback=my_broadcast_callback, timer_callback=on_timer)
    print(datetime.now().strftime("%H:%M:%S"))  # only hh:mm:ss
