def generate_trade_reports_csv():
    """
    Generate a CSV file of completed trades from trading_signals.csv.
    Output:
        trade_reports.csv
    """
    import pandas as pd

    signals_file = "/data/quant14/signals/z_trading_signals_EBX.csv"

    try:
        df = pd.read_csv(signals_file)
    except FileNotFoundError:
        print("Error: trading_signals.csv not found")
        return

    signals_df = df[df["Signal"] != 0].copy()
    if len(signals_df) == 0:
        print("No trades found")
        return

    trades = []
    position = None
    entry = None

    for idx, row in signals_df.iterrows():
        signal = row["Signal"]

        if position is None:
            if signal in [1, -1]:
                position = signal
                entry = row
        else:
            if signal == -position:
                trades.append({
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_time": entry["Time"],
                    "entry_price": float(entry["Price"]),
                    "exit_time": row["Time"],
                    "exit_price": float(row["Price"]),
                    "pnl": (float(row["Price"]) - float(entry["Price"])) * position
                })
                position = None
                entry = None

    if len(trades) == 0:
        print("No completed trades found")
        return

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv("trade_reports.csv", index=False)

    print("trade_reports.csv generated")

generate_trade_reports_csv()