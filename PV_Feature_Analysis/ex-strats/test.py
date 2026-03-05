import pandas as pd
import numpy as np

def main():
    df = pd.read_parquet("/data/quant14/EBY/day0.parquet", columns=["Time", "Price"])
    df['Signal'] = np.zeros(len(df))
    df["Signal"][0] = 1    
    df.to_csv("trade_signals_test.csv", index=False)

if __name__ == "__main__":
    main()    