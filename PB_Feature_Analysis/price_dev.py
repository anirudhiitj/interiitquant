import pandas as pd
import numpy as np

df = pd.read_csv("/data/quant14/EBY/combined.csv",usecols=["Time","Price"])
df["FormatTime"] = pd.to_timedelta(df["Time"])
df["Day"] = (df["FormatTime"].diff() < pd.Timedelta(0)).cumsum()

summary = df.groupby("Day", as_index=False)["Price"].agg(["max", "min"])

summary["diff"] = summary["max"] - summary["min"]

filtered = summary[summary["diff"] >= 0.2]

output='price_dev.txt'
with open(output, "w") as f:
    for day, row in filtered.iterrows():
        print(f"Day {row.name}: {row['diff']:.4f} (Max: {row['max']:.4f}, Min: {row['min']:.4f})")
        f.write(f"Day {row.name}: {row['diff']:.4f} (Max: {row['max']:.4f}, Min: {row['min']:.4f})\n")

