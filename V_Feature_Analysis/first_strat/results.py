import re
import pandas as pd
import numpy as np

# === READ LOG FILE ===
input_file = "all_days_output.txt"
print(f"📄 Reading log file: {input_file}")

with open(input_file, "r") as f:
    text = f.read()

# === EXTRACT DAILY METRICS USING REGEX ===
pattern = re.compile(
    r'0\s+0\s+([-+]?\d*\.\d+)\s+(\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)'
)
matches = pattern.findall(text)

if not matches:
    print("⚠️ No daily metric patterns found. Please check your regex or input format.")
    exit()

# === BUILD DAILY METRICS DATAFRAME ===
data = []
for m in matches:
    daily_return = float(m[0])
    daily_dd = float(m[1])
    daily_sharpe = float(m[2])
    daily_calmar = float(m[3])
    data.append([daily_return, daily_dd, daily_sharpe, daily_calmar])

df = pd.DataFrame(
    data,
    columns=["Daily_Return_%", "Daily_Max_Drawdown_%", "Daily_Sharpe", "Daily_Calmar"]
)

# === COMPUTE OVERALL PERFORMANCE METRICS ===
df["Cumulative_Return"] = (1 + df["Daily_Return_%"] / 100).cumprod()
total_return = df["Cumulative_Return"].iloc[-1] - 1
df["Peak"] = df["Cumulative_Return"].cummax()
df["Drawdown"] = (df["Cumulative_Return"] - df["Peak"]) / df["Peak"]
max_drawdown = abs(df["Drawdown"].min())

years = len(df) / 252  # assume 252 trading days per year
annual_return = ((1 + total_return) ** (1 / years) - 1)
calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.nan

# === SAVE DAILY METRICS CSV ===
csv_file = "daily_metrics.csv"
df.to_csv(csv_file, index=False)
print(f"✅ Daily metrics saved to: {csv_file}")

# === SAVE OVERALL SUMMARY TXT ===
summary_file = "overall_summary.txt"
with open(summary_file, "w") as f:
    f.write("========== OVERALL PERFORMANCE ==========\n")
    f.write(f"Days processed:        {len(df)}\n")
    f.write(f"Total Return:          {total_return*100:.2f}%\n")
    f.write(f"Annualized Return:     {annual_return*100:.2f}%\n")
    f.write(f"Maximum Drawdown:      {max_drawdown*100:.2f}%\n")
    f.write(f"Overall Calmar Ratio:  {calmar_ratio:.3f}\n")
    f.write("=========================================\n")

print(f"✅ Overall summary saved to: {summary_file}")

# === OPTIONAL: PRINT SUMMARY TO CONSOLE ===
print("\n========== OVERALL PERFORMANCE ==========")
print(f"Days processed:        {len(df)}")
print(f"Total Return:          {total_return*100:.2f}%")
print(f"Annualized Return:     {annual_return*100:.2f}%")
print(f"Maximum Drawdown:      {max_drawdown*100:.2f}%")
print(f"Overall Calmar Ratio:  {calmar_ratio:.3f}")
print("=========================================\n")
