import pandas as pd

# === Load your CSV ===
df = pd.read_csv("signals_all_days.csv")  # replace with your actual file path

# === Identify days (each time Time resets to 0) ===
df['day_id'] = (df['Time'] < df['Time'].shift()).cumsum()

# === Convert seconds to HH:MM:SS format (remove '0 days') ===
df['Time_hms'] = pd.to_timedelta(df['Time'], unit='s').dt.components
df['Time_hms'] = (
    df['Time_hms']['hours'].astype(str).str.zfill(2) + ':' +
    df['Time_hms']['minutes'].astype(str).str.zfill(2) + ':' +
    df['Time_hms']['seconds'].astype(str).str.zfill(2)
)

# === Process signals per day ===
def process_day(group):
    if (group['Signal'] == 1).any():
        first_one_idx = group.index[group['Signal'] == 1][0]
        group.loc[first_one_idx + 1:, 'Signal'] = 0
        group.loc[group.index[-1], 'Signal'] = -1
    return group

df = df.groupby('day_id', group_keys=False).apply(process_day)

# === Drop helper column ===
df = df.drop(columns=['day_id'])

# === Save processed file ===
df.to_csv("processed_signals.csv", index=False)

print("✅ Done! File saved as 'processed_signals.csv' with clean HH:MM:SS format.")
