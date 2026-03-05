import cudf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================
# READ CSV
# ======================================================
print("Reading CSV file...")
df = cudf.read_csv('trading_signals.csv')
print(f"Loaded {len(df):,} rows")

# ======================================================
# TIME PROCESSING
# ======================================================
print("Processing time data...")

# Convert Time column to pandas Timedelta
time_pd = pd.to_timedelta(df["Time"].to_pandas(), errors='coerce')

# Convert back to cuDF
df["Time"] = cudf.Series(time_pd)

# ======================================================
# DAY CALCULATION
# ======================================================
# Using pandas Timedelta(0) reference
df["Day"] = (df["Time"].diff().fillna(pd.Timedelta(0)) < pd.Timedelta(0)).cumsum()

# ======================================================
# POSITION CALCULATION
# ======================================================
print("Calculating positions...")

position = []
current_position = 0

signals = df["Signal"].to_pandas().values  # convert to numpy for faster iteration

for signal in signals:
    if signal == 1:
        current_position = 1 if current_position <= 0 else 0
    elif signal == -1:
        current_position = -1 if current_position >= 0 else 0
    position.append(current_position)

df["Position"] = cudf.Series(position)

# ======================================================
# DOWNSAMPLING FUNCTION
# ======================================================
def downsample(df, step=10):
    return df.iloc[::step]

# ======================================================
# SPECIFY DAYS TO PLOT
# ======================================================
days_to_plot = []  # change as needed

# Convert Day max to int for comparison
max_day = int(df["Day"].max())

# ======================================================
# PLOTTING LOOP
# ======================================================
for day_num in days_to_plot:
    if day_num > max_day:
        print(f"\nDay {day_num} does not exist (max day is {max_day}). Skipping...")
        continue

    print(f"\nCreating plot for Day {day_num}...")

    day_data = df[df["Day"] == day_num]
    day_data = downsample(day_data, step=10)

    if len(day_data) == 0:
        print(f"  No data for Day {day_num}. Skipping...")
        continue

    print(f"  Day {day_num}: {len(day_data):,} data points")

    # Convert to pandas for Plotly compatibility
    day_pd = day_data.to_pandas()

    # ======================================================
    # CREATE PLOT
    # ======================================================
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]],
        vertical_spacing=0.15,
        row_heights=[0.01, 0.99]
    )

    # Price trace
    fig.add_trace(
        go.Scattergl(
            x=day_pd.index,
            y=day_pd["Price"],
            name="Price",
            line=dict(color="#1f77b4", width=2),
            showlegend=False,
            hovertemplate='<b>Price</b><br>Value: $%{y:,.2f}<br>Index: %{x}<extra></extra>'
        ),
        row=2, col=1, secondary_y=False
    )

    # Long positions
    fig.add_trace(
        go.Scattergl(
            x=day_pd.index,
            y=[max(p, 0) for p in day_pd["Position"]],
            name="Long Position",
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(0, 128, 0, 0.5)",
            line_shape="hv"
        ),
        row=2, col=1, secondary_y=True
    )

    # Short positions
    fig.add_trace(
        go.Scattergl(
            x=day_pd.index,
            y=[min(p, 0) for p in day_pd["Position"]],
            name="Short Position",
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.5)",
            line_shape="hv"
        ),
        row=2, col=1, secondary_y=True
    )

    # Axes
    fig.update_xaxes(title_text="Index", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Position", secondary_y=True, row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"Day {day_num} - Price vs Trading Signals ({len(day_pd):,} points)",
        height=800,
        hovermode="x unified"
    )

    # Show
    fig.show()

    # ======================================================
    # DAY STATS
    # ======================================================
    signals_count = int((day_data["Signal"] != 0).sum())
    price_min = float(day_data["Price"].min())
    price_max = float(day_data["Price"].max())

    print(f"  Signals on Day {day_num}: {signals_count}")
    print(f"  Price range: ${price_min:.2f} - ${price_max:.2f}")

# ======================================================
# OVERALL SUMMARY
# ======================================================
print("\n=== OVERALL SUMMARY ===")
print(f"Total data points: {len(df):,}")
print(f"Total days: {max_day + 1}")
