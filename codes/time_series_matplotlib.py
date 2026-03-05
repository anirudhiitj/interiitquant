import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import cm

def combined_time_series_plot(data_df):
    warnings.filterwarnings('ignore')

    # Convert Time to timedelta
    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    
    # Detect day resets (when time decreases)
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))
    
    # Compute stats
    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price
    num_days = int(data_df["Day"].max()) + 1

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6), dpi=120)
    cmap = cm.get_cmap('tab10', num_days)  # 10 distinct colors
    
    # Plot each day
    for day, df_day in data_df.groupby("Day"):
        ax.plot(
            df_day["PlotIndex"], 
            df_day["Price"], 
            label=f"Day {day}", 
            color=cmap(day % 10), 
            linewidth=2
        )

    # X ticks at start of each day
    tick_positions = []
    tick_labels = []
    for day in range(num_days):
        day_data = data_df[data_df["Day"] == day]
        if len(day_data) > 0:
            tick_positions.append(day_data["PlotIndex"].iloc[0])
            tick_labels.append(f"Day {day}")
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)
    
    # Axes labels & title
    ax.set_title("Combined Multi-Day Price Analysis", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Time (Sequential across days)", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    
    # Grid & background
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=0.7)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set Y range with padding
    y_padding = price_range * 0.1
    ax.set_ylim(min_price - y_padding, max_price + y_padding)

    # Legend
    ax.legend(title="Days", fontsize=10, title_fontsize=11, frameon=True, loc='upper left')

    # Annotation box (stats)
    stats_text = (
        f'Number of Days: {num_days}\n'
        f'Initial Price: ${data_df["Price"].iloc[0]:.2f}\n'
        f'Final Price: ${data_df["Price"].iloc[-1]:.2f}\n'
        f'Max Price: ${max_price:.2f}\n'
        f'Min Price: ${min_price:.2f}\n'
        f'Range: ${price_range:.2f}'
    )

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment='top',
        bbox=dict(
            facecolor='lemonchiffon',
            edgecolor='black',
            boxstyle='round,pad=0.5'
        )
    )
    
    # Save and show
    out_path = "combined_time_series_plot_matplotlib.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: {out_path}")
    print(f"Number of Days: {num_days}")
    print(f"Price Range: ${min_price:.2f} - ${max_price:.2f}")

    return data_df

if __name__=="__main__":
    data_csv_path=f"/data/quant14/EBX/combined.csv"
    combined_df=pd.read_csv(data_csv_path,usecols=["Time","Price"])
    #df1 = pd.read_csv("/data/quant14/EBX/day0.csv")
    #df2 = pd.read_csv("/data/quant14/EBX/day1.csv")
    #df = pd.concat([df1, df2], ignore_index=True)
    combined_time_series_plot(combined_df)
    #combined_time_series_plot(pd.read_csv("day0_rand_signal.csv",usecols=["Time","Price"]))