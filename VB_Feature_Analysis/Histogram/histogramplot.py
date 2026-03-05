import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Set style to match the provided image (Seaborn Darkgrid)
sns.set_theme(style="darkgrid")

def plot_performance_by_direction(csv_path, output_path, cost_per_trade=0.03):
    # --- 1. Load Data ---
    if not os.path.exists(csv_path):
        print(f"❌ Error: Input file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Ensure column names are clean (remove surrounding whitespace if any)
    df.columns = df.columns.str.strip()

    # Update required columns to match trade_reports.csv (lowercase)
    required_cols = ['direction', 'pnl']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: CSV must contain columns: {required_cols}")
        print(f"   Found: {list(df.columns)}")
        return

    # --- 2. Data Processing ---
    
    # [CRITICAL] Apply Transaction Cost
    # The raw trade_reports.csv contains GROSS PnL. 
    # We subtract cost_per_trade (0.03) to get NET PnL matching your trade_pnl csv.
    df['net_pnl'] = df['pnl'] - cost_per_trade

    # Normalize direction to handle mixed cases (e.g., "LONG", "Long", "long")
    df['direction'] = df['direction'].astype(str).str.lower().str.strip()

    # Sort by Entry Time to ensure cumulative sum is chronological
    if 'entry_time' in df.columns:
        df = df.sort_values('entry_time')

    # Filter Longs
    df_longs = df[df['direction'] == 'long'].copy()
    
    # Filter Shorts
    df_shorts = df[df['direction'] == 'short'].copy()

    # Calculate Cumulative PnL using the NET PnL
    df_longs['Cumulative_PnL'] = df_longs['net_pnl'].cumsum()
    df_shorts['Cumulative_PnL'] = df_shorts['net_pnl'].cumsum()

    # Create 'Trade Count' axis (1st long trade, 2nd long trade, etc.)
    df_longs['Count'] = range(1, len(df_longs) + 1)
    df_shorts['Count'] = range(1, len(df_shorts) + 1)

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot Longs (Green)
    plt.plot(df_longs['Count'], df_longs['Cumulative_PnL'], 
             color='#2ca02c', # Standard "Tab Green"
             linewidth=2, 
             label=f"Longs (Total: {len(df_longs)})")

    # Plot Shorts (Red)
    plt.plot(df_shorts['Count'], df_shorts['Cumulative_PnL'], 
             color='#d62728', # Standard "Tab Red"
             linewidth=2, 
             label=f"Shorts (Total: {len(df_shorts)})")

    # --- 4. Styling ---
    plt.title("Performance by Direction (Long vs Short)", fontsize=16)
    plt.xlabel("Trade Count", fontsize=12)
    plt.ylabel("Cumulative PnL ($)", fontsize=12)
    
    # Add Legend (Upper Left)
    plt.legend(loc='upper left', fontsize=11, frameon=False)
    
    # Ensure zero line is visible if PnL dips below 0
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)

    # --- 5. Saving ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Performance Chart saved to: {output_path}")

if __name__ == "__main__":
    # Defaults updated for your specific path
    default_input = "/home/raid/Quant14/VB_Feature_Analysis/Histogram/trade_reports.csv"
    default_output = "performance_by_direction.png"

    parser = argparse.ArgumentParser(description="Generate Long vs Short Performance Chart.")
    parser.add_argument("--input", default=default_input, help="Path to trade_reports.csv")
    parser.add_argument("--output", default=default_output, help="Path to save the image")
    
    # Optional argument to adjust cost if needed later
    parser.add_argument("--cost", type=float, default=0.03, help="Transaction cost per trade (default: 0.03)")

    args = parser.parse_args()
    
    plot_performance_by_direction(args.input, args.output, args.cost)