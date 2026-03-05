import os
import re
import pandas as pd
import numpy as np
import dask_cudf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import math
from datetime import datetime, timedelta

# CONFIGURATION (copied from your code)
Strategy_Config = {
    'DATA_DIR': '/data/quant14/EBX/',
    'PRICE_COLUMN': 'Price',
    'TIME_COLUMN': 'Time',
    'KAMA_PERIOD': 15,
    'KAMA_FAST': 25,
    'KAMA_SLOW': 100,
    'EAMA_PERIOD': 15,
    'EAMA_FAST': 30,
    'EAMA_SLOW': 120,
    'SLOPE_WINDOW': 5,
    'SLOPE_THRESHOLD': 0.000075,
    'SLOPE_LOOKBACK': 5,
    'EXIT_COOLDOWN': 5,
    'MIN_SIGMA_THRESHOLD': 0.09,
    'WARMUP_PERIOD': 1800,
    'BLACKLISTED_DAYS': {},
}


def get_super_smoother_array(prices, period):
    """Ehlers Super Smoother for EAMA calculation"""
    n = len(prices)
    pi = math.pi
    a1 = math.exp(-1.414 * pi / period)
    b1 = 2 * a1 * math.cos(1.414 * pi / period)
    c1 = 1 - b1 + a1*a1
    c2 = b1
    c3 = -a1*a1

    ss = np.zeros(n)
    if n > 0: ss[0] = prices[0]
    if n > 1: ss[1] = prices[1]

    for i in range(2, n):
        ss[i] = c1*prices[i] + c2*ss[i-1] + c3*ss[i-2]

    return ss


def apply_recursive_filter(series, sc):
    """Apply adaptive recursive filter for KAMA/EAMA"""
    n = len(series)
    out = np.full(n, np.nan)

    start_idx = 0
    if np.isnan(series[0]):
        valid = np.where(~np.isnan(series))[0]
        if len(valid) > 0:
            start_idx = valid[0]
        else:
            return out

    out[start_idx] = series[start_idx]

    for i in range(start_idx + 1, n):
        c = sc[i] if not np.isnan(sc[i]) else 0
        val = series[i]
        if not np.isnan(val):
            out[i] = out[i-1] + c * (val - out[i-1])
        else:
            out[i] = out[i-1]

    return out


def calculate_kama(df, price_col="Price", config=None):
    """Calculate KAMA"""
    if config is None:
        config = Strategy_Config
    
    prices = df[price_col].values
    period = config['KAMA_PERIOD']
    fast = config['KAMA_FAST']
    slow = config['KAMA_SLOW']
    
    direction = df[price_col].diff(period).abs()
    volatility = df[price_col].diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0)

    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = (((er * (fast_sc - slow_sc)) + slow_sc)**2).values

    df['KAMA'] = apply_recursive_filter(prices, sc)
    return df


def calculate_eama(df, price_col="Price", config=None):
    """Calculate EAMA using Super Smoother + adaptive MA"""
    if config is None:
        config = Strategy_Config
    
    prices = df[price_col].values
    
    ss = get_super_smoother_array(prices, 30)
    df["SuperSmoother"] = ss
    
    period = config['EAMA_PERIOD']
    fast = config['EAMA_FAST']
    slow = config['EAMA_SLOW']
    
    ss_series = pd.Series(ss)
    direction = ss_series.diff(period).abs()
    volatility = ss_series.diff().abs().rolling(period).sum()
    er = (direction / volatility).fillna(0).values
    
    fast_sc = 2/(fast+1)
    slow_sc = 2/(slow+1)
    sc = ((er*(fast_sc-slow_sc))+slow_sc)**2

    df["EAMA"] = apply_recursive_filter(ss, sc)
    return df


def calculate_slope(series, window):
    """Calculate slope of a series using linear regression over a rolling window"""
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
        else:
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def check_day_eligibility(df, config):
    """Check if day is eligible for trading"""
    warmup_cutoff = config['WARMUP_PERIOD']
    warmup_mask = df['Time_sec'] <= warmup_cutoff
    
    if warmup_mask.sum() == 0:
        return False, 0.0, 0
    
    warmup_prices = df.loc[warmup_mask, config['PRICE_COLUMN']]
    sigma = warmup_prices.std()
    
    warmup_cutoff_idx = warmup_mask.sum()
    
    is_eligible = sigma > config['MIN_SIGMA_THRESHOLD']
    
    return is_eligible, sigma, warmup_cutoff_idx


def generate_signals_with_details(df: pd.DataFrame, config: dict):
    """Generate signals and return detailed info for visualization"""
    signals = np.zeros(len(df), dtype=np.float32)
    
    if df.empty or len(df) < 10:
        return signals, {}
    
    # Check eligibility
    is_eligible, sigma, warmup_cutoff_idx = check_day_eligibility(df, config)
    
    # Calculate indicators
    df = calculate_kama(df, config=config)
    df = calculate_eama(df, config=config)
    
    # Calculate difference and slope
    df['Diff'] = df['KAMA'] - df['EAMA']
    df['Diff_Slope'] = calculate_slope(df['Diff'], config['SLOPE_WINDOW'])
    
    # Detect crossovers
    df['KAMA_Prev'] = df['KAMA'].shift(1)
    df['EAMA_Prev'] = df['EAMA'].shift(1)
    
    cross_long = (df['KAMA'] < df['EAMA']) & (df['KAMA_Prev'] >= df['EAMA_Prev'])
    cross_short = (df['KAMA'] > df['EAMA']) & (df['KAMA_Prev'] <= df['EAMA_Prev'])
    
    df['Cross_Long'] = cross_long
    df['Cross_Short'] = cross_short
    
    # Generate signals
    position = 0
    last_exit_time = -999999
    
    entry_points = []
    exit_points = []
    
    for i in range(warmup_cutoff_idx, len(df)):
        current_time = df['Time_sec'].iloc[i]
        
        lookback_idx = i - config['SLOPE_LOOKBACK']
        if lookback_idx >= 0 and not pd.isna(df['Diff_Slope'].iloc[lookback_idx]):
            slope_lookback = abs(df['Diff_Slope'].iloc[lookback_idx])
        else:
            slope_lookback = 0
        
        cooldown_passed = (current_time - last_exit_time) >= config['EXIT_COOLDOWN']
        
        if position == 0 and cooldown_passed:
            if cross_long.iloc[i] and slope_lookback > config['SLOPE_THRESHOLD']:
                signals[i] = 1.0
                position = 1
                entry_points.append((i, 'LONG', slope_lookback))
            
            elif cross_short.iloc[i] and slope_lookback > config['SLOPE_THRESHOLD']:
                signals[i] = -1.0
                position = -1
                entry_points.append((i, 'SHORT', slope_lookback))
        
        elif position == 1:
            if cross_short.iloc[i]:
                signals[i] = -1.0
                position = 0
                last_exit_time = current_time
                exit_points.append((i, 'LONG_EXIT'))
        
        elif position == -1:
            if cross_long.iloc[i]:
                signals[i] = 1.0
                position = 0
                last_exit_time = current_time
                exit_points.append((i, 'SHORT_EXIT'))
    
    if position != 0:
        signals[-1] = -position
        exit_points.append((len(df)-1, 'EOD_EXIT'))
    
    details = {
        'is_eligible': is_eligible,
        'sigma': sigma,
        'warmup_cutoff_idx': warmup_cutoff_idx,
        'entry_points': entry_points,
        'exit_points': exit_points,
        'df': df
    }
    
    return signals, details


def load_day_data(day_num, config):
    """Load data for a specific day"""
    file_path = os.path.join(config['DATA_DIR'], f'day{day_num}.parquet')
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    required_cols = [config['TIME_COLUMN'], config['PRICE_COLUMN']]
    ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
    gdf = ddf.compute()
    df = gdf.to_pandas()
    
    if df.empty:
        return None
    
    df = df.reset_index(drop=True)
    df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
    
    return df


def plot_day_strategy(day_num, config, save_path=None):
    """Create comprehensive visualization for a single day"""
    print(f"\nProcessing Day {day_num}...")
    
    df = load_day_data(day_num, config)
    if df is None:
        print(f"Could not load data for day {day_num}")
        return
    
    signals, details = generate_signals_with_details(df.copy(), config)
    df = details['df']
    df['Signal'] = signals
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Day {day_num} - KAMA-EAMA Strategy Analysis', fontsize=16, fontweight='bold')
    
    # Convert time to datetime for better x-axis
    base_date = datetime(2024, 1, 1)
    df['DateTime'] = [base_date + timedelta(seconds=int(s)) for s in df['Time_sec']]
    
    warmup_idx = details['warmup_cutoff_idx']
    is_eligible = details['is_eligible']
    sigma = details['sigma']
    
    # === SUBPLOT 1: Price and Moving Averages ===
    ax1 = axes[0]
    ax1.plot(df['DateTime'], df[config['PRICE_COLUMN']], label='Price', color='black', linewidth=1, alpha=0.7)
    ax1.plot(df['DateTime'], df['KAMA'], label='KAMA', color='blue', linewidth=1.5)
    ax1.plot(df['DateTime'], df['EAMA'], label='EAMA', color='red', linewidth=1.5)
    
    # Shade warmup period
    if warmup_idx > 0:
        ax1.axvspan(df['DateTime'].iloc[0], df['DateTime'].iloc[warmup_idx], 
                    alpha=0.2, color='gray', label='Warmup (30min)')
    
    # Mark entry points
    for idx, direction, slope in details['entry_points']:
        color = 'green' if direction == 'LONG' else 'red'
        marker = '^' if direction == 'LONG' else 'v'
        ax1.scatter(df['DateTime'].iloc[idx], df[config['PRICE_COLUMN']].iloc[idx], 
                   color=color, marker=marker, s=200, zorder=5, edgecolors='black', linewidths=2,
                   label=f'{direction} Entry' if idx == details['entry_points'][0][0] else '')
    
    # Mark exit points
    for idx, exit_type in details['exit_points']:
        ax1.scatter(df['DateTime'].iloc[idx], df[config['PRICE_COLUMN']].iloc[idx], 
                   color='orange', marker='x', s=200, zorder=5, linewidths=3,
                   label='Exit' if idx == details['exit_points'][0][0] else '')
    
    ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    title1 = f'Price & Indicators (Sigma={sigma:.4f}, Eligible={is_eligible})'
    ax1.set_title(title1, fontsize=11)
    
    # === SUBPLOT 2: KAMA - EAMA Difference ===
    ax2 = axes[1]
    ax2.plot(df['DateTime'], df['Diff'], label='KAMA - EAMA', color='purple', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Shade crossover regions
    for i in range(len(df)):
        if df['Cross_Long'].iloc[i]:
            ax2.axvline(df['DateTime'].iloc[i], color='green', alpha=0.3, linewidth=2)
        if df['Cross_Short'].iloc[i]:
            ax2.axvline(df['DateTime'].iloc[i], color='red', alpha=0.3, linewidth=2)
    
    ax2.set_ylabel('KAMA - EAMA', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Difference (Crossovers marked with vertical lines)', fontsize=11)
    
    # === SUBPLOT 3: Slope with Threshold ===
    ax3 = axes[2]
    
    # Plot slope
    ax3.plot(df['DateTime'], df['Diff_Slope'], label='Diff Slope', color='brown', linewidth=1.5)
    
    # Plot threshold lines
    ax3.axhline(y=config['SLOPE_THRESHOLD'], color='green', linestyle='--', 
                linewidth=2, label=f'Threshold (+{config["SLOPE_THRESHOLD"]})')
    ax3.axhline(y=-config['SLOPE_THRESHOLD'], color='red', linestyle='--', 
                linewidth=2, label=f'Threshold (-{config["SLOPE_THRESHOLD"]})')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Shade regions where |slope| > threshold
    above_threshold = np.abs(df['Diff_Slope']) > config['SLOPE_THRESHOLD']
    for i in range(len(df)):
        if above_threshold.iloc[i]:
            ax3.axvspan(df['DateTime'].iloc[i], 
                       df['DateTime'].iloc[min(i+1, len(df)-1)],
                       alpha=0.1, color='yellow')
    
    # Mark slope values at entry points (lookback)
    for idx, direction, slope in details['entry_points']:
        lookback_idx = idx - config['SLOPE_LOOKBACK']
        if lookback_idx >= 0:
            color = 'green' if direction == 'LONG' else 'red'
            ax3.scatter(df['DateTime'].iloc[lookback_idx], df['Diff_Slope'].iloc[lookback_idx],
                       color=color, marker='o', s=150, zorder=5, edgecolors='black', linewidths=2)
    
    ax3.set_ylabel('Slope', fontsize=11, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f'Slope (window={config["SLOPE_WINDOW"]}, lookback={config["SLOPE_LOOKBACK"]}s)', fontsize=11)
    
    # === SUBPLOT 4: Signals ===
    ax4 = axes[3]
    
    # Plot signals as step function
    signal_times = df['DateTime']
    signal_values = df['Signal']
    
    ax4.step(signal_times, signal_values, where='post', color='blue', linewidth=2, label='Signal')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.fill_between(signal_times, 0, signal_values, step='post', alpha=0.3)
    
    # Mark entry/exit points
    for idx, direction, slope in details['entry_points']:
        color = 'green' if direction == 'LONG' else 'red'
        ax4.scatter(df['DateTime'].iloc[idx], signals[idx], 
                   color=color, s=200, zorder=5, edgecolors='black', linewidths=2)
    
    for idx, exit_type in details['exit_points']:
        ax4.scatter(df['DateTime'].iloc[idx], signals[idx], 
                   color='orange', marker='x', s=200, zorder=5, linewidths=3)
    
    ax4.set_ylabel('Signal', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Short (-1)', 'Flat (0)', 'Long (+1)'])
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Trading Signals', fontsize=11)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nDay {day_num} Summary:")
    print(f"  Eligible: {is_eligible} (Sigma: {sigma:.4f})")
    print(f"  Entry signals: {len(details['entry_points'])}")
    print(f"  Exit signals: {len(details['exit_points'])}")
    for idx, direction, slope in details['entry_points']:
        time_str = df['DateTime'].iloc[idx].strftime('%H:%M:%S')
        print(f"    {direction} @ {time_str} (slope={slope:.6f})")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()


def main():
    """Main function to plot multiple days"""
    config = Strategy_Config
    
    # Days to plot
    days_to_plot = [489, 396, 352, 315]
    
    print("="*80)
    print("KAMA-EAMA STRATEGY VISUALIZATION")
    print("="*80)
    print(f"Data Directory: {config['DATA_DIR']}")
    print(f"Days to plot: {days_to_plot}")
    print("="*80)
    
    for day_num in days_to_plot:
        save_path = f"day{day_num}_strategy_plot.png"
        plot_day_strategy(day_num, config, save_path=save_path)
        print()


if __name__ == "__main__":
    main()