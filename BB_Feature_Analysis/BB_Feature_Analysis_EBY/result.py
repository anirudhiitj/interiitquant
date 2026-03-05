import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "/data/quant14/EBY"
OUTPUT_DIR = "strategy_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
INITIAL_CAPITAL = 100000
RISK_FREE_RATE = 0.0
TRANSACTION_COST = 0.1

# STRATEGY PARAMETERS - Based on correlation analysis
LOOKBACK_WINDOW = 30  # Rolling window for feature calculation
ENTRY_THRESHOLD = 1.5  # Z-score threshold to enter
EXIT_THRESHOLD = 0.3   # Z-score threshold to exit
STOP_LOSS_PCT = 0.015  # 1.5% stop loss

# Features from correlation analysis (top correlated with Granger causality)
FEATURES_TO_USE = ['BB5_T1', 'BB13_T1', 'BB4_T1', 'BB6_T1', 'BB13_T2', 
                   'BB5_T2', 'BB11_T1', 'BB14_T1', 'BB12_T1', 'BB15_T1']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_daily_data(file_path):
    """Load a single day's data from CSV/Excel"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return None
        
        # Check for required columns
        if 'Price' not in df.columns and 'price' not in df.columns:
            # Try to find price column
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            if price_cols:
                df.rename(columns={price_cols[0]: 'Price'}, inplace=True)
            else:
                print(f"No price column found in {file_path}")
                return None
        
        if 'Price' not in df.columns:
            df.rename(columns={'price': 'Price'}, inplace=True)
        
        # Handle time column
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        elif 'time' in df.columns:
            df['Time'] = pd.to_datetime(df['time'])
        else:
            # Create time index if not present
            df['Time'] = pd.date_range(start='09:15:00', periods=len(df), freq='1S')
        
        df.set_index('Time', inplace=True)
        return df
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_zscore(series, window=20):
    """Calculate rolling z-score"""
    if len(series) < window:
        return pd.Series(np.nan, index=series.index)
    
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-8)
    return zscore

def calculate_rsi(series, period=14):
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================================
# STRATEGY CLASS
# ============================================================================

class MeanReversionStrategy:
    """
    Mean reversion strategy using Bollinger Band derivatives
    MODIFIED: Uses FULL CAPITAL for each position
    
    Entry Logic:
    - Long: When aggregate feature z-score < -ENTRY_THRESHOLD (oversold)
    - Short: When aggregate feature z-score > +ENTRY_THRESHOLD (overbought)
    
    Exit Logic:
    - Exit when z-score crosses back to neutral (abs(z) < EXIT_THRESHOLD)
    - Stop loss if price moves against position by STOP_LOSS_PCT
    """
    
    def __init__(self, capital=100000, features=None, 
                 lookback=30, entry_thresh=1.5, exit_thresh=0.3,
                 stop_loss_pct=0.015):
        
        self.initial_capital = capital
        self.capital = capital
        self.features = features or FEATURES_TO_USE
        self.lookback = lookback
        self.entry_thresh = entry_thresh
        self.exit_thresh = exit_thresh
        self.stop_loss_pct = stop_loss_pct
        
        # Trading state
        self.position = 0  # Current position: positive=long, negative=short, 0=flat
        self.entry_price = 0
        self.trades = []
        self.pnl_history = []
        self.position_history = []
        
    def calculate_features(self, df):
        """Calculate feature z-scores"""
        feature_zscores = pd.DataFrame(index=df.index)
        
        available_features = []
        for feature in self.features:
            if feature in df.columns:
                z = calculate_zscore(df[feature], window=self.lookback)
                feature_zscores[feature] = z
                available_features.append(feature)
        
        if len(available_features) == 0:
            print("WARNING: No features available")
            return None, []
        
        # Aggregate signal: mean z-score across all features
        aggregate_z = feature_zscores.mean(axis=1)
        
        return aggregate_z, available_features
    
    def calculate_position_size(self, price):
        """
        Calculate position size using ALL available capital
        
        Returns: number of shares (integer)
        """
        if price <= 0:
            return 0
        
        # Use 100% of capital
        max_shares = int(self.capital / price)
        
        return max_shares
    
    def generate_signal(self, z_score, price, current_capital):
        """
        Generate trading signal based on z-score
        
        Returns: (target_position, signal_type)
        """
        if np.isnan(z_score):
            return self.position, 'hold'
        
        # Check stop loss
        if self.position != 0 and self.entry_price > 0:
            price_change = (price - self.entry_price) / self.entry_price
            
            if self.position > 0 and price_change < -self.stop_loss_pct:
                return 0, 'stop_loss_long'
            elif self.position < 0 and price_change > self.stop_loss_pct:
                return 0, 'stop_loss_short'
        
        # Entry signals - calculate position size dynamically
        if self.position == 0:
            position_size = self.calculate_position_size(price)
            
            if z_score < -self.entry_thresh:
                # Oversold - go LONG (expecting mean reversion up)
                return position_size, 'enter_long'
            elif z_score > self.entry_thresh:
                # Overbought - go SHORT (expecting mean reversion down)
                return -position_size, 'enter_short'
        
        # Exit signals
        elif self.position > 0:  # Currently long
            if z_score > -self.exit_thresh:
                # Mean reversion complete
                return 0, 'exit_long'
        
        elif self.position < 0:  # Currently short
            if z_score < self.exit_thresh:
                # Mean reversion complete
                return 0, 'exit_short'
        
        return self.position, 'hold'
    
    def execute_trade(self, target_position, price, timestamp, signal_type):
        """Execute trade and update position"""
        if target_position == self.position:
            return
        
        # Calculate trade
        quantity = target_position - self.position
        trade_value = quantity * price
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'signal': signal_type,
            'price': price,
            'quantity': quantity,
            'position_before': self.position,
            'position_after': target_position,
            'trade_value': trade_value
        }
        
        self.trades.append(trade)
        
        # Update position
        if self.position != 0 and target_position == 0:
            # Closing position - realize P&L
            pnl = quantity * (price - self.entry_price)
            self.capital += pnl
        
        self.position = target_position
        
        if self.position != 0:
            self.entry_price = price
        else:
            self.entry_price = 0
    
    def run_backtest(self, df):
        """Run backtest on a single day's data"""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.pnl_history = []
        self.position_history = []
        
        # Calculate features
        aggregate_z, available_features = self.calculate_features(df)
        
        if aggregate_z is None:
            print("No features available for backtesting")
            return None, None
        
        print(f"Using features: {available_features}")
        
        # Iterate through data
        for i in range(self.lookback, len(df)):
            timestamp = df.index[i]
            price = df['Price'].iloc[i]
            z_score = aggregate_z.iloc[i]
            
            # Calculate current P&L
            if self.position != 0:
                unrealized_pnl = self.position * (price - self.entry_price)
            else:
                unrealized_pnl = 0
            
            current_capital = self.capital + unrealized_pnl
            
            # Generate and execute signal
            target_position, signal_type = self.generate_signal(z_score, price, current_capital)
            
            if target_position != self.position:
                self.execute_trade(target_position, price, timestamp, signal_type)
            
            # Record state
            self.pnl_history.append({
                'timestamp': timestamp,
                'price': price,
                'position': self.position,
                'capital': current_capital,
                'pnl': current_capital - self.initial_capital,
                'z_score': z_score
            })
            
            self.position_history.append(self.position)
        
        # Close all positions at end of day
        if self.position != 0:
            final_price = df['Price'].iloc[-1]
            final_timestamp = df.index[-1]
            self.execute_trade(0, final_price, final_timestamp, 'eod_close')
            
            final_capital = self.capital
        else:
            final_capital = self.capital
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.pnl_history)
        
        return results_df, final_capital

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def calculate_metrics(daily_returns, daily_pnls):
    """Calculate comprehensive performance metrics"""
    
    # Basic stats
    total_days = len(daily_returns)
    winning_days = np.sum(daily_pnls > 0)
    losing_days = np.sum(daily_pnls < 0)
    
    # Returns
    total_return = np.sum(daily_returns)
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    if std_return > 0:
        sharpe = (mean_return - RISK_FREE_RATE) / std_return * np.sqrt(252)
    else:
        sharpe = 0
    
    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        sortino = (mean_return - RISK_FREE_RATE) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sortino = np.inf if mean_return > 0 else 0
    
    # Maximum Drawdown
    cumulative = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    max_drawdown_pct = max_drawdown / INITIAL_CAPITAL if max_drawdown != 0 else 0
    
    # Calmar Ratio (return / max drawdown)
    if max_drawdown_pct < 0:
        calmar = total_return / abs(max_drawdown_pct)
    else:
        calmar = np.inf if total_return > 0 else 0
    
    # Win Rate
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Profit Factor
    gross_profit = np.sum(daily_pnls[daily_pnls > 0])
    gross_loss = np.abs(np.sum(daily_pnls[daily_pnls < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Average Win/Loss
    avg_win = np.mean(daily_pnls[daily_pnls > 0]) if winning_days > 0 else 0
    avg_loss = np.mean(daily_pnls[daily_pnls < 0]) if losing_days > 0 else 0
    
    # Total P&L across all days (meaningful for non-compounding strategy)
    total_pnl = np.sum(daily_pnls)
    
    metrics = {
        'Total Days': total_days,
        'Winning Days': winning_days,
        'Losing Days': losing_days,
        'Win Rate': win_rate,
        'Total Return (%)': total_return * 100,
        'Mean Daily Return (%)': mean_return * 100,
        'Std Daily Return (%)': std_return * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown ($)': max_drawdown,
        'Max Drawdown (%)': max_drawdown_pct * 100,
        'Calmar Ratio': calmar,
        'Profit Factor': profit_factor,
        'Gross Profit': gross_profit,
        'Gross Loss': gross_loss,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Total P&L ($)': total_pnl,
        'Avg Ending Capital Per Day': INITIAL_CAPITAL + (total_pnl / total_days)
    }
    
    return metrics

def plot_daily_results(results_df, date_str, output_dir):
    """Create dual-axis plot: Price and P&L over time"""
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Price on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Price', color=color, fontsize=12)
    ax1.plot(results_df['timestamp'], results_df['price'], color=color, linewidth=1, alpha=0.7, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # P&L on right axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('P&L ($)', color=color, fontsize=12)
    ax2.plot(results_df['timestamp'], results_df['pnl'], color=color, linewidth=1.5, label='P&L')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Fill P&L area
    ax2.fill_between(results_df['timestamp'], results_df['pnl'], 0, 
                      where=results_df['pnl'] >= 0, alpha=0.3, color='green', label='Profit')
    ax2.fill_between(results_df['timestamp'], results_df['pnl'], 0, 
                      where=results_df['pnl'] < 0, alpha=0.3, color='red', label='Loss')
    
    # Format
    plt.title(f'Intraday Trading: {date_str}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    # Save
    output_path = os.path.join(output_dir, f'pnl_{date_str}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: {output_path}")

# ============================================================================
# MAIN BACKTEST LOOP
# ============================================================================

def run_full_backtest():
    """Run backtest across all days"""
    
    print("\n" + "="*100)
    print("STARTING MULTI-DAY BACKTEST - FULL CAPITAL MODE")
    print("="*100)
    print(f"Data Path: {DATA_PATH}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Sizing: 100% of capital per trade")
    print(f"Features: {FEATURES_TO_USE[:5]}... ({len(FEATURES_TO_USE)} total)")
    print(f"Lookback: {LOOKBACK_WINDOW}, Entry: {ENTRY_THRESHOLD}, Exit: {EXIT_THRESHOLD}")
    print("="*100 + "\n")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Find all data files (day0.csv to day278.csv)
    all_files = []
    for day in range(279):
        file_path = os.path.join(DATA_PATH, f"day{day}.csv")
        if os.path.exists(file_path):
            all_files.append(file_path)
        else:
            print(f"WARNING: {file_path} not found")
    
    if len(all_files) == 0:
        print(f"ERROR: No data files found in {DATA_PATH}")
        print(f"Expected files: day0.csv, day1.csv, ..., day278.csv")
        return
    
    print(f"Found {len(all_files)}/279 data files")
    print(f"First file: {os.path.basename(all_files[0])}")
    print(f"Last file: {os.path.basename(all_files[-1])}")
    print()
    
    # Initialize strategy
    strategy = MeanReversionStrategy(
        capital=INITIAL_CAPITAL,
        features=FEATURES_TO_USE,
        lookback=LOOKBACK_WINDOW,
        entry_thresh=ENTRY_THRESHOLD,
        exit_thresh=EXIT_THRESHOLD,
        stop_loss_pct=STOP_LOSS_PCT
    )
    
    # Store results
    daily_pnls = []
    daily_returns = []
    daily_summaries = []
    
    # Run backtest for each day
    for day_idx, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        date_str = f"day{day_idx}"  # day0, day1, ..., day278
        
        print(f"[Day {day_idx}/{len(all_files)-1}] Processing: {file_name}")
        
        # Load data
        df = load_daily_data(file_path)
        
        if df is None or len(df) < LOOKBACK_WINDOW:
            print(f"  SKIPPED: Insufficient data")
            continue
        
        print(f"  Data points: {len(df):,}")
        print(f"  Price range: {df['Price'].min():.2f} - {df['Price'].max():.2f}")
        
        # Run backtest
        results_df, final_capital = strategy.run_backtest(df)
        
        if results_df is None:
            print(f"  SKIPPED: Backtest failed")
            continue
        
        # Calculate daily P&L
        day_pnl = final_capital - INITIAL_CAPITAL
        day_return = day_pnl / INITIAL_CAPITAL
        
        daily_pnls.append(day_pnl)
        daily_returns.append(day_return)
        
        # Summary
        num_trades = len(strategy.trades)
        
        summary = {
            'Date': date_str,
            'File': file_name,
            'PnL': day_pnl,
            'Return_Pct': day_return * 100,
            'Num_Trades': num_trades,
            'Final_Capital': final_capital
        }
        
        daily_summaries.append(summary)
        
        print(f"  Trades: {num_trades}, P&L: ${day_pnl:,.2f} ({day_return*100:+.2f}%)")
        
        # Plot results
        try:
            plot_daily_results(results_df, date_str, PLOTS_DIR)
        except Exception as e:
            print(f"  WARNING: Plot failed: {e}")
        
        print()
    
    # ========================================================================
    # AGGREGATE RESULTS
    # ========================================================================
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE - CALCULATING PERFORMANCE METRICS")
    print("="*100 + "\n")
    
    if len(daily_pnls) == 0:
        print("ERROR: No valid trading days")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(daily_returns), np.array(daily_pnls))
    
    # Print results
    print("PERFORMANCE SUMMARY")
    print("-" * 100)
    print(f"Total Trading Days:        {metrics['Total Days']}")
    print(f"Winning Days:              {metrics['Winning Days']} ({metrics['Win Rate']*100:.1f}%)")
    print(f"Losing Days:               {metrics['Losing Days']}")
    print()
    print(f"Total Return:              {metrics['Total Return (%)']:+.2f}%")
    print(f"Mean Daily Return:         {metrics['Mean Daily Return (%)']:+.4f}%")
    print(f"Std Daily Return:          {metrics['Std Daily Return (%)']:.4f}%")
    print()
    print(f"Sharpe Ratio:              {metrics['Sharpe Ratio']:.4f}  *** MOST IMPORTANT ***")
    print(f"Sortino Ratio:             {metrics['Sortino Ratio']:.4f}")
    print(f"Calmar Ratio:              {metrics['Calmar Ratio']:.4f}")
    print()
    print(f"Max Drawdown:              ${metrics['Max Drawdown ($)']:,.2f} ({metrics['Max Drawdown (%)']:.2f}%)")
    print(f"Profit Factor:             {metrics['Profit Factor']:.4f}")
    print()
    print(f"Gross Profit:              ${metrics['Gross Profit']:,.2f}")
    print(f"Gross Loss:                ${metrics['Gross Loss']:,.2f}")
    print(f"Average Win:               ${metrics['Average Win']:,.2f}")
    print(f"Average Loss:              ${metrics['Average Loss']:,.2f}")
    print()
    print(f"Initial Capital (per day): ${INITIAL_CAPITAL:,.2f}")
    print(f"Total P&L (279 days):      ${metrics['Total P&L ($)']:,.2f}")
    print(f"Avg Ending Capital/Day:    ${metrics['Avg Ending Capital Per Day']:,.2f}")
    print()
    print("NOTE: Strategy resets to $100k each day (no compounding)")
    print("      Using 100% of capital for each position")
    print("      'Total P&L' is the sum of daily profits/losses across 279 independent sessions")
    print("=" * 100)
    
    # Save results
    summary_df = pd.DataFrame(daily_summaries)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_summary.csv'), index=False)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)
    
    # Plot cumulative P&L
    plt.figure(figsize=(16, 8))
    cumulative_pnl = np.cumsum(daily_pnls)
    plt.plot(range(1, len(cumulative_pnl)+1), cumulative_pnl, linewidth=2, color='darkblue')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.fill_between(range(1, len(cumulative_pnl)+1), cumulative_pnl, 0, 
                     where=np.array(cumulative_pnl) >= 0, alpha=0.3, color='green')
    plt.fill_between(range(1, len(cumulative_pnl)+1), cumulative_pnl, 0, 
                     where=np.array(cumulative_pnl) < 0, alpha=0.3, color='red')
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Cumulative P&L ($)', fontsize=12)
    plt.title('Cumulative P&L Across All Trading Days (Full Capital Mode)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cumulative_pnl.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"Daily plots saved to: {PLOTS_DIR}/")
    print(f"Total plots generated: {len(all_files)}")
    print("\n" + "="*100)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_full_backtest()