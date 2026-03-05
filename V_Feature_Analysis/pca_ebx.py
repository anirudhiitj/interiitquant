import os
import glob
import re
import pandas as pd
import numpy as np
import dask_cudf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION - Copy from your original script
EBX_Config = {
    'DATA_DIR': '/data/quant14/EBX',
    'TIME_COLUMN': 'Time',
    'PRICE_COLUMN': 'Price',
    'FEATURE_COLUMN': 'PB9_T1',
    
    # P1 Parameters
    'P1_KF_TRANSITION_COV': 0.24,
    'P1_KF_OBSERVATION_COV': 0.7,
    'P1_KF_INITIAL_STATE_COV': 100,
    'P1_KAMA_WINDOW': 30,
    'P1_KAMA_FAST_PERIOD': 60,
    'P1_KAMA_SLOW_PERIOD': 300,
    'P1_KAMA_SLOPE_DIFF': 2,
    'P1_UCM_ALPHA': 0.8,
    'P1_UCM_BETA': 0.08,
    
    # P2 Parameters
    'P2_KAMA_WINDOW': 30,
    'P2_KAMA_FAST_PERIOD': 60,
    'P2_KAMA_SLOW_PERIOD': 300,
    'P2_KAMA_SLOPE_DIFF': 2,
    'P2_ATR_SPAN': 30,
    'P2_STD_PERIOD': 20,
}


def extract_day_num(filepath):
    """Extract day number from filename."""
    match = re.search(r'day(\d+)\.parquet', str(filepath))
    return int(match.group(1)) if match else -1


def prepare_p1_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare P1 (Kalman Filter) strategy features."""
    
    # KF_Trend
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=config['P1_KF_TRANSITION_COV'],
        observation_covariance=config['P1_KF_OBSERVATION_COV'],
        initial_state_mean=df[config['PRICE_COLUMN']].iloc[0],
        initial_state_covariance=config['P1_KF_INITIAL_STATE_COV']
    )
    filtered_state_means, _ = kf.filter(df[config['FEATURE_COLUMN']].dropna().values)
    smoothed_prices = pd.Series(filtered_state_means.squeeze(), index=df[config['FEATURE_COLUMN']].dropna().index)
    df["P1_KF_Trend"] = smoothed_prices.shift(1)
    df["P1_KF_Trend_Lagged"] = df["P1_KF_Trend"].shift(1)
    
    # KAMA
    price = df[config['FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P1_KAMA_WINDOW']
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()
    
    sc_fast = 2 / (config['P1_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P1_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2
    
    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]
    
    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])
    
    df["P1_KAMA"] = kama
    df["P1_KAMA_Slope"] = df["P1_KAMA"].diff(config['P1_KAMA_SLOPE_DIFF'])
    df["P1_KAMA_Slope_abs"] = df["P1_KAMA_Slope"].abs()
    
    # UCM
    series = df[config['FEATURE_COLUMN']]
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        raise ValueError(f"{config['FEATURE_COLUMN']} has no valid values")
    
    alpha = config['P1_UCM_ALPHA']
    beta = config['P1_UCM_BETA']
    n = len(series)
    
    mu = np.zeros(n)
    beta_slope = np.zeros(n)
    filtered = np.zeros(n)
    
    mu[first_valid_idx] = series.loc[first_valid_idx]
    beta_slope[first_valid_idx] = 0
    filtered[first_valid_idx] = mu[first_valid_idx] + beta_slope[first_valid_idx]
    
    for t in range(first_valid_idx + 1, n):
        if np.isnan(series.iloc[t]):
            mu[t] = mu[t-1]
            beta_slope[t] = beta_slope[t-1]
            filtered[t] = filtered[t-1]
            continue
        
        mu_pred = mu[t-1] + beta_slope[t-1]
        beta_pred = beta_slope[t-1]
        
        mu[t] = mu_pred + alpha * (series.iloc[t] - mu_pred)
        beta_slope[t] = beta_pred + beta * (series.iloc[t] - mu_pred)
        filtered[t] = mu[t] + beta_slope[t]
    
    df["P1_UCM"] = filtered
    df["P1_UCM_Lagged"] = df["P1_UCM"].shift(1)
    
    return df


def prepare_p2_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prepare P2 (KAMA + Volatility) strategy features."""
    
    # KAMA calculation
    price = df[config['FEATURE_COLUMN']].to_numpy(dtype=float)
    window = config['P2_KAMA_WINDOW']
    
    kama_signal = np.abs(pd.Series(price).diff(window))
    kama_noise = np.abs(pd.Series(price).diff()).rolling(window=window, min_periods=1).sum()
    er = (kama_signal / kama_noise.replace(0, np.nan)).fillna(0).to_numpy()

    sc_fast = 2 / (config['P2_KAMA_FAST_PERIOD'] + 1)
    sc_slow = 2 / (config['P2_KAMA_SLOW_PERIOD'] + 1)
    sc = ((er * (sc_fast - sc_slow)) + sc_slow) ** 2

    kama = np.full(len(price), np.nan)
    valid_start = np.where(~np.isnan(price))[0]
    if len(valid_start) == 0:
        raise ValueError("No valid price values found.")
    start = valid_start[0]
    kama[start] = price[start]

    for i in range(start + 1, len(price)):
        if np.isnan(price[i - 1]) or np.isnan(sc[i - 1]) or np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i - 1] * (price[i - 1] - kama[i - 1])

    df["P2_KAMA"] = kama
    df["P2_KAMA_Slope"] = df["P2_KAMA"].diff(config['P2_KAMA_SLOPE_DIFF']).shift(1)
    df["P2_KAMA_Slope_abs"] = df["P2_KAMA_Slope"].abs()

    # ATR calculation
    tr = df[config['FEATURE_COLUMN']].diff().abs()
    atr = tr.ewm(span=config['P2_ATR_SPAN'], adjust=False).mean()
    df["P2_ATR"] = atr.shift(1)
    df["P2_ATR_High"] = df["P2_ATR"].expanding(min_periods=1).max()

    # Standard Deviation
    std = df[config['FEATURE_COLUMN']].rolling(window=config['P2_STD_PERIOD'], min_periods=1).std()
    df["P2_STD"] = std.shift(1)
    df["P2_STD_High"] = df["P2_STD"].expanding(min_periods=1).max()
    
    return df


def extract_strategy_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract all intermediate features from both strategies for PCA analysis.
    Returns a dataframe with all features used for signal generation.
    """
    # Prepare P1 features
    df_p1 = prepare_p1_features(df.copy(), config)
    
    # Prepare P2 features
    df_p2 = prepare_p2_features(df.copy(), config)
    
    # Combine all features into one dataframe
    feature_df = pd.DataFrame({
        # Price data
        'Price': df[config['PRICE_COLUMN']],
        'Time_sec': df['Time_sec'],
        
        # P1 Strategy Features (Kalman Filter based)
        'P1_KF_Trend': df_p1['P1_KF_Trend'],
        'P1_KF_Trend_Lagged': df_p1['P1_KF_Trend_Lagged'],
        'P1_KAMA': df_p1['P1_KAMA'],
        'P1_KAMA_Slope': df_p1['P1_KAMA_Slope'],
        'P1_KAMA_Slope_abs': df_p1['P1_KAMA_Slope_abs'],
        'P1_UCM': df_p1['P1_UCM'],
        'P1_UCM_Lagged': df_p1['P1_UCM_Lagged'],
        
        # P2 Strategy Features (KAMA + Volatility based)
        'P2_KAMA': df_p2['P2_KAMA'],
        'P2_KAMA_Slope': df_p2['P2_KAMA_Slope'],
        'P2_KAMA_Slope_abs': df_p2['P2_KAMA_Slope_abs'],
        'P2_ATR': df_p2['P2_ATR'],
        'P2_ATR_High': df_p2['P2_ATR_High'],
        'P2_STD': df_p2['P2_STD'],
        'P2_STD_High': df_p2['P2_STD_High'],
    })
    
    return feature_df


def load_sample_days(config: dict, num_days: int = 50) -> pd.DataFrame:
    """
    Load a sample of days for analysis (to avoid memory issues).
    """
    data_dir = config['DATA_DIR']
    files_pattern = os.path.join(data_dir, "day*.parquet")
    all_files = glob.glob(files_pattern)
    sorted_files = sorted(all_files, key=extract_day_num)
    
    # Filter files (same logic as main script)
    filtered_files = [f for f in sorted_files if extract_day_num(f) > 79 or extract_day_num(f) < 60]
    
    if not filtered_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        print(f"Looking for pattern: {files_pattern}")
        return None
    
    # Sample evenly across the dataset
    sample_files = filtered_files[::max(1, len(filtered_files) // num_days)][:num_days]
    
    print(f"Loading {len(sample_files)} sample days for PCA analysis...")
    
    all_features = []
    for file_path in sample_files:
        day_num = extract_day_num(file_path)
        
        try:
            # Read data
            required_cols = [
                config['TIME_COLUMN'],
                config['PRICE_COLUMN'],
                config['FEATURE_COLUMN'],
            ]
            
            ddf = dask_cudf.read_parquet(file_path, columns=required_cols)
            gdf = ddf.compute()
            df = gdf.to_pandas()
            
            if df.empty:
                continue
            
            df = df.reset_index(drop=True)
            df['Time_sec'] = pd.to_timedelta(df[config['TIME_COLUMN']].astype(str)).dt.total_seconds().astype(int)
            
            # Extract features
            features = extract_strategy_features(df, config)
            features['day'] = day_num
            all_features.append(features)
            
            print(f"  ✓ Loaded day {day_num}")
            
        except Exception as e:
            print(f"  ✗ Error loading day {day_num}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No data loaded successfully")
    
    return pd.concat(all_features, ignore_index=True)


def perform_pca_analysis(feature_df: pd.DataFrame, output_dir: str = './pca_results'):
    """
    Perform comprehensive PCA analysis on strategy features.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PCA ANALYSIS OF STRATEGY SUBCOMPONENTS")
    print("="*80)
    
    # Select features for PCA (exclude time and price)
    feature_cols = [col for col in feature_df.columns if col not in ['Price', 'Time_sec', 'day']]
    
    # Remove rows with NaN values
    df_clean = feature_df[feature_cols].dropna()
    print(f"\nTotal samples: {len(feature_df):,}")
    print(f"Clean samples (no NaN): {len(df_clean):,}")
    print(f"Features analyzed: {len(feature_cols)}")
    
    # =========================================================================
    # 1. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("1. CORRELATION ANALYSIS")
    print("-"*80)
    
    correlation_matrix = df_clean[feature_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved correlation heatmap: {output_dir}/correlation_heatmap.png")
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:  # Threshold for high correlation
                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr))
    
    print(f"\nHighly correlated feature pairs (|r| > 0.7):")
    if high_corr_pairs:
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {feat1:25s} <-> {feat2:25s} : {corr:7.4f}")
    else:
        print("  None found")
    
    # =========================================================================
    # 2. STANDARDIZATION
    # =========================================================================
    print("\n" + "-"*80)
    print("2. FEATURE STANDARDIZATION")
    print("-"*80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    
    print(f"  ✓ Features standardized (mean=0, std=1)")
    print(f"\nFeature statistics after scaling:")
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    print(scaled_df.describe().loc[['mean', 'std']].round(4))
    
    # =========================================================================
    # 3. PCA COMPUTATION
    # =========================================================================
    print("\n" + "-"*80)
    print("3. PRINCIPAL COMPONENT ANALYSIS")
    print("-"*80)
    
    # Fit PCA with all components
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Variance explained
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"\nVariance explained by each component:")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"  PC{i+1:2d}: {var*100:6.2f}% (cumulative: {cum_var*100:6.2f}%)")
        if cum_var > 0.95:  # Stop after 95% variance explained
            break
    
    # =========================================================================
    # 4. SCREE PLOT
    # =========================================================================
    print("\n" + "-"*80)
    print("4. SCREE PLOT (Variance Explained)")
    print("-"*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Individual variance
    ax1.bar(range(1, len(explained_variance)+1), explained_variance * 100)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained (%)', fontsize=12)
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance * 100, 
             marker='o', linewidth=2, markersize=6)
    ax2.axhline(y=95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved scree plot: {output_dir}/scree_plot.png")
    
    # =========================================================================
    # 5. COMPONENT LOADINGS
    # =========================================================================
    print("\n" + "-"*80)
    print("5. PRINCIPAL COMPONENT LOADINGS")
    print("-"*80)
    
    # Get loadings (components)
    loadings = pca.components_
    
    # Create loadings dataframe
    n_components_to_show = min(5, len(loadings))
    loadings_df = pd.DataFrame(
        loadings[:n_components_to_show].T,
        columns=[f'PC{i+1}' for i in range(n_components_to_show)],
        index=feature_cols
    )
    
    print(f"\nTop {n_components_to_show} component loadings:")
    print(loadings_df.round(4))
    
    # Save to CSV
    loadings_df.to_csv(f'{output_dir}/component_loadings.csv')
    print(f"\n  ✓ Saved loadings to: {output_dir}/component_loadings.csv")
    
    # Plot loadings heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Principal Component Loadings', fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loadings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved loadings heatmap: {output_dir}/loadings_heatmap.png")
    
    # =========================================================================
    # 6. FEATURE CONTRIBUTIONS
    # =========================================================================
    print("\n" + "-"*80)
    print("6. TOP FEATURE CONTRIBUTIONS TO EACH PC")
    print("-"*80)
    
    for i in range(min(3, n_components_to_show)):
        print(f"\nPC{i+1} (explains {explained_variance[i]*100:.2f}% variance):")
        contrib = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
        for feat, val in contrib.head(5).items():
            original_loading = loadings_df.loc[feat, f'PC{i+1}']
            direction = "positive" if original_loading > 0 else "negative"
            print(f"  {feat:30s}: {abs(original_loading):6.4f} ({direction})")
    
    # =========================================================================
    # 7. BIPLOT (PC1 vs PC2)
    # =========================================================================
    print("\n" + "-"*80)
    print("7. BIPLOT (PC1 vs PC2)")
    print("-"*80)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sample points for visualization (too many points make it messy)
    sample_size = min(5000, len(X_pca))
    sample_indices = np.random.choice(len(X_pca), sample_size, replace=False)
    
    # Plot sample points
    ax.scatter(X_pca[sample_indices, 0], X_pca[sample_indices, 1], 
               alpha=0.3, s=1, c='gray')
    
    # Plot feature vectors
    scale_factor = 3
    for i, feature in enumerate(feature_cols):
        ax.arrow(0, 0, 
                loadings[0, i] * scale_factor, 
                loadings[1, i] * scale_factor,
                head_width=0.1, head_length=0.1, 
                fc='red', ec='red', alpha=0.6)
        ax.text(loadings[0, i] * scale_factor * 1.1, 
               loadings[1, i] * scale_factor * 1.1,
               feature, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance)', fontsize=12)
    ax.set_title('PCA Biplot: Feature Vectors in PC1-PC2 Space', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/biplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved biplot: {output_dir}/biplot.png")
    
    # =========================================================================
    # 8. STRATEGY SEPARATION ANALYSIS
    # =========================================================================
    print("\n" + "-"*80)
    print("8. STRATEGY SEPARATION ANALYSIS")
    print("-"*80)
    
    # Analyze how P1 and P2 features separate in PC space
    p1_features = [col for col in feature_cols if col.startswith('P1_')]
    p2_features = [col for col in feature_cols if col.startswith('P2_')]
    
    print(f"\nP1 Strategy features: {len(p1_features)}")
    print(f"P2 Strategy features: {len(p2_features)}")
    
    # Calculate average absolute loading for each strategy in each PC
    for i in range(min(3, n_components_to_show)):
        p1_avg_loading = loadings_df.loc[p1_features, f'PC{i+1}'].abs().mean()
        p2_avg_loading = loadings_df.loc[p2_features, f'PC{i+1}'].abs().mean()
        
        print(f"\nPC{i+1}:")
        print(f"  P1 avg |loading|: {p1_avg_loading:.4f}")
        print(f"  P2 avg |loading|: {p2_avg_loading:.4f}")
        
        if p1_avg_loading > p2_avg_loading * 1.5:
            print(f"  → PC{i+1} is dominated by P1 features")
        elif p2_avg_loading > p1_avg_loading * 1.5:
            print(f"  → PC{i+1} is dominated by P2 features")
        else:
            print(f"  → PC{i+1} contains mixed contributions")
    
    # =========================================================================
    # 9. INDEPENDENCE SCORE
    # =========================================================================
    print("\n" + "-"*80)
    print("9. STRATEGY INDEPENDENCE SCORE")
    print("-"*80)
    
    # Calculate correlation between P1 and P2 feature spaces
    p1_data = df_clean[p1_features].mean(axis=1)
    p2_data = df_clean[p2_features].mean(axis=1)
    
    correlation, p_value = pearsonr(p1_data, p2_data)
    independence_score = 1 - abs(correlation)
    
    print(f"\nP1-P2 Feature Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Independence Score: {independence_score:.4f}")
    print(f"\nInterpretation:")
    if independence_score > 0.7:
        print("  ✓ Strategies are HIGHLY INDEPENDENT - good alpha diversification")
    elif independence_score > 0.5:
        print("  ○ Strategies are MODERATELY INDEPENDENT - some overlap exists")
    else:
        print("  ✗ Strategies are HIGHLY CORRELATED - limited alpha diversification")
    
    # =========================================================================
    # 10. SUMMARY REPORT
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    summary = {
        'Total Features': len(feature_cols),
        'P1 Features': len(p1_features),
        'P2 Features': len(p2_features),
        'Samples Analyzed': len(df_clean),
        'Components for 95% Variance': n_components_95,
        'PC1 Variance Explained': f"{explained_variance[0]*100:.2f}%",
        'PC2 Variance Explained': f"{explained_variance[1]*100:.2f}%",
        'PC3 Variance Explained': f"{explained_variance[2]*100:.2f}%",
        'P1-P2 Correlation': f"{correlation:.4f}",
        'Independence Score': f"{independence_score:.4f}",
    }
    
    print("\nKey Findings:")
    for key, value in summary.items():
        print(f"  {key:30s}: {value}")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{output_dir}/pca_summary.csv', index=False)
    print(f"\n  ✓ Saved summary: {output_dir}/pca_summary.csv")
    
    print("\n" + "="*80)
    print(f"✓ All results saved to: {output_dir}/")
    print("="*80)
    
    return {
        'pca': pca,
        'scaler': scaler,
        'loadings_df': loadings_df,
        'explained_variance': explained_variance,
        'independence_score': independence_score,
        'feature_cols': feature_cols
    }


def main():
    config = EBX_Config
    
    print("\n" + "="*80)
    print("PCA ANALYSIS FOR EBX STRATEGY SUBCOMPONENTS")
    print("="*80)
    print("\nThis analysis will:")
    print("  1. Load sample days from your dataset")
    print("  2. Extract all P1 and P2 strategy features")
    print("  3. Compute correlation matrix")
    print("  4. Perform PCA to find independent components")
    print("  5. Visualize results and assess strategy independence")
    print("="*80)
    
    # Load sample data
    print(f"\nData directory: {config['DATA_DIR']}")
    feature_df = load_sample_days(config, num_days=10)
    
    if feature_df is None:
        print("\nERROR: Could not load data. Please check your DATA_DIR path.")
        return
    
    # Perform PCA analysis
    results = perform_pca_analysis(feature_df, output_dir='./pca_results')
    
    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("  1. Review correlation_heatmap.png to see feature relationships")
    print("  2. Check scree_plot.png to determine how many PCs capture variance")
    print("  3. Examine loadings_heatmap.png to understand PC composition")
    print("  4. Review biplot.png to visualize feature relationships")
    print("  5. Consider independence score when deciding strategy allocation")


if __name__ == "__main__":
    main()