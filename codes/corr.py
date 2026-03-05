import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import euclidean
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

def simple_correlation(series1, series2):
    """Pearson correlation coefficient."""
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    
    corr, pval = pearsonr(s1, s2)
    return {'correlation': corr, 'pvalue': pval, 'significant': pval < 0.05}


def lagged_correlation(series1, series2, max_lag=10):
    """Correlation at different time lags."""
    s1 = pd.Series(series1).dropna().reset_index(drop=True)
    s2 = pd.Series(series2).dropna().reset_index(drop=True)
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr, pval = pearsonr(s1[:-lag], s2[lag:])
        elif lag < 0:
            corr, pval = pearsonr(s1[-lag:], s2[:lag])
        else:
            corr, pval = pearsonr(s1, s2)
        results[lag] = {'correlation': corr, 'pvalue': pval}
    return results


def granger_causality(series1, series2, max_lag=10):
    """Test if series1 Granger-causes series2."""
    s1 = pd.Series(series1).dropna().reset_index(drop=True)
    s2 = pd.Series(series2).dropna().reset_index(drop=True)
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    
    data = pd.DataFrame({'series2': s2, 'series1': s1})
    
    try:
        gc_results = grangercausalitytests(data[['series2', 'series1']], 
                                          maxlag=max_lag, verbose=False)
        results = {}
        for lag in range(1, max_lag + 1):
            f_test = gc_results[lag][0]['ssr_ftest']
            results[lag] = {
                'f_statistic': f_test[0],
                'pvalue': f_test[1],
                'granger_causes': f_test[1] < 0.05
            }
        return results
    except Exception as e:
        return {'error': str(e)}


def spearman_correlation(series1, series2):
    """Spearman's rank correlation (non-parametric)."""
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    
    corr, pval = spearmanr(s1, s2)
    return {'correlation': corr, 'pvalue': pval, 'significant': pval < 0.05}


def kendall_correlation(series1, series2):
    """Kendall's tau correlation (rank-based)."""
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len], s2[:min_len]
    
    corr, pval = kendalltau(s1, s2)
    return {'correlation': corr, 'pvalue': pval, 'significant': pval < 0.05}


def dynamic_time_warping(series1, series2, window=None):
    """
    Dynamic Time Warping distance between two time series.
    Lower distance = more similar patterns.
    
    Parameters:
    -----------
    window : int, optional
        Sakoe-Chiba window constraint (None = no constraint)
    """
    s1 = np.array(pd.Series(series1).dropna())
    s2 = np.array(pd.Series(series2).dropna())
    
    n, m = len(s1), len(s2)
    
    # Initialize DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Set window constraint
    if window is None:
        window = max(n, m)
    
    # Compute DTW
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
    
    distance = dtw[n, m]
    
    # Normalize by path length
    normalized_distance = distance / (n + m)
    
    return {
        'dtw_distance': distance,
        'normalized_distance': normalized_distance,
        'path_length': n + m
    }


def mutual_information(series1, series2, bins=20):
    """
    Mutual Information - measures statistical dependence (linear & non-linear).
    Higher MI = stronger relationship.
    
    Parameters:
    -----------
    bins : int
        Number of bins for discretization (for discrete MI)
    """
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()
    min_len = min(len(s1), len(s2))
    s1, s2 = s1[:min_len].values, s2[:min_len].values
    
    # Method 1: Continuous MI (using sklearn)
    s1_reshape = s1.reshape(-1, 1)
    mi_continuous = mutual_info_regression(s1_reshape, s2, random_state=42)[0]
    
    # Method 2: Discrete MI (binning approach)
    s1_binned = pd.cut(s1, bins=bins, labels=False, duplicates='drop')
    s2_binned = pd.cut(s2, bins=bins, labels=False, duplicates='drop')
    
    # Remove NaN from binning
    valid = ~(pd.isna(s1_binned) | pd.isna(s2_binned))
    s1_binned = s1_binned[valid]
    s2_binned = s2_binned[valid]
    
    if len(s1_binned) > 0:
        mi_discrete = mutual_info_score(s1_binned, s2_binned)
    else:
        mi_discrete = 0
    
    # Normalized MI (0 to 1)
    h1 = -np.sum(np.histogram(s1_binned, bins=bins)[0] / len(s1_binned) * 
                 np.log2(np.histogram(s1_binned, bins=bins)[0] / len(s1_binned) + 1e-10))
    h2 = -np.sum(np.histogram(s2_binned, bins=bins)[0] / len(s2_binned) * 
                 np.log2(np.histogram(s2_binned, bins=bins)[0] / len(s2_binned) + 1e-10))
    
    normalized_mi = mi_discrete / max(h1, h2) if max(h1, h2) > 0 else 0
    
    return {
        'mutual_information_continuous': mi_continuous,
        'mutual_information_discrete': mi_discrete,
        'normalized_mi': normalized_mi
    }


def analyze_all(series1, series2, max_lag=10, dtw_window=50, mi_bins=20):
    """Run all correlation and causality analyses."""
    results = {}
    
    results['simple_correlation'] = simple_correlation(series1, series2)
    results['lagged_correlation'] = lagged_correlation(series1, series2, max_lag)
    results['granger_s1_causes_s2'] = granger_causality(series1, series2, max_lag)
    results['granger_s2_causes_s1'] = granger_causality(series2, series1, max_lag)
    results['spearman_correlation'] = spearman_correlation(series1, series2)
    results['kendall_correlation'] = kendall_correlation(series1, series2)
    results['dynamic_time_warping'] = dynamic_time_warping(series1, series2, dtw_window)
    results['mutual_information'] = mutual_information(series1, series2, mi_bins)
    
    return results


def print_report(results):
    """Print analysis report."""
    
    sc = results['simple_correlation']
    print(f"\nSimple Correlation: {sc['correlation']:.4f}")
    
    lc = results['lagged_correlation']
    best_lag = max(lc.items(), key=lambda x: abs(x[1]['correlation']))
    print(f"\nLagged Correlation (best at lag {best_lag[0]}): {best_lag[1]['correlation']:.4f}")
    
    gc1 = results['granger_s1_causes_s2']
    if 'error' not in gc1:
        sig_lags = [lag for lag, data in gc1.items() if data['granger_causes']]
        print(f"\nGranger Causality S1→S2: {'YES' if sig_lags else 'NO'}")
    
    sp = results['spearman_correlation']
    print(f"\nSpearman Correlation: {sp['correlation']:.4f}")
    
    kc = results['kendall_correlation']
    print(f"\nKendall Correlation: {kc['correlation']:.4f}")
    
    dtw = results['dynamic_time_warping']
    print(f"\nDynamic Time Warping Distance: {dtw['normalized_distance']:.4f}")
    
    mi = results['mutual_information']
    print(f"\nMutual Information: {mi['mutual_information_continuous']:.4f}")


# Example usage
if __name__ == "__main__":
    filename = 'random_25000.xlsx'
    
    try:
        df = pd.read_excel(filename)
        series1 = df.iloc[:, 0]
        series2 = df.iloc[:, 1]
        
        results = analyze_all(series1, series2, max_lag=10, dtw_window=100, mi_bins=20)
        print_report(results)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
    except Exception as e:
        print(f"Error: {e}")