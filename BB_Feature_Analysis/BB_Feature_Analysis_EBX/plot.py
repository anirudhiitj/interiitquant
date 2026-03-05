import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the data
df = pd.read_csv('/home/raid/Quant14/BB_Feature_Analysis/BB_Feature_Analysis_EBX/correlation_results_20251022_082143.csv')

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('BB Feature Analysis - Comprehensive Overview', fontsize=16, fontweight='bold')

# Plot 1: Top 20 Features by Absolute Pearson Correlation
ax1 = axes[0, 0]
top_20 = df.nlargest(20, 'Abs_Pearson')
ax1.barh(range(len(top_20)), top_20['Abs_Pearson'], color='steelblue')
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['Feature'], fontsize=8)
ax1.set_xlabel('Absolute Pearson Correlation')
ax1.set_title('Top 20 Features by Absolute Pearson Correlation')
ax1.invert_yaxis()

# Plot 2: Pearson vs Spearman Correlation
ax2 = axes[0, 1]
scatter = ax2.scatter(df['Pearson'], df['Spearman'], 
                     c=df['Abs_Pearson'], cmap='viridis', 
                     alpha=0.6, s=50)
ax2.plot([-1, 1], [-1, 1], 'r--', alpha=0.5, label='y=x')
ax2.set_xlabel('Pearson Correlation')
ax2.set_ylabel('Spearman Correlation')
ax2.set_title('Pearson vs Spearman Correlation')
ax2.legend()
plt.colorbar(scatter, ax=ax2, label='Abs Pearson')

# Plot 3: Distribution of Valid Data Percentage
ax3 = axes[0, 2]
ax3.hist(df['Valid_Pct'], bins=30, color='coral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Valid Data Percentage (%)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Valid Data Percentage')
ax3.axvline(df['Valid_Pct'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["Valid_Pct"].mean():.2f}%')
ax3.legend()

# Plot 4: Mutual Information vs Pearson Correlation
ax4 = axes[1, 0]
scatter2 = ax4.scatter(df['Abs_Pearson'], df['Mutual_Info'], 
                      c=df['Valid_Pct'], cmap='plasma', 
                      alpha=0.6, s=50)
ax4.set_xlabel('Absolute Pearson Correlation')
ax4.set_ylabel('Mutual Information')
ax4.set_title('Mutual Information vs Absolute Pearson Correlation')
plt.colorbar(scatter2, ax=ax4, label='Valid %')

# Plot 5: Granger Causality Distribution
ax5 = axes[1, 1]
granger_counts = df['Granger→Price'].value_counts()
colors = ['#2ecc71' if x == 'YES' else '#e74c3c' for x in granger_counts.index]
ax5.bar(granger_counts.index, granger_counts.values, color=colors, edgecolor='black')
ax5.set_ylabel('Count')
ax5.set_title('Granger Causality to Price Distribution')
for i, v in enumerate(granger_counts.values):
    ax5.text(i, v + 5, str(v), ha='center', fontweight='bold')

# Plot 6: Best Lag Distribution
ax6 = axes[1, 2]
lag_counts = df['Best_Lag'].value_counts().sort_index()
ax6.bar(lag_counts.index, lag_counts.values, color='teal', edgecolor='black', alpha=0.7)
ax6.set_xlabel('Best Lag')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Best Lag Values')
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('bb_feature_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional individual plots
# Plot 7: Heatmap of top correlations
fig2, ax = plt.subplots(figsize=(12, 10))
top_30 = df.nlargest(30, 'Abs_Pearson')
corr_data = top_30[['Pearson', 'Spearman', 'Kendall']].T
sns.heatmap(corr_data, annot=False, cmap='RdYlGn', center=0, 
            xticklabels=top_30['Feature'], yticklabels=['Pearson', 'Spearman', 'Kendall'],
            cbar_kws={'label': 'Correlation'}, ax=ax)
ax.set_title('Top 30 Features - Correlation Heatmap', fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('bb_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("BB FEATURE ANALYSIS SUMMARY")
print("="*60)
print(f"Total Features: {len(df)}")
print(f"Average Valid Data %: {df['Valid_Pct'].mean():.2f}%")
print(f"Features with Granger Causality: {(df['Granger→Price'] == 'YES').sum()}")
print(f"Average Absolute Pearson Correlation: {df['Abs_Pearson'].mean():.4f}")
print(f"\nTop 5 Features by Correlation:")
print(df.nlargest(5, 'Abs_Pearson')[['Feature', 'Pearson', 'Abs_Pearson']])
print("="*60)