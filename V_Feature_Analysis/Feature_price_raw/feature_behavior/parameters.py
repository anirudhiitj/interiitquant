import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Path to your day file (use any representative day)
DATA_PATH = "/data/quant14/EBX/day10.csv"
FEATURE = "V2_T8"

df = pd.read_csv(DATA_PATH)

mu = df[FEATURE].mean()
sigma = df[FEATURE].std()
sk = skew(df[FEATURE])
kurt = kurtosis(df[FEATURE], fisher=False)  # Pearson definition (3 = normal)
threshold = mu + 3 * sigma

print("------------------------------------------------------------")
print(f"Feature: {FEATURE}")
print(f"Mean (μ): {mu:.4f}")
print(f"Std. Deviation (σ): {sigma:.4f}")
print(f"Skewness: {sk:.4f}")
print(f"Kurtosis: {kurt:.4f}")
print(f"Spike Threshold (μ + 3σ): {threshold:.4f}")
print("------------------------------------------------------------")
