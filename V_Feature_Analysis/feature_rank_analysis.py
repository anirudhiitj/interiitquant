import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================
INPUT_FILE = Path("V_Feature_Analysis/EBX_combined.txt")
OUTPUT_FILE = Path("V_Feature_Analysis/feature_ranking.csv")

# ======================================================================
# REGEX PATTERNS
# ======================================================================
pattern = {
    "feature": re.compile(r"FEATURE:\s*([A-Za-z0-9_]+)"),
    "pearson": re.compile(r"Pearson:\s*([+-]?\d*\.\d+|\d+)"),
    "spearman": re.compile(r"Spearman:\s*([+-]?\d*\.\d+|\d+)"),
    "kendall": re.compile(r"Kendall:\s*([+-]?\d*\.\d+|\d+)"),
    "mutual_info": re.compile(r"Mutual\s*Info:\s*([+-]?\d*\.\d+|\d+)"),
    "lag": re.compile(r"Best\s*Lag:\s*([+-]?\d+)\s*periods"),
    "lag_corr": re.compile(r"Best\s*Lag\s*Corr:\s*([+-]?\d*\.\d+|\d+)"),
    "granger": re.compile(r"Granger.?Price:\s*(YES|NO)", re.IGNORECASE)
}

# ======================================================================
# PARSE LOGIC
# ======================================================================
features = []
block = {}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        # Detect start of a new feature
        if "FEATURE:" in line and block:
            features.append(block)
            block = {}

        for key, regex in pattern.items():
            match = regex.search(line)
            if match:
                block[key] = match.group(1)

    # Append last block at EOF
    if block:
        features.append(block)

# ======================================================================
# BUILD DATAFRAME
# ======================================================================
df = pd.DataFrame(features)
numeric_cols = ["pearson", "spearman", "kendall", "mutual_info", "lag", "lag_corr"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df["granger"] = df["granger"].str.upper().map({"YES": 1, "NO": 0})

print(f"✅ Parsed {len(df)} features successfully.")
print("Sample:")
print(df.head(5))
print("\nColumns extracted:", list(df.columns))

# ======================================================================
# COMPUTE COMPOSITE INFORMATION SCORE
# ======================================================================
if df["lag"].abs().max() == 0:
    df["lag_norm"] = 0
else:
    df["lag_norm"] = df["lag"].abs() / df["lag"].abs().max()

# Suggested weight configuration 
w = {
    "pearson": 0.20,
    "spearman": 0.15,
    "kendall": 0.05,
    "mutual_info": 0.15,
    "lag": 0.25,
    "lag_corr": 0.10,
    "granger": 0.10
}

df["info_score"] = (
    w["pearson"] * df["pearson"].abs() +
    w["spearman"] * df["spearman"].abs() +
    w["kendall"] * df["kendall"].abs() +
    w["mutual_info"] * df["mutual_info"] +
    w["lag"] * (1 - df["lag_norm"]) +
    w["lag_corr"] * df["lag_corr"].abs() +
    w["granger"] * df["granger"]
)

df.sort_values("info_score", ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# ======================================================================
# SAVE OUTPUT + SUMMARY
# ======================================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved ranked feature table → {OUTPUT_FILE}")

# Summary stats
print("\nTop 10 features:")
print(df[["feature", "info_score", "pearson", "spearman", "mutual_info", "lag", "granger"]].head(10))

print("\nGranger YES count:", df["granger"].sum())

# ======================================================================
# VISUALIZE TOP FEATURES
# ======================================================================
topn = 15
plt.figure(figsize=(10, 6))
plt.barh(df["feature"][:topn][::-1], df["info_score"][:topn][::-1])
plt.xlabel("Composite Information Score")
plt.title(f"Top {topn} Most Informative Volatility Features")
plt.tight_layout()
plt.show()
