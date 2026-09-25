import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

analytics_dir = os.path.dirname(__file__)
csv_path = os.path.join(analytics_dir, "titanic.csv")

# 1. Load Titanic dataset & save offline fallback
try:
    df = sns.load_dataset("titanic")
except Exception:
    df = pd.read_csv(csv_path)

df.to_csv(csv_path, index=False)

# Profiling
print("--- Data Profile ---")
print(f"Shape: {df.shape}")
print(df.info())
print(df.describe())

missing_pct = df.isnull().mean() * 100
print("\nMissing Values (%):\n", missing_pct[missing_pct > 0])

# 2. Missing value strategy based on thresholds
# - deck (77.10% missing): drop column (>30% threshold)
# - embark_town / embarked (<5% missing): drop rows
# - age (19.87% missing): impute median
df_clean = df.drop(columns=["deck"]).dropna(subset=["embarked", "embark_town"]).copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

# 3. IQR Outliers & Skewness
def report_iqr(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = series[(series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)]
    print(f"{name} IQR Outliers: {len(outliers)}")

report_iqr(df_clean["age"], "Age")
report_iqr(df_clean["fare"], "Fare")

mean_f, med_f, mode_f = df_clean["fare"].mean(), df_clean["fare"].median(), df_clean["fare"].mode()[0]
print(f"Fare Stats - Mean: {mean_f:.2f}, Median: {med_f:.2f}, Mode: {mode_f:.2f}")

# 4. Bivariate Analysis & Restricted 6-Column Heatmap
print("\nSurvival Rates:")
print("By Sex:\n", df_clean.groupby("sex")["survived"].mean())
print("By Pclass:\n", df_clean.groupby("pclass")["survived"].mean())
print("By Sex & Pclass:\n", df_clean.groupby(["sex", "pclass"])["survived"].mean())

num_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (6 Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(analytics_dir, "correlation_heatmap.png"))
plt.close()

# 5. Exploratory Z-score Standardization Check
age_z = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
fare_z = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()
print(f"Z-Score Check -> Age Mean: {age_z.mean():.4f}, Std: {age_z.std():.4f}")
print(f"Z-Score Check -> Fare Mean: {fare_z.mean():.4f}, Std: {fare_z.std():.4f}")