import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REPORT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_dataset.csv")
OUT_PATH = os.path.join(REPORT_DIR, "eda_summary.txt")

TARGET = "liver_cancer"

# Load data
df = pd.read_csv(DATA_PATH)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"EDA Report for {DATA_PATH}\n\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Column types:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Missing values per column:\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("Class balance (target):\n")
    f.write(str(df[TARGET].value_counts()) + "\n\n")
    f.write("Descriptive stats (numerical):\n")
    f.write(str(df.describe(include=[np.number])) + "\n\n")
    f.write("Descriptive stats (categorical):\n")
    f.write(str(df.describe(include=["object", "category"])) + "\n\n")
    f.write("Correlation matrix (numerical):\n")
    f.write(str(df.corr(numeric_only=True)) + "\n\n")

# Optional: Save plots
plot_dir = os.path.join(REPORT_DIR, "figs")
os.makedirs(plot_dir, exist_ok=True)

# Class balance plot
plt.figure(figsize=(4,3))
sns.countplot(x=TARGET, data=df)
plt.title("Class Balance")
plt.savefig(os.path.join(plot_dir, "class_balance.png"), bbox_inches="tight")
plt.close()

# Histograms for numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(4,3))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Histogram: {col}")
    plt.savefig(os.path.join(plot_dir, f"hist_{col}.png"), bbox_inches="tight")
    plt.close()

# Boxplots for numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(4,3))
    sns.boxplot(x=df[col].dropna())
    plt.title(f"Boxplot: {col}")
    plt.savefig(os.path.join(plot_dir, f"box_{col}.png"), bbox_inches="tight")
    plt.close()

# Barplots for categorical columns
for col in df.select_dtypes(include=["object", "category"]).columns:
    plt.figure(figsize=(5,3))
    df[col].value_counts().plot.bar()
    plt.title(f"Barplot: {col}")
    plt.savefig(os.path.join(plot_dir, f"bar_{col}.png"), bbox_inches="tight")
    plt.close()

print(f"EDA report saved to {OUT_PATH}\nPlots saved to {plot_dir}")
