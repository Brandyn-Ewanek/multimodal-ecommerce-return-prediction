import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
DATA_DIR = os.path.join(BASE_DIR, "Data")
PLOT_DIR = os.path.join(BASE_DIR, "Analysis_Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

file_neg = os.path.join(DATA_DIR, "thesis_dataset_returned_TARGETS.parquet")
file_pos = os.path.join(DATA_DIR, "thesis_dataset_positive_TARGETS.parquet")

# Set Visual Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 1. LOAD & PREPARE ---
print("🔄 Loading datasets for Deep Dive...")
df_neg = pd.read_parquet(file_neg)
df_pos = pd.read_parquet(file_pos)

# Add Type Labels for Plotting
df_neg['dataset_type'] = 'Returned (Negative)'
df_pos['dataset_type'] = 'Perfect (Positive)'

# Combine for holistic analysis
df = pd.concat([df_neg, df_pos], ignore_index=True)

# Sample for dense plots 
df_sample = df.sample(n=10000, random_state=42)

print(f"✅ Data Loaded. Total Rows: {len(df):,}")

# --- 2. DISTRIBUTION ANALYSIS ---
print("📊 Generating Distribution Plots...")

# A. Return Likelihood Distribution 
plt.figure()
sns.histplot(data=df, x='return_likelihood', hue='dataset_type', bins=50, kde=True, palette=['red', 'green'], alpha=0.6)
plt.title('Distribution of Return Likelihood Scores (The "Signal")', fontsize=16)
plt.xlabel('Return Risk Score (0=Safe, 1=Risk)')
plt.ylabel('Count of Products')
plt.savefig(os.path.join(PLOT_DIR, "01_Distribution_ReturnLikelihood.png"))
plt.close()

# B. Visual Score Distribution 
plt.figure()
sns.histplot(data=df, x='visual_score', hue='dataset_type', bins=50, kde=True, palette=['red', 'green'], alpha=0.6)
plt.title('Distribution of Visual Discrepancy Scores', fontsize=16)
plt.xlabel('Visual Score (0=Accurate, 1=Misleading)')
plt.savefig(os.path.join(PLOT_DIR, "02_Distribution_VisualScore.png"))
plt.close()

# C. Description Quality Distribution
plt.figure()
sns.histplot(data=df, x='description_quality_score', hue='dataset_type', bins=50, kde=True, palette=['red', 'green'], alpha=0.6)
plt.title('Distribution of Description Quality', fontsize=16)
plt.xlabel('Quality Score')
plt.savefig(os.path.join(PLOT_DIR, "03_Distribution_DescQuality.png"))
plt.close()

# --- 3. CORRELATION ANALYSIS ---
print("🔗 Generating Correlation Plots...")

# A. Heatmap of Key Metrics
cols_to_corr = ['return_likelihood', 'visual_score', 'description_quality_score', 'word_count', 'rating']
corr_matrix = df[cols_to_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix: What drives Returns?', fontsize=16)
plt.savefig(os.path.join(PLOT_DIR, "04_Correlation_Heatmap.png"))
plt.close()

# B. Word Count vs Return Risk
# Binning word counts to see trends
df['word_count_bin'] = pd.cut(df['word_count'], bins=[0, 20, 50, 100, 200, 500, 1000], labels=['0-20', '20-50', '50-100', '100-200', '200-500', '500+'])

plt.figure()
sns.boxplot(data=df, x='word_count_bin', y='return_likelihood', hue='dataset_type', palette=['red', 'green'])
plt.title('Impact of Description Length on Return Risk', fontsize=16)
plt.xlabel('Word Count Range')
plt.ylabel('Return Likelihood')
plt.savefig(os.path.join(PLOT_DIR, "05_WordCount_vs_Risk.png"))
plt.close()

# --- 4. CATEGORY & KEYWORD ANALYSIS ---
print("🔠 Generating Category/Keyword Plots...")

# A. Return Risk by Category
cat_risk = df.groupby('category')['return_likelihood'].mean().sort_values()

plt.figure(figsize=(12, 6))
sns.barplot(x=cat_risk.values, y=cat_risk.index, palette='viridis')
plt.title('Average Return Risk by Category', fontsize=16)
plt.xlabel('Mean Return Risk')
plt.savefig(os.path.join(PLOT_DIR, "06_Risk_by_Category.png"))
plt.close()

# B. Top 20 "Trigger Keywords" for Returns
# We filter only for the Negative dataset to see what specific words drove the high scores
top_triggers = df_neg['trigger_keyword'].value_counts().head(20).index
trigger_risk = df_neg[df_neg['trigger_keyword'].isin(top_triggers)].groupby('trigger_keyword')['return_likelihood'].mean().sort_values()

plt.figure(figsize=(12, 8))
sns.barplot(x=trigger_risk.values, y=trigger_risk.index, palette='magma')
plt.title('Top 20 Trigger Keywords sorted by Severity', fontsize=16)
plt.xlabel('Mean Return Risk Score')
plt.savefig(os.path.join(PLOT_DIR, "07_Keyword_Severity.png"))
plt.close()

# C. Scatter: Visual Score vs Return Risk
# Using the Sample dataframe to prevent blob mess
plt.figure()
sns.regplot(data=df_sample, x='visual_score', y='return_likelihood', scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('Relationship: Visual Mismatch vs. Return Risk', fontsize=16)
plt.xlabel('Visual Discrepancy Score')
plt.ylabel('Return Likelihood')
plt.savefig(os.path.join(PLOT_DIR, "08_Scatter_Visual_vs_Return.png"))
plt.close()

print("\n" + "="*50)
print(f"✅ ANALYSIS COMPLETE.")
print(f"   All plots saved to: {PLOT_DIR}")
print("="*50)