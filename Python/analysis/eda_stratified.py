import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
DATA_DIR = os.path.join(BASE_DIR, "Data")
PLOT_DIR = os.path.join(BASE_DIR, "Analysis_Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Input Files
file_neg = os.path.join(DATA_DIR, "thesis_dataset_returned_TARGETS.parquet")
file_pos = os.path.join(DATA_DIR, "thesis_dataset_positive_TARGETS.parquet")

# --- CATEGORY DEFINITIONS ---
GROUP_A_HIGH_VISUAL = [
    "raw_review_Amazon_Fashion",
    "raw_review_Clothing_Shoes_and_Jewelry", 
    "raw_review_Beauty_and_Personal_Care",
    "raw_review_Home_and_Kitchen"
]

GROUP_B_LOW_VISUAL = [
    "raw_review_Electronics",
    "raw_review_Cell_Phones_and_Accessories",
    "raw_review_Tools_and_Home_Improvement",
    "raw_review_Automotive",
    "raw_review_Sports_and_Outdoors"
]

def get_group(cat):
    if cat in GROUP_A_HIGH_VISUAL: return "Group A (High Visual)"
    if cat in GROUP_B_LOW_VISUAL: return "Group B (Functional)"
    return "Other"

def annotate_r2(data, x, y, ax, color='black', loc='upper left'):
    """Calculates and writes Pearson R and R-Squared on the plot"""
    clean_data = data[[x, y]].dropna()
    if len(clean_data) < 2: return
    
    r, p = stats.pearsonr(clean_data[x], clean_data[y])
    r2 = r**2
    
    text = f"$R^2 = {r2:.3f}$\n$p = {p:.1e}$"
    ax.text(0.05, 0.90, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- 1. LOAD DATA ---
print("🔄 Loading Target Data...")
df_neg = pd.read_parquet(file_neg)
df_pos = pd.read_parquet(file_pos)
df = pd.concat([df_neg, df_pos], ignore_index=True)

# Apply Grouping
df['Visual_Group'] = df['category'].apply(get_group)

# Sampling for Scatter Plots 
df_sample = df.sample(n=min(20000, len(df)), random_state=42)

print(f"✅ Loaded {len(df):,} rows.")

# --- 2. PLOT 1: GROUP A vs GROUP B (The Thesis Test) ---
print("📊 Generating Group Comparison Plot...")

plt.figure(figsize=(12, 6))
g = sns.lmplot(
    data=df_sample, 
    x='visual_score', 
    y='return_likelihood', 
    col='Visual_Group', 
    hue='Visual_Group',
    palette={'Group A (High Visual)': 'magenta', 'Group B (Functional)': 'blue'},
    scatter_kws={'alpha': 0.1},
    line_kws={'lw': 2},
    height=6, aspect=1.2
)

# Annotate R2 for each subplot
for ax, title in zip(g.axes.flatten(), g.col_names):
    ax.set_title(title, fontsize=14)
    subset = df[df['Visual_Group'] == title]
    annotate_r2(subset, 'visual_score', 'return_likelihood', ax)

plt.subplots_adjust(top=0.85)
g.fig.suptitle('Thesis Test: Does Visual Mismatch Predict Returns differently by Group?', fontsize=16)
output_path = os.path.join(PLOT_DIR, "Thesis_RQ3_Group_Comparison_R2.png")
plt.savefig(output_path)
plt.close()
print(f"   Saved: {output_path}")

# --- 3. PLOT 2: DETAILED BREAKDOWN BY CATEGORY ---
print("📊 Generating Granular Category Plot...")

categories = df['category'].unique()
# Create a grid of plots
num_cats = len(categories)
cols = 3
rows = (num_cats // cols) + (1 if num_cats % cols > 0 else 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, cat in enumerate(categories):
    ax = axes[i]
    subset = df[df['category'] == cat]
    
    # Determine color based on group
    color = 'magenta' if cat in GROUP_A_HIGH_VISUAL else 'blue'
    
    # Scatter Plot with Regression
    sns.regplot(
        data=subset.sample(n=min(2000, len(subset)), random_state=42), # Sample per cat for speed
        x='visual_score', 
        y='return_likelihood', 
        ax=ax, 
        color=color,
        scatter_kws={'alpha': 0.1}, 
        line_kws={'color': 'black'}
    )
    
    # Titles and Stats
    clean_cat_name = cat.replace('raw_review_', '')
    ax.set_title(f"{clean_cat_name}", fontsize=12)
    ax.set_xlabel("Visual Discrepancy")
    ax.set_ylabel("Return Risk")
    
    # Add R2
    annotate_r2(subset, 'visual_score', 'return_likelihood', ax)

# Hide empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_cat = os.path.join(PLOT_DIR, "Thesis_Granular_Category_R2.png")
plt.savefig(output_path_cat)
plt.close()
print(f"   Saved: {output_path_cat}")

print("\n✅ Stratified R2 Analysis Complete.")