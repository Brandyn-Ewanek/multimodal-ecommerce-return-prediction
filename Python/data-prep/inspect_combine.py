import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
DATA_DIR = os.path.join(BASE_DIR, "Data")

file_neg = os.path.join(DATA_DIR, "thesis_dataset_returned_TARGETS.parquet")
file_pos = os.path.join(DATA_DIR, "thesis_dataset_positive_TARGETS.parquet")

def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

# --- 1. INSPECT NEGATIVE (RETURNS) ---
print_header("DATASET 1: RETURNED ITEMS (Negative)")
if os.path.exists(file_neg):
    df_neg = pd.read_parquet(file_neg)
    print(f"✅ Loaded: {len(df_neg):,} rows")
    print(f"   Columns: {list(df_neg.columns)}")
    
    # Check the score distribution
    if 'return_likelihood' in df_neg.columns:
        stats = df_neg['return_likelihood'].describe()
        print("\n   --- Return Likelihood Stats ---")
        print(f"   Mean Score: {stats['mean']:.4f} (Should be HIGH, > 0.6)")
        print(f"   Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
        
        # Simple Text Histogram
        high_risk = len(df_neg[df_neg['return_likelihood'] > 0.7])
        print(f"   High Risk (>0.7): {high_risk:,} ({high_risk/len(df_neg):.1%})")
    else:
        print(" CRITICAL: 'return_likelihood' column missing!")
else:
    print(" File not found.")
    df_neg = pd.DataFrame()

# --- 2. INSPECT POSITIVE (PERFECT MATCHES) ---
print_header("DATASET 2: POSITIVE ITEMS (Good)")
if os.path.exists(file_pos):
    df_pos = pd.read_parquet(file_pos)
    print(f"✅ Loaded: {len(df_pos):,} rows")
    
    if 'return_likelihood' in df_pos.columns:
        stats = df_pos['return_likelihood'].describe()
        print("\n   --- Return Likelihood Stats ---")
        print(f"   Mean Score: {stats['mean']:.4f} (Should be LOW, < 0.3)")
        print(f"   Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
        
        # Simple Text Histogram
        low_risk = len(df_pos[df_pos['return_likelihood'] < 0.3])
        print(f"   Low Risk (<0.3): {low_risk:,} ({low_risk/len(df_pos):.1%})")
    else:
        print(" CRITICAL: 'return_likelihood' column missing!")
else:
    print(" File not found.")
    df_pos = pd.DataFrame()

# --- 3. SIMULATED TRAINING SET ---
print_header("COMBINED TRAINING PREVIEW")

if not df_neg.empty and not df_pos.empty:
    total_rows = len(df_neg) + len(df_pos)
    print(f" Total Training Samples: {total_rows:,}")
    
    # Class Balance
    neg_ratio = len(df_neg) / total_rows
    pos_ratio = len(df_pos) / total_rows
    
    print(f"   Balance: {neg_ratio:.1%} Returned  vs  {pos_ratio:.1%} Positive")
    
    if 0.4 < neg_ratio < 0.6:
        print("    Dataset is Well Balanced.")
    else:
        print("    Dataset is Imbalanced .")
        
    # Check Common Columns
    common_cols = set(df_neg.columns) & set(df_pos.columns)
    print(f"   Shared Columns for Training: {list(common_cols)}")
    
else:
    print(" Cannot combine: One or both datasets are missing.")

print("\n")