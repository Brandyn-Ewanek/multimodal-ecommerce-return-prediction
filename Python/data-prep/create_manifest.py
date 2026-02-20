import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
DATA_DIR = os.path.join(BASE_DIR, "Data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Inputs
file_meta = os.path.join(DATA_DIR, "thesis_metadata_images.parquet")
file_neg_targets = os.path.join(DATA_DIR, "thesis_dataset_returned_TARGETS.parquet")
file_pos_targets = os.path.join(DATA_DIR, "thesis_dataset_positive_TARGETS.parquet")

# Output
manifest_out = os.path.join(DATA_DIR, "FINAL_THESIS_GROUND_TRUTH.csv")

# --- 1. LOAD DATA ---
print("🔄 Loading Datasets...")

# Load Metadata (Source of Descriptions)
if not os.path.exists(file_meta):
    print(" CRITICAL: Metadata file not found. We cannot get descriptions!")
    exit()
df_meta = pd.read_parquet(file_meta)
#  the ASIN and the Description
df_desc = df_meta[['parent_asin', 'description', 'title']].drop_duplicates(subset=['parent_asin'])
print(f"   Loaded Descriptions for {len(df_desc):,} items.")

# Load Negative Targets (Returns)
if os.path.exists(file_neg_targets):
    df_neg = pd.read_parquet(file_neg_targets)
    # We take the ASIN and the Score
    df_neg = df_neg[['parent_asin', 'return_likelihood', 'category']] 
    print(f"   Loaded {len(df_neg):,} Negative Targets.")
else:
    print("⚠️ Negative Targets not found. Skipping.")
    df_neg = pd.DataFrame()

# Load Positive Targets (Perfect Matches)
if os.path.exists(file_pos_targets):
    df_pos = pd.read_parquet(file_pos_targets)
    df_pos = df_pos[['parent_asin', 'return_likelihood', 'category']]
    print(f"   Loaded {len(df_pos):,} Positive Targets.")
else:
    print("⚠️ Positive Targets not found (Run judge_pos.py first!).")
    df_pos = pd.DataFrame()

# --- 2. MERGE & COMBINE ---
print("\n🔗 Combining Datasets...")

# Stack positive and negative together
df_combined = pd.concat([df_neg, df_pos], ignore_index=True)

# Merge with Descriptions
# Inner join: We only keep items where we have BOTH a score AND a description
df_final = df_combined.merge(df_desc, on='parent_asin', how='inner')

print(f"   Merged Count: {len(df_final):,}")

# --- 3. VERIFY IMAGES ON DISK ---
print("\n🔍 Verifying Local Images...")

valid_rows = []
missing_count = 0

# Get set of existing files for speed
existing_files = set(os.listdir(IMAGE_DIR))

for idx, row in tqdm(df_final.iterrows(), total=len(df_final)):
    asin = row['parent_asin']
    filename = f"{asin}.jpg"
    
    if filename in existing_files:
        row_data = {
            'sample_id': asin,
            'image_path': os.path.join(IMAGE_DIR, filename),
            'text': str(row['description']), # The Product Description
            'title': str(row['title']),
            'category': row['category'],
            'return_likelihood': float(row['return_likelihood']) # The Target
        }
        valid_rows.append(row_data)
    else:
        missing_count += 1

# --- 4. SAVE MANIFEST ---
df_manifest = pd.DataFrame(valid_rows)

print("\n" + "="*30)
print(f" MANIFEST REPORT")
print(f"   Total Valid Samples: {len(df_manifest):,}")
print(f"   Missing Images:      {missing_count:,}")
print("="*30)

if len(df_manifest) > 0:
    df_manifest.to_csv(manifest_out, index=False)
    print(f" Saved Manifest to: {manifest_out}")
    print("   You are ready for Phase 1A Training!")
else:
    print(" Manifest is empty. Check your paths.")