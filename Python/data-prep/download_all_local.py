import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# --- 1. LOCAL CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product" 

# Inputs
meta_file = os.path.join(BASE_DIR, "Data", "thesis_metadata_images.parquet")
neg_file_in = os.path.join(BASE_DIR, "Data", "thesis_dataset_returned_DEDUPED_NO_SHORT.parquet")
pos_file_in = os.path.join(BASE_DIR, "Data", "thesis_dataset_positive_DEDUPED_NO_SHORT.parquet")

# Outputs
output_folder = os.path.join(BASE_DIR, "images")
os.makedirs(output_folder, exist_ok=True)

neg_out = os.path.join(BASE_DIR, "Data", "thesis_dataset_returned_FINAL.parquet")
pos_out = os.path.join(BASE_DIR, "Data", "thesis_dataset_positive_FINAL.parquet")

# --- 2. LOAD & PREPARE ---
print(" Loading Metadata...")
try:
    df_meta = pd.read_parquet(meta_file)
    print(f"   Targets Loaded: {len(df_meta):,}")
except Exception as e:
    print(f" Error loading metadata: {e}")
    print("   Make sure you downloaded the .parquet files to the Data folder!")
    exit()

print(" Checking local disk state...")
existing_files = set(os.listdir(output_folder))
print(f"   Found {len(existing_files):,} images already done.")

# Identify what is left to do
df_meta['filename'] = df_meta['parent_asin'] + ".jpg"
df_missing = df_meta[~df_meta['filename'].isin(existing_files)].copy()

print(f"   🚀 Queued for Download: {len(df_missing):,}")

# --- 3. THE DOWNLOAD WORKER ---
def download_image(row):
    url = row['image_url']
    asin = row['parent_asin']
    
    # Sanitize URL (Handle lists/arrays)
    try:
        if hasattr(url, '__len__') and not isinstance(url, str):
            url = url[0] if len(url) > 0 else None
    except:
        return None

    if not url or not isinstance(url, str):
        return None

    file_path = os.path.join(output_folder, f"{asin}.jpg")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            # Verify it's a real image (not a "Access Denied" HTML page)
            img = Image.open(BytesIO(response.content))
            img.verify() 
            
            # Save
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return asin
    except:
        pass # Fail silently (404s are normal)
    
    return None

# --- 4. EXECUTE (MASSIVE PARALLELISM) ---
if len(df_missing) > 0:
    print("\n Starting Download Stream (50 Threads)...")
    print("   (This will take a few hours. Do not close this window.)")
    
    rows = df_missing.to_dict('records')
    
    # Adjust max_workers based on your internet. 50 is usually safe.
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(tqdm(executor.map(download_image, rows), total=len(rows), desc="Downloading"))

# --- 5. BUILD FINAL DATASETS ---
print("\n Building Final Parquet Files...")

# Re-scan folder to see what we actually got
final_files = set(os.listdir(output_folder))
valid_asins = {f.replace(".jpg", "") for f in final_files if f.endswith(".jpg")}

print(f"   Total Valid Images: {len(valid_asins):,}")

def save_final(path_in, path_out, name):
    try:
        df = pd.read_parquet(path_in)
        initial = len(df)
        
        # Keep only rows with valid images
        df_final = df[df['parent_asin'].isin(valid_asins)].copy()
        
        # Add local absolute path
        df_final['image_path'] = df_final['parent_asin'].apply(lambda x: os.path.join(output_folder, f"{x}.jpg"))
        
        print(f"\n   --- {name} ---")
        print(f"   Original: {initial:,}")
        print(f"   Final:    {len(df_final):,}")
        
        if len(df_final) > 0:
            df_final.to_parquet(path_out, index=False)
            print(f"   Saved: {path_out}")
    except Exception as e:
        print(f"   Could not process {name}: {e}")

save_final(neg_file_in, neg_out, "NEGATIVE")
save_final(pos_file_in, pos_out, "POSITIVE")

print("\n✅ DONE. You are ready for the Gemini Judge.")