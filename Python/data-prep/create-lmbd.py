import os
import zipfile
import lmdb
import pickle
from PIL import Image
from tqdm import tqdm
import io
import pandas as pd

# --- CONFIG ---
ZIP_PATH = "/opt/dlami/nvme/images.zip"
LMDB_PATH = "/opt/dlami/nvme/thesis_images.lmdb"
MANIFEST_FILE = "/data/FINAL_THESIS_GROUND_TRUTH.csv"

#  map 1TB to ensure we have address space 
MAP_SIZE = 1099511627776 

def create_dataset():
    if os.path.exists(LMDB_PATH):
        print(f"❌ Error: {LMDB_PATH} already exists. Please remove it first.")
        return

    print(f"🚀 Starting Conversion: Zip -> LMDB (High Performance DB)")
    
    # 1. Load manifest 
    df = pd.read_csv(MANIFEST_FILE)
    valid_ids = set()
    for x in df['sample_id'].unique():
        try: valid_ids.add(str(int(float(x))))
        except: valid_ids.add(str(x))
    
    # 2. Open LMDB Environment
    env = lmdb.open(LMDB_PATH, map_size=MAP_SIZE)
    
    # 3. Stream Zip and Write to LMDB
    count = 0
    with env.begin(write=True) as txn:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            # List all files in zip
            all_files = z.namelist()
            
            for filename in tqdm(all_files, desc="Processing Images"):
                if not filename.endswith((".jpg", ".jpeg")): continue
                
                # Extract ID from filename (e.g., "images/B001.jpg" -> "B001")
                base = os.path.basename(filename)
                file_id = os.path.splitext(base)[0]
                
                # Only save if it's in our CSV 
                if file_id in valid_ids:
                    try:
                        # Read bytes
                        img_bytes = z.read(filename)
                        
                        # Resize to 224x224 
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        img = img.resize((224, 224))
                        
                        # Save back to bytes
                        out_io = io.BytesIO()
                        img.save(out_io, format='JPEG', quality=90)
                        final_bytes = out_io.getvalue()
                        
                        # Write to DB
                        txn.put(file_id.encode('ascii'), final_bytes)
                        count += 1
                        
                        # Commit every 1000 images 
                        if count % 1000 == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                            
                    except Exception as e:
                        print(f"Skipping {file_id}: {e}")

    print(f"🎉 Success! {count} images saved to {LMDB_PATH}")
    print("   You can now delete the zip file if you need space.")

if __name__ == "__main__":
    create_dataset()