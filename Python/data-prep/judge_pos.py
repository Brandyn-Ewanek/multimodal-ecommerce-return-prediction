import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions
import json
import re
from tqdm import tqdm
import time
import os

# --- 1. CONFIGURATION ---
# ⚠️ Replace with your actual API Key if needed
API_KEY = "YOUR_API_KEY"
genai.configure(api_key=API_KEY)

# Using the fast model
model = genai.GenerativeModel('models/gemini-2.5-flash-lite') 

# PATHS
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
DATA_DIR = os.path.join(BASE_DIR, "Data")

# Input: The raw positive reviews we collected earlier
file_in = os.path.join(DATA_DIR, "thesis_dataset_positive_FINAL.parquet")
# Output: The scored targets 
file_out = os.path.join(DATA_DIR, "thesis_dataset_positive_TARGETS.parquet")

SAVE_INTERVAL = 100 

# --- 2. CLEANER ---
def clean_json_response(text):
    try:
        # Strip markdown code blocks if present
        if "```" in text:
            text = text.replace("```json", "").replace("```", "")
        
        # Find the actual JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return text
    except:
        return text

# --- 3. THE "POSITIVE" PROMPT ---
def analyze_positive_targets(text, category):
    prompt = f"""
    Positive Review for {category}: "{text}"
    
    This customer was HAPPY. Analyze why the image/text succeeded. Output JSON ONLY.
    
    Keys Required:
    1. "visual_score": (float 0.00-1.00) Probability IMAGE is misleading. (Since this is a positive review, this should be very low, e.g., 0.01 - 0.10).
    2. "return_likelihood": (float 0.00-1.00) Probability of return. 
    3. "description_quality": (float 0.00-1.00) Accuracy of text description. (Should be high, e.g. 0.90+).
    4. "winning_term": (string) The specific adjective used to praise the visual (e.g. "Buttery", "Vibrant", "Sturdy").
    5. "best_visual_feature": (string) The specific part praised (e.g. "Hemline", "Logo", "Texture").
    6. "fit_result": (string) "True to Size", "Runs Large", "Runs Small", or "Not Mentioned".
    7. "lighting_score": (int 1-10) Inferred quality of product representation.
    8. "sophistication": (string) "Novice" vs "Expert".
    """
    
    retries = 0
    while retries < 3:
        try:
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            clean_text = clean_json_response(response.text)
            data = json.loads(clean_text)
            
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else None
            
            # Validation: Ensure key fields exist
            if data and "visual_score" in data:
                return data
            return None
            
        except exceptions.ResourceExhausted:
            # If we hit the rate limit, pause and retry
            time.sleep(2 * (retries + 1))
            retries += 1
        except Exception:
            # Skip rows that cause errors
            return None 
    return None

# --- 4. EXECUTION LOOP ---
print(" Loading Positive Dataset...")
if not os.path.exists(file_in):
    print(f" Input file not found: {file_in}")
    exit()

df_source = pd.read_parquet(file_in)
print(f"   Source Rows: {len(df_source):,}")

# Resume Logic
if os.path.exists(file_out):
    print("   Resuming from previous save...")
    try:
        df_scored = pd.read_parquet(file_out)
        scored_indices = set(df_scored.index)
        print(f"   Found {len(scored_indices):,} completed rows.")
    except:
        scored_indices = set()
else:
    print("   Starting Fresh...")
    scored_indices = set()

# Filter and Shuffle
df_todo = df_source.loc[~df_source.index.isin(scored_indices)].sample(frac=1, random_state=42).copy()
print(f"   Remaining to Judge: {len(df_todo):,}")

results = []
indices = []
print("\n⚡ Positive Scoring Started (Target: ~20,000 rows)...")
print("   (Press Ctrl+C to stop when you have enough)")

try:
    for idx, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Judging"):
        
        data = analyze_positive_targets(row['text'], row['category'])
        
        if data:
            res_row = row.to_dict()
            
            # --- MAPPING SCHEMA ---
            res_row['visual_score'] = float(data.get('visual_score', 0.05))
            res_row['return_likelihood'] = float(data.get('return_likelihood', 0.05))
            res_row['description_quality_score'] = float(data.get('description_quality', 0.95))
            
            # Metadata
            res_row['winning_term'] = str(data.get('winning_term', ''))
            res_row['best_visual_feature'] = str(data.get('best_visual_feature', ''))
            res_row['fit_result'] = str(data.get('fit_result', 'Not Mentioned'))
            res_row['lighting_score'] = int(data.get('lighting_score', 8))
            res_row['buyer_sophistication'] = str(data.get('sophistication', 'Novice'))
            
            results.append(res_row)
            indices.append(idx)
        
        # Save every chunk
        if len(results) >= SAVE_INTERVAL:
            new_chunk = pd.DataFrame(results, index=indices)
            if os.path.exists(file_out):
                try:
                    existing_df = pd.read_parquet(file_out)
                    updated_df = pd.concat([existing_df, new_chunk], ignore_index=False)
                except:
                    updated_df = new_chunk
            else:
                updated_df = new_chunk
            
            # Write safely
            updated_df.to_parquet(file_out)
            results = []
            indices = []
            time.sleep(0.1) # Brief pause to be nice to the CPU

except KeyboardInterrupt:
    print("\n🛑 Process stopped by user.")

# Final Save on Exit
if len(results) > 0:
    new_chunk = pd.DataFrame(results, index=indices)
    if os.path.exists(file_out):
        existing_df = pd.read_parquet(file_out)
        updated_df = pd.concat([existing_df, new_chunk], ignore_index=False)
    else:
        updated_df = new_chunk
    updated_df.to_parquet(file_out)

print(f"\n✅ SAVED. Total Rows in {file_out}: {len(pd.read_parquet(file_out)):,}")