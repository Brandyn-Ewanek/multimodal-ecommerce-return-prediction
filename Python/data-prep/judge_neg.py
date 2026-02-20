import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions
import json
import re
from tqdm import tqdm
import time
import os

# --- 1. CONFIGURATION ---
API_KEY = "YOUR_API_KEY"
genai.configure(api_key=API_KEY)

#  MODEL: 2.5 Flash Lite
model = genai.GenerativeModel('models/gemini-2.5-flash-lite') 

# PATHS
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
file_in = os.path.join(BASE_DIR, "Data", "thesis_dataset_returned_FINAL.parquet")
file_out = os.path.join(BASE_DIR, "Data", "thesis_dataset_returned_TARGETS.parquet")

SAVE_INTERVAL = 100 

# --- 2. CLEANER ---
def clean_json_response(text):
    try:
        if "```" in text:
            text = text.replace("```json", "").replace("```", "")
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            return text[start:end]
        return text
    except:
        return text

# --- 3. THE "HIGH PRECISION" PROMPT (9 TARGETS) ---
def analyze_targets(text, category):
    prompt = f"""
    Review for {category}: "{text}"
    
    Analyze this return. Output a single JSON object.
    
    Keys Required:
    1. "visual_score": (float 0.00-1.00) Probability IMAGE is misleading. Use 2 decimal precision (e.g. 0.15, 0.87).
    2. "return_likelihood": (float 0.00-1.00) Probability of return. Use 2 decimal precision.
    3. "description_quality": (float 0.00-1.00) Accuracy of text. Use 2 decimal precision.
    4. "defect_category": (string) "Color", "Size", "Texture", "Design", "Quality".
    5. "visual_element": (string) Specific detail (e.g. "Hemline", "Logo").
    6. "fix": (string) Short advice to seller.
    7. "keywords": (string) Visual words used.
    8. "misleading_word": (string) The specific word in the description the user disagreed with (e.g. "Navy", "Silk"). If none, use "None".
    9. "correction_word": (string) The word the user would use instead (e.g. "Teal", "Polyester").
    """
    
    retries = 0
    while retries < 3:
        try:
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            clean_text = clean_json_response(response.text)
            data = json.loads(clean_text)
            
            # Handle list vs dict
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else None
            
            # Validation
            if data and "visual_score" in data:
                return data
            return None
            
        except exceptions.ResourceExhausted:
            time.sleep(2 * (retries + 1))
            retries += 1
        except Exception:
            return None 
    return None

# --- 4. INDEX PRESERVATION ---
print("🔄 Loading datasets...")
if not os.path.exists(file_in):
    print(f"❌ Input file not found: {file_in}")
    exit()

df_source = pd.read_parquet(file_in)

# Check for existing progress
if os.path.exists(file_out):
    print("   Resuming...")
    try:
        df_scored = pd.read_parquet(file_out)
        scored_indices = set(df_scored.index)
        print(f"   Found {len(scored_indices):,} completed rows.")
    except:
        print("   ⚠️ Output file empty/corrupt. Starting fresh.")
        scored_indices = set()
else:
    print("   Starting Fresh...")
    scored_indices = set()

# SHUFFLE logic
df_todo = df_source.loc[~df_source.index.isin(scored_indices)].sample(frac=1, random_state=42).copy()

print(f"   Remaining (Shuffled): {len(df_todo):,}")

# --- 5. EXECUTION LOOP ---
results = []
indices = [] # Track original indices
print(f"\n⚡ Speed Run Started (Model: gemini-2.5-flash-lite)...")

for idx, row in tqdm(df_todo.iterrows(), total=len(df_todo), desc="Judging"):
    
    data = analyze_targets(row['text'], row['category'])
    
    if data:
        res_row = row.to_dict()
        
        # 9 Targets
        res_row['visual_score'] = data.get('visual_score', 0.0)
        res_row['return_likelihood'] = data.get('return_likelihood', 0.0)
        res_row['description_quality_score'] = data.get('description_quality', 0.0)
        res_row['defect_category'] = data.get('defect_category', 'Other')
        res_row['visual_element_mismatch'] = data.get('visual_element', '')
        res_row['actionable_fix'] = data.get('fix', '')
        res_row['visual_keywords'] = str(data.get('keywords', ''))
        res_row['misleading_term'] = data.get('misleading_word', '')
        res_row['correction_term'] = data.get('correction_word', '')
        
        results.append(res_row)
        indices.append(idx) # CAPTURE THE ORIGINAL INDEX
    
    # --- SAVE BATCH ---
    if len(results) >= SAVE_INTERVAL:
        new_chunk = pd.DataFrame(results, index=indices)
        
        if os.path.exists(file_out):
            try:
                existing_df = pd.read_parquet(file_out)
                updated_df = pd.concat([existing_df, new_chunk], ignore_index=False) # Keep indices
            except:
                updated_df = new_chunk
        else:
            updated_df = new_chunk
        
        updated_df.to_parquet(file_out)
        
        results = []
        indices = []
        time.sleep(0.1) 

# --- FINAL SAVE ---
if len(results) > 0:
    new_chunk = pd.DataFrame(results, index=indices)
    if os.path.exists(file_out):
        existing_df = pd.read_parquet(file_out)
        updated_df = pd.concat([existing_df, new_chunk], ignore_index=False)
    else:
        updated_df = new_chunk
    updated_df.to_parquet(file_out)

print(f"\n DONE. Saved to: {file_out}")