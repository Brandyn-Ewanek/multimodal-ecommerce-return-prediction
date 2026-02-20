import pandas as pd
import os

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
file_path = os.path.join(BASE_DIR, "Data", "thesis_dataset_returned_TARGETS.parquet")

# --- EXECUTION ---
print(f"🔍 Inspecting: {file_path} ...")

if not os.path.exists(file_path):
    print("❌ File not found yet. The judge script hasn't saved the first batch (100 rows) yet.")
    print("   Wait a few minutes and try again.")
    exit()

try:
    df = pd.read_parquet(file_path)
    print(f"✅ File Loaded. Total Rows Scored So Far: {len(df):,}")
    
    # Pick 5 random rows to check
    sample_size = min(10, len(df))
    sample = df.sample(sample_size)
    
    print("\n" + "="*80)
    
    for idx, row in sample.iterrows():
        print(f"📂 CATEGORY: {row.get('category', 'N/A')}")
        print(f"💬 REVIEW:  \"{row.get('text', '')[:200]}...\"") # Show first 200 chars
        print("-" * 40)
        
        # The Critical Targets
        print(f"🎯 VISUAL SCORE:      {row.get('visual_score', 'N/A')}  (0.0 - 1.0)")
        print(f"📉 RETURN RISK:       {row.get('return_likelihood', 'N/A')}")
        print(f"📝 DESC QUALITY:      {row.get('description_quality_score', 'N/A')}")
        
        # The Business Intel
        print(f"🚫 MISLEADING WORD:   {row.get('misleading_term', 'N/A')}")
        print(f"✅ CORRECTION:        {row.get('correction_term', 'N/A')}")
        print(f"🔧 FIX ADVICE:        {row.get('actionable_fix', 'N/A')}")
        
        print("="*80 + "\n")

except Exception as e:
    print(f"❌ Error reading file: {e}")