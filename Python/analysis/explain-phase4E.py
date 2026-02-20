import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import lmdb
import io
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SiglipProcessor, SiglipModel
from peft import LoraConfig, get_peft_model
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = '/data' if os.path.exists('/data') else '.'
LMDB_PATH = "/opt/dlami/nvme/thesis_images.lmdb"
MANIFEST_FILE = os.path.join(BASE_DIR, "FINAL_THESIS_GROUND_TRUTH.csv")

MODEL_DIR = "Results_Phase4E_LMDB_Professional"
MODEL_PATH = os.path.join(MODEL_DIR, "Phase4E_model_epoch_3.pth") 
OUTPUT_DIR = "Master_SHAP_Visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5 
MAX_LEN = 64
RISK_LIMIT = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_LIST = [
    'raw_review_Amazon_Fashion', 'raw_review_Beauty_and_Personal_Care',
    'raw_review_Clothing_Shoes_and_Jewelry', 'raw_review_Home_and_Kitchen',
    'raw_review_Electronics', 'raw_review_Cell_Phones_and_Accessories',
    'raw_review_Tools_and_Home_Improvement', 'raw_review_Automotive',
    'raw_review_Sports_and_Outdoors'
]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORY_LIST)}
NUM_CATEGORIES = len(CATEGORY_LIST)

# --- 1. THE EXACT TRAINED MODEL ARCHITECTURE ---
class ThesisSiglipClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        peft = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")
        self.backbone = get_peft_model(base, peft)
        
        # This is the exact Embedding layer from your Phase 4E training script!
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, 32)
        
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768 + 32, 768), 
            nn.BatchNorm1d(768), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(768, 2)
        )

    def forward(self, pixel_values, input_ids, attention_mask, categories):
        if categories.dim() > 1:
            categories = categories.squeeze()
        if categories.dim() == 0:
            categories = categories.unsqueeze(0)

        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        combined = torch.cat((out.image_embeds, out.text_embeds, self.cat_embed(categories)), dim=1)
        return self.fusion(combined)

# --- 2. LMDB HELPER ---
def get_image_from_lmdb(sid, txn):
    img_bytes = txn.get(str(sid).encode('ascii'))
    if img_bytes:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return Image.new('RGB', (224, 224), color='black')

# --- 3. MAIN EXPLAINER SCRIPT ---
def run_master_explainer():
    print(" Initializing Master SHAP Explainer...")
    
    # Load Model
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = ThesisSiglipClassifier().to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Data
    df = pd.read_csv(MANIFEST_FILE)
    risky_df = df[df['return_likelihood'] >= RISK_LIMIT].sample(NUM_SAMPLES, random_state=42)
    
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    txn = env.begin(write=False)

    print(f" Generating Text & Image explanations for {len(risky_df)} samples...")

    all_text_shap_values = []

    for idx, row in risky_df.iterrows():
        sid = row.get('sample_id', row.get('parent_asin', 'unknown'))
        print(f"\n Processing Sample: {sid}")
        
        raw_text = str(row['text'])[:150] 
        cat_idx = torch.tensor([CAT_TO_IDX.get(row['category'], 0)]).to(DEVICE)
        
        # Base Image
        pil_img = get_image_from_lmdb(sid, txn).resize((224, 224))
        img_array = np.array(pil_img)

        # ---------------------------------------------------------
        # PART A: TEXT EXPLANATION (Force Plot)
        # ---------------------------------------------------------
        def text_predict_wrapper(text_list):
            with torch.no_grad():
                inputs = processor(text=list(text_list), images=[pil_img]*len(text_list), 
                                   return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
                
                cats = cat_idx.repeat(len(text_list))
                mask = inputs['attention_mask'].to(DEVICE) if 'attention_mask' in inputs else torch.ones_like(inputs['input_ids']).to(DEVICE)
                
                out = model(inputs['pixel_values'].to(DEVICE), inputs['input_ids'].to(DEVICE), mask, cats)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                return probs

        text_explainer = shap.Explainer(text_predict_wrapper, shap.maskers.Text(processor.tokenizer))
        text_shap_vals = text_explainer([raw_text])
        all_text_shap_values.append(text_shap_vals)

        # Save Text Plot
        shap.plots.force(text_shap_vals[0, :, 1], matplotlib=True, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f"SHAP_{sid}_TEXT.png"), bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # PART B: IMAGE EXPLANATION (High-Fidelity Heatmap)
        # ---------------------------------------------------------
        def image_predict_wrapper(img_arrays):
            with torch.no_grad():
                pil_images = [Image.fromarray(img.astype(np.uint8)) for img in img_arrays]
                inputs = processor(text=[raw_text]*len(pil_images), images=pil_images, 
                                   return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
                
                cats = cat_idx.repeat(len(pil_images))
                mask = inputs['attention_mask'].to(DEVICE) if 'attention_mask' in inputs else torch.ones_like(inputs['input_ids']).to(DEVICE)
                
                out = model(inputs['pixel_values'].to(DEVICE), inputs['input_ids'].to(DEVICE), mask, cats)
                return torch.softmax(out, dim=1).cpu().numpy()

        image_masker = shap.maskers.Image("inpaint_telea", (224, 224, 3))
        image_explainer = shap.Explainer(image_predict_wrapper, image_masker, output_names=["Safe", "Risky"])
        
        # Run SHAP (Max evals keeps it from running forever)
        img_shap_vals = image_explainer(img_array[np.newaxis, ...], max_evals=300, batch_size=32)

        # Save Image Plot
        shap.image_plot(img_shap_vals, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, f"SHAP_{sid}_IMAGE.png"), bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # PART C: GLOBAL TEXT BAR CHART
    # ---------------------------------------------------------
    print("\n Generating Global Text Importance Chart...")
    try:
        all_vals = []
        all_words = []
        
        # Extract and flatten every single word and its SHAP value
        for e in all_text_shap_values:
            all_vals.extend(e.values[0, :, 1]) 
            all_words.extend(e.data[0, :])
            
        # Convert to flat numpy arrays
        vals_flat = np.array(all_vals)
        words_flat = np.array(all_words)
        
        # Rebuild a 1D Global Explanation object
        global_text = shap.Explanation(values=vals_flat, data=words_flat)
        
        plt.figure(figsize=(12, 8))
        shap.plots.bar(global_text, max_display=20, show=False)
        plt.title("Phase 4E: Top 20 Words Driving 'Risky' Predictions")
        plt.savefig(os.path.join(OUTPUT_DIR, "SHAP_Global_Text_BarChart.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not generate Global Bar Chart: {e}")

if __name__ == "__main__":
    run_master_explainer()