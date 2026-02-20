import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipProcessor, SiglipModel
from peft import LoraConfig, get_peft_model
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import time
import lmdb
import io

# --- CONFIGURATION ---
if os.path.exists('/data'):
    BASE_DIR = '/data'
    print("☁️ AWS ENVIRONMENT DETECTED: Using /data")
MANIFEST_FILE = os.path.join(BASE_DIR, "FINAL_THESIS_GROUND_TRUTH.csv") 
LMDB_PATH = "/opt/dlami/nvme/thesis_images.lmdb"


# Output Dirs (Phase 4C - Weighted Loss 2x)
RESULTS_DIR = "Results_Phase4C_Weighted_Entailment_2x"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32 
EPOCHS = 3            
LEARNING_RATE = 5e-4 
MAX_LEN = 64
TEST_SPLIT = 0.20

# --- THRESHOLDS ---
SAFE_LIMIT = 0.4
RISK_LIMIT = 0.6

# --- 1. CATEGORY MAPPING ---
CATEGORY_LIST = [
    'raw_review_Amazon_Fashion',
    'raw_review_Beauty_and_Personal_Care',
    'raw_review_Clothing_Shoes_and_Jewelry',
    'raw_review_Home_and_Kitchen',
    'raw_review_Electronics',
    'raw_review_Cell_Phones_and_Accessories',
    'raw_review_Tools_and_Home_Improvement',
    'raw_review_Automotive',
    'raw_review_Sports_and_Outdoors'
]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORY_LIST)}
NUM_CATEGORIES = len(CATEGORY_LIST)

# --- 2. DATASET CLASS ---
class ThesisEntailmentDataset(Dataset):
    def __init__(self, df, processor, split_name, lmdb_path):
        self.df = df
        self.processor = processor
        self.split_name = split_name
        self.lmdb_path = lmdb_path
        self.env = None # Opened lazily for multiprocessing

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.env is None: 
            self._init_db()
        
        row = self.df.iloc[idx]
        text = str(row['text'])
        
        # 1. Clean ID
        raw_id = row.get('sample_id', row.get('parent_asin', idx))
        try: sid = str(int(float(raw_id)))
        except: sid = str(raw_id)
            
        # 2. FAST READ FROM LMDB
        img_bytes = self.txn.get(sid.encode('ascii'))
        
        if img_bytes:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            image = Image.new('RGB', (224, 224), color='black') # Fallback

        # 3. Text & Image Processing via SigLIP Processor
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN
        )

        cat_idx = CAT_TO_IDX.get(row['category'], 0)
        
        # 4. ENTAILMENT TARGET LOGIC (Binary Label)
        raw_score = float(row['return_likelihood'])
        label = 1 if raw_score >= RISK_LIMIT else 0
        target = torch.tensor(label, dtype=torch.long)

        # 5. SAFELY HANDLE ATTENTION MASK
        if 'attention_mask' in inputs:
            att_mask = inputs['attention_mask'].squeeze(0)
        else:
            input_shape = inputs['input_ids'].squeeze(0).shape
            att_mask = torch.ones(input_shape, dtype=torch.long)

        return {
            'sample_id': sid,
            'split': self.split_name,
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': att_mask,
            'category': torch.tensor(cat_idx, dtype=torch.long),
            'target': target
        }

# --- 3. THE MODEL (SigCLIP LoRA) ---
class ThesisSiglipClassifier(nn.Module):
    def __init__(self):
        super(ThesisSiglipClassifier, self).__init__()
        
        print("   Loading SigCLIP (google/siglip-base-patch16-224)...")
        base_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        
        self.backbone = get_peft_model(base_model, peft_config)
            
        vis_dim = 768
        txt_dim = 768
        cat_dim = 32
        
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, cat_dim)
        
        input_dim = vis_dim + txt_dim + cat_dim 
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 2) 
        )

    def forward(self, pixel_values, input_ids, attention_mask, categories):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values
        )
        
        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds
        cat_feat = self.cat_embed(categories)
        
        combined = torch.cat((img_embeds, txt_embeds, cat_feat), dim=1)
        return self.fusion(combined)

# --- 4. ANALYSIS ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 4C: Learning Curve (Weighted Loss 2x)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase4C_loss_curve.png"))
    plt.close()

    y_true = full_df['actual_label']
    y_pred = full_df['pred_label']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Safe', 'Risky'], yticklabels=['Safe', 'Risky'])
    plt.title('Phase 4C Confusion Matrix (Weighted 2x)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase4C_confusion_matrix.png"))
    plt.close()

    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    acc_train = accuracy_score(train_data['actual_label'], train_data['pred_label'])
    acc_test = accuracy_score(test_data['actual_label'], test_data['pred_label'])
    report = classification_report(test_data['actual_label'], test_data['pred_label'])
    
    stats_text = (
        f"Phase 4C (Weighted Loss 2x) Results:\n"
        f"------------------------------------\n"
        f"Accuracy (Train): {acc_train:.4f}\n"
        f"Accuracy (Test):  {acc_test:.4f}\n"
        f"\nClassification Report (Test):\n{report}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase4C_metrics.txt"), "w") as f:
        f.write(stats_text)

    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase4C_predictions_full.csv"), index=False)
    print(f"📋 Full predictions saved to: Phase4C_predictions_full.csv")

# --- 5. MAIN TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Phase 4C (Weighted Loss 2x) on {device}...")
    
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found.")
        return

    df = pd.read_csv(MANIFEST_FILE)
    
    # 1. Split & Filter
    train_df_full, val_df_full = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
    
    def filter_middle(d):
        return d[(d['return_likelihood'] <= SAFE_LIMIT) | (d['return_likelihood'] >= RISK_LIMIT)].copy()

    train_df = filter_middle(train_df_full)
    val_df = filter_middle(val_df_full)
    print(f"   📚 Data Loaded (Middle Dropped) -> Train: {len(train_df):,} | Test: {len(val_df):,}")
    
    try:
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    except:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    train_loader = DataLoader(ThesisEntailmentDataset(train_df, processor, 'train', LMDB_PATH), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ThesisEntailmentDataset(val_df, processor, 'test', LMDB_PATH), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ThesisSiglipClassifier().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # --- ⚖️ WEIGHTED LOSS CHANGED HERE ---
    # Safe (0) = 1.0
    # Risky (1) = 2.0 (The Balanced Approach)
    class_weights = torch.tensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {'train_loss': [], 'val_loss': []}

    print("\n⚡ Training Started...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            cats = batch['category'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(pixel_values, input_ids, mask, cats)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cats = batch['category'].to(device)
                target = batch['target'].to(device)
                
                output = model(pixel_values, input_ids, mask, cats)
                loss = criterion(output, target)
                v_loss += loss.item()

        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)

        print(f"   ✅ Ep {epoch+1}: Train Loss {avg_t_loss:.4f} | Test Loss {avg_v_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase4C_model_epoch_{epoch+1}.pth"))

    print(f"\n⏱️ Training finished in {(time.time()-start_time)/60:.1f} minutes.")

    # --- FINAL PREDICTIONS ---
    print("\n🔮 Generating Full Predictions...")
    model.eval()
    all_results = []
    
    def scan(loader, split_name):
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Scanning {split_name}"):
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cats = batch['category'].to(device)
                target = batch['target'].to(device)
                
                output = model(pixel_values, input_ids, mask, cats)
                preds = torch.argmax(output, dim=1).cpu().numpy().flatten()
                actuals = target.cpu().numpy().flatten()
                
                for i in range(len(preds)):
                    all_results.append({
                        'sample_id': batch['sample_id'][i],
                        'split': batch['split'][i],
                        'actual_label': actuals[i],
                        'pred_label': preds[i]
                    })

    scan(val_loader, "Test")
    scan(DataLoader(ThesisEntailmentDataset(train_df, processor, 'train', LMDB_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True), "Train")

    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    print(f"\n🎉 Phase 4C Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()