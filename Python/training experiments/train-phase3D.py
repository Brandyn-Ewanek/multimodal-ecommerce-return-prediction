import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
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

# Output Dirs (Phase 3D - Gated Forensic CLIP)
RESULTS_DIR = "Results_Phase3D_CLIP_Gated"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-6    # Low LR
MAX_LEN = 77
TEST_SPLIT = 0.20

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
class ThesisCLIPDataset(Dataset): 
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

        # 3. Text & Image Processing via Hugging Face Processor
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN
        )

        cat_idx = CAT_TO_IDX.get(row['category'], 0)
        target = torch.tensor([float(row['return_likelihood'])], dtype=torch.float)

        # 4. Safely Handle Attention Mask
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

# --- 3. THE MODEL (Gated Forensic CLIP) ---
class ThesisGatedCLIPNet(nn.Module):
    def __init__(self):
        super(ThesisGatedCLIPNet, self).__init__()
        
        print("   Loading CLIP Model (openai/clip-vit-base-patch32)...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # 🔥 UNFREEZE
        for param in self.clip.parameters():
            param.requires_grad = True
            
        # Dimensions
        vis_dim = 512
        txt_dim = 512
        cat_dim = 32
        
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, cat_dim)
        
        # Inputs: Img(512) + Txt(512) + Cat(32) + Cos(1) + Euc(1) = 1058
        input_dim = vis_dim + txt_dim + cat_dim + 2
        hidden_dim = 512
        
        # --- GATING MECHANISM (The "Smart Filter") ---
        # Path A: Signal
        self.signal_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Path B: Gate (Sigmoid)
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid() 
        )
        
        # Final Regressor
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, pixel_values, input_ids, attention_mask, categories):
        # 1. Extract Features
        outputs = self.clip(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values
        )
        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds
        
        # 2. Geometric Metrics
        img_norm = F.normalize(img_embeds, p=2, dim=1)
        txt_norm = F.normalize(txt_embeds, p=2, dim=1)
        
        cosine_sim = torch.sum(img_norm * txt_norm, dim=1).unsqueeze(1)
        euclidean_dist = torch.norm(img_embeds - txt_embeds, p=2, dim=1, keepdim=True)
            
        # 3. Category
        cat_feat = self.cat_embed(categories)
        
        # 4. Concatenate
        combined = torch.cat((img_embeds, txt_embeds, cat_feat, cosine_sim, euclidean_dist), dim=1)
        
        # 5. GATED FUSION
        signal = self.signal_layer(combined)
        gate = self.gate_layer(combined)
        
        gated_features = signal * gate
        
        return self.sigmoid(self.regressor(gated_features))

# --- 4. ANALYSIS ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 3D: Learning Curve (Gated Forensic CLIP)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase3D_loss_curve.png"))
    plt.close()

    # 2. Residual Plot
    full_df['residual'] = full_df['pred_score'] - full_df['actual_score']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x='actual_score', y='residual', hue='split', alpha=0.2, palette={'train': 'blue', 'test': 'red'})
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Phase 3D Residuals: Gated CLIP Errors')
    plt.xlabel('Actual Return Risk')
    plt.ylabel('Error (Predicted - Actual)')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase3D_residuals.png"))
    plt.close()

    # 3. Metrics
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 3D (Gated Forensic CLIP) Results:\n"
        f"-------------------------------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase3D_metrics.txt"), "w") as f:
        f.write(stats_text)

    # 4. Save CSV
    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase3D_predictions_full.csv"), index=False)
    print(f"📋 Full predictions saved to: Phase3D_predictions_full.csv")

# --- 5. MAIN TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"🚀 Starting Phase 3D (Gated Forensic CLIP) on {device}...")
    
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found.")
        return

    df = pd.read_csv(MANIFEST_FILE)
    
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
    print(f"   📚 Train: {len(train_df):,} | Test: {len(val_df):,}")

    print("   Initializing CLIP Processor...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_loader = DataLoader(ThesisCLIPDataset(train_df, processor, 'train', LMDB_PATH), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ThesisCLIPDataset(val_df, processor, 'test', LMDB_PATH), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ThesisGatedCLIPNet().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

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
            if scaler:
                with torch.amp.autocast('cuda'):
                    output = model(pixel_values, input_ids, mask, cats)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase3D_model_epoch_{epoch+1}.pth"))

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
                preds = output.cpu().numpy().flatten()
                actuals = target.cpu().numpy().flatten()
                
                for i in range(len(preds)):
                    all_results.append({
                        'sample_id': batch['sample_id'][i],
                        'split': batch['split'][i],
                        'actual_score': actuals[i],
                        'pred_score': preds[i]
                    })

    scan(val_loader, "Test")
    scan(DataLoader(ThesisCLIPDataset(train_df, processor, 'train', LMDB_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=8), "Train")

    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    print(f"\n🎉 Phase 3D Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()