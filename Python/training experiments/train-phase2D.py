import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import DistilBertTokenizer, DistilBertModel
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


# Output Dirs (Phase 2D - Euclidean)
RESULTS_DIR = "Results_Phase2D_Euclidean"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 64
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
class ThesisDataset(Dataset):
    def __init__(self, df, tokenizer, transform, split_name, lmdb_path):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
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
            image = self.transform(image)
        else:
            image = torch.zeros((3, 224, 224)) # Fallback tensor

        # 3. Text Processing (DistilBERT)
        encoding = self.tokenizer.encode_plus(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )

        cat_idx = CAT_TO_IDX.get(row['category'], 0)
        
        return {
            'sample_id': sid,
            'split': self.split_name,
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': torch.tensor(cat_idx, dtype=torch.long),
            'target': torch.tensor([float(row['return_likelihood'])], dtype=torch.float)
        }

# --- 3. THE MODEL (Phase 1B Base + Euclidean Distance) ---
class ThesisEuclideanNet(nn.Module):
    def __init__(self):
        super(ThesisEuclideanNet, self).__init__()
        
        # 1. Vision Leg
        self.visual = models.resnet18(pretrained=True)
        self.visual.fc = nn.Identity() # Output: 512
        
        # 2. Text Leg
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Output: 768
        
        # 3. Category Leg
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, 32) 
        # Output: 32

        # 4. ALIGNMENT LAYER
        # Shrink Text (768) -> 512 to match Image for distance calc
        self.text_align = nn.Linear(768, 512)
        
        # Fusion Input:
        # Image (512) + Text (768) + Category (32) + EuclideanDist (1) = 1313
        self.fusion = nn.Sequential(
            nn.Linear(1313, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask, categories):
        # A. Extract Core Features
        vis_feat = self.visual(images)   # [Batch, 512]
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :] # [Batch, 768]
        cat_feat = self.cat_embed(categories) # [Batch, 32]
        
        # B. Calculate Euclidean Distance
        # 1. Align Text to Image Space
        txt_aligned = self.text_align(txt_feat) # [Batch, 512]
        
        # 2. Calculate Distance (p=2 is Euclidean)
        # Result shape: [Batch] -> Unsqueeze to [Batch, 1]
        euclidean_dist = torch.norm(vis_feat - txt_aligned, p=2, dim=1, keepdim=True)
        
        # C. Fusion
        combined = torch.cat((vis_feat, txt_feat, cat_feat, euclidean_dist), dim=1)
        
        return self.sigmoid(self.fusion(combined))

# --- 4. ANALYSIS ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 2D: Learning Curve (Euclidean Only)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase2D_loss_curve.png"))
    plt.close()

    # 2. Residual Plot
    full_df['residual'] = full_df['pred_score'] - full_df['actual_score']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x='actual_score', y='residual', hue='split', alpha=0.2, palette={'train': 'blue', 'test': 'red'})
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Phase 2D Residuals: Euclidean Errors')
    plt.xlabel('Actual Return Risk')
    plt.ylabel('Error (Predicted - Actual)')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase2D_residuals.png"))
    plt.close()

    # 3. Metrics
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 2D (Euclidean Distance) Results:\n"
        f"--------------------------------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase2D_metrics.txt"), "w") as f:
        f.write(stats_text)

    # 4. Save CSV
    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase2D_predictions_full.csv"), index=False)
    print(f"📋 Full predictions saved to: Phase2D_predictions_full.csv")

# --- 5. MAIN TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"🚀 Starting Phase 2D (Euclidean Distance) on {device}...")
    
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found.")
        return

    df = pd.read_csv(MANIFEST_FILE)
    
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
    print(f"   📚 Train: {len(train_df):,} | Test: {len(val_df):,}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Loaders
    train_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train', LMDB_PATH), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ThesisDataset(val_df, tokenizer, transform, 'test', LMDB_PATH), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ThesisEuclideanNet().to(device)
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
            img = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            cats = batch['category'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    output = model(img, ids, mask, cats)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(img, ids, mask, cats)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            t_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cats = batch['category'].to(device)
                target = batch['target'].to(device)
                
                output = model(img, ids, mask, cats)
                loss = criterion(output, target)
                v_loss += loss.item()

        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)

        print(f"   ✅ Ep {epoch+1}: Train Loss {avg_t_loss:.4f} | Test Loss {avg_v_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase2D_model_epoch_{epoch+1}.pth"))

    print(f"\n⏱️ Training finished in {(time.time()-start_time)/60:.1f} minutes.")

    # --- FINAL PREDICTIONS ---
    print("\n🔮 Generating Full Predictions...")
    model.eval()
    all_results = []
    
    def scan(loader, split_name):
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Scanning {split_name}"):
                img = batch['image'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cats = batch['category'].to(device)
                target = batch['target'].to(device)
                
                output = model(img, ids, mask, cats)
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
    scan(DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train', LMDB_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=8), "Train")

    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    print(f"\n🎉 Phase 2D Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()