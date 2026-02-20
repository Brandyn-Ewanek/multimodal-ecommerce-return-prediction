import os
import pandas as pd
import torch
import torch.nn as nn
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

# --- ⚙️ CONFIGURATION ---
if os.path.exists('/data'):
    BASE_DIR = '/data'
    print("☁️ AWS ENVIRONMENT DETECTED: Using /data")
MANIFEST_FILE = os.path.join(BASE_DIR, "FINAL_THESIS_GROUND_TRUTH.csv") 
LMDB_PATH = "/opt/dlami/nvme/thesis_images.lmdb"

# Output Dirs (Phase 1D - 9 Epochs)
RESULTS_DIR = "Results_Phase1D_Fashion_9Epochs"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 9               # INCREASED from 3 to 9
LEARNING_RATE = 1e-5     # SLOWER (was 2e-5)
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

# --- 1. DATASET CLASS ---
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
            image = torch.zeros((3, 224, 224)) # Fallback

        # 3. Text Processing
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

# --- 2. THE MODEL (Fashion Specialist) ---
class ThesisFashionNet(nn.Module):
    def __init__(self):
        super(ThesisFashionNet, self).__init__()
        
        self.visual = models.resnet18(pretrained=True)
        self.visual.fc = nn.Identity() 
        
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask):
        vis_feat = self.visual(images)
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]
        combined = torch.cat((vis_feat, txt_feat), dim=1)
        return self.sigmoid(self.fusion(combined))

# --- 3. ANALYSIS ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 1D (9 Epochs): Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1D_9ep_loss_curve.png"))
    plt.close()

    # Metrics
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 1D (9 Epochs - Fashion Only) Results:\n"
        f"-------------------------------------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase1D_9ep_metrics.txt"), "w") as f:
        f.write(stats_text)

    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase1D_9ep_predictions.csv"), index=False)

# --- 4. MAIN TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"🚀 Starting Phase 1D (9 Epochs) on {device}...")
    
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found.")
        return

    full_df = pd.read_csv(MANIFEST_FILE)
    
    # FILTER: Only Keep Amazon Fashion
    df = full_df[full_df['category'] == 'raw_review_Amazon_Fashion'].copy()
    print(f"   ✂️  Fashion Items: {len(df):,}")
    
    if len(df) == 0:
        print("❌ Error: No Fashion items found!")
        return

    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train', LMDB_PATH), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(ThesisDataset(val_df, tokenizer, transform, 'test', LMDB_PATH), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ThesisFashionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}

    print("\n⚡ Training Started (9 Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            img = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    output = model(img, ids, mask)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(img, ids, mask)
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
                target = batch['target'].to(device)
                output = model(img, ids, mask)
                loss = criterion(output, target)
                v_loss += loss.item()

        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)

        print(f"   ✅ Ep {epoch+1}: Train Loss {avg_t_loss:.4f} | Test Loss {avg_v_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase1D_9ep_epoch_{epoch+1}.pth"))

    # Predictions
    print("\n🔮 Generating Predictions...")
    model.eval()
    all_results = []
    
    # Scanner
    def scan(loader, split):
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Scanning {split}"):
                img = batch['image'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                
                output = model(img, ids, mask)
                preds = output.cpu().numpy().flatten()
                actuals = batch['target'].cpu().numpy().flatten()
                
                for i in range(len(preds)):
                    all_results.append({
                        'sample_id': batch['sample_id'][i],
                        'split': batch['split'][i],
                        'actual_score': actuals[i],
                        'pred_score': preds[i]
                    })
    
    scan(val_loader, "Test")
    scan(DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=False), "Train")

    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    print(f"\n🎉 Phase 1D (9 Epochs) Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()