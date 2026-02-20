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

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 64
TEST_SPLIT = 0.20

# Paths
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
MANIFEST_FILE = os.path.join(BASE_DIR, "Data", "FINAL_THESIS_GROUND_TRUTH.csv") 

# Output Dirs (New Folder for Phase 1B)
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Phase1B_CategoryAware")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# --- 2. DATASET CLASS (Updated for Categories) ---
class ThesisDataset(Dataset):
    def __init__(self, df, tokenizer, transform, split_name):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.split_name = split_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        img_path = row['image_path']
        cat_name = row['category']
        
        # ID
        if 'sample_id' in row: sample_id = str(row['sample_id'])
        elif 'parent_asin' in row: sample_id = str(row['parent_asin'])
        else: sample_id = str(idx)

        # Image
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except (UnidentifiedImageError, OSError, Exception):
            image = torch.zeros((3, 224, 224))

        # Text
        encoding = self.tokenizer.encode_plus(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )

        # Category (One-Hot Index)
        cat_idx = CAT_TO_IDX.get(cat_name, 0) # Default to 0 if unknown
        cat_tensor = torch.tensor(cat_idx, dtype=torch.long)

        # Target
        target = torch.tensor([float(row['return_likelihood'])], dtype=torch.float)

        return {
            'sample_id': sample_id,
            'split': self.split_name,
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category': cat_tensor,  # <--- NEW INPUT
            'target': target
        }

# --- 3. THE MODEL (Now with 3 Inputs) ---
class ThesisCategoryNet(nn.Module):
    def __init__(self):
        super(ThesisCategoryNet, self).__init__()
        
        # 1. Vision Leg
        self.visual = models.resnet18(pretrained=True)
        self.visual.fc = nn.Identity() # Output: 512
        
        # 2. Text Leg
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Output: 768
        
        # 3. Category Leg (Embedding)
        self.cat_embed = nn.Embedding(NUM_CATEGORIES, 32) 
        # Output: 32
        
        # Fusion: 512 + 768 + 32 = 1312 Inputs
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask, categories):
        # Vision
        vis_feat = self.visual(images)
        
        # Text
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]
        
        # Category
        cat_feat = self.cat_embed(categories)
        
        # Concatenate All Three
        combined = torch.cat((vis_feat, txt_feat, cat_feat), dim=1)
        
        return self.sigmoid(self.fusion(combined))

# --- 4. ANALYSIS (Standard) ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 1B: Learning Curve (Category Aware)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1B_loss_curve.png"))
    plt.close()

    # 2. Residual Plot 
    full_df['residual'] = full_df['pred_score'] - full_df['actual_score']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x='actual_score', y='residual', hue='split', alpha=0.2, palette={'train': 'blue', 'test': 'red'})
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Phase 1B Residuals: Category Aware Errors')
    plt.xlabel('Actual Return Risk')
    plt.ylabel('Error (Predicted - Actual)')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1B_residuals.png"))
    plt.close()

    # 3. Metrics
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 1B (Category Aware) Results:\n"
        f"----------------------------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase1B_metrics.txt"), "w") as f:
        f.write(stats_text)

    # 4. Save CSV
    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase1B_predictions_full.csv"), index=False)
    print(f"📋 Full predictions saved to: Phase1B_predictions_full.csv")

# --- 5. MAIN TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"🚀 Starting Phase 1B (Category Aware) on {device}...")
    
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found.")
        return

    df = pd.read_csv(MANIFEST_FILE)
    
    # Validation: Ensure categories match
    unknown_cats = set(df['category'].unique()) - set(CATEGORY_LIST)
    if unknown_cats:
        print(f"⚠️ Warning: Found unknown categories in data: {unknown_cats}")
        print("   They will be mapped to index 0 (Fashion).")

    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
    print(f"   📚 Train: {len(train_df):,} | Test: {len(val_df):,}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Loaders
    train_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ThesisDataset(val_df, tokenizer, transform, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = ThesisCategoryNet().to(device)
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
                    # Pass category to model
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
        
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase1B_model_epoch_{epoch+1}.pth"))

    print(f"\n⏱️ Training finished in {(time.time()-start_time)/60:.1f} minutes.")

    # --- FINAL PREDICTIONS ---
    print("\n🔮 Generating Full Predictions...")
    model.eval()
    all_results = []
    
    # Eval Train Loader (No Shuffle)
    train_eval_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=False)

    def predict_loader(loader):
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                img = batch['image'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                cats = batch['category'].to(device)
                target = batch['target'].to(device)
                
                output = model(img, ids, mask, cats)
                
                preds = output.cpu().numpy().flatten()
                actuals = target.cpu().numpy().flatten()
                ids_list = batch['sample_id']
                split_list = batch['split']
                
                for i in range(len(preds)):
                    all_results.append({
                        'sample_id': ids_list[i],
                        'split': split_list[i],
                        'actual_score': actuals[i],
                        'pred_score': preds[i]
                    })

    print("   ...Scanning Test Set...")
    predict_loader(val_loader)
    print("   ...Scanning Train Set...")
    predict_loader(train_eval_loader)

    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    print(f"\n🎉 Phase 1B Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()