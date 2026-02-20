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

# --- ⚙️ CONFIGURATION ---
BATCH_SIZE = 32       # Increased slightly due to AMP (Mixed Precision)
EPOCHS = 3            
LEARNING_RATE = 2e-5  
MAX_LEN = 64
TEST_SPLIT = 0.20     

# Paths
BASE_DIR = r"C:\Users\maxx9\Desktop\Data Science\14. IU\03. ImageText Product"
MANIFEST_FILE = os.path.join(BASE_DIR, "Data", "FINAL_THESIS_GROUND_TRUTH.csv") 

# Output Dirs
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Phase1A_Baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. DATASET CLASS ---
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
        
        # Safe ID Loading
        if 'sample_id' in row: sample_id = str(row['sample_id'])
        elif 'parent_asin' in row: sample_id = str(row['parent_asin'])
        else: sample_id = str(idx)

        # Image Loading with Safety Guard
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except (UnidentifiedImageError, OSError, Exception):
            # Return a black square if image is corrupt
            image = torch.zeros((3, 224, 224))

        # TARGET: Single Value (Return Likelihood)
        # We wrap it in a list so it becomes a tensor of shape [1]
        target = torch.tensor([float(row['return_likelihood'])], dtype=torch.float)

        encoding = self.tokenizer.encode_plus(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )

        return {
            'sample_id': sample_id,
            'split': self.split_name,
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': target
        }

# --- 2. THE MODEL (Single Head) ---
class ThesisBaselineNet(nn.Module):
    def __init__(self):
        super(ThesisBaselineNet, self).__init__()
        # Vision: ResNet18
        self.visual = models.resnet18(pretrained=True)
        self.visual.fc = nn.Identity() # Removes the classification layer (Output: 512 dims)
        
        # Text: DistilBERT
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Fusion: Concat(512 + 768) -> Hidden -> Output(1)
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1) # <--- CHANGED TO 1 OUTPUT
        )
        self.sigmoid = nn.Sigmoid() # Force output between 0 and 1

    def forward(self, images, input_ids, attention_mask):
        vis_feat = self.visual(images)
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :] # CLS token
        
        combined = torch.cat((vis_feat, txt_feat), dim=1)
        return self.sigmoid(self.fusion(combined))

# --- 3. ANALYSIS & PLOTTING ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 1A: Learning Curve (MSE Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1A_loss_curve.png"))
    plt.close()

    # 2. Residual Plot (The "Truth" Check)
    full_df['residual'] = full_df['pred_score'] - full_df['actual_score']
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x='actual_score', y='residual', hue='split', alpha=0.2, palette={'train': 'blue', 'test': 'red'})
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Residual Analysis: Prediction Error')
    plt.xlabel('Actual Return Risk')
    plt.ylabel('Error (Predicted - Actual)')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1A_residuals.png"))
    plt.close()

    # 3. Metrics
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 1A Results:\n"
        f"-----------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase1A_metrics.txt"), "w") as f:
        f.write(stats_text)

    # 4. Save CSV
    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase1A_predictions_full.csv"), index=False)
    print(f"📋 Full predictions saved to: Phase1A_predictions_full.csv")

# --- 4. MAIN LOOP ---
def run_training():
    # Check Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None # Mixed Precision
    
    print(f"🚀 Starting Phase 1A on {device}...")
    if scaler: print("   ⚡ Mixed Precision (AMP) Enabled for speed.")
    
    # Load Manifest
    if not os.path.exists(MANIFEST_FILE):
        print(f"❌ Error: Manifest not found at {MANIFEST_FILE}")
        return

    df = pd.read_csv(MANIFEST_FILE)
    
    if 'return_likelihood' not in df.columns:
        print("❌ CRITICAL: 'return_likelihood' column missing from manifest!")
        return

    # Split
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)
    print(f"   📚 Train: {len(train_df):,} | Test: {len(val_df):,}")

    # Setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Data Loaders
    train_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ThesisDataset(val_df, tokenizer, transform, 'test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model & Optimization
    model = ThesisBaselineNet().to(device)
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
            target = batch['target'].to(device)

            optimizer.zero_grad()
            
            # AMP Forward Pass
            if scaler:
                with torch.cuda.amp.autocast():
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
                
                # Validation doesn't need scaling
                output = model(img, ids, mask)
                loss = criterion(output, target)
                v_loss += loss.item()

        # Logging
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)

        print(f"   ✅ Ep {epoch+1}: Train Loss {avg_t_loss:.4f} | Test Loss {avg_v_loss:.4f}")
        
        # --- 🛡️ SAFETY SAVE (Epoch End) ---
        save_path = os.path.join(RESULTS_DIR, f"Phase1A_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"   💾 Checkpoint Saved: {os.path.basename(save_path)}")

    print(f"\n⏱️ Training finished in {(time.time()-start_time)/60:.1f} minutes.")

    # --- FINAL PREDICTIONS ---
    print("\n🔮 Generating Full Predictions...")
    model.eval()
    all_results = []
    
    # We use a new loader for Train Eval to disable Shuffle
    train_eval_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=False)

    def predict_loader(loader):
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                img = batch['image'].to(device)
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                target = batch['target'].to(device)
                
                output = model(img, ids, mask)
                
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

    # Convert to DF and Analyze
    final_df = pd.DataFrame(all_results)
    analyze_and_save(history, final_df)
    
    print(f"\n🎉 Phase 1A Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()