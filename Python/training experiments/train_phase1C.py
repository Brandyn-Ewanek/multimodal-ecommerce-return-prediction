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

# Output Dirs (Phase 1C)
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "Phase1C_OneHot")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 1. CATEGORY LIST ---
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

# --- 2. DATASET CLASS (One-Hot Version) ---
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
        
        # Image
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224))

        # Text
        encoding = self.tokenizer.encode_plus(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )

        # ONE-HOT ENCODING
        one_hot = torch.zeros(NUM_CATEGORIES)
        cat_idx = CAT_TO_IDX.get(cat_name, 0)
        one_hot[cat_idx] = 1.0

        # Target
        target = torch.tensor([float(row['return_likelihood'])], dtype=torch.float)

        return {
            'sample_id': str(row.get('sample_id', idx)),
            'split': self.split_name,
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'category_onehot': one_hot,
            'target': target
        }

# --- 3. THE MODEL (One-Hot Fusion) ---
class ThesisOneHotNet(nn.Module):
    def __init__(self):
        super(ThesisOneHotNet, self).__init__()
        self.visual = models.resnet18(pretrained=True)
        self.visual.fc = nn.Identity() 
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Fusion: 512 (Vis) + 768 (Text) + 9 (One-Hot) = 1289
        self.fusion = nn.Sequential(
            nn.Linear(512 + 768 + NUM_CATEGORIES, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, input_ids, attention_mask, one_hot_cats):
        vis_feat = self.visual(images)
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state[:, 0, :]
        
        # Concatenate Image, Text, and the raw One-Hot Vector
        combined = torch.cat((vis_feat, txt_feat, one_hot_cats), dim=1)
        return self.sigmoid(self.fusion(combined))

# --- 4. CONSISTENT ANALYTICS ---
def analyze_and_save(history, full_df):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--s', label='Val Loss')
    plt.title('Phase 1C: Learning Curve (One-Hot Categories)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1C_loss_curve.png"))
    plt.close()

    # 2. Residual Plot 
    full_df['residual'] = full_df['pred_score'] - full_df['actual_score']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_df, x='actual_score', y='residual', hue='split', alpha=0.2, palette={'train': 'blue', 'test': 'red'})
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.title('Phase 1C Residuals: One-Hot Error Analysis')
    plt.xlabel('Actual Return Risk')
    plt.ylabel('Error (Predicted - Actual)')
    plt.savefig(os.path.join(RESULTS_DIR, "Phase1C_residuals.png"))
    plt.close()

    # 3. Metrics Text
    train_data = full_df[full_df['split'] == 'train']
    test_data = full_df[full_df['split'] == 'test']
    r2_train = r2_score(train_data['actual_score'], train_data['pred_score'])
    r2_test = r2_score(test_data['actual_score'], test_data['pred_score'])
    mae_test = mean_absolute_error(test_data['actual_score'], test_data['pred_score'])
    
    stats_text = (
        f"Phase 1C (One-Hot) Results:\n"
        f"---------------------------\n"
        f"R2 Score (Train): {r2_train:.4f}\n"
        f"R2 Score (Test):  {r2_test:.4f}\n"
        f"MAE (Test):       {mae_test:.4f}\n"
    )
    print("\n" + stats_text)
    with open(os.path.join(RESULTS_DIR, "Phase1C_metrics.txt"), "w") as f:
        f.write(stats_text)

    full_df.to_csv(os.path.join(RESULTS_DIR, "Phase1C_predictions_full.csv"), index=False)

# --- 5. TRAINING LOOP ---
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"🚀 Starting Phase 1C (One-Hot) on {device}...")
    df = pd.read_csv(MANIFEST_FILE)
    train_df, val_df = train_test_split(df, test_size=TEST_SPLIT, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ThesisDataset(val_df, tokenizer, transform, 'test'), batch_size=BATCH_SIZE, shuffle=False)
    
    model = ThesisOneHotNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        for batch in loop:
            img, ids, mask = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device)
            cats, target = batch['category_onehot'].to(device), batch['target'].to(device)

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
                output = model(batch['image'].to(device), batch['input_ids'].to(device), 
                               batch['attention_mask'].to(device), batch['category_onehot'].to(device))
                v_loss += criterion(output, batch['target'].to(device)).item()

        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        print(f"   ✅ Ep {epoch+1}: Train Loss {history['train_loss'][-1]:.4f} | Test Loss {history['val_loss'][-1]:.4f}")
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"Phase1C_model_epoch_{epoch+1}.pth"))

    # Final Inference
    model.eval()
    all_results = []
    train_eval_loader = DataLoader(ThesisDataset(train_df, tokenizer, transform, 'train'), batch_size=BATCH_SIZE, shuffle=False)
    
    def predict_loader(loader):
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                output = model(batch['image'].to(device), batch['input_ids'].to(device), 
                               batch['attention_mask'].to(device), batch['category_onehot'].to(device))
                preds, actuals = output.cpu().numpy().flatten(), batch['target'].cpu().numpy().flatten()
                for i in range(len(preds)):
                    all_results.append({'sample_id': batch['sample_id'][i], 'split': batch['split'][i], 
                                        'actual_score': actuals[i], 'pred_score': preds[i]})

    predict_loader(val_loader)
    predict_loader(train_eval_loader)
    analyze_and_save(history, pd.DataFrame(all_results))
    print(f"\n🎉 Phase 1C Complete! Results in: {RESULTS_DIR}")

if __name__ == "__main__":
    run_training()