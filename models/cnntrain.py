# cnntrain.py – 3‑class Cross‑Entropy version
# -----------------------------------------------------------------------------
# Trains an MLP that outputs 3 logits (low / med / high) instead of a single
# regression value. Loss is CrossEntropy, and evaluation uses argmax.
# -----------------------------------------------------------------------------

import os, sys, warnings
from collections import Counter

import numpy as np
from PIL import Image

import torch, clip, torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from langtest import extract_body_ratios

# ----------------------------- CLI -----------------------------
if len(sys.argv) != 2:
    print("Usage: python cnntrain.py <style>")
    sys.exit(1)

STYLE = sys.argv[1].lower()
VALID = ["formal","casual","athleisure"]
if STYLE not in VALID:
    sys.exit(f"❌ Invalid style. Choose from: {VALID}")

# ----------------------------- Device + CLIP -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_prep = clip.load("ViT-B/32", device=DEVICE)

# ----------------------------- Feature extraction -----------------------------

def extract_embedding(path:str):
    img = clip_prep(Image.open(path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return clip_model.encode_image(img).cpu().squeeze().numpy()

def extract_features(path:str):
    clip_emb = extract_embedding(path)
    tlr, sym = extract_body_ratios(path)
    if tlr is None or sym is None:
        raise ValueError("Pose detection failed")
    tlr  = np.clip(tlr/3,0,1)
    sym  = np.clip(sym/2,0,1)
    return np.concatenate((clip_emb,[tlr*0.2,sym*0.2]))

# ----------------------------- Load dataset -----------------------------
BASE = "../outfits"; style_dir = os.path.join(BASE, STYLE)
X, y = [], []
label_map = {"low":0, "med":1, "high":2}

for lbl in ["high","med","low"]:
    folder = os.path.join(style_dir,lbl)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue
        path = os.path.join(folder,fname)
        try:
            X.append(extract_features(path))
            y.append(label_map[lbl])
        except Exception as e:
            warnings.warn(f"Skip {fname}: {e}")

X = np.asarray(X,dtype=np.float32)
Y = np.asarray(y,dtype=np.int64)

# ----------------------------- Train / test split -----------------------------
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=42)

# ----------------------------- Model -----------------------------
class OutfitMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,384), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(384,128), nn.ReLU(),
            nn.Linear(128,3)  # logits for 3 classes
        )
    def forward(self,x):
        return self.net(x)

model = OutfitMLP(X.shape[1]).to(DEVICE)

# ----------------------------- DataLoader -----------------------------
X_tr_t = torch.tensor(X_tr,device=DEVICE)
Y_tr_t = torch.tensor(Y_tr,device=DEVICE)
train_ds = TensorDataset(X_tr_t,Y_tr_t)

class_counts = Counter(Y_tr)
weights = torch.tensor([1.0/class_counts[c] for c in Y_tr],dtype=torch.float32,device=DEVICE)

def make_sampler(labels):
    freq = Counter(labels)               # e.g. {'low': 210, 'med': 430, 'high': 790}
    weights = [1.0 / freq[y] for y in labels]
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

loader = DataLoader(train_ds,batch_size=32,sampler=WeightedRandomSampler(weights,len(weights)))

# ----------------------------- Optim & loss -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = optim.Adam(model.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)

# ----------------------------- Train -----------------------------
print("🧠 Training 3‑class MLP …")
best_acc=0; patience=10; no_improve=0
for epoch in range(100):
    model.train(); tot=0
    for xb,yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); optimizer.step(); tot+=loss.item()
    scheduler.step()

    # quick val
    model.eval();
    with torch.no_grad():
        preds = model(torch.tensor(X_te,device=DEVICE)).cpu().argmax(1).numpy()
    acc = accuracy_score(Y_te,preds)
    print(f"Epoch {epoch:02d}  loss {tot/len(loader):.4f}  val_acc {acc*100:.2f}%")
    if acc>best_acc:
        best_acc, no_improve = acc,0
        torch.save(model.state_dict(),f"fit_mlp_{STYLE}.pt")
    else:
        no_improve+=1
        if no_improve==patience:
            print("Early stop."); break

print("✅ Best val accuracy:",best_acc*100)

# ----------------------------- Final report -----------------------------
model.load_state_dict(torch.load(f"fit_mlp_{STYLE}.pt",map_location=DEVICE))
model.eval();
with torch.no_grad(): final_preds = model(torch.tensor(X_te,device=DEVICE)).cpu().argmax(1).numpy()
print("\n📊 Classification Report:")
print(classification_report(Y_te,final_preds,target_names=["low","med","high"]))
