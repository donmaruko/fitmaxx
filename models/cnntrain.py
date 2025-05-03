# cnntrain.py
import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from langtest import extract_body_ratios
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# -----------------------------
# CLI Args
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python cnntrain.py <style>")
    sys.exit(1)

input_style = sys.argv[1].lower()
valid_styles = ['formal', 'casual', 'athleisure']
if input_style not in valid_styles:
    print(f"❌ Invalid style. Choose from: {valid_styles}")
    sys.exit(1)

# -----------------------------
# Setup Device + CLIP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def extract_embedding(image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image).cpu().squeeze().numpy()
    return embedding

def extract_features(image_path):
    clip_emb = extract_embedding(image_path)
    torso_leg_ratio, limb_symmetry = extract_body_ratios(image_path)
    if torso_leg_ratio is None or limb_symmetry is None:
        raise ValueError("Pose detection failed.")

    torso_leg_ratio = min(max(torso_leg_ratio / 3.0, 0), 1)
    limb_symmetry = min(max(limb_symmetry / 2.0, 0), 1)

    return np.concatenate((clip_emb, [torso_leg_ratio, limb_symmetry]))

# -----------------------------
# Load Dataset
# -----------------------------
base_dir = "../outfits"
style_dir = os.path.join(base_dir, input_style)
X, y = [], []

for label in ["high", "med", "low"]:
    folder = os.path.join(style_dir, label)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder, fname)
        try:
            feat = extract_features(path)
            X.append(feat)
            if label == "high":
                y.append(1.0)
            elif label == "med":
                y.append(np.random.uniform(0.45, 0.55))  # mid-range fuzz
            else:
                y.append(0.0)
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# -----------------------------
# Split and Stratify
# -----------------------------
y_strat = np.round(y * 2).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_strat, random_state=42
)

# -----------------------------
# Define MLP Model
# -----------------------------
class OutfitMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
mlp = OutfitMLP(input_dim).to(device)

# -----------------------------
# Prepare Training
# -----------------------------
X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device).unsqueeze(1)
train_dataset = TensorDataset(X_train_t, y_train_t)

# Weighted Sampling for imbalance
y_bins = np.round(y_train * 2).astype(int)
class_counts = Counter(y_bins)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = torch.tensor([class_weights[b] for b in y_bins], dtype=torch.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

# Optim + Loss + LR scheduler
criterion = nn.SmoothL1Loss(beta=0.1)
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# -----------------------------
# Training Loop
# -----------------------------
print("🧠 Training MLP...")
for epoch in range(100):
    mlp.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = mlp(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
mlp.eval()
X_test_t = torch.tensor(X_test, device=device)
with torch.no_grad():
    preds = mlp(X_test_t).cpu().squeeze().numpy()

y_test_binned = np.round(y_test * 2).astype(int)
preds_binned = np.round(preds * 2).astype(int)

print("\n📊 Class Distribution (Test):", Counter(y_test_binned))
print("📊 Class Distribution (Predicted):", Counter(preds_binned))
print("\n📊 Evaluation Report:")
print(classification_report(y_test_binned, preds_binned, target_names=["low", "med", "high"]))
print(f"🎯 Accuracy: {accuracy_score(y_test_binned, preds_binned) * 100:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
torch.save(mlp.state_dict(), f"fit_mlp_{input_style}.pt")
print(f"✅ Model weights saved to fit_mlp_{input_style}.pt")
