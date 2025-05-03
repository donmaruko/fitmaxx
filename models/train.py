# train.py
import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from langtest import extract_body_ratios

# -----------------------------
# CLI Args
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python train.py <style>")
    sys.exit(1)

input_style = sys.argv[1].lower()
valid_styles = ['formal', 'casual', 'athleisure']
if input_style not in valid_styles:
    print(f"❌ Invalid style. Choose from: {valid_styles}")
    sys.exit(1)

# -----------------------------
# Setup CLIP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().squeeze().numpy()
    return embedding

def extract_features(image_path):
    clip_emb = extract_embedding(image_path)
    torso_leg_ratio, limb_symmetry = extract_body_ratios(image_path)
    if torso_leg_ratio is None or limb_symmetry is None:
        raise ValueError("Pose detection failed.")
    return np.concatenate((clip_emb, [torso_leg_ratio, limb_symmetry]))

# -----------------------------
# Build Dataset
# -----------------------------
base_dir = "../outfits"
X, y = [], []

style_dir = os.path.join(base_dir, input_style)
for label in ["high", "low"]:
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
            y.append(1 if label == "high" else 0)
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Define and Train MLP
# -----------------------------
import torch.nn as nn
import torch.optim as optim

class OutfitMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
mlp = OutfitMLP(input_dim).to(device)

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train, device=device).unsqueeze(1)

criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=1e-3)

print("🧠 Training MLP...")
for epoch in range(20):
    mlp.train()
    optimizer.zero_grad()
    outputs = mlp(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch} — Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
mlp.eval()
X_test_t = torch.tensor(X_test, device=device)
with torch.no_grad():
    preds = mlp(X_test_t).cpu().squeeze().numpy()

y_pred = (preds >= 0.5).astype(int)
print("\n📊 Evaluation Report:")
print(classification_report(y_test, y_pred))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

torch.save(mlp.state_dict(), f"fit_mlp_{input_style}.pt")
print(f"✅ Model weights saved to fit_mlp_{input_style}.pt")
