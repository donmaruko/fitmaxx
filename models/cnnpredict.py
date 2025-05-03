# cnnpredict.py
import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from langtest import extract_body_ratios
import torch.nn as nn

# -----------------------------
# CLI Args
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python cnnpredict.py <image_path> <style>")
    sys.exit(1)

image_path = sys.argv[1]
input_style = sys.argv[2].lower()
valid_styles = ['formal', 'casual', 'athleisure']
if input_style not in valid_styles:
    print(f"❌ Invalid style. Choose from: {valid_styles}")
    sys.exit(1)

# -----------------------------
# Device + CLIP Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# MLP Model Definition (Same as training)
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

# -----------------------------
# Feature Extraction (CLIP + Pose)
# -----------------------------
def extract_features(image_path):
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_emb = clip_model.encode_image(image).cpu().squeeze().numpy()

    torso_leg_ratio, limb_symmetry = extract_body_ratios(image_path)
    if torso_leg_ratio is None or limb_symmetry is None:
        raise ValueError("Pose detection failed.")

    torso_leg_ratio = min(max(torso_leg_ratio / 3.0, 0), 1)
    limb_symmetry = min(max(limb_symmetry / 2.0, 0), 1)

    return np.concatenate((clip_emb, [torso_leg_ratio, limb_symmetry]))

# -----------------------------
# Load MLP Model and Predict
# -----------------------------
try:
    features = extract_features(image_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    input_dim = features.shape[0]
    model = OutfitMLP(input_dim).to(device)

    model_path = f"fit_mlp_{input_style}.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        score = model(features_tensor).item()
except Exception as e:
    print(f"❌ Failed to process image: {e}")
    sys.exit(1)

# -----------------------------
# Output Drip Score
# -----------------------------
score_percent = round(score * 100, 2)
print(f"\n🖼️  Analyzing: {image_path}")
print(f"🎯 Style: {input_style}")
print(f"💧 Drip Score: {score_percent} / 100")

if score_percent >= 75:
    print("🔥 Fit is elite. Certified drip!")
elif score_percent >= 50:
    print("🤝 Mid-tier outfit. Balanced but could use more styling.")
else:
    print("🧢 Needs work. Focus on proportions, layering, or accessories.")
