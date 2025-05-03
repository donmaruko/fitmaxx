# predict.py
import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
from langtest import extract_body_ratios

if len(sys.argv) != 3:
    print("Usage: python predict.py <image_path> <style>")
    sys.exit(1)

input_image = sys.argv[1]
input_style = sys.argv[2].lower()
valid_styles = ['formal', 'casual', 'athleisure']
if input_style not in valid_styles:
    print(f"❌ Invalid style. Choose from: {valid_styles}")
    sys.exit(1)

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
# Define MLP (must match train.py)
# -----------------------------
import torch.nn as nn

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

try:
    test_feat = extract_features(input_image).astype(np.float32)
    input_dim = test_feat.shape[0]
    mlp = OutfitMLP(input_dim).to(device)
    mlp.load_state_dict(torch.load(f"fit_mlp_{input_style}.pt", map_location=device))
    mlp.eval()

    test_tensor = torch.tensor(test_feat, device=device).unsqueeze(0)
    with torch.no_grad():
        prob = mlp(test_tensor).item()
except Exception as e:
    print(f"❌ Failed to process: {e}")
    sys.exit(1)

score = round(prob * 100, 2)
print(f"\n🖼️  Analyzing: {input_image}")
print(f"🎯 Style: {input_style}")
print(f"💧 Drip Score: {score} / 100")

if score >= 75:
    print("🔥 Fit is elite. Certified drip!")
elif score >= 50:
    print("👌 Decent outfit. Consider tweaking silhouette or color harmony.")
else:
    print("🧢 Needs work. Focus on proportions, layering, or accessories.")
