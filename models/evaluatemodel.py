import os
import sys
import torch
import clip
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from langtest import extract_body_ratios  # Reuse your existing pose feature code

# -----------------------------
# CLI Args
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python evaluatemodel.py <image_path> <style>")
    print("Example: python evaluatemodel.py myfit.jpg formal")
    sys.exit(1)

input_image = sys.argv[1]
input_style = sys.argv[2].lower()
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
    """Returns [clip_embedding..., torso_leg_ratio, limb_symmetry]"""
    clip_emb = extract_embedding(image_path)
    torso_leg_ratio, limb_symmetry = extract_body_ratios(image_path)
    if torso_leg_ratio is None or limb_symmetry is None:
        raise ValueError("Failed to extract pose ratios.")
    return np.concatenate((clip_emb, [torso_leg_ratio, limb_symmetry]))

# -----------------------------
# Training Phase
# -----------------------------
base_dir = "../outfits"
X, y = [], []

style_dir = os.path.join(base_dir, input_style)
if not os.path.exists(style_dir):
    print(f"❌ Style folder not found: {style_dir}")
    sys.exit(1)

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

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = clf.predict(X_test)
print("\n📊 Evaluation Report:")
print(classification_report(y_test, y_pred))
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

joblib.dump(clf, f"fit_classifier_{input_style}.pkl")
print(f"✅ Classifier saved to fit_classifier_{input_style}.pkl")

# -----------------------------
# Predict on Input Image
# -----------------------------
try:
    test_feat = extract_features(input_image)
except Exception as e:
    print(f"❌ Failed to process input image: {e}")
    sys.exit(1)

proba = clf.predict_proba([test_feat])[0][1]  # probability of "high"
score = round(proba * 100, 2)

print(f"\n🖼️  Analyzing: {input_image}")
print(f"🎯 Style: {input_style}")
print(f"💧 Drip Score: {score} / 100")
if score >= 75:
    print("🔥 Fit is elite. Certified drip!")
elif score >= 50:
    print("👌 Decent outfit. Consider tweaking silhouette or color harmony.")
else:
    print("🧢 Needs work. Focus on proportions, layering, or accessories.")
