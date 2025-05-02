import clip
import torch
import os
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_path = "outfits"
label_map = {"high": 1, "low": 0}
X, y = [], []

for style in os.listdir(base_path):
    for tier in ["high", "low"]:
        tier_path = os.path.join(base_path, style, tier)
        if not os.path.isdir(tier_path):
            continue
        for fname in os.listdir(tier_path):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(tier_path, fname)
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy().flatten()
                X.append(embedding)
                y.append(label_map[tier])

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["low", "high"]))
