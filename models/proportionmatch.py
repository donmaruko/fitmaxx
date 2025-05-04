# proportionmatch_boosted.py
"""
Final boosted proportion scorer (batch capable).
- Rewards fits that land inside ideal proportion zones.
- Penalizes fits that miss proportion standards badly.
- Softens base score using distance.
- Supports single image or folder.

Requires:
    pip install pandas numpy scikit-learn mediapipe pillow opencv-python
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import cv2

from proportion1 import ProportionScorer

# --- Feature extraction ---
def extract_features_from_image(img_path: Path) -> np.ndarray:
    scorer = ProportionScorer()
    img = Image.open(img_path).convert("RGB")
    lm = scorer.extract_landmarks(img)
    if lm is None:
        return None

    body = scorer.compute_body_metrics(lm)
    brk = scorer.detect_clothing_breakpoints(np.array(img), lm)
    vis = scorer.compute_visible_ratios(brk, lm)

    return np.array([
        body["torso_leg_ratio"],
        body["arm_symmetry_diff"],
        body["leg_symmetry_diff"],
        vis["vis_leg_torso_ratio"],
        vis["sleeve_sym"],
        vis["hem_sym"],
    ])

# --- Boosted scoring ---
def boosted_score(name: str, new_features: np.ndarray, good_feats: np.ndarray):
    dists = euclidean_distances([new_features], good_feats)
    avg_dist = dists.mean()

    base_score = max(20, 100 - (avg_dist ** 0.45) * 100)

    torso_leg_ratio = new_features[0]
    arm_sym_diff = new_features[1]
    leg_sym_diff = new_features[2]
    vis_leg_torso_ratio = new_features[3]
    sleeve_sym = new_features[4]
    hem_sym = new_features[5]

    boost = 0
    penalty = 0

    # Boosts
    if 0.6 <= torso_leg_ratio <= 0.75:
        boost += 6
    if 1.25 <= vis_leg_torso_ratio <= 1.7:
        boost += 6
    if arm_sym_diff < 0.08:
        boost += 3
    if leg_sym_diff < 0.08:
        boost += 3
    if sleeve_sym < 0.4:
        boost += 3
    if hem_sym < 0.1:
        boost += 3
    if boost == 24:
        boost += 3  # full bonus for perfect boosts

    # Penalties
    if torso_leg_ratio > 0.78:
        penalty += 5.1
    if torso_leg_ratio > 0.81:
        penalty += 10
    if vis_leg_torso_ratio < 1.35:
        penalty += 5.1

    final_score = base_score + boost - penalty
    final_score = max(0, min(100, final_score))

    scaled_score = round(final_score * 0.7)

    # Rating interpretation
    if scaled_score >= 35:
        print("✅ Proportions are balanced")
    else:
        print("❌ Proportions are unbalanced")
    
# --- Main CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosted proportion match scorer")
    parser.add_argument("--image", type=Path, help="Path to a single image")
    parser.add_argument("--folder", type=Path, help="Path to a folder of images")
    parser.add_argument("--csv", type=Path, required=True, help="CSV file with good proportion data")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    good_features = df[[ 
        "torso_leg_ratio", "arm_symmetry_diff", "leg_symmetry_diff",
        "vis_leg_torso_ratio", "sleeve_sym", "hem_sym"
    ]].values

    if args.image:
        new_features = extract_features_from_image(args.image)
        if new_features is None:
            print(f"❌ Could not process {args.image.name}: no full body detected.")
        else:
            boosted_score(args.image.name, new_features, good_features)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"❌ Folder not found: {args.folder}")
        else:
            for img_path in folder.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                new_features = extract_features_from_image(img_path)
                if new_features is None:
                    print(f"❌ Could not process {img_path.name}: no full body detected.")
                    continue
                boosted_score(img_path.name, new_features, good_features)
    else:
        print("❌ Please provide either --image or --folder")