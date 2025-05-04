# proportion_ml_pipeline.py
"""
Command‑line helper for a hackathon‑friendly proportion‑rating pipeline.

1️⃣  Extract numeric proportion features into a CSV               
    python proportion_ml_pipeline.py extract --images_folder data/all --output_csv proportion_data.csv
    (CSV is created with an empty "label" column – quickly mark 1 for GOOD, 0 for BAD.)

2️⃣  Train a tiny Random‑Forest classifier on the labelled CSV     
    python proportion_ml_pipeline.py train --csv proportion_data.csv --model_out proportion_model.pkl

3️⃣  Predict on a brand‑new image                                  
    python proportion_ml_pipeline.py predict --model proportion_model.pkl --image new_photo.jpg

Requires:
    pip install mediapipe opencv-python pillow numpy scikit-learn joblib pandas
    (and your own `proportion_only.py` with the ProportionScorer class.)
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image

# import the proportion scorer you already built
from proportion1 import ProportionScorer  # make sure this is on PYTHONPATH

FEATURE_NAMES = [
    "torso_leg_ratio", "arm_symmetry_diff", "leg_symmetry_diff",
    "vis_leg_torso_ratio", "sleeve_sym", "hem_sym",
]


def extract_features(images_folder: Path, output_csv: Path):
    """Walk through images_folder, compute features, dump to CSV."""
    scorer = ProportionScorer()
    rows = []

    for img_path in images_folder.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"⚠️  {img_path.name}: {e}")
            continue

        lm = scorer.extract_landmarks(img)
        if lm is None:
            print(f"⏭️  {img_path.name}: no full body detected")
            continue

        body = scorer.compute_body_metrics(lm)
        brk = scorer.detect_clothing_breakpoints(np.array(img.convert("RGB")), lm)
        vis = scorer.compute_visible_ratios(brk, lm)

        rows.append({
            "img_name": str(img_path),
            **body,
            **vis,
            "label": ""  # fill 1 for good, 0 for bad
        })

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"✅ Wrote {len(rows)} rows → {output_csv}")


def train_model(csv_file: Path, model_out: Path):
    """Train RandomForest on the labelled CSV and save the model."""
    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column (0=bad,1=good)")

    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    joblib.dump(clf, model_out)
    print(f"✅ Model saved → {model_out} | validation accuracy: {acc:.2%}")


def predict_image(model_path: Path, image_path: Path):
    """Predict proportion quality for one image."""
    clf = joblib.load(model_path)
    scorer = ProportionScorer()

    img = Image.open(image_path)
    lm = scorer.extract_landmarks(img)
    if lm is None:
        print("❌ Could not detect a person in the image.")
        return

    body = scorer.compute_body_metrics(lm)
    brk = scorer.detect_clothing_breakpoints(np.array(img.convert("RGB")), lm)
    vis = scorer.compute_visible_ratios(brk, lm)

    feats = [
        body["torso_leg_ratio"], body["arm_symmetry_diff"], body["leg_symmetry_diff"],
        vis["vis_leg_torso_ratio"], vis["sleeve_sym"], vis["hem_sym"],
    ]

    pred = clf.predict([feats])[0]
    proba = clf.predict_proba([feats])[0][pred]

    label = "GOOD" if pred == 1 else "BAD"
    print(f"Prediction ➜ {label} proportions  (confidence {proba:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proportion ML pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    p_e = sub.add_parser("extract", help="Extract features from images → CSV")
    p_e.add_argument("--images_folder", type=Path, required=True)
    p_e.add_argument("--output_csv", type=Path, required=True)

    # train
    p_t = sub.add_parser("train", help="Train model from labelled CSV")
    p_t.add_argument("--csv", type=Path, required=True)
    p_t.add_argument("--model_out", type=Path, required=True)

    # predict
    p_p = sub.add_parser("predict", help="Predict proportion rating for one image")
    p_p.add_argument("--model", type=Path, required=True)
    p_p.add_argument("--image", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "extract":
        extract_features(args.images_folder, args.output_csv)
    elif args.cmd == "train":
        train_model(args.csv, args.model_out)
    elif args.cmd == "predict":
        predict_image(args.model, args.image)