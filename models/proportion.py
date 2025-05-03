# proportion.py  ───────────────────────────────────────────────────────────────
import sys
from pathlib import Path
from typing import Optional, Dict

import mediapipe as mp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ── Mediapipe pose setup ──────────────────────────────────────────────────────
mp_pose = mp.solutions.pose

# ── (Optional) device setup for future torch use ─────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Placeholder preprocess (unused in this snippet) ──────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ──────────────────────────────────────────────────────────────────────────────
#  Proportion / silhouette scorer  (identical to your original, just moved)
# ──────────────────────────────────────────────────────────────────────────────
class ProportionScorer:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True)

    # --------------------------------------------------------------------- #
    def extract_landmarks(self, image: Image.Image) -> Optional[Dict[str, np.ndarray]]:
        image_np = np.array(image.convert("RGB"))
        results = self.pose.process(image_np)
        if not results.pose_landmarks:
            return None
        lm = results.pose_landmarks.landmark
        kp = lambda idx: np.array([lm[idx].x, lm[idx].y])
        return {
            "left_shoulder":  kp(11), "right_shoulder": kp(12),
            "left_hip":       kp(23), "right_hip":      kp(24),
            "left_knee":      kp(25), "right_knee":     kp(26),
            "left_ankle":     kp(27), "right_ankle":    kp(28),
            "left_elbow":     kp(13), "right_elbow":    kp(14),
            "left_wrist":     kp(15), "right_wrist":    kp(16),
        }

    # --------------------------------------------------------------------- #
    def compute_body_metrics(self, lc: Dict[str, np.ndarray]) -> Dict[str, float]:
        dist = lambda a, b: np.linalg.norm(a - b)

        torso_len = (dist(lc["left_shoulder"], lc["left_hip"]) +
                     dist(lc["right_shoulder"], lc["right_hip"])) / 2

        leg_len   = (dist(lc["left_hip"], lc["left_knee"]) +
                     dist(lc["left_knee"], lc["left_ankle"]) +
                     dist(lc["right_hip"], lc["right_knee"]) +
                     dist(lc["right_knee"], lc["right_ankle"])) / 2

        arm_sym   = abs((dist(lc["left_shoulder"], lc["left_elbow"]) +
                         dist(lc["left_elbow"],   lc["left_wrist"])) -
                        (dist(lc["right_shoulder"], lc["right_elbow"]) +
                         dist(lc["right_elbow"],   lc["right_wrist"])))

        leg_sym   = abs((dist(lc["left_hip"], lc["left_knee"]) +
                         dist(lc["left_knee"], lc["left_ankle"])) -
                        (dist(lc["right_hip"], lc["right_knee"]) +
                         dist(lc["right_knee"], lc["right_ankle"])))

        return {
            "torso_leg_ratio":   torso_len / (leg_len + 1e-6),
            "arm_symmetry_diff": arm_sym,
            "leg_symmetry_diff": leg_sym,
        }

    # --------------------------------------------------------------------- #
    # (unchanged) simple score if you ever want to use it
    def score(self, m: Dict[str, float]) -> float:
        torso_leg_score = max(0, 1 - abs(m["torso_leg_ratio"] - 1.0))
        symmetry_score  = max(0, 1 - (m["arm_symmetry_diff"] + m["leg_symmetry_diff"]) / 2)
        return round((torso_leg_score * 0.6 + symmetry_score * 0.4) * 100, 2)

    # --------------------------------------------------------------------- #
    def evaluate_image(self, img_path: str) -> Dict[str, Optional[float]]:
        image     = Image.open(img_path)
        landmarks = self.extract_landmarks(image)
        if landmarks is None:
            return {"error": "Could not detect full body."}
        metrics = self.compute_body_metrics(landmarks)
        return metrics | {"final_score": self.score(metrics)}


# ──────────────────────────────────────────────────────────────────────────────
#  Command‑line helper
# ──────────────────────────────────────────────────────────────────────────────
def _print_raw_metrics(img_file: Path) -> None:
    scorer     = ProportionScorer()
    image      = Image.open(img_file)
    landmarks  = scorer.extract_landmarks(image)

    if landmarks is None:
        print(f"[{img_file.name}]  ❌  Could not detect a full body.")
        return

    m = scorer.compute_body_metrics(landmarks)
    print(f"\n[{img_file.name}]")
    print("-" * (len(img_file.name) + 2))
    print(f" Torso ÷ Leg ratio : {m['torso_leg_ratio']:.3f}")
    print(f" Arm symmetry diff : {m['arm_symmetry_diff']:.3f}")
    print(f" Leg symmetry diff : {m['leg_symmetry_diff']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python proportion.py <image1> [image2 image3 …]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.is_file():
            print(f"File not found: {path}")
            continue
        _print_raw_metrics(path)
