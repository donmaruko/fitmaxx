# cnnpredict.py – Vision‑GPT outfit grader (FULL VERSION)
# -----------------------------------------------------------------------------
# • Scores outfit aesthetics with your local MLP (unchanged).
# • Uses GPT‑4o vision to identify garments + give body‑aware advice.
# • Adds body‑build descriptor (long‑torso / balanced / long‑leg) from pose
#   ratios so GPT can reference proportion tweaks.
# •     python cnnpredict.py <image> <style> [--no-vision] [--debug]
# -----------------------------------------------------------------------------

import os, sys, argparse, warnings, base64
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import clip
import torch.nn as nn

from langtest import extract_body_ratios  # <- your pose util

# -----------------------------------------------------------------------------
# Command‑line interface
# -----------------------------------------------------------------------------
p = argparse.ArgumentParser(description="Outfit scoring + GPT styling advice")
p.add_argument("image", help="photo (jpg/png)")
p.add_argument("style", choices=["formal", "casual", "athleisure"])
p.add_argument("--no-vision", action="store_true", help="disable GPT‑vision")
p.add_argument("--debug", action="store_true")
a = p.parse_args()
IMG_PATH, STYLE = a.image, a.style.lower()
USE_VISION = not a.no_vision and os.getenv("OPENAI_VISION", "1") != "0"
DEBUG = a.debug
if not os.path.isfile(IMG_PATH):
    sys.exit(f"❌ Image not found: {IMG_PATH}")

# -----------------------------------------------------------------------------
# Torch + CLIP setup (for drip features)
# -----------------------------------------------------------------------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEV)

# -----------------------------------------------------------------------------
# Local aesthetic model (same architecture as in cnntrain.py)
# -----------------------------------------------------------------------------
class OutfitMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 logits, **no sigmoid**
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# Helper: extract CLIP + pose features
# -----------------------------------------------------------------------------

def extract_feats(path: str) -> Tuple[np.ndarray, float, float]:
    """Return (feature‑vector, raw torso/leg ratio, raw symmetry ratio)."""
    img_t = clip_preprocess(Image.open(path)).unsqueeze(0).to(DEV)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t).cpu().squeeze().numpy()

    tl_raw, sym_raw = extract_body_ratios(path)  # may return None
    tl_raw = 0.0 if tl_raw is None else tl_raw
    sym_raw = 0.0 if sym_raw is None else sym_raw

    # scaled values passed to the MLP
    tl_scaled = np.clip(tl_raw / 3.0, 0, 1) * 0.2
    sym_scaled = np.clip(sym_raw / 2.0, 0, 1) * 0.2

    return np.concatenate((emb, [tl_scaled, sym_scaled])), tl_raw, sym_raw

# -----------------------------------------------------------------------------
# NEW: helper to map score → bucket & headline
# -----------------------------------------------------------------------------

def category_from_score(pct: float):
    if pct > 80:  # 81‑100 → elite
        return "elite", "🔥 Elite fit – certified drip!"
    elif 60 <= pct <= 80:
        return "good", "👍 Good fit – you’re on the right track."
    elif 51 <= pct <= 59:
        return "needs_work", "⚠️ This fit could use some work."
    else:
        return "bad", "🚨 This fit needs some work."

# -----------------------------------------------------------------------------
# Compute drip score
# -----------------------------------------------------------------------------
try:
    feats, tl_ratio, sym_ratio = extract_feats(IMG_PATH)
    mlp = OutfitMLP(len(feats)).to(DEV)

    weights_path = f"fit_mlp_{STYLE}.pt"
    raw = torch.load(weights_path, map_location=DEV)

    # unwrap {"state_dict": …} bundles
    meta = {}
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
        raw = raw["state_dict"]

    # adapt key prefixes if needed
    if any(k.startswith("model.") for k in raw):
        raw = {k.replace("model.", "net.", 1): v for k, v in raw.items()}

    mlp.load_state_dict(raw, strict=True)
    mlp.eval()
    logits = mlp(torch.tensor(feats).float().unsqueeze(0).to(DEV))
    probs = torch.softmax(logits, dim=1).cpu()[0]  # [p_low, p_mid, p_high]

    score_val = (probs[1] * 0.5 + probs[2] * 1.0).item()  # 0‑1

    if DEBUG and "val_acc" in meta:
        print(f"[DEBUG] Checkpoint val‑accuracy: {meta['val_acc']*100:.2f}%")
except Exception as e:
    sys.exit(f"❌ Failed to score: {e}")

score_pct = round(score_val * 100, 2)

# -----------------------------------------------------------------------------
# Print bucket headline (single source of truth)
# -----------------------------------------------------------------------------
bucket, headline = category_from_score(score_pct)
print(
    f"\n🖼️  Analyzing: {IMG_PATH}\n🎯 Style: {STYLE}\n💧 Drip Score: {score_pct} / 100"
)
print(headline)

# -----------------------------------------------------------------------------
# Early exit if vision disabled
# -----------------------------------------------------------------------------
if not USE_VISION:
    if DEBUG:
        print("[DEBUG] Vision advice skipped (disabled).")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Determine body‑build descriptor for the prompt
# -----------------------------------------------------------------------------
if tl_ratio == 0:
    build = "unknown proportions"
elif tl_ratio > 0.55:
    build = "long torso / shorter legs"
elif tl_ratio < 0.45:
    build = "short torso / longer legs"
else:
    build = "balanced proportions"

# -----------------------------------------------------------------------------
# OpenAI SDK setup (supports ≥1.0 or legacy 0.x)
# -----------------------------------------------------------------------------
try:
    from openai import OpenAI  # ≥1.0

    oa = OpenAI()
    legacy = False
except ImportError:
    import openai as oa  # legacy

    legacy = True

# -----------------------------------------------------------------------------
# GPT prompt templates & bucket‑specific extras
# -----------------------------------------------------------------------------
# ─── GPT prompt templates & bucket‑specific extras ────────────────────────────
# (put this right after the OpenAI SDK setup)

# Every response MUST begin with a bullet list of garments.
COMMON_FORMAT = (
    "### Main Garments:\n"
    "- <item 1>\n- <item 2>\n- …\n"
)

sys_prompt_elite = (
    "You are a professional fashion stylist. "
    "First, **identify each garment visible in the photo**. "
    "Then explain why the outfit flatters the wearer’s body proportions and "
    "fits the chosen style. Discuss silhouette, texture, color harmony, "
    "and visual balance. No improvement tips.\n\n"
    "Use this exact format:\n"
    + COMMON_FORMAT +
    "### Why It Works:\n"
    "- <reason 1>\n- <reason 2>"
)

sys_prompt_improve = (
    "You are a professional fashion stylist. "
    "First, **identify each garment visible in the photo**. "
    "Then list two strengths of the outfit, followed by three proportion‑aware "
    "improvements that would bring it closer to the target style. "
    "Reference principles such as rule‑of‑thirds, color harmony, vertical "
    "lines, and contrast.\n\n"
    "Use this exact format:\n"
    + COMMON_FORMAT +
    "### Strengths:\n"
    "- <point 1>\n- <point 2>\n"
    "### Improvements:\n"
    "- <tip 1>\n- <tip 2>\n- <tip 3>"
)

extra = {
    "elite": "",
    "good": (
        "Emphasise the existing strengths before listing improvements. "
        "Remember to follow the required format."
    ),
    "needs_work": (
        "Briefly note the main weaknesses, then give improvements. "
        "Remember to follow the required format."
    ),
    "bad": (
        "First explain why the fit doesn’t hit (silhouette, palette, proportion), "
        "then give concrete improvements. Remember to follow the required format."
    ),
}

sys_prompt = sys_prompt_elite if bucket == "elite" else sys_prompt_improve


# -----------------------------------------------------------------------------
# Build the user prompt for GPT‑4o vision
# -----------------------------------------------------------------------------
user_prompt = (
    f"Target style: {STYLE}. Drip score: {score_pct}. Category: {bucket}. "
    f"Body build: {build}. {extra[bucket]} "
    "Always start your response with the ### Main Garments section."
)


# -----------------------------------------------------------------------------
# Build multimodal message payload
# -----------------------------------------------------------------------------
with open(IMG_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

img_block = {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
}
msgs = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [{"type": "text", "text": user_prompt}, img_block]},
]

try:
    if legacy:
        rsp = oa.ChatCompletion.create(
            model="gpt-4o-mini", messages=msgs, temperature=0.7
        )
        reply = rsp["choices"][0]["message"]["content"]
    else:
        rsp = oa.chat.completions.create(
            model="gpt-4o-mini", messages=msgs, temperature=0.7
        )
        reply = rsp.choices[0].message.content

    print("\n🧠 Fit Analysis:\n" + reply)
except Exception as e:
    warnings.warn(
        f"FitMaxx vision failed: {e}. Rerun with --no-vision to skip."
    )

if DEBUG:
    print("[DEBUG] FitMaxx vision completed.")
