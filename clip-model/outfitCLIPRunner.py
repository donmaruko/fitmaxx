import clip
import torch
import os
import sys
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np

"""
This uses CLIP's multimodal understanding to compare fit images to high-drip reference fits.
It captures general aesthetic alignment, meaning: "Does this look like other good outfits in this style category?"
"""

# Load the CLIP model + preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Dataset Class (must be above usage)
# -----------------------------
class OutfitDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return preprocess(Image.open(self.paths[idx]))

# -----------------------------
# Build reference_db by scanning folders
# -----------------------------
reference_db = {}  # style → tier → list of embeddings

base_path = "outfits"
styles = os.listdir(base_path)

for style in styles:
    style_path = os.path.join(base_path, style)
    if not os.path.isdir(style_path):
        continue

    reference_db[style] = {}

    for tier in os.listdir(style_path):  # e.g. high, low
        tier_path = os.path.join(style_path, tier)
        if not os.path.isdir(tier_path):
            continue

        image_paths = [os.path.join(tier_path, f) for f in os.listdir(tier_path) if f.endswith(('.jpg', '.png'))]
        if len(image_paths) == 0:
            print(f"[!] No images found in {tier_path}")
            continue

        print(f"Processing {len(image_paths)} images in {style}/{tier}...")

        dataset = OutfitDataset(tier_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                encoded = model.encode_image(batch).cpu()
                embeddings.extend(encoded)

        reference_db[style][tier] = embeddings

# -----------------------------
# Encode text prompts for aesthetic classification
# -----------------------------
text_inputs = torch.cat([
    clip.tokenize("a formal outfit"),
    clip.tokenize("a casual fit"),
    clip.tokenize("an athleisure outfit")
]).to(device)

with torch.no_grad():
    text_embeddings = model.encode_text(text_inputs)

# -----------------------------
# Score functions
# -----------------------------
def score_image(image_path, reference_embeddings):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu()
    
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    reference_embeddings = torch.stack(reference_embeddings)
    reference_embeddings /= reference_embeddings.norm(dim=-1, keepdim=True)

    similarities = (reference_embeddings @ image_embedding.T).squeeze()
    return similarities.mean().item()

def image_to_style_similarity(image_path, text_embeddings):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu().float()
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.cpu().float()
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        similarities = (text_embeddings @ image_embedding.T).squeeze()
        return similarities  # returns similarity to all 3 style prompts

def compute_drip_score(image_path, style, reference_db):
    if style not in reference_db:
        raise ValueError(f"[!] Style '{style}' not found in reference database.")
    if "high" not in reference_db[style] or "low" not in reference_db[style]:
        raise ValueError(f"[!] Missing 'high' or 'low' tier in '{style}' references.")
    
    sim_high = score_image(image_path, reference_db[style]["high"])
    sim_low = score_image(image_path, reference_db[style]["low"])

    # Raw score range is [-1, 1] → normalize to [0, 100]
    raw_score = sim_high - sim_low
    drip_score = round((raw_score + 1) / 2 * 100, 2)

    return {
        "style": style,
        "similarity_to_high": round(sim_high, 4),
        "similarity_to_low": round(sim_low, 4),
        "drip_score": drip_score
    }

def interpret_drip_score(score):
    if score > 75:
        return "🔥 Certified drip! Clean fit that aligns with high-tier aesthetics."
    elif score > 50:
        return "👌 Decent fit, but room for improvement."
    else:
        return "🧢 Needs work. Fit leans toward low-drip examples."

# this is boilerplate code that extracts top and bottom halves of the person, include this in every model
def extract_features(image_path):
    """
    Basic shared feature extraction for all models.
    Segments top and bottom halves of the image for further processing.
    Returns: dict with 'top_image', 'bottom_image' as PIL Images
    """
    image = Image.open(image_path).convert('RGB')
    w, h = image.size

    # Crop top and bottom halves
    top_crop_box = (0, 0, w, h // 2)
    bottom_crop_box = (0, h // 2, w, h)

    top_image = image.crop(top_crop_box)
    bottom_image = image.crop(bottom_crop_box)

    features = {
        "original": image,
        "top_image": top_image,
        "bottom_image": bottom_image,
        "width": w,
        "height": h,
    }

    return features

# -----------------------------
# Example Usage (CLI-style)
# -----------------------------
test_image_path = "clip-model/goodgym.jpeg"  # Replace with user upload in production
chosen_style = sys.argv[1] if len(sys.argv) > 1 else "athleisure"

if os.path.exists(test_image_path):
    try:
        result = compute_drip_score(test_image_path, chosen_style, reference_db)
        print(f"\n👕 Fit Analysis for: {result['style'].capitalize()}")
        print(f"Similarity to High Drip: {result['similarity_to_high']}")
        print(f"Similarity to Low Drip:  {result['similarity_to_low']}")
        print(f"💧 Drip Score: {result['drip_score']} / 100")
        print(f"🧠 Interpretation: {interpret_drip_score(result['drip_score'])}")

        text_sim = image_to_style_similarity(test_image_path, text_embeddings)
        print(f"\nText style similarity (formal, casual, athleisure): {text_sim.tolist()}")
    except ValueError as e:
        print(str(e))
else:
    print("[!] No test image found. Drop one in and update 'myfit.jpg'")

features = extract_features("clip-model/goodgym.jpeg")
features["top_image"].show()
features["bottom_image"].show()
