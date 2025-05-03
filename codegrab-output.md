# Project Structure

```
hunterhacks/
└── models/
    ├── fusionrunner.py
    ├── outfitCLIPRunner.py
    ├── silhouette.py
    └── train-classifier.py
```

# Project Files

## File: `models/fusionrunner.py`

```python
import os
import sys
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
chosen_style = sys.argv[1] if len(sys.argv) > 1 else "athleisure"
test_image_path = "myfit.png"

# -----------------------------
# Define State Schema
# -----------------------------
class SilhouetteState(BaseModel):
    image_path: str
    style: str
    landmarks: list = Field(default_factory=list)
    image_shape: tuple = None
    proportions: dict = None
    score: int = 0
    reason: list = Field(default_factory=list)

# -----------------------------
# Pose Extraction using MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True)

def extract_pose_landmarks(state: SilhouetteState) -> SilhouetteState:
    image_path = state.image_path
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        return state.model_copy(update={
            "landmarks": results.pose_landmarks.landmark,
            "image_shape": image.shape
        })
    return state.model_copy(update={"landmarks": None})

# -----------------------------
# Analyze Silhouette
# -----------------------------
def calculate_proportions(state: SilhouetteState) -> SilhouetteState:
    landmarks = state.landmarks
    if not landmarks:
        return state.model_copy(update={"proportions": None})

    def dist(p1, p2):
        return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))

    image_h, image_w, _ = state.image_shape
    shoulder = dist(landmarks[11], landmarks[12])
    torso = dist(landmarks[11], landmarks[23])
    legs = dist(landmarks[23], landmarks[25]) + dist(landmarks[25], landmarks[27])

    torso_leg_ratio = torso / legs if legs else 0
    proportions = {
        "shoulder_width": shoulder,
        "torso_length": torso,
        "leg_length": legs,
        "torso_leg_ratio": torso_leg_ratio
    }

    return state.model_copy(update={"proportions": proportions})

# -----------------------------
# Style-based Rule Scoring
# -----------------------------
def score_by_style(state: SilhouetteState) -> SilhouetteState:
    proportions = state.proportions
    style = state.style

    if not proportions:
        return state.model_copy(update={"score": 0, "reason": ["No landmarks detected."]})

    score = 50
    reason = []

    if style == "streetwear":
        if proportions["torso_leg_ratio"] < 0.8:
            score += 10
            reason.append("Cropped top with longer legs matches streetwear aesthetic.")
    elif style == "formal":
        if proportions["shoulder_width"] > 0.15:
            score += 10
            reason.append("Broad shoulders align with formal silhouettes.")
    elif style == "athleisure":
        if 0.8 < proportions["torso_leg_ratio"] < 1.2:
            score += 10
            reason.append("Balanced proportions suitable for functional athleisure.")

    return state.model_copy(update={"score": score, "reason": reason})

# -----------------------------
# CLIP Model Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Segment Person from Background
# -----------------------------
def segment_person(image_path):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = segmenter.process(image_rgb)

        if results.segmentation_mask is None:
            print("[!] No person detected, using original image.")
            return Image.open(image_path)

        mask = results.segmentation_mask > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        person_only = np.where(mask[..., None], image, bg_image)
        person_only_rgb = cv2.cvtColor(person_only, cv2.COLOR_BGR2RGB)
        return Image.fromarray(person_only_rgb)

# -----------------------------
# Dataset and Reference Embeddings
# -----------------------------
class OutfitDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return preprocess(Image.open(self.paths[idx]))

reference_db = {}
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outfits"))
style_path = os.path.join(base_path, chosen_style)

if not os.path.isdir(style_path):
    raise FileNotFoundError(f"[!] Style folder '{chosen_style}' does not exist in outfits/")

reference_db[chosen_style] = {}

for tier in os.listdir(style_path):  # high/low
    tier_path = os.path.join(style_path, tier)
    if not os.path.isdir(tier_path):
        continue

    image_paths = [os.path.join(tier_path, f) for f in os.listdir(tier_path) if f.lower().endswith(('.jpg', '.png'))]
    if not image_paths:
        print(f"[!] No images found in {tier_path}")
        continue

    print(f"Processing {len(image_paths)} images in {chosen_style}/{tier}...")

    dataset = OutfitDataset(tier_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            encoded = model.encode_image(batch).cpu()
            embeddings.extend(encoded)

    reference_db[chosen_style][tier] = {
        "embeddings": embeddings,
        "paths": image_paths
    }

# -----------------------------
# Scoring Functions
# -----------------------------
def score_image(image_path, reference_embeddings):
    person_cropped = segment_person(image_path)
    image = preprocess(person_cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu()

    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    reference_embeddings = torch.stack(reference_embeddings)
    reference_embeddings /= reference_embeddings.norm(dim=-1, keepdim=True)
    similarities = (reference_embeddings @ image_embedding.T).squeeze()
    return similarities.mean().item()

def compute_drip_score_cosine(image_path, style, reference_db):
    if style not in reference_db:
        raise ValueError(f"[!] Style '{style}' not found.")
    if "high" not in reference_db[style] or "low" not in reference_db[style]:
        raise ValueError(f"[!] Missing 'high' or 'low' tier in '{style}'.")

    high_embeds = reference_db[style]["high"]["embeddings"]
    low_embeds = reference_db[style]["low"]["embeddings"]
    sim_high = score_image(image_path, high_embeds)
    sim_low = score_image(image_path, low_embeds)
    raw_score = sim_high - sim_low
    drip_score = round((raw_score + 1) / 2 * 100, 2)
    return {
        "style": style,
        "similarity_to_high": round(sim_high, 4),
        "similarity_to_low": round(sim_low, 4),
        "drip_score": drip_score
    }

# -----------------------------
# Generate Style Tip
# -----------------------------
tip_prompt = PromptTemplate.from_template("""
You are a fashion stylist analyzing a user's outfit.
Style: {style}
Score: {score}
Reasoning: {reason}

Generate a short, helpful critique (1-2 sentences) about how their fit aligns with the style. Be specific and stylish.
""")

llm = ChatOpenAI(temperature=0.7)
style_tip_chain = tip_prompt | llm

def generate_style_tip(state: SilhouetteState) -> SilhouetteState:
    tip = style_tip_chain.invoke({
        "style": state.style,
        "score": state.score,
        "reason": ", ".join(state.reason)
    })
    return state.model_copy(up1date={"style_tip": tip.content})

# -----------------------------
# Build LangGraph Pipeline
# -----------------------------
workflow = StateGraph(SilhouetteState)
workflow.add_node("extract_pose", RunnableLambda(extract_pose_landmarks))
workflow.add_node("analyze_silhouette", RunnableLambda(calculate_proportions))
workflow.add_node("score_silhouette", RunnableLambda(score_by_style))
workflow.add_node("generate_tip", RunnableLambda(generate_style_tip))

workflow.set_entry_point("extract_pose")
workflow.add_edge("extract_pose", "analyze_silhouette")
workflow.add_edge("analyze_silhouette", "score_silhouette")
workflow.add_edge("score_silhouette", "generate_tip")

graph = workflow.compile()

# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(test_image_path):
        print("[!] No test image found. Please drop one into 'models/myfit.png'")
        sys.exit(1)

    silhouette_result = graph.invoke({"image_path": test_image_path, "style": chosen_style})
    print("\nSilhouette Evaluation")
    print("Score:", silhouette_result["score"])
    print("Reasoning:", silhouette_result["reason"])

    try:
        result = compute_drip_score_cosine(test_image_path, chosen_style, reference_db)
        print(f"\n👕 [Cosine] Fit Analysis for: {result['style'].capitalize()}")
        print(f"Similarity to High Drip: {result['similarity_to_high']}")
        print(f"Similarity to Low Drip:  {result['similarity_to_low']}")
        print("\n💬 Style Tip:")
        print(silhouette_result["style_tip"])
        print(f"💧 Drip Score: {result['drip_score']} / 100")
    except ValueError as e:
        print(str(e))

```

## File: `models/outfitCLIPRunner.py`

```python
import clip
import torch
import os
import sys
from PIL import Image
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
import numpy as np
import joblib  # for loading our classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Load the CLIP model + preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Optionally load the classifier if using it.
# Ensure you have trained and saved this model (drip_classifier.pkl) beforehand.
USE_CLASSIFIER = "--use-clf" in sys.argv
if USE_CLASSIFIER:
    clf = joblib.load("drip_classifier.pkl")
    print("[*] Using classifier-based scoring.")

# -----------------------------
# Dataset Class (must be above usage)
# -----------------------------
class OutfitDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return preprocess(Image.open(self.paths[idx]))

# -----------------------------
# Build reference_db by scanning folders
# -----------------------------
reference_db = {}  # style → tier → list of embeddings, and paths stored separately if needed

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

        image_paths = [os.path.join(tier_path, f) for f in os.listdir(tier_path) if f.lower().endswith(('.jpg', '.png'))]
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

        # For classifier training or visualization, you might also store paths:
        reference_db[style][tier] = {"embeddings": embeddings, "paths": image_paths}

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

def compute_drip_score_cosine(image_path, style, reference_db):
    # Existing cosine similarity–based scoring
    if style not in reference_db:
        raise ValueError(f"[!] Style '{style}' not found in reference database.")
    if "high" not in reference_db[style] or "low" not in reference_db[style]:
        raise ValueError(f"[!] Missing 'high' or 'low' tier in '{style}' references.")
    
    high_embeds = reference_db[style]["high"]["embeddings"]
    low_embeds = reference_db[style]["low"]["embeddings"]
    sim_high = score_image(image_path, high_embeds)
    sim_low = score_image(image_path, low_embeds)
    raw_score = sim_high - sim_low
    drip_score = round((raw_score + 1) / 2 * 100, 2)
    return {
        "style": style,
        "similarity_to_high": round(sim_high, 4),
        "similarity_to_low": round(sim_low, 4),
        "drip_score": drip_score
    }

def compute_drip_score_classifier(image_path, style):
    # Using the classifier to predict probability of a high-drip fit.
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().flatten()
    # Assuming the classifier was trained on normalized embeddings, ensure normalization:
    embedding = embedding / np.linalg.norm(embedding)
    # Predict probability; clf.predict_proba returns array with columns [prob_low, prob_high]
    prob = clf.predict_proba([embedding])[0]
    p_high = prob[1]
    drip_score = round(p_high * 100, 2)
    return {
        "style": style,
        "drip_score": drip_score,
        "confidence": p_high,
    }

def interpret_drip_score(score):
    if score > 75:
        return "🔥 Certified drip! Clean fit that aligns with high-tier aesthetics."
    elif score > 50:
        return "👌 Decent fit, but room for improvement."
    else:
        return "🧢 Needs work. Fit leans toward low-drip examples."

# -----------------------------
# Basic shared feature extraction (top/bottom segmentation) 
# -----------------------------
def extract_features(image_path):
    """
    Basic shared feature extraction for all models.
    Segments top and bottom halves of the image for further processing.
    Returns: dict with 'top_image', 'bottom_image' as PIL Images.
    """
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    top_crop_box = (0, 0, w, h // 2)
    bottom_crop_box = (0, h // 2, w, h)
    top_image = image.crop(top_crop_box)
    bottom_image = image.crop(bottom_crop_box)
    return {"original": image, "top_image": top_image, "bottom_image": bottom_image, "width": w, "height": h}

# -----------------------------
# Example Usage (CLI-style)
# -----------------------------
test_image_path = "models/myfit.png"  # Replace with actual test image path
chosen_style = sys.argv[1] if len(sys.argv) > 1 else "athleisure"

if not os.path.exists(test_image_path):
    print("[!] No test image found. Drop one in and update the test image path.")
    sys.exit(1)

try:
    if USE_CLASSIFIER:
        result = compute_drip_score_classifier(test_image_path, chosen_style)
        print(f"\n👕 [Classifier] Fit Analysis for: {result['style'].capitalize()}")
        print(f"💧 Drip Score: {result['drip_score']} / 100")
        print(f"🧠 Confidence (prob. high): {result['confidence']:.4f}")
        print(f"🧠 Interpretation: {interpret_drip_score(result['drip_score'])}")
    else:
        result = compute_drip_score_cosine(test_image_path, chosen_style, reference_db)
        print(f"\n👕 [Cosine] Fit Analysis for: {result['style'].capitalize()}")
        print(f"Similarity to High Drip: {result['similarity_to_high']}")
        print(f"Similarity to Low Drip:  {result['similarity_to_low']}")
        print(f"💧 Drip Score: {result['drip_score']} / 100")
        print(f"🧠 Interpretation: {interpret_drip_score(result['drip_score'])}")

    text_sim = image_to_style_similarity(test_image_path, text_embeddings)
    print(f"\nText style similarity (formal, casual, athleisure): {text_sim.tolist()}")

    # Show extracted top and bottom halves for visual sanity check.
    features = extract_features(test_image_path)
    features["top_image"].show()
    features["bottom_image"].show()

except ValueError as e:
    print(str(e))


# -----------------------------
# Visual + Explainable Output: Find most similar reference fit and plot embeddings using UMAP/t-SNE
# -----------------------------
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def find_most_similar_fit(image_path, reference_data):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image).cpu()
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    ref_embeddings = torch.stack(reference_data["embeddings"])
    ref_embeddings /= ref_embeddings.norm(dim=-1, keepdim=True)
    similarities = cosine_similarity(image_embedding.numpy(), ref_embeddings.numpy())[0]
    
    # Debugging: Print similarities
    print(f"Similarities: {similarities}")
    print(f"Reference paths: {reference_data['paths']}")
    
    best_idx = int(np.argmax(similarities))
    return reference_data["paths"][best_idx], similarities[best_idx]

def plot_embeddings_umap(user_embedding, high_embeddings, low_embeddings):
    try:
        import umap.umap_ as umap
    except ImportError:
        print("Please install umap-learn (pip install umap-learn)")
        return
    
    # Combine embeddings
    all_embeddings = torch.stack(high_embeddings + low_embeddings + [user_embedding]).numpy()
    labels = (["high"] * len(high_embeddings)) + (["low"] * len(low_embeddings)) + ["user"]
    
    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(all_embeddings)

    plt.figure(figsize=(8, 6))
    for lab in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == lab]
        coords = reduced[idxs]
        if lab == "high":
            plt.scatter(coords[:, 0], coords[:, 1], c="green", label="High Drip")
        elif lab == "low":
            plt.scatter(coords[:, 0], coords[:, 1], c="red", label="Low Drip")
        else:
            plt.scatter(coords[:, 0], coords[:, 1], c="blue", label="User Fit", edgecolor="black", s=100)
    plt.title("UMAP Projection of CLIP Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example: Use only high and low reference images for chosen style.
if chosen_style in reference_db and "high" in reference_db[chosen_style] and "low" in reference_db[chosen_style]:
    ref_high_data = reference_db[chosen_style]["high"]
    ref_low_data = reference_db[chosen_style]["low"]
    # Find nearest fit from high-drip references:
    best_fit_path, best_sim = find_most_similar_fit(test_image_path, ref_high_data)
    print(f"\n📸 Your fit is closest to: {best_fit_path} (Similarity: {best_sim:.4f})")
    print(f"Reference embeddings for {style} (high): {len(ref_high_data['embeddings'])}")
    print(f"Reference embeddings for {style} (low): {len(ref_low_data['embeddings'])}")
    
    # For UMAP visualization, get one test embedding:
    image = preprocess(Image.open(test_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        user_embedding = model.encode_image(image).cpu().squeeze()
    plot_embeddings_umap(user_embedding, ref_high_data["embeddings"], ref_low_data["embeddings"])

else:
    print(f"[!] Insufficient reference images for style {chosen_style} to generate visualization.")

```

## File: `models/silhouette.py`

```python
import cv2
import mediapipe as mp
import numpy as np
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

# -----------------------------
# Define State Schema
# -----------------------------
class SilhouetteState(BaseModel):
    image_path: str
    style: str
    landmarks: list = Field(default_factory=list)
    image_shape: tuple = None
    proportions: dict = None
    score: int = 0
    reason: list = Field(default_factory=list)

# -----------------------------
# 1. Pose Extraction using MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True)

def extract_pose_landmarks(state: SilhouetteState) -> SilhouetteState:
    image_path = state.image_path
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)
    if results.pose_landmarks:
        return state.model_copy(update={
            "landmarks": results.pose_landmarks.landmark,
            "image_shape": image.shape
        })
    return state.model_copy(update={"landmarks": None})

# -----------------------------
# 2. Analyze Silhouette
# -----------------------------
def calculate_proportions(state: SilhouetteState) -> SilhouetteState:
    landmarks = state.landmarks
    if not landmarks:
        return state.model_copy(update={"proportions": None})

    def dist(p1, p2):
        return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))

    image_h, image_w, _ = state.image_shape
    shoulder = dist(landmarks[11], landmarks[12])
    torso = dist(landmarks[11], landmarks[23])
    legs = dist(landmarks[23], landmarks[25]) + dist(landmarks[25], landmarks[27])

    torso_leg_ratio = torso / legs if legs else 0
    proportions = {
        "shoulder_width": shoulder,
        "torso_length": torso,
        "leg_length": legs,
        "torso_leg_ratio": torso_leg_ratio
    }

    return state.model_copy(update={"proportions": proportions})

# -----------------------------
# 3. Style-based Rule Scoring
# -----------------------------
def score_by_style(state: SilhouetteState) -> SilhouetteState:
    proportions = state.proportions
    style = state.style

    if not proportions:
        return state.model_copy(update={"score": 0, "reason": ["No landmarks detected."]})

    score = 50
    reason = []

    if style == "streetwear":
        if proportions["torso_leg_ratio"] < 0.8:
            score += 10
            reason.append("Cropped top with longer legs matches streetwear aesthetic.")
    elif style == "formal":
        if proportions["shoulder_width"] > 0.15:
            score += 10
            reason.append("Broad shoulders align with formal silhouettes.")
    elif style == "athleisure":
        if 0.8 < proportions["torso_leg_ratio"] < 1.2:
            score += 10
            reason.append("Balanced proportions suitable for functional athleisure.")

    return state.model_copy(update={"score": score, "reason": reason})

# -----------------------------
# 4. Build LangGraph Pipeline
# -----------------------------
workflow = StateGraph(SilhouetteState)

workflow.add_node("extract_pose", RunnableLambda(extract_pose_landmarks))
workflow.add_node("analyze_silhouette", RunnableLambda(calculate_proportions))
workflow.add_node("score_silhouette", RunnableLambda(score_by_style))

workflow.set_entry_point("extract_pose")
workflow.add_edge("extract_pose", "analyze_silhouette")
workflow.add_edge("analyze_silhouette", "score_silhouette")

graph = workflow.compile()

# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    result = graph.invoke({"image_path": "models/myfit.png", "style": "athleisure"})
    print("\nSilhouette Evaluation")
    print("Score:", result["score"])
    print("Reasoning:", result["reason"])

```

## File: `models/train-classifier.py`

```python
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

```

