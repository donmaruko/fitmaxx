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
from typing import Optional
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
chosen_style = sys.argv[1] if len(sys.argv) > 1 else "athleisure"
test_image_path = "models/myfit.png"

# -----------------------------
# Define State Schema
# -----------------------------
class SilhouetteState(BaseModel):
    image_path: str
    style: str
    landmarks: Optional[list] = Field(default_factory=list)
    image_shape: Optional[tuple] = None
    proportions: Optional[dict] = None
    score: int = 0
    reason: Optional[list] = Field(default_factory=list)
    style_tip: Optional[str] = None

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

workflow.set_finish_point("generate_tip") 

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
