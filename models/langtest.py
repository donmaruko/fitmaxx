from langgraph.graph import StateGraph
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional
import clip
import torch
from PIL import Image
import os
import numpy as np
import onnxruntime as ort
from torch.nn.functional import cosine_similarity as torch_cos_sim
from rembg import remove
from io import BytesIO


llm = ChatOpenAI(model="gpt-3.5-turbo")  # or "gpt-4" if available

# -----------------------------
# Setup and CLIP model loading
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

reference_db = {}  # Assumes this is populated earlier as in your original code

# -----------------------------
# Tools
# -----------------------------
@tool
def interpret_score_tool(query: str) -> str:
    """Interpret a drip score and provide style-specific feedback. Input should be '<style>,<score>'."""
    try:
        style, score_str = query.split(",")
        score = float(score_str)
    except Exception:
        return "❌ Invalid input format. Use '<style>,<score>' (e.g., athleisure,67.5)."

    if score > 75:
        return f"🔥 Your {style} fit is elite. Certified drip!"
    elif score > 50:
        return f"👌 Your {style} outfit is decent. Consider tweaking the silhouette or layering."
    else:
        return f"🧢 Your {style} fit needs work. Focus on color harmony and proportions."


@tool
def recommend_outfit_tool(style: str) -> str:
    """Recommend ideal outfit pieces based on the chosen style."""
    return f"Suggested {style} pieces: layered jackets, tapered pants, and balanced tones. Try accessorizing with minimal items."

def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image and return person-only cropped version."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()  # <-- extract raw bytes

    output_bytes = remove(img_bytes, force_return_bytes=True)
    return Image.open(BytesIO(output_bytes)).convert("RGB")


# -----------------------------
# Image scoring
# -----------------------------
def score_image_cosine(image_path, style, reference_db):
    def score_image(image_path, reference_embeddings):
        raw_image = Image.open(image_path)
        person_only_image = remove_background(raw_image)  # 👈 crop background
        image = preprocess(person_only_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = model.encode_image(image).cpu()
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        reference_embeddings = torch.stack(reference_embeddings)
        reference_embeddings /= reference_embeddings.norm(dim=-1, keepdim=True)
        similarities = (reference_embeddings @ image_embedding.T).squeeze()
        return similarities.mean().item()

    high_embeds = reference_db[style]["high"]["embeddings"]
    low_embeds = reference_db[style]["low"]["embeddings"]
    sim_high = score_image(image_path, high_embeds)
    sim_low = score_image(image_path, low_embeds)
    raw_score = sim_high - sim_low
    drip_score = round((raw_score + 1) / 2 * 100, 2)
    return drip_score


# -----------------------------
# LangGraph: Branching Logic
# -----------------------------
def branch_logic(state):
    score = state["drip_score"]
    return "recommend" if score >= 75 else "improve"

# -----------------------------
# LangGraph Pipeline
# -----------------------------

# -----------------------------
# Build reference_db by scanning folders
# -----------------------------
class OutfitDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        return preprocess(Image.open(self.paths[idx]))

base_path = "../outfits"
if not os.path.exists(base_path):
    raise FileNotFoundError(f"[!] Could not find outfits folder at {base_path}")
styles = os.listdir(base_path)

for style in styles:
    style_path = os.path.join(base_path, style)
    if not os.path.isdir(style_path):
        continue

    reference_db[style] = {}

    for tier in os.listdir(style_path):  # e.g., 'high', 'low'
        tier_path = os.path.join(style_path, tier)
        if not os.path.isdir(tier_path):
            continue

        image_paths = [os.path.join(tier_path, f) for f in os.listdir(tier_path) if f.lower().endswith(('.jpg', '.png'))]
        if len(image_paths) == 0:
            print(f"[!] No images found in {tier_path}")
            continue

        dataset = OutfitDataset(tier_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                encoded = model.encode_image(batch).cpu()
                embeddings.extend(encoded)

        reference_db[style][tier] = {"embeddings": embeddings, "paths": image_paths}


class OutfitState(TypedDict, total=False):
    image_path: str
    style: str
    drip_score: float
    feedback: str
    recommendation: str

graph = StateGraph(OutfitState)

def score_node(state):
    score = score_image_cosine(state["image_path"], state["style"], reference_db)
    return {**state, "drip_score": score}

def improvement_node(state):
    input_str = f"{state['style']},{state['drip_score']}"
    msg = interpret_score_tool.invoke(input_str)
    return {**state, "feedback": msg}


def recommendation_node(state):
    agent = initialize_agent(
        [recommend_outfit_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    msg = agent.run(f"Suggest fits for {state['style']}")
    return {**state, "recommendation": msg}


# Define nodes
graph.add_node("score", score_node)
graph.add_node("recommend", recommendation_node)
graph.add_node("improve", improvement_node)

# Define graph structure
graph.set_entry_point("score")
graph.add_conditional_edges("score", branch_logic)

# -----------------------------
# Run graph
# -----------------------------
workflow = graph.compile()

# Example usage
if __name__ == "__main__":
    state = {"image_path": "tester.jpeg", "style": "athleisure"}
    final_state = workflow.invoke(state)
    print("\n===== FINAL OUTFIT ANALYSIS =====")
    print(f"🖼️  Image Path: {final_state['image_path']}")
    print(f"🎯 Style Target: {final_state['style']}")
    print(f"💧 Drip Score: {final_state['drip_score']} / 100")

    if "recommendation" in final_state:
        print(f"🧥 Recommendation:\n{final_state['recommendation']}")
    if "feedback" in final_state:
        print(f"🔍 Feedback:\n{final_state['feedback']}")
    print("=================================\n")
