from langgraph.graph import StateGraph
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from typing import TypedDict, Optional
import matplotlib.pyplot as plt
import clip
import torch
from PIL import Image
import os
import numpy as np
import onnxruntime as ort
from torch.nn.functional import cosine_similarity as torch_cos_sim
from rembg import remove
from io import BytesIO
import mediapipe as mp

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("❌ Please set your OPENAI_API_KEY environment variable before running this script.")
llm = ChatOpenAI(model="gpt-4o-mini")  # or "gpt-4" if available

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

@tool
def proportion_feedback_tool(data: str) -> str:
    """Give outfit adjustment advice based on torso-to-leg ratio and limb symmetry.
    Input format: '<torso_leg_ratio>,<limb_symmetry>'."""
    try:
        torso_str, sym_str = data.split(",")
        torso_ratio = float(torso_str)
        symmetry_diff = float(sym_str)
    except Exception:
        return "❌ Invalid input format. Use '<torso_leg_ratio>,<limb_symmetry>' (e.g., 1.03,0.08)."

    torso_msg = ""
    if torso_ratio > 1.05:
        torso_msg = "👕 You have a long torso. Use high-rise pants, tucked tops, and avoid long outerwear."
    elif torso_ratio < 0.95:
        torso_msg = "👖 You have long legs. Balance it with cropped jackets or mid-rise bottoms."
    else:
        torso_msg = "✅ Your proportions are balanced. You have flexibility in layering and structure."

    symmetry_msg = ""
    if symmetry_diff < 0.1:
        symmetry_msg = "🦿 Your limbs are symmetrical. You can experiment with asymmetry and standout pieces."
    else:
        symmetry_msg = "🪞 Slight asymmetry detected. Consider clean lines and even layering."

    return f"{torso_msg}\n{symmetry_msg}"


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image and return person-only cropped version."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()  # <-- extract raw bytes

    output_bytes = remove(img_bytes, force_return_bytes=True)
    return Image.open(BytesIO(output_bytes)).convert("RGB")

#
# Body Ratios
#

def extract_body_ratios(image_path: str) -> tuple[Optional[float], Optional[float]]:
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        image = np.array(Image.open(image_path).convert("RGB"))
        results = pose.process(image)

        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark

        def distance(a, b):
            return np.linalg.norm(np.array([lm[a].x, lm[a].y]) - np.array([lm[b].x, lm[b].y]))

        # Key landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        # Measurements
        torso_length = distance(LEFT_SHOULDER, LEFT_HIP) + distance(RIGHT_SHOULDER, RIGHT_HIP) / 2
        leg_length = (distance(LEFT_HIP, LEFT_KNEE) + distance(LEFT_KNEE, LEFT_ANKLE) +
                      distance(RIGHT_HIP, RIGHT_KNEE) + distance(RIGHT_KNEE, RIGHT_ANKLE)) / 2

        left_arm = distance(LEFT_SHOULDER, LEFT_ELBOW) + distance(LEFT_ELBOW, LEFT_WRIST)
        right_arm = distance(RIGHT_SHOULDER, RIGHT_ELBOW) + distance(RIGHT_ELBOW, RIGHT_WRIST)
        left_leg = distance(LEFT_HIP, LEFT_KNEE) + distance(LEFT_KNEE, LEFT_ANKLE)
        right_leg = distance(RIGHT_HIP, RIGHT_KNEE) + distance(RIGHT_KNEE, RIGHT_ANKLE)

        torso_leg_ratio = round(torso_length / leg_length, 2) if leg_length != 0 else None
        limb_symmetry = round(abs(left_arm - right_arm) + abs(left_leg - right_leg), 2)

        return torso_leg_ratio, limb_symmetry

# -----------------------------
# Image scoring
# -----------------------------
def score_image_cosine(image_path, style, reference_db):
    def get_embedding(image_path):
        raw_image = Image.open(image_path)
        person_only_image = remove_background(raw_image)
        image = preprocess(person_only_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).cpu().squeeze()
            embedding /= embedding.norm()
        return embedding

    test_embed = get_embedding(image_path)
    high_embeds = reference_db[style]["high"]["embeddings"]
    low_embeds = reference_db[style]["low"]["embeddings"]

    high_stack = torch.stack(high_embeds)
    low_stack = torch.stack(low_embeds)

    high_stack /= high_stack.norm(dim=-1, keepdim=True)
    low_stack /= low_stack.norm(dim=-1, keepdim=True)

    sim_to_high = torch_cos_sim(test_embed, high_stack).mean().item()
    sim_to_low = torch_cos_sim(test_embed, low_stack).mean().item()

    # Normalize position between low/high anchors
    drip_score = (sim_to_high - sim_to_low) / (1e-5 + (sim_to_high + sim_to_low))
    drip_score = round((drip_score + 1) / 2 * 100, 2)

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
    torso_leg_ratio: Optional[float]
    limb_symmetry: Optional[float]
    proportion_feedback: Optional[str]

graph = StateGraph(OutfitState)

def score_node(state):
    score = score_image_cosine(state["image_path"], state["style"], reference_db)
    torso_leg_ratio, limb_symmetry = extract_body_ratios(state["image_path"])
    return {
        **state,
        "drip_score": score,
        "torso_leg_ratio": torso_leg_ratio,
        "limb_symmetry": limb_symmetry,
    }


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

def body_proportion_node(state):
    if state['torso_leg_ratio'] is None or state['limb_symmetry'] is None:
        return {**state, "proportion_feedback": "⚠️ Couldn't detect full body posture. Try a clearer full-body photo."}
    
    input_str = f"{state['torso_leg_ratio']},{state['limb_symmetry']}"
    msg = proportion_feedback_tool.invoke(input_str)
    return {**state, "proportion_feedback": msg}



# Define nodes
graph.add_node("score", score_node)
graph.add_node("recommend", recommendation_node)
graph.add_node("improve", improvement_node)
graph.add_node("body_feedback", body_proportion_node)
graph.add_edge("score", "body_feedback")


# Define graph structure
graph.set_entry_point("score")
graph.add_conditional_edges("body_feedback", branch_logic)


# -----------------------------
# Run graph
# -----------------------------
workflow = graph.compile()

# Example usage
if __name__ == "__main__":
    state = {"image_path": "uglysuit.jpg", "style": "formal"}
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

    if "torso_leg_ratio" in final_state:
        print(f"📏 Torso-to-Leg Ratio: {final_state['torso_leg_ratio']:.2f}")
    if "limb_symmetry" in final_state:
        print(f"🦿 Limb Symmetry Difference: {final_state['limb_symmetry']:.2f}")

    if "proportion_feedback" in final_state:
        print(f"🧍 Proportion-Based Advice:\n{final_state['proportion_feedback']}")
