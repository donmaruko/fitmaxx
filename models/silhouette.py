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
