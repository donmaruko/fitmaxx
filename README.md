# 👔👟 FitMaxx — Your Personal AI Outfit Wingman

FitMaxx is a modular AI web app that analyzes outfit photos and rates how well they match a chosen **style** — such as `formal`, `casual`, or even custom aesthetics like `alternative` or `techwear`.

The model rates your outfit from 0 to 100 based on how well it fits the selected style. It highlights what works, suggests style-specific improvements, and delivers everything through a Gradio web interface.

---

## 🎯 Vision

Fashion should be accessible to everyone, but it's often gatekept by obscure rules and elitist standards. This project aims to break those barriers by promoting fashion literacy and helping anyone understand how to dress well - regardless of background or body type. We've trained models to objectively assess outfit quality based on built-in parameters like proportions.

Our project scores how well an outfit complements your unique body proportions - not society's idea of the "ideal" body. The goal is to make fashion more inclusive, helping people feel confident and informed in their style choices, no matter what style they choose.

---

## 🧠 How It Works

FitMaxx uses a lightweight **Multilayer Perceptron (MLP) classifier** powered by **CLIP** features to:

1. **Learn outfit quality** by training on labeled images organized by style and quality tier (`high`, `med`, `low`).
2. **Predict outfit quality** by extracting semantic and body-proportion features from a new outfit photo and scoring it based on learned patterns.

FitMaxx includes a proportion-aware scoring module. It complements the aesthetic score by evaluating **body proportions and clothing balance**, using pose estimate and heuristic-based scoring. It computes a **boosted balance score** based on how well the user dresses to their body proportions.

---

## 🗂️ Project Structure

This repo comes with sample model weights based on the **Formal** and **Casual** styles to play around with.

```plaintext
fitmaxx/
├── frontend/
│ └── ui.py # Gradio interface
├── models/
│ ├── train.py               # Train model on images in outfits/<style>/
│ ├── predict.py             # Core prediction logic
│ ├── langtest.py            # Pose/body proportion helper
│ ├── proportion*            # Body proportion model files
│ └── *.pt # Model weights
├── outfits/
│ └── <style>/               # e.g. casual/, formal/, vintage/
│       ├── high/            # Good outfits for that style
│       ├── med/             # Decent outfits for that style
│       └── low/             # Bad outfits for that style
├── requirements.txt
├── .gitignore 
└── README.md 
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/donmaruko/fitmaxx.git
cd fitmaxx

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."

# Launch the UI
python frontend/ui.py
```

Then open `http://127.0.0.1:7860` in your browser. Try it with sample images of you in your nicest outfit and let the grader run!

---

## 🛠️ Train Your Own Style (Modular)

FitMaxx is style-agnostic - you can teach it any aesthetic by giving it data:

1. Create a new directory under `outfits/`, for example `outfits/athleisure/`

2. Add images into `high/`, `med/`, `low/` subfolders
> These represent good, bad, and mediocre examples of outfits in that style.

3. Train the model:
```bash
python models/train.py athleisure
```
This would create its own weight files within `models/`

4. Update the `predict.py` and `ui.py` files to accommodate this new style
> WORK IN PROGRESS

Now your model will know how to evaluate athleisure outfits!

---

## 🔧 Built With

| Technologies             | Purpose                                                |
|--------------------------|--------------------------------------------------------|
| **CLIP (ViT-B/32)**      | Pretrained vision model for feature extraction         |
| **LangChain**            | Extract body ratio data                                |
| **NumPy and PIL**        | Image loading, feature concatenation                   |
| **MediaPipe**            | Pose analysis for proportion-aware feedback            |
| **PyTorch**              | MLP training and inference                             |
| **scikit-learn**         | Train/test split, weighted sampling, classification    |
| **rembg**                | Background removal for isolation                       |
| **Gradio**               | Elegant web UI                                         |
| **Pillow / torchvision** | Image I/O and preprocessing                            |
| **OpenAI API**           | Vision-based feedback rationalization                  |

---

## 📌 TODO / Future Ideas

- Add support for adding new styles and improve modularity
- Wrap the pipeline in a scalable API service to support async training
- Let users upload new training data through the UI
- Visualization of score attribution (heatmaps / CAM)
- Batch scoring for image folders
- Let users define a new style using natural language descriptions instead of image folders
- Use a vector database to store outfit embeddings for similarity search, custom style matching, or per-user models
- Use self-supervised learning to pretrain outfit embeddings on unlabeled data to reduce dependency on manual labels

---
