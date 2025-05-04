# 👔👟 FitMaxx — Your Personal AI Outfit Wingman

FitMaxx is a modular AI web app that analyzes outfit photos and rates how well they match a chosen **style** — such as `formal`, `casual`, or even custom aesthetics like `alternative` or `techwear`.

The AI gives you a 0–100 “drip score,” explains what works, and suggests specific improvements — all through a clean Gradio web interface.

---

## 🧠 How It Works

FitMaxx uses a lightweight Convolutional Neural Network (CNN) combined with a classifier to:

1. **Learn outfit quality** by training on labeled images organized by style and quality tier (`high`, `med`, `low`).
2. **Predict outfit quality** by extracting features from a new outfit photo and scoring it against what it learned.

---

## 🗂️ Project Structure

This repo comes with sample model weights based on the **Formal** and **Casual** styles to play around with.

```plaintext
fitmaxx/
├── frontend/
│ └── ui.py # Gradio interface
├── models/
│ ├── train.py # Train model on images in outfits/<style>/
│ ├── predict.py # Core prediction logic
│ ├── langtest.py # Pose/body proportion helper
│ └── *.pt # Model weights
├── outfits/
│ └── <style>/ # e.g. casual/, formal/, vintage/
│       ├── high/ # Good outfits for that style
│       ├── med/  # Decent outfits for that style
│       └── low/  # Bad outfits for that style
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
- These represent good, bad, and mediocre examples of outfits in that style.

3. Train the model:
```bash
python models/train.py athleisure
```
- This would create its own weight files within `models/`

4. Update the `predict.py` and `ui.py` files to accommodate this new style
- WORK IN PROGRESS

Now your model will know how to evaluate athleisure outfits!

---

## 🔧 Built With

- **PyTorch** - MLP training and inference
- **Gradio** - Elegant web UI
- **Pillow / torchvision** - image I/O and preprocessing
- **OpenAI API** - Feedback rationalization

---

## 📌 TODO / Future Ideas

- Add support for adding new styles and improve modularity

- Let users upload new training data through the UI

- Visualization of score attribution (heatmaps / CAM)

- Batch scoring for image folders

---