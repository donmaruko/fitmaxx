# frontend/ui.py
import os, re, subprocess, pathlib, gradio as gr

# ---------- paths ----------
ROOT       = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "models"
CNNSCRIPT  = MODEL_DIR / "cnnpredict.py"

# ---------- inference helper ----------
def grade_fit(img_path: str, style: str) -> str:
    if not img_path:
        return "### ❌ Please upload an image."

    img_path = pathlib.Path(img_path).resolve()

    proc = subprocess.run(
        ["python", str(CNNSCRIPT), str(img_path), style],
        text=True, capture_output=True,
        cwd=MODEL_DIR,
        env=os.environ,
    )

    if proc.returncode != 0:
        return f"### ❌ Error\n```\n{proc.stderr}\n```"

    md = proc.stdout

    # ── POST‑PROCESS ─────────────────────────────────────────
    # 1. drop the camera‑emoji “Analyzing:” line
    md = re.sub(r"(?m)^🖼️.*\n?", "", md)

    # 2. inject hard line‑breaks (<br>) before each header token
    #    so they render on separate lines
    md = re.sub(r" +💧",  "  \n💧",  md)   # two spaces → hard <br>
    md = re.sub(r" +👍",  "  \n👍",  md)
    md = re.sub(r" +🚨",  "  \n🚨",  md)
    md = re.sub(r" +❗",  "  \n❗",  md)
    # Ensure "Style" and "Drip Score" are on separate lines
    md = re.sub(r"(🎯 Style: .+?) (💧 Drip Score: .+)", r"\1  \n\2", md)

    # 3. add an extra blank line after the feedback line for spacing
    md = re.sub(r"(💧.*|👍.*|🚨.*|❗.*)$", r"\1\n", md, flags=re.M)

    return md

# ---------- UI theme & CSS ----------
THEME = (
    gr.themes.Soft(primary_hue="purple")
    .set(body_text_color="#EDEDED", body_background_fill="#1e1e1e")
)

CSS = """
.gradio-container { max-width: 1100px; margin:auto; }
#title  { text-align:center; font-size:2.2rem; margin-bottom:.5em }
#result { border:1px solid #333; border-radius:.5rem; padding:1rem; }
"""

# ---------- Gradio app ----------
with gr.Blocks(title="FitMaxx", theme=THEME, css=CSS) as demo:
    gr.Markdown("👔👟 **FitMaxx** — *your personal outfit grader*", elem_id="title")

    with gr.Row(equal_height=True):
        img = gr.Image(
            type="filepath",
            label="Upload your outfit photo",
            interactive=True,
            height=600,
        )

        with gr.Column():
            style    = gr.Dropdown(
                ["formal", "casual"],
                value="formal",
                label="Which style are you going for?",
            )
            rate_btn = gr.Button("🚀 Rate my fit!")
            out_md   = gr.Markdown(elem_id="result")

    rate_btn.click(grade_fit, [img, style], out_md)

if __name__ == "__main__":
    demo.launch()
