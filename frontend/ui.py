# frontend/ui.py
import os, re, subprocess, pathlib, gradio as gr

# ---------- paths ----------
ROOT       = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "models"
CNNSCRIPT  = MODEL_DIR / "predict.py"

# ---------- inference helper ----------
def grade_fit(img_path: str, style: str) -> str:
    if not img_path:
        return "### ❌ Please upload an image."

    img_path = pathlib.Path(img_path).resolve()

    # Run predict.py
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
    md = re.sub(r"(?m)^🖼️.*\n?", "", md)
    md = re.sub(r" +💧",  "  \n💧",  md)
    md = re.sub(r" +👍",  "  \n👍",  md)
    md = re.sub(r" +🚨",  "  \n🚨",  md)
    md = re.sub(r" +❗",  "  \n❗",  md)
    md = re.sub(r"(🎯 Style: .+?) (💧 Drip Score: .+)", r"\1  \n\2", md)
    md = re.sub(r"(💧.*|👍.*|🚨.*|❗.*)$", r"\1\n", md, flags=re.M)

    # Run proportionmatch.py and append result
    prop_proc = subprocess.run(
        ["python", "proportionmatch.py", "--image", str(img_path), "--csv", "proportiondata.csv"],
        text=True, capture_output=True,
        cwd=MODEL_DIR,
        env=os.environ,
    )

    # Combine stdout and stderr lines
    all_lines = (prop_proc.stdout + prop_proc.stderr).strip().splitlines()

    if prop_proc.returncode == 0:
        for line in all_lines:
            if "Proportions are" in line:
                insertion = f"\n🧍 {line.strip()}\n"

                if "🧠 Fit Analysis" in md:
                    md = md.replace("🧠 Fit Analysis", insertion + "\n🧠 Fit Analysis")
                else:
                    md += insertion
                break
    else:
        md += "\n⚠️ Could not analyze proportions.\n"
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
    gr.Markdown("👔👟 **FitMaxx** — *your personal outfit wingman*", elem_id="title")

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
