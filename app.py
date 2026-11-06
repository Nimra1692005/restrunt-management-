"""Smart Table — web app (Gradio).

Ek hi app me teen tareeke:
  1. Live webcam  → browser ka camera, har frame analyze, green/red + timer.
  2. Photo/shot   → ek tasveer upload ya camera snap, usi waqt result.
  3. Video clip   → pura clip process hokar annotated video.

Ye file repo ke root par chalti hai aur bache hue modules reuse karti hai:
  detection.py / seated_logic.py / state.py / drawing.py / colab_run.py

Deploy: Hugging Face Spaces (SDK = Gradio) — README/DEPLOY.md me steps.
Local test:  pip install gradio && python app.py
"""

import os
import time

import gradio as gr

import detection
import drawing
from seated_logic import ChairTracker, best_person_overlap
from state import TableManager
from colab_run import run_video


# ------------------------------------------------------------------ single photo
_det = None


def _detector():
    global _det
    if _det is None:
        _det = detection.Detector()
    return _det


def analyze_photo(img):
    """Ek BGR ndarray (photo/snap) -> annotated image + text result."""
    if img is None:
        return None, "Koi image nahi aayi."
    det = _detector()
    persons, chairs = det.detect(img)

    now = time.time()
    seats = {}
    seated_count = 0
    for i, ch in enumerate(chairs, start=1):
        seated, ratio = best_person_overlap(ch["box"], [p["box"] for p in persons])
        seats[i] = _FakeSeat(seated, ch["box"], now, ratio)
        if seated:
            seated_count += 1
    tracks = [{"id": i, "box": c["box"], "detected": True}
              for i, c in enumerate(chairs, start=1)]

    drawing.draw(img, tracks, seats, {}, persons, now)
    summary = (f"Chairs: {len(chairs)} | Seated: {seated_count} | "
               f"Free: {len(chairs) - seated_count} | Persons: {len(persons)}")
    return img, summary


class _FakeSeat:
    """Single-photo ke liye halka seat object (drawing.draw ko chahiye)."""
    def __init__(self, seated, box, now, ratio):
        self.seated = seated
        self.box = box
        self._since = now if seated else None
        self.ratio = ratio

    def raw_elapsed(self, now=None):
        return 0.0  # ek photo me time ka matlab nahi


# ------------------------------------------------------------------ live webcam
# Persistent state — taake smoothing + timer frames ke beech zinda rahe.
_live = {"tracker": None, "manager": None, "det": None, "reset_at": 0.0}


def _reset_live_if_idle():
    if time.time() - _live["reset_at"] > 60:
        _live["tracker"] = ChairTracker()
        _live["manager"] = TableManager()
    _live["reset_at"] = time.time()


def analyze_webcam(frame):
    """Gradio webcam: har frame aata hai; annotated frame wapas karo."""
    if frame is None:
        return None
    _reset_live_if_idle()
    if _live["det"] is None:
        _live["det"] = _detector()
    if _live["tracker"] is None:
        _live["tracker"] = ChairTracker()
        _live["manager"] = TableManager()

    now = time.time()
    persons, chairs = _live["det"].detect(frame)
    tracks = _live["tracker"].update(chairs)
    _live["manager"].update(tracks, persons, now=now)
    drawing.draw(frame, tracks, _live["manager"].seats,
                 _live["manager"].tables, persons, now)
    return frame


# ------------------------------------------------------------------ video clip
def analyze_video(path):
    if not path or not os.path.exists(path):
        return None, "Video file nahi mila."
    out = "/tmp/output.mp4" if os.name != "nt" else "output.mp4"
    stats = run_video(path, out, log_path="events.csv")
    return out, f"Done: {stats}"


# ------------------------------------------------------------------ UI
def build_ui():
    with gr.Blocks(title="Smart Table Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🍽️ Smart Table Management — CV Demo")
        gr.Markdown("Green box = free chair · Red box = seated · "
                    "Timer = kitni der se seated hai.")

        with gr.Tab("📸 Photo / Snapshot"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="numpy",
                                      sources=["upload", "webcam"],
                                      label="Upload ya camera se lo")
                    run_img = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="Result")
                    txt_out = gr.Textbox(label="Summary")
            run_img.click(analyze_photo, img_in, [img_out, txt_out])

        with gr.Tab("🎥 Live Webcam"):
            gr.Markdown("Camera permission **Allow** karo. State frames ke beech "
                        "rehti hai, is liye **timer chalega**.")
            cam = gr.Webcam(stream=True, label="Live camera")
            live_out = gr.Image(label="Annotated live feed")
            cam.stream(analyze_webcam, cam, live_out, cache_examples=False)

        with gr.Tab("🎬 Video clip"):
            vid_in = gr.Video(label="Upload clip (mp4)")
            run_vid = gr.Button("Process video")
            vid_out = gr.Video(label="Annotated output")
            vid_txt = gr.Textbox(label="Stats")
            run_vid.click(analyze_video, vid_in, [vid_out, vid_txt])

    return demo


# Module-level `demo` — Hugging Face Spaces (Gradio SDK) ise automatically
# dhoondhta hai aur chala deta hai. `python app.py` bhi chalta hai.
demo = build_ui()

if __name__ == "__main__":
    demo.launch()
