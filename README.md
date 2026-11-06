# Smart Table Management — Computer Vision Demo

A webcam watches a restaurant's chairs and tables. The system tells you, live,
whether each seat is **free (green box)** or **occupied / seated (red box)**,
and starts a **timer** the moment someone sits down.

> Scope of this demo only. Planned later (not here): "waiter not arrived in
> 5 min" alerts, "table not cleaned" alerts, and a host view of which table
> frees up first.

---

## How it works (plain words)

1. **Detect** — YOLOv8 finds every *person* and *chair* in each video frame.
2. **Track** — each physical chair gets a stable id. If a seated person hides
   the chair, the system *remembers* its last position for a few seconds so the
   seat (and its timer) does not disappear.
3. **Decide "seated"** — a chair is seated when a person's box covers enough of
   the chair box. The answer is smoothed over several frames so it never flickers.
4. **Time it** — the timer starts the frame a chair flips to *seated* and resets
   when it flips back to *free*.
5. **Draw + log** — green/red boxes and `mm:ss` timers are drawn; every sit/leave
   is appended to `events.csv`.

No faces or customer images are stored — only ids, timers and event rows.

---

## Install

Requires Python 3.9+ .

```bash
cd smart_table
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

On a laptop **without a GPU**, install the CPU build of PyTorch first so the
download stays small:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first run auto-downloads `yolov8n.pt` (the smallest YOLOv8, ~6 MB). Later
runs work offline.

---

## Run

```bash
python table_demo.py                 # default webcam (index 0)
python table_demo.py --camera 1      # a different webcam
python table_demo.py --video clip.mp4   # use a recorded video instead
python table_demo.py --log my_events.csv  # custom CSV log path
```

Press **q** or **ESC** to quit.

### No space to install Python? Use Google Colab
If your laptop can't install Python/torch (disk space), run the whole demo in
the browser on free **Google Colab** — nothing is installed on your machine.
See [`COLAB.md`](COLAB.md) for copy-paste cells. It processes an uploaded video
and shows an annotated `output.mp4` (same logic, no webcam window needed).

---

## Files

| File | Responsibility |
|---|---|
| `table_demo.py` | entry point (thin wrapper) |
| `main.py` | the frame loop: wire everything together |
| `config.py` | **all tunable settings live here** |
| `detection.py` | YOLOv8 person/chair detection + opening camera/video |
| `seated_logic.py` | box overlap math + chair tracking/memory |
| `state.py` | seated smoothing + per-chair/per-table timers |
| `drawing.py` | green/red boxes, on-screen timers, status line |
| `logging_util.py` | CSV event log |
| `colab_run.py` | headless (no-window) video runner for Colab |
| `app.py` | **web app (Gradio): live webcam + photo + video** |
| `requirements.txt` | dependencies |

Docs: `README.md` (ye file), `COLAB.md` (Colab run steps), `DEPLOY.md` (Hugging
Face Spaces pe web app deploy karna).

---

## Tuning quick guide (edit `config.py`)

- Seat flickers on/off → raise `SMOOTH_ON` / `SMOOTH_OFF`.
- Standing people counted as seated → raise `OVERLAP_RATIO`.
- Hidden chairs lost too quickly → raise `CHAIR_MEMORY_FRAMES`.
- Wrong/missing chairs → lower `CONF_CHAIR` slightly.
- Group chairs into tables → set `TABLES = {"T1": [1, 2, 3]}` after you see the
  on-screen chair ids.

---

## Tests

Automated tests (no camera needed) live in `tests/`:

```bash
pip install pytest
pytest -v
```
