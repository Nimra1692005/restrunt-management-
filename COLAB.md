# Run the demo on Google Colab (no laptop install, no storage needed)

Colab gives you a free Python + GPU/CPU in the browser. You do **not** install
anything on your laptop. Because Colab has no video window, we process a
**video file** and play back an annotated `output.mp4` in the notebook.

---

## 1. Get the code into Colab

**Easiest — upload a zip:**
1. On your laptop, right-click the `smart_table` folder → *Send to → Compressed (zipped) folder*.
2. Open a new [Colab notebook](https://colab.research.google.com/).
3. Left sidebar → **Files** → **Upload to session storage** → pick the `.zip`.
4. Run this cell to unzip:

```python
import zipfile, os
zipfile.ZipFile("smart_table.zip").extractall(".")
os.chdir("smart_table")            # move into the project folder
!ls
```

---

## 2. Install the libraries (runs on Colab's disk, not yours)

```python
!pip install -q ultralytics opencv-python-headless
```

First import downloads `yolov8n.pt` (~6 MB) automatically — no action needed.

---

## 3. Give Colab a video to watch

Pick **one**:

**Option A — upload a clip** you recorded on your phone (recommended for the
demo): sidebar → Files → Upload `clip.mp4`.

**Option B — grab a short stock clip** of people sitting at café tables:

```python
# a public sample clip (internet required)
!wget -q -O clip.mp4 https://media.w3.org/2011/02/demo-ffmpeg-bunny.mp4
```
*(bunny is just a smoke test of the pipeline; for a real demo use your café clip.)*

---

## 4. Run the Smart Table logic

```python
from colab_run import run_video
stats = run_video("clip.mp4", "output.mp4")
print(stats)   # frames processed, seated events, time, output file
```

---

## 5. Watch the annotated result + the event log

```python
from IPython.display import Video, display, HTML
display(Video("output.mp4", embed=True, width=720))

# show the CSV event log as a small table
import pandas as pd
display(pd.read_csv("events.csv"))
```

Green box = free chair, red box = seated with a running `mm:ss` timer.

---

## Tips / troubleshooting

- **Slow?** Run on CPU is fine for a short clip. For faster, menu *Runtime →
  Change runtime type → TPU/GPU*, but our model runs CPU-only here already.
- **`output.mp4` won't play in notebook:** the script auto-falls back to
  `output.avi`; download it and open in VLC, or convert with `!ffmpeg`.
- **No café clip on hand:** record 30–60 s on your phone (2–4 chairs, one
  person sits, one walks past, then leaves) and upload it — this is also the
  test media Roles 3/6 wanted.
- This Colab run uses the **identical** `detection.py / seated_logic.py /
  state.py / drawing.py`, so tuning `config.py` affects both desktop and Colab.
