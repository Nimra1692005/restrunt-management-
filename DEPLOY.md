# Deploy the Smart Table web app (free, shareable link)

Ye app teen kaam karti hai — **live webcam**, **photo analyze**, **video clip** —
sab browser me, ek link se. Sabse aasaan free host: **Hugging Face Spaces**.

---

## A) Hugging Face Spaces (recommended)

1. https://huggingface.co/new-space banao
   - **Space name:** `smart-table-demo`
   - **License:** MIT
   - **SDK:** **Gradio**
   - **Hardware:** free CPU
   - **Create**
2. Space banne ke baad **Files** tab → **Add files → Upload files**:
   ye poori `smart_table` folder ki files upload karo (ya repo connect karo, neeche):
   `app.py, config.py, detection.py, seated_logic.py, state.py, drawing.py,
   logging_util.py, colab_run.py, requirements.txt`
   - ⚠️ `*.pt`, `__pycache__`, `clip.mp4`, `events.csv` upload **mat** karo
     (weights khud download ho jayenge; media/logs privacy ke liye bahar).
3. `app.py` root par hona chahiye (Spaces Gradio SDK root se chalata hai).
4. **Show Details → Restart Space.** Thoda wait karo; `app.py` pehli baar
   `yolov8n.pt` download karega.
5. Link khulega: `https://<user>-smart-table-demo.hf.space` — ye khol ke **Allow
   camera** karo → Live Webcam tab me timer chalega.

### (Optional) GitHub repo se connect
Spaces settings me **Repository → Link to existing repository** se aap ka
`restrunt-management-` repo sync kar sakte ho — files dobara upload nahi karni.

---

## B) Locally test karne se pehle (kisi bhi machine pe)
```bash
pip install -r requirements.txt
python app.py
# browser me http://127.0.0.1:7860
```
Live webcam ke liye `http://localhost` par browser camera permission chahiye.

---

## Kya-kya milega
| Tab | Kaam |
|---|---|
| 📸 Photo / Snapshot | Ek tasveer upload → green/red boxes + "Seated: X / Free: Y" summary |
| 🎥 Live Webcam | Browser camera → har frame analyze, **timer frames ke beech chalta hai** |
| 🎬 Video clip | pura clip daalo → annotated `output.mp4` wapas |

---

## Troubleshooting
- **Space build error / import fail:** `requirements.txt` me `gradio` aur
  `opencv-python-headless` hone chahiye (hain). Purani `.pyc`/`.pt` upload mat karo.
- **Webcam black on Spaces:** browser permission block na ho; HTTPS (Spaces) pe
  camera chalta hai. Chrome/Edge behtar.
- **Slow (~3–5 FPS):** normal hai (free CPU). Frames kam, magar demo chalega.
- **Live timer na badhe:** `config.SMOOTH_ON` 1–2 kar do taake jaldi red ho.
