"""Colab / headless adapter — run the demo with NO local install and NO GUI.

Why this exists: Google Colab has no OpenCV window (cv2.imshow) and cannot
open your laptop webcam directly. So instead of showing a live window, this
script reads a *video file you uploaded*, runs the exact same detection /
seated / timer logic, and writes an annotated ``output.mp4`` that you can
play right inside the notebook.

Typical Colab flow (see COLAB.md):
    1. upload the smart_table folder as a zip, unzip it
    2. !pip install ultralytics opencv-python-headless
    3. from colab_run import run_video
    4. run_video("clip.mp4", "output.mp4")   # then display output.mp4

It reuses every module from the desktop app, so the maths is identical.
"""

import time

import cv2

import config
import detection
import drawing
from seated_logic import ChairTracker
from state import TableManager
from logging_util import EventLogger


def run_video(input_path, output_path="output.mp4", log_path="events.csv",
              max_frames=None):
    """Process a video file and write an annotated copy. Returns stats dict."""
    cap = detection.open_source(video=input_path)
    if cap is None:
        raise FileNotFoundError(f"could not open video: {input_path}")

    # match output settings to the source
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
    if not out.isOpened():
        # mp4v missing on some builds -> fall back to AVI/MJPG so we never crash
        output_path = output_path.rsplit(".", 1)[0] + ".avi"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"),
                              src_fps, (w, h))

    detector = detection.Detector()
    tracker = ChairTracker()
    manager = TableManager()
    logger = EventLogger(log_path)
    prev_seated = {}

    processed = seated_frames = 0
    t0 = time.time()
    try:
        while True:
            frame = detection.read_frame(cap)
            if frame is None:
                break
            now = time.time()
            persons, chairs = detector.detect(frame)
            tracks = tracker.update(chairs)
            active = manager.update(tracks, persons, now=now)
            manager.prune({tr["id"] for tr in tracks})

            for cid, seat in active.items():
                was = prev_seated.get(cid, False)
                if seat.seated != was:
                    logger.log(cid, "seated" if seat.seated else "free",
                               None, None if seat.seated else seat.last_duration)
                prev_seated[cid] = seat.seated
                if seat.seated:
                    seated_frames += 1

            drawing.draw(frame, tracks, manager.seats, manager.tables,
                         persons, now)
            fps = (processed + 1) / max(now - t0, 1e-6)
            drawing.draw_status(frame, fps, f"file:{input_path}")
            out.write(frame)

            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        out.release()
        logger.close()

    return {
        "frames": processed,
        "seated_events": seated_frames,
        "elapsed_sec": round(time.time() - t0, 1),
        "output": output_path,
    }


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "clip.mp4"
    print(run_video(inp))
