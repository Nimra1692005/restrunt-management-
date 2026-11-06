"""Application entry point / frame loop for the Smart Table demo.

Wires together: detection -> chair tracking -> seated state + timers ->
drawing -> CSV event logging. Run via ``python table_demo.py``.

Keys:  q / ESC = quit.
"""

import argparse
import sys
import time

import cv2

import config
import detection
import drawing
from seated_logic import ChairTracker
from state import TableManager
from logging_util import EventLogger


def build_table_lookup():
    """Map chair_id -> table name using config.TABLES (empty if ungrouped)."""
    lookup = {}
    for tname, ids in config.TABLES.items():
        for cid in ids:
            lookup[cid] = tname
    return lookup


def run(source_index=None, video=None, log_path="events.csv"):
    # --- model + source (with clear, friendly errors) ------------------
    try:
        detector = detection.Detector()
    except Exception as e:
        print(f"[error] could not load YOLO model '{config.MODEL_NAME}': {e}")
        print("        check your internet (first run downloads weights) "
              "or run: pip install -r requirements.txt")
        return 1

    cap = detection.open_source(index=source_index, video=video)
    if cap is None:
        where = video if video else f"camera index {source_index or config.CAMERA_INDEX}"
        print(f"[error] could not open video source: {where}")
        print("        try --camera 1, or pass --video path/to/clip.mp4")
        return 1

    tracker = ChairTracker()
    manager = TableManager()
    logger = EventLogger(log_path)
    prev_seated = {}            # chair_id -> bool (for change detection)
    table_of = build_table_lookup()

    source_label = "video" if video else "webcam"
    empty_streak = 0
    t_last_fps, frames_since_fps, fps = time.time(), 0, 0.0

    print("Running. Press 'q' or ESC to quit.")
    try:
        while True:
            frame = detection.read_frame(cap)
            if frame is None:
                empty_streak += 1
                if empty_streak >= config.MAX_EMPTY_FRAMES:
                    print("[warn] too many empty frames; stopping.")
                    break
                continue
            empty_streak = 0

            now = time.time()
            try:
                persons, chairs = detector.detect(frame)
            except Exception as e:
                # A single bad frame should not kill the whole demo.
                print(f"[warn] detection failed on a frame: {e}")
                continue
            tracks = tracker.update(chairs)
            active = manager.update(tracks, persons, now=now)
            manager.prune({tr["id"] for tr in tracks})

            # --- CSV events on seated/free transitions -----------------
            for cid, seat in active.items():
                was = prev_seated.get(cid, False)
                if seat.seated != was:
                    logger.log(
                        chair_id=cid,
                        state="seated" if seat.seated else "free",
                        table=table_of.get(cid),
                        # duration is meaningful on "free": how long they sat
                        duration_sec=(None if seat.seated
                                      else seat.last_duration),
                    )
                prev_seated[cid] = seat.seated

            # --- draw --------------------------------------------------
            drawing.draw(frame, tracks, manager.seats, manager.tables,
                         persons, now)
            drawing.draw_status(frame, fps, source_label)
            cv2.imshow(config.WINDOW_NAME, frame)

            # --- fps + key handling -----------------------------------
            frames_since_fps += 1
            if now - t_last_fps >= config.FPS_EVERY:
                fps = frames_since_fps / (now - t_last_fps)
                t_last_fps, frames_since_fps = now, 0

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="Smart Table management demo")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                   help="webcam index (default from config)")
    p.add_argument("--video", default=config.VIDEO_PATH,
                   help="path to a video file instead of a webcam")
    p.add_argument("--log", default="events.csv", help="CSV event log path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    return run(source_index=args.camera, video=args.video, log_path=args.log)


if __name__ == "__main__":
    sys.exit(main())
