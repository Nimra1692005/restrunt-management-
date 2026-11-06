"""Central configuration for the Smart Table demo.

Every tunable knob lives here so behaviour can be adjusted in one place
without touching the logic files.
"""

# ---------------------------------------------------------------- model
MODEL_NAME = "yolov8n.pt"   # smallest YOLOv8 — runs fine on a laptop CPU
DEVICE = "cpu"              # "cpu" or 0 for CUDA
IMGSZ = 640                 # inference input size; 640 is a good CPU speed/accuracy trade-off
CONF_PERSON = 0.35          # min confidence for a person detection
CONF_CHAIR = 0.25           # min confidence for a chair detection (chairs are hard)

# ---------------------------------------------------------------- sources
CAMERA_INDEX = 0            # default webcam
VIDEO_PATH = None           # set to a file path (or pass on command line) to use a video
MAX_EMPTY_FRAMES = 30       # give up if this many blank/undecodable frames in a row

# ---------------------------------------------------------------- seated logic
OVERLAP_RATIO = 0.45       # intersection area / chair area needed to call a chair seated
                              # (0.45 avoids counting a person merely STANDING in
                              #  front of a chair; a seated body covers >50% of the seat)
SMOOTH_ON = 3               # consecutive "seated" frames before a chair flips red
SMOOTH_OFF = 6              # consecutive "free" frames before a chair flips green
                              # (OFF > ON so brief detection drops don't reset timers)

# ---------------------------------------------------------------- chair memory
# A seated person often hides the chair completely. We keep the chair's last
# known box for this many frames so the seated state (and timer) survives.
CHAIR_MEMORY_FRAMES = 90    # ~3 s at 30 fps
CHAIR_MEMORY_MAX = 6        # never remember more than this many lost chairs (safety)

# ---------------------------------------------------------------- tables
# YOLO detects chairs reliably but not "your restaurant's table".
# A table = a named group of chair ids. Chair ids are stable per physical
# chair, assigned by the ChairTracker in seated_logic.py (IoU matching).
# For a first demo leave TABLES empty (each chair stands alone). Once the
# chair ids are visible on screen, group them here, e.g. {"T1": [1, 2, 3]}.
TABLES = {}                # e.g. {"T1": [1, 2, 3], "T2": [4, 5]}

# ---------------------------------------------------------------- drawing
WINDOW_NAME = "Smart Table demo"
SHOW_DETECTIONS = True      # draw raw person boxes too (debug aid, toggle off on stage)
FPS_EVERY = 1.0             # print FPS once per second instead of every frame
