"""Detection layer: turn a raw video frame into person boxes and chair boxes.

Only YOLOv8 (via ultralytics) is used, filtered to the COCO classes
`person` (0) and `chair` (56). Nothing here knows about timers or drawing.
"""

import cv2
from ultralytics import YOLO

import config

COCO_PERSON = 0
COCO_CHAIR = 56


class Detector:
    """Wrap a YOLOv8 model and expose clean (persons, chairs) per frame."""

    def __init__(self, model_name=None, device=None):
        self.model = YOLO(model_name or config.MODEL_NAME)
        self.device = device or config.DEVICE

    def detect(self, frame):
        """Return (persons, chairs): lists of dicts with box, conf."""
        results = self.model.predict(
            frame,
            conf=0.20,               # low floor; we re-filter per class below
            imgsz=config.IMGSZ,
            device=self.device,
            verbose=False,
        )
        persons, chairs = [], []
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return persons, chairs
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(cls)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            item = {"box": (x1, y1, x2, y2), "conf": float(conf[i])}
            if cls[i] == COCO_PERSON and conf[i] >= config.CONF_PERSON:
                persons.append(item)
            elif cls[i] == COCO_CHAIR and conf[i] >= config.CONF_CHAIR:
                chairs.append(item)
        return persons, chairs


def open_source(index=None, video=None):
    """Open a webcam or a video file. Returns a cv2.VideoCapture or None."""
    if video:
        cap = cv2.VideoCapture(video)
    else:
        cap = cv2.VideoCapture(index if index is not None else config.CAMERA_INDEX)
    if not cap.isOpened():
        return None
    return cap


def read_frame(cap):
    """Read one frame. Return an BGR ndarray, or None on failure/blank."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    # Guard against an all-zero blank frame some drivers return on warm-up.
    if not frame.any():
        return None
    return frame
