"""Seated logic: geometry + chair tracking/memory + per-frame seated decision.

Two jobs live here:

1. ChairTracker — give every physical chair a stable id across frames, and
   remember a chair's last box for a while when it disappears (a seated
   person hides it). This is what stops chairs from "vanishing" mid-demo.

2. seated_overlap — decide if a person box covers a chair box enough to say
   "someone is sitting there".

No drawing, no timers — those live in other modules.
"""

import config


# ---------------------------------------------------------------- geometry
def intersection_area(a, b):
    """Pixel area where two (x1, y1, x2, y2) boxes overlap (0 if none)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap_ratio(person_box, chair_box):
    """Fraction of the CHAIR covered by the person (0..1).

    Using chair area as the denominator is deliberate: when someone sits,
    their body covers most of the chair even if their full bounding box
    (head to toe) is much larger.
    """
    ca = box_area(chair_box)
    if ca <= 0:
        return 0.0
    return intersection_area(person_box, chair_box) / ca


def iou(a, b):
    """Standard intersection-over-union, used to match chairs frame-to-frame."""
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def is_seated(person_box, chair_box, ratio=None):
    ratio = config.OVERLAP_RATIO if ratio is None else ratio
    return overlap_ratio(person_box, chair_box) >= ratio


# ---------------------------------------------------------------- tracking
class ChairTracker:
    """Track chairs over time and remember chairs hidden by a seated person.

    Each frame we get fresh chair boxes. We match them to existing tracked
    chairs by IoU. Unmatched tracked chairs are NOT dropped instantly — they
    are kept (with their last box) for ``CHAIR_MEMORY_FRAMES`` frames so a
    person sitting down (who hides the chair) does not delete the seat.
    """

    def __init__(self, iou_match=0.25, memory_frames=None):
        self.iou_match = iou_match
        self.memory_frames = (config.CHAIR_MEMORY_FRAMES
                              if memory_frames is None else memory_frames)
        self._next_id = 1
        # id -> {"box": (x1,y1,x2,y2), "seen": frames since last detected}
        self.tracks = {}

    def update(self, detected_chairs):
        """Feed this frame's chair boxes; return list of current tracks.

        Returns dicts: {id, box, detected(bool)} where detected=True means a
        chair box was seen this frame, False means we are relying on memory.
        """
        det_boxes = [c["box"] for c in detected_chairs]
        used = set()

        # 1) match existing tracks to detections (greedy best-IoU)
        for tid, tr in self.tracks.items():
            best, best_iou = -1, self.iou_match
            for di, db in enumerate(det_boxes):
                if di in used:
                    continue
                v = iou(tr["box"], db)
                if v > best_iou:
                    best, best_iou = di, v
            if best >= 0:
                tr["box"] = det_boxes[best]
                tr["seen"] = 0
                used.add(best)
            else:
                tr["seen"] += 1

        # 2) create new tracks for unmatched detections
        for di, db in enumerate(det_boxes):
            if di in used:
                continue
            self.tracks[self._next_id] = {
                "box": db, "seen": 0,
            }
            self._next_id += 1

        # 3) drop tracks lost too long (older than memory window)
        dead = [tid for tid, tr in self.tracks.items()
                if tr["seen"] > self.memory_frames]
        for tid in dead:
            del self.tracks[tid]

        # 4) safety cap: never remember more than CHAIR_MEMORY_MAX lost chairs
        lost = [(tid, tr) for tid, tr in self.tracks.items() if tr["seen"] > 0]
        if len(lost) > config.CHAIR_MEMORY_MAX:
            # forget the oldest-lost ones first
            lost.sort(key=lambda kv: kv[1]["seen"], reverse=True)
            for tid, _ in lost[config.CHAIR_MEMORY_MAX:]:
                self.tracks.pop(tid, None)

        return [
            {"id": tid, "box": tr["box"], "detected": tr["seen"] == 0}
            for tid, tr in self.tracks.items()
        ]


def best_person_overlap(chair_box, person_boxes, ratio=None):
    """Return (seated_bool, best_ratio) of any person covering the chair."""
    ratio = config.OVERLAP_RATIO if ratio is None else ratio
    best = 0.0
    for pb in person_boxes:
        r = overlap_ratio(pb, chair_box)
        if r > best:
            best = r
    return best >= ratio, best
