"""Drawing / UI layer. Takes state and paints boxes, labels and timers.

Green box = free chair, red box = seated chair. Each seated chair (and each
seated table) shows a mm:ss timer. Person boxes are optional (debug).
"""

import cv2

import config

GREEN = (60, 200, 60)
RED = (40, 40, 220)
BLUE = (255, 150, 40)      # person debug boxes
GRAY = (150, 150, 150)     # remembered-but-currently-hidden chair (dashed look)


def fmt_timer(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _put(frame, text, x, y, color, scale=0.6, thick=2):
    """Draw text with a dark background rectangle so it stays readable."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(0, min(x, frame.shape[1] - tw - 4))
    y = max(th + 4, y)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, tracks, seats, tables, persons, now):
    """Paint everything onto `frame` (modified in place)."""
    # chairs / seats
    for tr in tracks:
        cid = tr["id"]
        seat = seats.get(cid)
        x1, y1, x2, y2 = tr["box"]
        if seat is not None and seat.seated:
            color = RED
            label = f"seat{cid} {fmt_timer(seat.raw_elapsed(now))}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        elif not tr["detected"]:
            # chair currently hidden but remembered, not seated
            color = GRAY
            _draw_dashed_rect(frame, (x1, y1), (x2, y2), color)
            label = f"seat{cid} ?"
        else:
            color = GREEN
            label = f"seat{cid} free"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        _put(frame, label, x1, y1, color)

    # tables: outline the bounding box of their chairs + one timer
    for t in tables.values():
        boxes = [seats[cid].box for cid in t.chair_ids
                 if cid in seats and seats[cid].box is not None]
        if not boxes:
            continue
        gx1 = min(b[0] for b in boxes)
        gy1 = min(b[1] for b in boxes)
        gx2 = max(b[2] for b in boxes)
        gy2 = max(b[3] for b in boxes)
        color = RED if t.seated else GREEN
        _draw_dashed_rect(frame, (gx1, gy1), (gx2, gy2), color, step=16)
        txt = f"{t.name} {fmt_timer(t.elapsed(now))}" if t.seated else f"{t.name} free"
        _put(frame, txt, gx1, gy1 - 24, color, scale=0.7)

    # optional person debug boxes
    if config.SHOW_DETECTIONS:
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 1)

    return frame


def _draw_dashed_rect(frame, pt1, pt2, color, thickness=2, step=10):
    (x1, y1), (x2, y2) = pt1, pt2
    for x in range(x1, x2, step * 2):
        cv2.line(frame, (x, y1), (min(x + step, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + step, x2), y2), color, thickness)
    for y in range(y1, y2, step * 2):
        cv2.line(frame, (x1, y), (x1, min(y + step, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + step, y2)), color, thickness)


def draw_status(frame, fps, source_label):
    _put(frame, f"{source_label}  {fps:4.1f} FPS", 8, 24, (255, 255, 255), 0.6, 1)
    return frame
