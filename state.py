"""Timers and per-seat state machine (with temporal smoothing).

A SeatState holds, for one tracked chair id:
  * seated (the stable, smoothed boolean we display)
  * on_run / off_run (consecutive raw frames arguing for seated / free)
  * seated_since (wall-clock time the current sit started)

We only flip ``seated`` on when the raw signal has agreed for SMOOTH_ON
frames, and only flip it off after SMOOTH_OFF frames — this kills the
flicker you get when YOLO occasionally misses a person or chair.

TableState groups chairs into tables (config.TABLES) and runs a table-level
timer that is active while ANY of its chairs is seated.
"""

import time

import config


class SeatState:
    def __init__(self, chair_id):
        self.chair_id = chair_id
        self.seated = False
        self.on_run = 0
        self.off_run = 0
        self.seated_since = None   # time.time() when seated flipped True
        self.last_duration = 0.0   # seconds of the most recent completed sit
        self.box = None            # last known chair box (for drawing)

    def raw_elapsed(self, now=None):
        """Seconds the current seat has been held, or 0 if free."""
        if not self.seated or self.seated_since is None:
            return 0.0
        return ((now or time.time()) - self.seated_since)

    def update(self, raw_seated, box, now=None):
        """Feed this frame's raw seated decision; update smoothed state.

        Returns True if the stable seated state CHANGED this frame (so callers
        can log a seat/leave event).
        """
        now = now or time.time()
        self.box = box
        changed = False
        if raw_seated:
            self.on_run += 1
            self.off_run = 0
            if not self.seated and self.on_run >= config.SMOOTH_ON:
                self.seated = True
                self.seated_since = now
                changed = True
        else:
            self.off_run += 1
            self.on_run = 0
            if self.seated and self.off_run >= config.SMOOTH_OFF:
                # remember how long this sit lasted before clearing it
                self.last_duration = now - (self.seated_since or now)
                self.seated = False
                self.seated_since = None
                changed = True
        return changed


class TableState:
    """Timer for a named group of chairs."""

    def __init__(self, name, chair_ids):
        self.name = name
        self.chair_ids = set(chair_ids)
        self.seated = False
        self.seated_since = None

    def update(self, seats, now=None):
        now = now or time.time()
        any_seated = any(seats[cid].seated for cid in self.chair_ids if cid in seats)
        if any_seated and not self.seated:
            self.seated = True
            self.seated_since = now
        elif not any_seated and self.seated:
            self.seated = False
            self.seated_since = None

    def elapsed(self, now=None):
        if not self.seated or self.seated_since is None:
            return 0.0
        return ((now or time.time()) - self.seated_since)


class TableManager:
    """Owns all SeatState + TableState objects and advances them each frame."""

    def __init__(self, tables=None):
        self.tables_cfg = tables if tables is not None else config.TABLES
        self.seats = {}     # chair_id -> SeatState
        self.tables = {
            name: TableState(name, ids) for name, ids in self.tables_cfg.items()
        }

    def _get_seat(self, chair_id):
        if chair_id not in self.seats:
            self.seats[chair_id] = SeatState(chair_id)
        return self.seats[chair_id]

    def update(self, tracks, persons, now=None):
        """tracks: from ChairTracker; persons: list of person dicts.

        Returns dict chair_id -> SeatState for the chairs seen this frame.
        """
        now = now or time.time()
        person_boxes = [p["box"] for p in persons]
        active = {}
        # import here to avoid a circular import at module load
        from seated_logic import best_person_overlap

        for tr in tracks:
            seat = self._get_seat(tr["id"])
            raw, _ratio = best_person_overlap(tr["box"], person_boxes)
            seat.update(raw, tr["box"], now=now)
            active[tr["id"]] = seat

        for t in self.tables.values():
            t.update(self.seats, now=now)
        return active

    def prune(self, live_ids):
        """Forget seats whose chair track is gone (keeps memory bounded)."""
        for cid in list(self.seats.keys()):
            if cid not in live_ids and not self.seats[cid].seated:
                del self.seats[cid]
