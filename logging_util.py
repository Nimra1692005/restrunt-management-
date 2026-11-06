"""CSV event logging for seat state changes.

Only non-personal data is stored: chair id, table name (if any), the new
state, and timestamps. No faces, no images — per the project's privacy rule.
"""

import csv
import os
import time


class EventLogger:
    def __init__(self, path="events.csv"):
        self.path = path
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(
                ["timestamp", "wall_clock", "chair_id", "table", "state",
                 "duration_sec"])

    def log(self, chair_id, state, table=None, duration_sec=None):
        now = time.time()
        self._w.writerow([
            f"{now:.3f}",
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            chair_id, table or "", state,
            f"{duration_sec:.1f}" if duration_sec is not None else "",
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
