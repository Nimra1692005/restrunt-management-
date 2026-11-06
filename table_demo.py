#!/usr/bin/env python3
"""Smart Table Management demo — starter entry point.

Thin wrapper around main.py so you can just run:

    python table_demo.py                 # default webcam
    python table_demo.py --camera 1      # another webcam
    python table_demo.py --video clip.mp4

All the real logic lives in the modules next to this file:
    config.py       -> every tunable setting
    detection.py    -> YOLOv8 person/chair detection
    seated_logic.py -> overlap math + chair tracking/memory
    state.py        -> seated smoothing + timers
    drawing.py      -> green/red boxes and on-screen timers
    logging_util.py -> CSV event log
    main.py         -> the frame loop
"""

import sys

from main import main

if __name__ == "__main__":
    sys.exit(main())
