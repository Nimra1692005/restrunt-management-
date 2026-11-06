"""Automated tests for the pure logic (fake boxes only — no camera/YOLO).

Run:  pytest -v
"""

import config
import seated_logic as sl
from state import SeatState


# ------------------------------------------------------ overlap math
def test_intersection_area_basic():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert sl.intersection_area(a, b) == 25  # 5x5 overlap


def test_intersection_area_no_overlap():
    assert sl.intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0


def test_box_area_and_zero_guard():
    assert sl.box_area((0, 0, 4, 5)) == 20
    # inverted box -> area clamped to 0, overlap_ratio must not crash
    assert sl.box_area((10, 10, 5, 5)) == 0
    assert sl.overlap_ratio((0, 0, 10, 10), (5, 5, 0, 0)) == 0.0


def test_iou_identical_is_one():
    box = (10, 10, 20, 20)
    assert sl.iou(box, box) == 1.0


def test_overlap_ratio_chair_fully_covered():
    chair = (100, 100, 200, 200)          # area 10000
    person = (90, 90, 210, 210)           # engulfs the chair
    assert sl.overlap_ratio(person, chair) == 1.0


# ------------------------------------------------------ seated decision
def test_seated_when_person_sits_on_chair():
    chair = (100, 100, 200, 200)
    person = (110, 90, 190, 200)          # ~80% of chair covered
    seated, ratio = sl.best_person_overlap(chair, [person])
    assert ratio > config.OVERLAP_RATIO
    assert seated is True


def test_not_seated_when_person_only_walks_past():
    chair = (100, 100, 200, 200)
    person = (150, 150, 210, 360)         # clipping only a corner (~25%)
    seated, ratio = sl.best_person_overlap(chair, [person])
    assert ratio < config.OVERLAP_RATIO   # standing person is NOT seated
    assert seated is False


def test_empty_room_not_seated():
    chair = (100, 100, 200, 200)
    seated, ratio = sl.best_person_overlap(chair, [])
    assert seated is False and ratio == 0.0


# ------------------------------------------------------ smoothing
def test_smoothing_on_needs_consecutive_frames():
    s = SeatState(1)
    chair = (0, 0, 10, 10)
    # one stray "seated" frame must NOT flip the state
    s.update(True, chair, now=0)
    assert s.seated is False
    # a single drop in the run resets progress toward ON
    s.update(False, chair, now=1)
    assert s.seated is False
    # now feed SMOOTH_ON consecutive seated frames
    for i in range(config.SMOOTH_ON):
        s.update(True, chair, now=100 + i)
    assert s.seated is True


def test_smoothing_off_is_slower_than_on():
    # config guarantees leaving needs more frames than sitting
    assert config.SMOOTH_OFF > config.SMOOTH_ON
    s = SeatState(1)
    chair = (0, 0, 10, 10)
    for i in range(config.SMOOTH_ON):
        s.update(True, chair, now=200 + i)
    assert s.seated is True
    # brief flicker of "free" should NOT immediately reset
    s.update(False, chair, now=300)
    assert s.seated is True
    # sustained free frames eventually flip it off
    for i in range(config.SMOOTH_OFF):
        s.update(False, chair, now=400 + i)
    assert s.seated is False


# ------------------------------------------------------ timer start/reset
def test_timer_starts_and_resets():
    s = SeatState(1)
    chair = (0, 0, 10, 10)
    for i in range(config.SMOOTH_ON):
        s.update(True, chair, now=1000 + i)
    assert s.seated is True
    seated_at = 1000 + (config.SMOOTH_ON - 1)  # now of the frame that flipped on
    assert s.raw_elapsed(now=seated_at + 8) == 8.0

    # flip to free; last_duration should capture how long they sat
    for i in range(config.SMOOTH_OFF):
        s.update(False, chair, now=seated_at + 10 + i)
    assert s.seated is False
    assert s.raw_elapsed() == 0.0
    # last now was (seated_at + 10 + (SMOOTH_OFF-1)); minus seated_since(=seated_at)
    assert s.last_duration == 10 + (config.SMOOTH_OFF - 1)


# ------------------------------------------------------ chair memory
def test_chair_id_stable_and_remembered_when_hidden():
    tracker = sl.ChairTracker(memory_frames=3)
    chair = (50, 50, 150, 150)
    first = tracker.update([{"box": chair, "conf": 0.5}])
    cid = first[0]["id"]
    assert first[0]["detected"] is True

    # chair disappears next frame (person sat down and hides it)
    still = tracker.update([])
    assert len(still) == 1
    assert still[0]["id"] == cid           # same id kept
    assert still[0]["detected"] is False   # now relying on memory
    assert still[0]["box"] == chair        # remembered box

    # after exceeding the memory window it is finally forgotten
    tracker.update([])
    tracker.update([])
    gone = tracker.update([])
    assert gone == []
