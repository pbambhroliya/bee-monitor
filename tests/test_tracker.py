import numpy as np

from beemon.detector import Detections
from beemon.tracker import BeeTracker, ByteTrackAdapter, TrackerConfig


def run(tracker, frames, fps=15.0):
    for i, (boxes, conf) in enumerate(frames):
        tracker.update(i / fps, i, Detections(np.array(boxes).reshape(-1, 4), np.array(conf), np.zeros(len(conf))))
    return [t for t in tracker.finalize() if t.ever_confirmed]


def box(cx, cy, w=30, h=24):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_fast_small_bee_keeps_one_identity_with_missed_detections():
    # 30 px bee moving 40 px per frame: boxes in consecutive frames do not overlap (IoU = 0)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(40):
        if i % 4 == 2:  # every 4th frame the detector misses the bee
            frames.append(([], []))
            continue
        frames.append(([box(100 + 40 * i + rng.normal(0, 1), 300 + rng.normal(0, 1))], [0.8]))
    tracks = run(BeeTracker(TrackerConfig(max_gate_px=250), (2000, 1080)), frames)
    assert len(tracks) == 1
    detected = [r for r in tracks[0].rows if r[9]]
    gaps = [r for r in tracks[0].rows if not r[9]]
    # gaps are kept as predicted rows (hard positives); the first miss (frame 2) happens before the
    # track is confirmed (min_hits=3), so only 9 of the 10 misses are recorded
    assert len(detected) == 30 and len(gaps) == 9


def test_two_bees_no_identity_switch():
    frames = []
    for i in range(30):
        frames.append(([box(100 + 10 * i, 200), box(800 - 10 * i, 600)], [0.9, 0.9]))
    tracks = run(BeeTracker(TrackerConfig(), (1920, 1080)), frames)
    assert len(tracks) == 2
    for track in tracks:
        ys = {round(r[4], -2) for r in track.rows}
        assert len(ys) == 1  # each track stays on its own bee


def test_low_confidence_continues_but_never_starts_a_track():
    frames = [([box(100 + 5 * i, 100)], [0.8]) for i in range(5)]
    frames += [([box(125 + 5 * i, 100), box(900, 900)], [0.2, 0.2]) for i in range(5)]  # weak detections
    tracks = run(BeeTracker(TrackerConfig(), (1920, 1080)), frames)
    assert len(tracks) == 1
    assert sum(r[9] for r in tracks[0].rows) == 10  # the 5 weak detections continued the track


def test_track_ends_when_bee_leaves_the_frame():
    frames = [([box(1880 + 20 * i, 500)], [0.9]) for i in range(3)]
    frames += [([], [])] * 3
    frames += [([box(1890, 500)], [0.9])] * 4  # a new bee appears at the same edge later
    tracks = run(BeeTracker(TrackerConfig(min_hits=2), (1920, 1080)), frames)
    assert len(tracks) == 2


def test_bytetrack_adapter_runs():
    frames = [([box(100 + 3 * i, 200)], [0.9]) for i in range(10)]
    tracks = run(ByteTrackAdapter(TrackerConfig(), (1920, 1080), 15.0), frames)
    assert len(tracks) >= 1
    assert sum(len(t.rows) for t in tracks) >= 8
