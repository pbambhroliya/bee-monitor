"""Synthetic test data: a textured "flower field" video with moving dark blobs ("bees") and known tracks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beemon.detector import Detections  # noqa: E402


def make_background(width=640, height=360, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (60, 140, 70)  # green
    for _ in range(250):  # purple "flowers"
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.circle(image, (x, y), int(rng.integers(4, 12)), (200, 150, 170), -1)
    noise = rng.normal(0, 6, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def bee_paths(n_frames=90):
    """Two moving bees and one static dark spot (a 'flower bud' false positive)."""
    t = np.arange(n_frames)
    bee1 = np.stack([60 + t * 5.0, 100 + 20 * np.sin(t / 10)], axis=1)          # left -> right
    bee2 = np.stack([500 - t * 3.0, 260 - t * 1.0], axis=1)                     # right -> up-left
    bud = np.tile(np.array([[320.0, 60.0]]), (n_frames, 1))                     # never moves
    return {1: bee1, 2: bee2, 3: bud}


def stamp_index(frame, index):
    """Write the frame index as an 8-bit barcode (8x8 px blocks) into the bottom-left corner."""
    h = frame.shape[0]
    for bit in range(8):
        value = 255 if (index >> bit) & 1 else 0
        frame[h - 12:h - 4, bit * 8:(bit + 1) * 8] = value


def read_index(frame):
    h = frame.shape[0]
    return sum(1 << bit for bit in range(8) if frame[h - 10:h - 6, bit * 8 + 2:bit * 8 + 6].mean() > 127)


@pytest.fixture
def synthetic_video(tmp_path):
    """Writes a 90-frame 30 fps video and returns (path, paths dict, size)."""
    width, height, n = 640, 360, 90
    background = make_background(width, height)
    paths = bee_paths(n)
    path = tmp_path / "SYNTH01.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height))
    for i in range(n):
        frame = background.copy()
        for key, xy in paths.items():
            cx, cy = xy[i]
            wobble = 2 if key != 3 else 0
            cv2.ellipse(frame, (int(cx), int(cy)), (9 + (i % 2) * wobble, 6), 0, 0, 360, (20, 40, 60), -1)
        stamp_index(frame, i)
        writer.write(frame)
    writer.release()
    return path, paths, (width, height)


class FakeDetector:
    """Returns the ground-truth boxes with noise, missed detections and confidence values."""

    names = {0: "bee"}

    def __init__(self, paths, fps=30.0, miss_rate=0.2, seed=1):
        self.paths = paths
        self.fps = fps
        self.miss_rate = miss_rate
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, image):
        i = read_index(image)  # frame index from the barcode stamped into the synthetic video
        boxes, conf = [], []
        for key, xy in self.paths.items():
            if i >= len(xy) or self.rng.random() < self.miss_rate:
                continue
            cx, cy = xy[i] + self.rng.normal(0, 1.0, 2)
            boxes.append([cx - 10, cy - 7, cx + 10, cy + 7])
            conf.append(0.7 if key != 3 else 0.6)
        self.calls += 1
        return Detections(np.array(boxes).reshape(-1, 4), np.array(conf), np.zeros(len(conf)))


@pytest.fixture
def fake_detector_factory():
    return FakeDetector
