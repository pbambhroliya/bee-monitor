"""Live preview while a video is being processed: results/<video>/live.jpg (+ counters in progress.json).

The picture is redrawn at most once per ``every_s`` seconds, so it costs almost nothing. The
app (page "Track a video") shows it and refreshes itself.
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from beemon.detector import Detections
from beemon.render import _color


class LivePreview:
    def __init__(self, out_dir: Path, every_s: float = 1.0, width: int = 960, min_hits: int = 3, trail_s: float = 2.0):
        self.path = Path(out_dir) / "live.jpg"
        self.every_s = float(every_s)
        self.width = int(width)
        self.min_hits = int(min_hits)
        self.trail_s = float(trail_s)
        self.last = 0.0
        self.hits: Counter = Counter()
        self.trails: dict[int, deque] = defaultdict(deque)
        self.bees_now = 0

    def observe(self, frame: np.ndarray, t: float, dets: Detections, matched: list[tuple[int, int]]) -> None:
        """Call for every frame the detector ran on."""
        centers = dets.centers if len(dets) else np.zeros((0, 2))
        for track_id, j in matched:
            self.hits[track_id] += 1
            trail = self.trails[track_id]
            trail.append((t, float(centers[j, 0]), float(centers[j, 1])))
            while trail and t - trail[0][0] > self.trail_s:
                trail.popleft()
        confirmed_now = [tid for tid, _ in matched if self.hits[tid] >= self.min_hits]
        self.bees_now = len(confirmed_now)
        if time.time() - self.last >= self.every_s:
            self.last = time.time()
            self._render(frame, t, dets, matched)

    def stats(self) -> dict:
        return {"bees_now": self.bees_now,
                "tracks_so_far": sum(1 for h in self.hits.values() if h >= self.min_hits)}

    def _render(self, frame: np.ndarray, t: float, dets: Detections, matched: list[tuple[int, int]]) -> None:
        height, width = frame.shape[:2]
        scale = self.width / width
        image = cv2.resize(frame, (self.width, int(height * scale)), interpolation=cv2.INTER_AREA)
        matched_dets = {j for _, j in matched}
        for j in range(len(dets)):  # detections that belong to no track: thin grey
            if j in matched_dets:
                continue
            x1, y1, x2, y2 = (dets.xyxy[j] * scale).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (170, 170, 170), 1)
        for track_id, j in matched:
            confirmed = self.hits[track_id] >= self.min_hits
            color = _color(track_id) if confirmed else (0, 220, 255)
            trail = [(int(x * scale), int(y * scale)) for _, x, y in self.trails[track_id]]
            if len(trail) > 1:
                cv2.polylines(image, [np.array(trail, np.int32)], False, color, 2)
            x1, y1, x2, y2 = (dets.xyxy[j] * scale).astype(int)
            cv2.rectangle(image, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 2 if confirmed else 1)
            label = f"#{track_id} {dets.conf[j]:.2f}" if confirmed else "?"
            cv2.putText(image, label, (x1 - 3, max(12, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        stats = self.stats()
        header = f"t={t:6.1f}s   insects now: {stats['bees_now']}   tracks so far: {stats['tracks_so_far']}"
        cv2.rectangle(image, (0, 0), (self.width, 26), (0, 0, 0), -1)
        cv2.putText(image, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:  # write atomically: the dashboard never reads a half-written picture
            tmp = self.path.with_suffix(".tmp.jpg")
            tmp.write_bytes(buffer.tobytes())
            os.replace(tmp, self.path)
