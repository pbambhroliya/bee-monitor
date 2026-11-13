"""Multi-object tracking for small, fast insects.

BeeTrack (``tracker.type: beetrack``) keeps the proven ideas of ByteTrack and Polytrack but
adapts the matching to bees that are only ~20-40 px large:

* ByteTrack's two stages: confident detections are matched first; weak detections
  (``low_conf`` .. ``high_conf``) may only CONTINUE an existing track, never start one.
  This recovers bees the detector only half-sees without adding false tracks.
* Matching by CENTRE DISTANCE inside a speed gate instead of box overlap (IoU). A 30 px bee
  that moves 30 px between processed frames has IoU 0 with its previous box, so IoU-based
  trackers (stock ByteTrack) break such tracks; a distance gate does not.
* Constant-velocity Kalman filter in pixel/second units, so frame skipping and variable time
  steps are handled correctly.
* The gate grows with the time a track has been lost (``max_speed_px_s`` x seconds lost),
  like Polytrack's "re-detect within a radius of the predicted position".
* Polytrack's exit rule: a track predicted to leave the frame is ended immediately.

``ByteTrackAdapter`` (``tracker.type: bytetrack``) wraps the original Ultralytics ByteTrack
with the same interface, as a baseline for the report.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import linear_sum_assignment

from beemon.detector import Detections

ROW_FIELDS = (
    "frame", "time_s", "track_id", "cx", "cy", "w", "h", "conf", "cls_id", "detected", "motion_in", "motion_ratio",
)


@dataclass
class TrackerConfig:
    high_conf: float = 0.35
    low_conf: float = 0.10
    new_track_conf: float = 0.40
    min_hits: int = 3
    max_speed_px_s: float = 1200.0
    min_gate_px: float = 25.0
    max_gate_px: float = 250.0
    max_lost_s: float = 1.0
    tentative_max_lost_s: float = 0.3
    size_ratio_max: float = 3.0
    edge_margin_px: float = 15.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "TrackerConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


class KalmanCV:
    """Constant-velocity Kalman filter on the box centre; time in seconds."""

    _H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def __init__(self, cx: float, cy: float, meas_std: float = 4.0, accel_std: float = 1500.0, vel_std: float = 400.0):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([meas_std**2, meas_std**2, vel_std**2, vel_std**2])
        self.meas_var = meas_std**2
        self.accel_var = accel_std**2

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        a, b, c = dt**4 / 4, dt**3 / 2, dt**2
        Q = self.accel_var * np.array([[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, cx: float, cy: float) -> None:
        H = self._H
        S = H @ self.P @ H.T + np.eye(2) * self.meas_var
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([cx, cy]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]

    def clamp_speed(self, max_speed: float) -> None:
        speed = float(np.hypot(*self.x[2:]))
        if speed > max_speed > 0:
            self.x[2:] *= max_speed / speed


class Track:
    def __init__(self, track_id: int, frame: int, t: float, box: np.ndarray, conf: float, cls: int,
                 motion_in: float, motion_ratio: float):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.id = track_id
        self.kf = KalmanCV(cx, cy)
        self.t = t
        self.w, self.h = float(box[2] - box[0]), float(box[3] - box[1])
        self.hits = 1
        self.first_t = self.last_det_t = t
        self.last_det_frame = frame
        self.state = "tentative"
        self.ever_confirmed = False
        self.rows: list[tuple] = []
        self.cls_votes: dict[int, float] = defaultdict(float)
        self._add(frame, t, cx, cy, self.w, self.h, conf, cls, True, motion_in, motion_ratio)

    def _add(self, frame, t, cx, cy, w, h, conf, cls, detected, motion_in, motion_ratio):
        self.rows.append((int(frame), float(t), self.id, float(cx), float(cy), float(w), float(h), float(conf),
                          int(cls), bool(detected), float(motion_in), float(motion_ratio)))
        if detected:
            self.cls_votes[int(cls)] += float(conf)

    def predict_to(self, t: float) -> None:
        self.kf.predict(t - self.t)
        self.t = t

    def update(self, frame, t, box, conf, cls, motion_in, motion_ratio, cfg: TrackerConfig) -> None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = float(box[2] - box[0]), float(box[3] - box[1])
        self.kf.update(cx, cy)
        self.kf.clamp_speed(cfg.max_speed_px_s)
        self.w, self.h = 0.6 * self.w + 0.4 * w, 0.6 * self.h + 0.4 * h
        self.hits += 1
        self.last_det_t, self.last_det_frame = t, frame
        if self.state == "tentative" and self.hits >= cfg.min_hits:
            self.state, self.ever_confirmed = "confirmed", True
        elif self.state == "lost":
            self.state = "confirmed"
        self._add(frame, t, cx, cy, w, h, conf, cls, True, motion_in, motion_ratio)

    def mark_missed(self, frame: int, t: float) -> None:
        """Record the predicted position (gap row) and mark the track lost."""
        if self.ever_confirmed:
            cx, cy = self.kf.pos
            self._add(frame, t, cx, cy, self.w, self.h, 0.0, -1, False, 0.0, 0.0)
        if self.state == "confirmed":
            self.state = "lost"

    def trim(self) -> None:
        """Drop trailing predicted rows (after the last detection the bee is gone)."""
        while self.rows and not self.rows[-1][9]:
            self.rows.pop()

    @property
    def size(self) -> float:
        return float(np.sqrt(max(1.0, self.w * self.h)))


class BeeTracker:
    name = "beetrack"

    def __init__(self, cfg: TrackerConfig, frame_size: tuple[int, int]):
        self.cfg = cfg
        self.width, self.height = frame_size
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    # ----------------------------------------------------------------- matching
    def _associate(self, tracks: list[Track], det_idx, dets: Detections, t: float):
        det_idx = [int(j) for j in det_idx]
        if not tracks or not det_idx:
            return [], list(tracks), det_idx
        cfg = self.cfg
        idx = np.asarray(det_idx)
        centers = dets.centers[idx]
        sizes = np.sqrt(np.maximum(1.0, dets.wh[idx, 0] * dets.wh[idx, 1]))
        conf = dets.conf[idx]
        cost = np.full((len(tracks), len(idx)), 1e6)
        for i, track in enumerate(tracks):
            px, py = track.kf.pos
            dist = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            lost_for = max(t - track.last_det_t, 1e-3)
            gate = float(np.clip(cfg.max_speed_px_s * lost_for, cfg.min_gate_px, cfg.max_gate_px))
            ratio = np.maximum(sizes / track.size, track.size / sizes)
            ok = (dist <= gate) & (ratio <= cfg.size_ratio_max)
            cost[i, ok] = dist[ok] / gate + 0.1 * (1.0 - conf[ok])
        rows, cols = linear_sum_assignment(cost)
        matches, used_t, used_d = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] < 1e5:
                matches.append((tracks[r], det_idx[c]))
                used_t.add(r)
                used_d.add(c)
        rest_tracks = [tr for i, tr in enumerate(tracks) if i not in used_t]
        rest_dets = [d for k, d in enumerate(det_idx) if k not in used_d]
        return matches, rest_tracks, rest_dets

    def _leaving(self, track: Track) -> bool:
        px, py = track.kf.pos
        m = self.cfg.edge_margin_px
        if px < 0 or py < 0 or px > self.width or py > self.height:
            return True
        vx, vy = track.kf.vel
        last = next((r for r in reversed(track.rows) if r[9]), None)
        if last is None:
            return False
        lx, ly = last[3], last[4]
        return (lx < m and vx < 0) or (ly < m and vy < 0) or (lx > self.width - m and vx > 0) or (
            ly > self.height - m and vy > 0)

    # ----------------------------------------------------------------- API
    def update(self, t: float, frame: int, dets: Detections) -> list[tuple[int, int]]:
        """Process one frame. Returns ``(track_id, detection_index)`` pairs matched or created."""
        cfg = self.cfg
        for track in self.active:
            track.predict_to(t)
        n = len(dets)
        motion_in = dets.motion_in if dets.motion_in is not None else np.zeros(n, np.float32)
        motion_ratio = dets.motion_ratio if dets.motion_ratio is not None else np.zeros(n, np.float32)
        conf = dets.conf
        high = np.flatnonzero(conf >= cfg.high_conf)
        low = np.flatnonzero((conf >= cfg.low_conf) & (conf < cfg.high_conf))

        pool = [tr for tr in self.active if tr.state in ("confirmed", "lost")]
        tentative = [tr for tr in self.active if tr.state == "tentative"]
        m1, rest_pool, rest_high = self._associate(pool, high, dets, t)
        m2, _, rest_high = self._associate(tentative, rest_high, dets, t)
        m3, _, _ = self._associate([tr for tr in rest_pool if tr.state == "confirmed"], low, dets, t)

        matched: list[tuple[int, int]] = []
        matched_ids = set()
        for track, j in m1 + m2 + m3:
            track.update(frame, t, dets.xyxy[j], conf[j], dets.cls[j], motion_in[j], motion_ratio[j], cfg)
            matched.append((track.id, j))
            matched_ids.add(track.id)

        still_active = []
        for track in self.active:
            if track.id in matched_ids:
                still_active.append(track)
                continue
            lost_for = t - track.last_det_t
            if track.state == "tentative":
                (self._finish(track) if lost_for > cfg.tentative_max_lost_s else still_active.append(track))
                continue
            track.mark_missed(frame, t)
            if lost_for > cfg.max_lost_s or self._leaving(track):
                self._finish(track)
            else:
                still_active.append(track)
        self.active = still_active

        for j in rest_high:
            if conf[j] >= cfg.new_track_conf:
                track = Track(self._next_id, frame, t, dets.xyxy[j], conf[j], dets.cls[j], motion_in[j], motion_ratio[j])
                if cfg.min_hits <= 1:
                    track.state, track.ever_confirmed = "confirmed", True
                self._next_id += 1
                self.active.append(track)
                matched.append((track.id, j))
        return matched

    def has_active(self) -> bool:
        return any(tr.state in ("confirmed", "tentative") for tr in self.active)

    def _finish(self, track: Track) -> None:
        track.trim()
        self.finished.append(track)

    def finalize(self) -> list[Track]:
        for track in self.active:
            self._finish(track)
        self.active = []
        return self.finished


class _BTResults:
    """Minimal results object understood by Ultralytics' BYTETracker."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return _BTResults(self.xyxy[index], self.conf[index], self.cls[index])

    @property
    def xywh(self):
        xy = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
        wh = self.xyxy[:, 2:] - self.xyxy[:, :2]
        return np.concatenate([xy, wh], axis=1)


class ByteTrackAdapter:
    """Ultralytics ByteTrack (IoU matching) with the BeeTracker interface — baseline only."""

    name = "bytetrack"

    def __init__(self, cfg: TrackerConfig, frame_size: tuple[int, int], effective_fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=cfg.high_conf,
            track_low_thresh=cfg.low_conf,
            new_track_thresh=cfg.new_track_conf,
            track_buffer=max(1, int(round(cfg.max_lost_s * effective_fps))),
            match_thresh=0.8,
            fuse_score=True,
        )
        try:
            self._tracker = BYTETracker(args, frame_rate=int(round(effective_fps)))
        except TypeError:
            self._tracker = BYTETracker(args)
        self.width, self.height = frame_size
        self._tracks: dict[int, Track] = {}

    def update(self, t: float, frame: int, dets: Detections) -> list[tuple[int, int]]:
        n = len(dets)
        output = np.asarray(self._tracker.update(_BTResults(dets.xyxy, dets.conf, dets.cls), img=None))
        if n == 0 or output.size == 0:
            return []
        motion_in = dets.motion_in if dets.motion_in is not None else np.zeros(n, np.float32)
        motion_ratio = dets.motion_ratio if dets.motion_ratio is not None else np.zeros(n, np.float32)
        matched = []
        centers = dets.centers
        for row in output.reshape(-1, output.shape[-1]):
            track_id, score = int(row[4]), float(row[5])
            cx, cy = (row[0] + row[2]) / 2, (row[1] + row[3]) / 2
            dist = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy) + 1e3 * (np.abs(dets.conf - score) > 1e-5)
            j = int(np.argmin(dist))
            track = self._tracks.get(track_id)
            if track is None:
                track = Track(track_id, frame, t, dets.xyxy[j], dets.conf[j], dets.cls[j], motion_in[j], motion_ratio[j])
                track.state, track.ever_confirmed = "confirmed", True
                self._tracks[track_id] = track
            else:
                track._add(frame, t, *centers[j], *dets.wh[j], dets.conf[j], dets.cls[j], True, motion_in[j],
                           motion_ratio[j])
                track.last_det_t, track.last_det_frame = t, frame
            matched.append((track_id, j))
        return matched

    def has_active(self) -> bool:
        return bool(self._tracker.tracked_stracks or self._tracker.lost_stracks)

    def finalize(self) -> list[Track]:
        return sorted(self._tracks.values(), key=lambda tr: tr.id)


def make_tracker(kind: str, cfg: TrackerConfig, frame_size: tuple[int, int], effective_fps: float):
    kind = (kind or "beetrack").lower()
    if kind == "beetrack":
        return BeeTracker(cfg, frame_size)
    if kind == "bytetrack":
        return ByteTrackAdapter(cfg, frame_size, effective_fps)
    raise ValueError(f"Unknown tracker type {kind!r} (use beetrack or bytetrack)")
