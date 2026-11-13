"""Video reading: metadata and sequential frame access with neighbouring frames for motion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm"}


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def stem(self) -> str:
        return Path(self.path).stem

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class FramePack:
    """One frame to process plus the frames ``offset`` before and after it (for motion)."""

    index: int
    time_s: float
    frame: np.ndarray
    prev: np.ndarray | None = None
    next: np.ndarray | None = None


def find_videos(source: str | Path) -> list[Path]:
    source = Path(source).expanduser()
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    raise FileNotFoundError(f"No video file or folder at {source}")


def probe(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Video reports invalid metadata (fps={fps}, size={width}x{height}): {path}")
    if count <= 0:  # some containers do not report a frame count: count by reading
        capture = cv2.VideoCapture(str(path))
        count = 0
        while capture.grab():
            count += 1
        capture.release()
    return VideoInfo(str(path), fps, width, height, count)


def frame_range(info: VideoInfo, start_s: float | None = 0.0, end_s: float | None = None) -> tuple[int, int]:
    """Return ``(first, last_exclusive)`` frame indices for a time window."""
    start = max(0, int(round((start_s or 0.0) * info.fps)))
    end = info.frame_count if end_s in (None, 0) else min(info.frame_count, int(round(end_s * info.fps)))
    if end <= start:
        raise ValueError(f"Empty time window: start {start_s}s, end {end_s}s, video {info.duration_s:.1f}s")
    return start, end


SEEK_GAP = 40  # frames; decoding runs at ~80-100 frames/s on a laptop, a jump costs ~0.4 s


def iter_frames(path: str | Path, targets: Iterable[int], offset: int = 0) -> Iterator[FramePack]:
    """Read the video sequentially and yield the requested frames.

    For every target frame ``k`` the pack also holds frames ``k - offset`` and ``k + offset``
    (clamped to the available range) so motion can be computed. Frames that are not needed
    are skipped with ``grab()`` (no colour conversion), which keeps extraction fast.
    """
    wanted = sorted({int(t) for t in targets if int(t) >= 0})
    if not wanted:
        return
    offset = max(0, int(offset))
    info = probe(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    first = max(0, wanted[0] - offset)
    if first > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    buffer: dict[int, np.ndarray] = {}
    pointer = 0
    index = first
    last_read = None

    def pack(target: int) -> FramePack:
        frame = buffer[target]
        if offset == 0:
            return FramePack(target, target / info.fps, frame)
        prev_index = max(target - offset, min(buffer))
        next_index = min(target + offset, last_read)
        return FramePack(target, target / info.fps, frame, buffer.get(prev_index, frame), buffer.get(next_index, frame))

    try:
        while pointer < len(wanted):
            gap = wanted[pointer] - offset - index
            if gap > SEEK_GAP:  # far ahead: jumping is faster than decoding every frame in between
                index = wanted[pointer] - offset
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                continue
            if index < wanted[pointer] - offset:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            buffer[index] = frame
            last_read = index
            while pointer < len(wanted) and wanted[pointer] + offset <= index:
                if wanted[pointer] in buffer:
                    yield pack(wanted[pointer])
                pointer += 1
            if pointer < len(wanted):
                keep_from = wanted[pointer] - offset
                for old in [k for k in buffer if k < keep_from]:
                    del buffer[old]
            index += 1
        # end of video: emit remaining targets that were read, using the last frame as "next"
        while pointer < len(wanted):
            if wanted[pointer] in buffer:
                yield pack(wanted[pointer])
            pointer += 1
    finally:
        capture.release()


def read_frames(path: str | Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Read a handful of specific frames (sequential, frame-accurate)."""
    return {pack.index: pack.frame for pack in iter_frames(path, indices, offset=0)}
