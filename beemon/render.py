"""Annotated output video: boxes, track ids and trails, coloured by the final filter decision."""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from beemon.video import iter_frames

GREY = (150, 150, 150)
SHORT_REASON = {"too short": "short", "static object": "static", "background hotspot": "hotspot",
                "low confidence": "lowconf", "classifier: not an insect": "not insect"}


def _color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _cached_frames(frame_dir: Path, fps: float):
    """Frames saved (already small) during processing: ``<index>.jpg``."""
    for path in sorted(frame_dir.glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is not None:
            index = int(path.stem)
            yield type("Pack", (), {"index": index, "time_s": index / fps, "frame": image})()


def render_video(video_path: str, tracks: pd.DataFrame, summary: pd.DataFrame, out_path: Path, cfg: dict,
                 video: dict, log: Callable[[str], None] = print, frame_dir: Path | None = None) -> Path | None:
    ocfg = cfg["output"]
    scale = float(ocfg.get("video_scale", 0.5))
    trail_s = float(ocfg.get("trail_s", 1.5))
    draw_rejected = bool(ocfg.get("draw_rejected", True))
    stride = int(video["frame_stride"])
    start = int(round(video["start_s"] * video["fps"]))
    end = int(round(video["end_s"] * video["fps"]))
    targets = list(range(start, end, stride))
    width, height = int(video["width"] * scale) // 2 * 2, int(video["height"] * scale) // 2 * 2
    fps_out = video["fps"] / stride

    by_frame: dict[int, list] = defaultdict(list)
    if len(tracks):
        for row in tracks.itertuples(index=False):
            by_frame[int(row.frame)].append(row)
    status = summary.set_index("track_id") if len(summary) else pd.DataFrame()
    valid_ids = set(summary.loc[summary["valid"].astype(bool), "track_id"]) if len(summary) else set()
    started: set[int] = set()
    trails: dict[int, deque] = defaultdict(deque)

    raw_path = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (width, height))
    if not writer.isOpened():
        log("  could not open a video writer; skipping annotated video")
        return None
    log(f"  rendering annotated video ({len(targets)} frames)...")
    try:
        cached = frame_dir is not None and Path(frame_dir).is_dir() and any(Path(frame_dir).glob("*.jpg"))
        source = _cached_frames(Path(frame_dir), video["fps"]) if cached else iter_frames(video_path, targets, 0)
        for pack in source:
            image = pack.frame if pack.frame.shape[:2] == (height, width) else \
                cv2.resize(pack.frame, (width, height), interpolation=cv2.INTER_AREA)
            present = 0
            for row in by_frame.get(pack.index, []):
                tid = int(row.track_id)
                valid = tid in valid_ids
                if not valid and not draw_rejected:
                    continue
                color = _color(tid) if valid else GREY
                cx, cy, w, h = row.cx * scale, row.cy * scale, row.w * scale, row.h * scale
                trail = trails[tid]
                trail.append((pack.time_s, int(cx), int(cy)))
                while trail and pack.time_s - trail[0][0] > trail_s:
                    trail.popleft()
                if len(trail) > 1:
                    cv2.polylines(image, [np.array([(x, y) for _, x, y in trail], np.int32)], False, color, 2)
                p1 = (int(cx - w / 2) - 3, int(cy - h / 2) - 3)
                p2 = (int(cx + w / 2) + 3, int(cy + h / 2) + 3)
                cv2.rectangle(image, p1, p2, color, 2 if row.detected else 1)
                if valid:
                    present += 1
                    started.add(tid)
                    label = f"#{tid} {row.cls_name if isinstance(row.cls_name, str) else ''}"
                else:
                    reason = status.loc[tid, "reason"] if tid in status.index else ""
                    label = f"#{tid} {SHORT_REASON.get(reason, reason)}"
                cv2.putText(image, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)
            header = f"t={pack.time_s:6.1f}s  insects now: {present}  visits so far: {len(started)}"
            cv2.rectangle(image, (0, 0), (width, 26), (0, 0, 0), -1)
            cv2.putText(image, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()

    ffmpeg = _ffmpeg()
    if ffmpeg:  # re-encode to H.264 so browsers (and the dashboard) can play it
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(out_path)]
        if subprocess.run(command, check=False).returncode == 0:
            raw_path.unlink(missing_ok=True)
            log(f"  annotated video: {out_path}")
            return out_path
    raw_path.replace(out_path)
    log(f"  annotated video (mp4v; install imageio-ffmpeg for browser playback): {out_path}")
    return out_path
