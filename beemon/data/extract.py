"""Frames out of videos: for labelling (with optional model pre-labels), tracking clips, motion previews."""

from __future__ import annotations

import csv
import json
import random
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from beemon.data.common import frame_name, load_split, write_cvat_yolo_package, xyxy_to_yolo
from beemon.motion import frame_difference, mie_image
from beemon.video import find_videos, frame_range, iter_frames, probe

MANIFEST_FIELDS = ["image", "video", "video_path", "frame", "time_s", "fps", "width", "height"]


def _read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_frames(source, out_dir, every_s: float = 3.0, max_per_video: int | None = 80, start_s: float = 0.0,
                   end_s: float | None = None, split_yaml=None, include_test: bool = False, prelabel=None,
                   prelabel_conf: float = 0.25, prelabel_imgsz: int = 1280, prelabel_device=None,
                   only_with_detections: bool = False, empty_fraction: float = 0.2, seed: int = 0,
                   only_test: bool = False, log: Callable[[str], None] = print) -> Path:
    """Sample frames for labelling into ``out_dir/images`` and write ``manifest.csv``.

    With ``prelabel`` (a detector .pt), boxes predicted by that model are written as a CVAT
    "YOLO 1.1" package (``cvat_prelabels.zip``) so annotators only correct them.
    """
    videos = find_videos(source)
    if not videos:
        raise SystemExit(f"No videos found in {source}. Put the videos there first "
                         "(python -m beemon download ..., or copy them), then run this again.")
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    manifest = _read_manifest(manifest_path)
    existing = {row["image"] for row in manifest}
    split = load_split(split_yaml)
    rng = random.Random(seed)

    model = names = None
    if prelabel:
        from ultralytics import YOLO

        model = YOLO(str(prelabel))
        names = [str(model.names[i]) for i in sorted(model.names)]
    prelabels: list[tuple[str, np.ndarray]] = []

    if only_test and not any(v.stem in split["test_videos"] for v in videos):
        raise SystemExit(f"--only-test: none of the videos in {source} is listed as a test video in the split file "
                         f"({split['test_videos']}). Edit configs/split.yaml first.")
    for video in videos:
        if only_test and video.stem not in split["test_videos"]:
            continue
        if video.stem in split["test_videos"] and not (include_test or only_test):
            log(f"skip {video.name}: listed as a TEST video in the split file (use --include-test to override)")
            continue
        info = probe(video)
        first, last = frame_range(info, start_s, end_s)
        step = max(1, int(round(every_s * info.fps)))
        candidates = list(range(first, last, step))
        # spread the frames over the whole video; when keeping only frames with detections,
        # look at 4x more candidates so the cap is still filled from the whole video
        limit = None if not max_per_video else max_per_video * (4 if only_with_detections else 1)
        if limit and len(candidates) > limit:
            keep = np.linspace(0, len(candidates) - 1, limit).round().astype(int)
            candidates = [candidates[i] for i in sorted(set(keep))]
        log(f"{video.name}: {len(candidates)} candidate frames")
        saved = 0
        for pack in iter_frames(video, candidates, 0):
            name = frame_name(video.stem, pack.index) + ".jpg"
            if name in existing:
                continue
            labels = np.zeros((0, 5))
            if model is not None:
                result = model.predict(pack.frame, imgsz=prelabel_imgsz, conf=prelabel_conf, verbose=False,
                                       **({"device": prelabel_device} if prelabel_device else {}))[0]
                if result.boxes is not None and len(result.boxes):
                    labels = xyxy_to_yolo(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                          info.width, info.height)
                if only_with_detections and len(labels) == 0 and rng.random() > empty_fraction:
                    continue
            if max_per_video and saved >= max_per_video:
                break
            cv2.imwrite(str(image_dir / name), pack.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({"image": name, "video": video.stem, "video_path": str(video.resolve()),
                             "frame": pack.index, "time_s": round(pack.time_s, 3), "fps": info.fps,
                             "width": info.width, "height": info.height})
            existing.add(name)
            if model is not None:
                prelabels.append((name, labels))
            saved += 1
        log(f"  saved {saved} frames")

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    if model is not None and prelabels:
        package = write_cvat_yolo_package(out_dir / "cvat_prelabels.zip", names, prelabels)
        log(f"pre-labels for CVAT: {package}  (model classes: {names})")
    log(f"{len(manifest)} frames in {image_dir}, manifest: {manifest_path}")
    return out_dir


def make_clip(video, out_dir, start_s: float, duration_s: float, stride: int = 1,
              log: Callable[[str], None] = print) -> Path:
    """Export a short clip as numbered JPEG frames (+ zip) for track annotation in CVAT."""
    info = probe(video)
    first, last = frame_range(info, start_s, start_s + duration_s)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    targets = list(range(first, last, max(1, stride)))
    zip_path = out_dir / "clip_frames.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for i, pack in enumerate(iter_frames(video, targets, 0), start=1):
            name = f"{i:06d}.jpg"
            ok, buffer = cv2.imencode(".jpg", pack.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (frames_dir / name).write_bytes(buffer.tobytes())
            archive.writestr(name, buffer.tobytes())
    meta = {"video": Path(video).stem, "video_path": str(Path(video).resolve()), "start_frame": first,
            "stride": max(1, stride), "n_frames": len(targets), "fps": info.fps,
            "start_s": first / info.fps, "end_s": last / info.fps}
    (out_dir / "clip.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"clip: {len(targets)} frames -> {zip_path}\n"
        f"process the SAME window with:  beemon process {video} --set video.start_s={meta['start_s']:.3f} "
        f"--set video.end_s={meta['end_s']:.3f} --set video.frame_stride={max(1, stride)}")
    return out_dir


def motion_preview(video, out_path, time_s: float = 10.0, offsets=(1, 2, 4, 8), blur: int = 5,
                   width: int = 640, local_kernel: int = 0) -> Path:
    """Side-by-side picture of the frame, the frame difference and the MIE image for several offsets.

    Use it to choose ``motion.offset_frames``: bees should light up clearly, swaying flowers not.
    """
    info = probe(video)
    center = min(info.frame_count - 1, int(round(time_s * info.fps)))
    needed = {center} | {max(0, center - o) for o in offsets} | {min(info.frame_count - 1, center + o) for o in offsets}
    frames = {pack.index: pack.frame for pack in iter_frames(video, sorted(needed), 0)}
    current = frames[center]
    height = int(current.shape[0] * width / current.shape[1])

    def small(img):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (width, 28), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    rows = []
    for o in offsets:
        prev = frames[max(0, center - o)]
        nxt = frames[min(info.frame_count - 1, center + o)]
        fd = frame_difference(prev, current, nxt, blur)
        fd_view = cv2.applyColorMap(cv2.convertScaleAbs(fd, alpha=3.0), cv2.COLORMAP_INFERNO)
        panels = [label(small(current), f"frame {center} (t={center / info.fps:.1f}s)"),
                  label(small(fd_view), f"motion, offset n={o} (x3 brightness)")]
        if local_kernel:
            local = frame_difference(prev, current, nxt, blur, local_kernel)
            local_view = cv2.applyColorMap(cv2.convertScaleAbs(local, alpha=3.0), cv2.COLORMAP_INFERNO)
            panels.append(label(small(local_view), f"local motion (kernel {local_kernel}), n={o}"))
            fd = local
        panels.append(label(small(mie_image(current, fd)), f"MIE image, n={o}"))
        rows.append(np.hstack(panels))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path
