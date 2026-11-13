"""Build YOLO datasets from labelled frames (split by VIDEO), in RGB or motion-informed (MIE) mode, and tile them."""

from __future__ import annotations

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from beemon.data.common import (
    cvat_listed, find_label_files, list_images, load_class_names, load_split, parse_frame_name, read_labels,
    write_labels,
    yolo_to_xyxy, data_yaml_split_dirs, label_path_for,
)
from beemon.detector import tile_windows
from beemon.motion import frame_difference, mie_image
from beemon.video import VIDEO_EXTENSIONS, iter_frames

SPLITS = ("train", "val", "test")


def _manifest(batch: Path) -> dict[str, dict]:
    path = batch / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {Path(row["image"]).stem: row for row in csv.DictReader(handle)}


def _assign_splits(videos: list[str], split: dict, log) -> dict[str, str]:
    test = set(split["test_videos"])
    val = set(split["val_videos"])
    rest = sorted(v for v in videos if v not in test and v not in val)
    if not val and rest:
        rng = random.Random(split.get("seed", 42))
        shuffled = rest[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * float(split.get("val_fraction", 0.2))))) if len(shuffled) >= 3 else 0
        val = set(shuffled[:n_val])
    assignment = {}
    for video in videos:
        assignment[video] = "test" if video in test else "val" if video in val else "train"
    if not any(s == "val" for s in assignment.values()):
        log("WARNING: fewer than 3 training videos -> no video-level validation split possible; "
            "20% of the training FRAMES are used for validation instead (scores will be optimistic).")
    return assignment


def is_auto_batch(batch: Path) -> bool:
    return (Path(batch) / "labels_auto").is_dir() or (Path(batch) / "labels_suggested").is_dir()


def _remap(labels: np.ndarray, names: list[str], target: list[str], single_class: bool, path, classes_yaml):
    unknown = sorted({names[int(c)] for c in labels[:, 0] if int(c) < len(names) and names[int(c)] not in target})
    if unknown and not single_class:
        raise ValueError(f"Labels {unknown} in {path} are not in {classes_yaml} ({target}). "
                         "Rename them in CVAT or add them to the class file.")
    mapping = {i: (0 if single_class else target.index(n)) for i, n in enumerate(names)
               if single_class or n in target}
    keep = np.array([int(c) in mapping for c in labels[:, 0]])
    labels = labels[keep].copy()
    labels[:, 0] = [mapping[int(c)] for c in labels[:, 0]]
    return labels


def _find_video(row: dict | None, video: str, videos_dir: Path | None) -> Path | None:
    if row and row.get("video_path") and Path(row["video_path"]).is_file():
        return Path(row["video_path"])
    if videos_dir:
        for path in Path(videos_dir).rglob(f"{video}.*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
    return None


def build_dataset(batches: list[str | Path], out: str | Path, classes_yaml, split_yaml=None, mode: str = "rgb",
                  mie_offset: int = 4, mie_blur: int = 5, videos_dir=None, single_class: bool = False,
                  mie_local_kernel: int = 0, missing_as_empty: bool = False, overwrite: bool = False,
                  human_only: list | None = None, log: Callable[[str], None] = print) -> Path:
    """Collect labelled frames from one or more labelling batches into an Ultralytics dataset.

    Each batch folder contains ``images/`` (from ``beemon extract`` or ``beemon workflow label``),
    ``manifest.csv`` and labels: the unzipped CVAT "YOLO 1.1" export (anywhere inside the batch)
    and/or automatic labels in ``labels_auto/``. Human labels win over automatic ones; for batches
    listed in ``human_only`` only the human-reviewed frames are used. A frame that occurs in several
    batches is used once.
    """
    mode = mode.lower()
    if mode not in ("rgb", "mie"):
        raise ValueError("mode must be rgb or mie")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} already exists. Use a new name (e.g. own_rgb_v2) or --overwrite.")
        shutil.rmtree(out)
    target_names = load_class_names(classes_yaml)
    split = load_split(split_yaml)

    human_only = {Path(b).resolve() for b in (human_only or [])}
    chosen: dict[str, tuple] = {}  # stem -> (image_path, labels, video, frame, manifest_row, is_auto)
    for batch in map(Path, batches):
        manifest = _manifest(batch)
        label_files = find_label_files(batch)
        listed = cvat_listed(batch)
        missing = skipped_auto = 0
        for image in list_images(batch / "images"):
            entry = label_files.get(image.stem)
            if entry is not None and entry[2] and batch.resolve() in human_only:
                entry = None
                skipped_auto += 1
            if entry is None and image.stem in listed:
                entry = (None, None, False)  # in the CVAT task but no boxes -> CVAT wrote no file: background
            if entry is None and not missing_as_empty:
                missing += 1
                continue
            path, names, is_auto = entry if entry else (None, None, False)
            labels = read_labels(path) if path else np.zeros((0, 5))
            if len(labels):
                labels = _remap(labels, names or target_names, target_names, single_class, path, classes_yaml)
            row = manifest.get(image.stem)
            parsed = parse_frame_name(image.stem)
            video = row["video"] if row else parsed[0] if parsed else None
            frame = int(row["frame"]) if row else parsed[1] if parsed else None
            if video is None:
                raise ValueError(f"Cannot tell which video {image.name} comes from (no manifest row, name not "
                                 "<video>__f<frame>). Extract frames with `beemon extract`.")
            previous = chosen.get(image.stem)
            if previous is not None and not previous[5] and is_auto:
                continue  # a human label of the same frame from another batch wins
            chosen[image.stem] = (image, labels, video, frame, row, is_auto)
        if missing:
            hint = (" (not checked by a human yet — finish them in the bee labeller)" if is_auto_batch(batch)
                    else " (use --missing-as-empty if unlabelled frames really contain no insect)")
            log(f"{batch.name}: {missing} images without labels skipped{hint}")
        if skipped_auto:
            log(f"{batch.name}: {skipped_auto} automatic labels ignored (only human-reviewed frames used)")
    items = [value[:5] for value in chosen.values()]
    if not items:
        raise ValueError("No labelled images found. Did you unzip the CVAT export into the batch folder?")
    n_auto = sum(1 for value in chosen.values() if value[5])
    log(f"{len(items)} labelled frames: {len(items) - n_auto} human, {n_auto} automatic")

    assignment = _assign_splits(sorted({it[2] for it in items}), split, log)
    if not any(s == "val" for s in assignment.values()):
        rng = random.Random(split.get("seed", 42))
        frame_val = {id(it) for it in items if assignment[it[2]] == "train" and rng.random() < 0.2}
    else:
        frame_val = set()

    names = ["insect"] if single_class else target_names
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    counts = {s: Counter() for s in SPLITS}
    n_images = Counter()
    split_rows = []
    by_video: dict[str, list] = defaultdict(list)
    for it in items:
        by_video[it[2]].append(it)

    for video, video_items in by_video.items():
        mie_frames = {}
        if mode == "mie":
            video_path = _find_video(video_items[0][4], video, Path(videos_dir) if videos_dir else None)
            if video_path is None:
                raise FileNotFoundError(f"MIE mode needs the source video of {video}. Put it in --videos-dir.")
            wanted = sorted({it[3] for it in video_items})
            for pack in iter_frames(video_path, wanted, mie_offset):
                fd = frame_difference(pack.prev, pack.frame, pack.next, mie_blur, mie_local_kernel)
                mie_frames[pack.index] = mie_image(pack.frame, fd)
        for it in video_items:
            image, labels, _, frame, _ = it
            s = "val" if id(it) in frame_val else assignment[video]
            dest = out / "images" / s / (image.stem + ".jpg")
            if mode == "mie":
                if frame not in mie_frames:
                    log(f"WARNING: frame {frame} of {video} could not be read; skipped")
                    continue
                cv2.imwrite(str(dest), mie_frames[frame], [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif image.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copy2(image, dest)
            else:
                cv2.imwrite(str(dest), cv2.imread(str(image)))
            write_labels(out / "labels" / s / (image.stem + ".txt"), labels)
            n_images[s] += 1
            counts[s].update(int(c) for c in labels[:, 0])
            split_rows.append({"image": dest.name, "video": video, "frame": frame, "split": s, "boxes": len(labels)})

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    info = {
        "mode": mode,
        "mie_offset": mie_offset if mode == "mie" else None,
        "mie_local_kernel": mie_local_kernel if mode == "mie" else None,
        "single_class": single_class,
        "batches": [str(Path(b).resolve()) for b in batches],
        "videos": {s: sorted(v for v, a in assignment.items() if a == s) for s in SPLITS},
        "images": dict(n_images),
        "boxes_per_class": {s: {names[c]: n for c, n in sorted(counts[s].items())} for s in SPLITS},
        "empty_images": {s: sum(1 for r in split_rows if r["split"] == s and r["boxes"] == 0) for s in SPLITS},
    }
    (out / "dataset_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    with (out / "split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "video", "frame", "split", "boxes"])
        writer.writeheader()
        writer.writerows(split_rows)
    log(f"dataset written to {out}")
    for s in SPLITS:
        log(f"  {s:5s}: {n_images[s]:5d} images, boxes {dict(info['boxes_per_class'][s])}, "
            f"empty {info['empty_images'][s]}, videos {info['videos'][s]}")
    if n_images["test"] == 0:
        log("WARNING: the test split is empty — list your test videos in configs/split.yaml and label frames from them.")
    return out


def tile_dataset(data_yaml, out, size: int = 960, overlap: float = 0.2, empty_keep: float = 0.25,
                 min_visible: float = 0.5, splits=SPLITS, seed: int = 0, log: Callable[[str], None] = print) -> Path:
    """Cut every image into overlapping tiles (SAHI-style training). Train with imgsz = tile size."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists")
    rng = random.Random(seed)
    totals = Counter()
    for s in splits:
        dirs = data_yaml_split_dirs(data_yaml, s)
        if not dirs:
            continue
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for folder in dirs:
            for image_path in list_images(folder):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                labels = read_labels(label_path_for(image_path))
                boxes = yolo_to_xyxy(labels, width, height)
                areas = np.maximum(1e-6, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
                for x0, y0, x1, y1 in tile_windows(width, height, size, overlap):
                    cx1, cy1 = np.clip(boxes[:, 0], x0, x1), np.clip(boxes[:, 1], y0, y1)
                    cx2, cy2 = np.clip(boxes[:, 2], x0, x1), np.clip(boxes[:, 3], y0, y1)
                    visible = (cx2 - cx1) * (cy2 - cy1) / areas
                    keep = visible >= min_visible
                    tw, th = x1 - x0, y1 - y0
                    if not keep.any() and rng.random() > empty_keep:
                        continue
                    tile_labels = np.stack([
                        labels[keep, 0], ((cx1 + cx2) / 2 - x0)[keep] / tw, ((cy1 + cy2) / 2 - y0)[keep] / th,
                        (cx2 - cx1)[keep] / tw, (cy2 - cy1)[keep] / th], axis=1) if keep.any() else np.zeros((0, 5))
                    stem = f"{image_path.stem}__t{x0}_{y0}"
                    cv2.imwrite(str(out / "images" / s / f"{stem}.jpg"), image[y0:y1, x0:x1],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    write_labels(out / "labels" / s / f"{stem}.txt", tile_labels)
                    totals[s] += 1
    new = {"path": str(out.resolve()), **{s: f"images/{s}" for s in splits if totals[s]}, "names": data["names"]}
    (out / "data.yaml").write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    (out / "dataset_info.yaml").write_text(yaml.safe_dump(
        {"tiled_from": str(data_yaml.resolve()), "tile_size": size, "overlap": overlap, "tiles": dict(totals)},
        sort_keys=False), encoding="utf-8")
    log(f"tiles written to {out}: {dict(totals)}  -> train with --imgsz {size}, process with detector.tiles.enabled")
    return out
