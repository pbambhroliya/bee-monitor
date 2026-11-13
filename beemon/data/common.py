"""Shared helpers for labels, frame names, CVAT packages and split files."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAME_PATTERN = re.compile(r"^(?P<video>.+)__f(?P<frame>\d+)$")


def frame_name(video_stem: str, frame: int) -> str:
    """File stem for a frame: ``<video>__f0001234``. Encodes where the frame came from."""
    return f"{video_stem}__f{int(frame):07d}"


def parse_frame_name(stem: str) -> tuple[str, int] | None:
    match = NAME_PATTERN.match(stem)
    return (match.group("video"), int(match.group("frame"))) if match else None


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_labels(path: str | Path) -> np.ndarray:
    """YOLO label file -> (N, 5) array [cls, cx, cy, w, h] (normalised)."""
    path = Path(path)
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float64)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(v) for v in parts[:5]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def write_labels(path: str | Path, labels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in np.asarray(labels).reshape(-1, 5)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1, 5)
    cx, cy, w, h = labels[:, 1] * width, labels[:, 2] * height, labels[:, 3] * width, labels[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_yolo(xyxy: np.ndarray, cls: np.ndarray, width: int, height: int) -> np.ndarray:
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    x1, y1 = np.clip(xyxy[:, 0], 0, width), np.clip(xyxy[:, 1], 0, height)
    x2, y2 = np.clip(xyxy[:, 2], 0, width), np.clip(xyxy[:, 3], 0, height)
    out = np.stack([np.asarray(cls, dtype=np.float64), (x1 + x2) / 2 / width, (y1 + y2) / 2 / height,
                    (x2 - x1) / width, (y2 - y1) / height], axis=1)
    return out[(out[:, 3] > 0) & (out[:, 4] > 0)]


def load_class_names(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else data
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if not names:
        raise ValueError(f"No class names in {path}")
    return [str(n) for n in names]


def load_split(path: str | Path | None) -> dict:
    if not path or not Path(path).is_file():
        return {"test_videos": [], "val_videos": [], "val_fraction": 0.2, "seed": 42}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data["test_videos"] = [str(v) for v in (data.get("test_videos") or [])]
    data["val_videos"] = [str(v) for v in (data.get("val_videos") or [])]
    data.setdefault("val_fraction", 0.2)
    data.setdefault("seed", 42)
    return data


def write_cvat_yolo_package(zip_path: str | Path, names: list[str], items: list[tuple[str, np.ndarray]]) -> Path:
    """Write annotations in CVAT's "YOLO 1.1" import format.

    ``items`` are ``(image_file_name, labels)``. In CVAT: create the task from the images, then
    Actions -> Upload annotations -> YOLO 1.1 -> this zip.
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("obj.names", "\n".join(names) + "\n")
        archive.writestr("obj.data", f"classes = {len(names)}\nnames = data/obj.names\ntrain = data/train.txt\n"
                                     "backup = backup/\n")
        archive.writestr("train.txt", "\n".join(f"data/obj_train_data/{name}" for name, _ in items) + "\n")
        for name, labels in items:
            lines = [f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}" for r in np.asarray(labels).reshape(-1, 5)]
            archive.writestr(f"obj_train_data/{Path(name).stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
    return zip_path


AUTO_LABEL_DIR = "labels_auto"
_NOT_LABELS = {"train.txt", "valid.txt", "test.txt", "obj.names", "classes.txt", "review_list.txt",
               "random_sample.txt"}
_NOT_LABEL_DIRS = {"images", "review", "labels_suggested"}  # lists, CVAT packages, suggestions: never labels


def _names_near(path: Path, root: Path) -> list[str] | None:
    """obj.names / classes.txt in the label file's folder or the nearest parent inside ``root``."""
    for folder in [path.parent, *path.parent.parents]:
        for candidate in ("obj.names", "classes.txt"):
            file = folder / candidate
            if file.is_file():
                return [ln.strip() for ln in file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if folder == root:
            break
    return None


def find_label_files(folder: str | Path) -> dict[str, tuple[Path, list[str] | None, bool]]:
    """All YOLO label txt files in a labelling batch: ``stem -> (path, class names, is_auto)``.

    Human labels (e.g. an unzipped CVAT "YOLO 1.1" export) always win over automatic labels
    (written by `beemon workflow label` into ``labels_auto/``) for the same frame.
    """
    folder = Path(folder)
    found: dict[str, tuple[Path, list[str] | None, bool]] = {}
    for path in sorted(folder.rglob("*.txt")):
        if path.name in _NOT_LABELS or _NOT_LABEL_DIRS & set(path.relative_to(folder).parts):
            continue
        is_auto = AUTO_LABEL_DIR in path.relative_to(folder).parts
        if path.stem in found and is_auto and not found[path.stem][2]:
            continue  # keep the human label
        found[path.stem] = (path, _names_near(path, folder), is_auto)
    return found


def cvat_listed(folder: str | Path) -> set[str]:
    """Pictures listed in an unzipped CVAT export (its train.txt) — they were part of the CVAT task.

    CVAT writes no label file for a picture without boxes, so a listed picture without a label file
    is a reviewed, insect-free picture. (The bee labeller always writes a file, also an empty one.)
    """
    folder = Path(folder)
    stems = set()
    for listing in folder.rglob("train.txt"):
        parts = set(listing.relative_to(folder).parts)
        if parts & {"images", "review", AUTO_LABEL_DIR, "labels_suggested", "labels_human"}:
            continue
        stems |= {Path(line.strip()).stem for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()}
    return stems


def data_yaml_split_dirs(data_yaml: str | Path, split: str) -> list[Path]:
    """Image folders of one split of an Ultralytics data.yaml (supports a single dir or a list)."""
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    entry = data.get(split)
    if not entry:
        return []
    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entries = entry if isinstance(entry, list) else [entry]
    return [Path(e) if Path(e).is_absolute() else root / e for e in entries]


def label_path_for(image_path: Path) -> Path:
    """Ultralytics convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")
