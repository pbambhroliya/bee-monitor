"""Labelling batches on disk — no dependencies besides the standard library and Pillow.

A batch folder looks like this (made by `beemon extract`, `beemon workflow label` or
`beemon labeling-package`):

    <batch>/images/*.jpg              the pictures
    <batch>/review/review_list.txt    optional: only these pictures need labelling (review mode)
    <batch>/review/cvat_review.zip    optional: suggested boxes for them
    <batch>/labels_auto/*.txt         optional: suggested boxes (automatic labels)
    <batch>/cvat_prelabels.zip        optional: suggested boxes (from `beemon extract --prelabel`)
    <batch>/labels_human/*.txt        ← written here: one YOLO file per finished picture (+ obj.names, log.csv)
"""

from __future__ import annotations

import csv
import io
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

DEFAULT_CLASSES = ["bee", "bumblebee", "hoverfly", "other_insect", "moth"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
HUMAN_DIR = "labels_human"


def _read_yolo_text(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(v) for v in parts[:5]])
    return rows


def _zip_labels(path: Path) -> tuple[dict[str, list], list[str] | None]:
    labels, names = {}, None
    if not path.is_file():
        return labels, names
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith("obj.names"):
                names = [n.strip() for n in archive.read(name).decode().splitlines() if n.strip()]
            elif name.endswith(".txt") and "obj_train_data/" in name:
                labels[Path(name).stem] = _read_yolo_text(archive.read(name).decode())
    return labels, names


@dataclass
class Batch:
    root: Path
    images: list[Path]
    classes: list[str]
    review_mode: bool = False
    suggestions: dict[str, list] = field(default_factory=dict)  # stem -> YOLO rows

    @property
    def out_dir(self) -> Path:
        return self.root / HUMAN_DIR

    def done(self) -> set[str]:
        return {p.stem for p in self.out_dir.glob("*.txt")} if self.out_dir.is_dir() else set()

    def boxes_for(self, image: Path) -> tuple[list[dict], bool]:
        """Boxes in pixels for the picture: the saved human labels if any, else the suggestions."""
        width, height = image_size(image)
        saved = self.out_dir / f"{image.stem}.txt"
        rows = _read_yolo_text(saved.read_text()) if saved.is_file() else self.suggestions.get(image.stem, [])
        boxes = [{"x1": (cx - w / 2) * width, "y1": (cy - h / 2) * height, "x2": (cx + w / 2) * width,
                  "y2": (cy + h / 2) * height, "cls": min(int(c), len(self.classes) - 1)}
                 for c, cx, cy, w, h in rows]
        return boxes, saved.is_file()

    def save(self, image: Path, boxes: list[dict], annotator: str) -> None:
        width, height = image_size(image)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "obj.names").write_text("\n".join(self.classes) + "\n")
        lines = []
        for b in boxes:
            x1, x2 = sorted((max(0.0, min(width, b["x1"])), max(0.0, min(width, b["x2"]))))
            y1, y2 = sorted((max(0.0, min(height, b["y1"])), max(0.0, min(height, b["y2"]))))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{int(b['cls'])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        (self.out_dir / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        log = self.out_dir / "log.csv"
        new = not log.is_file()
        with log.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["image", "annotator", "boxes", "time"])
            writer.writerow([image.name, annotator, len(lines), time.strftime("%Y-%m-%d %H:%M:%S")])

    def typical_box_size(self) -> float:
        sizes = []
        for rows in list(self.suggestions.values())[:400]:
            for _, _, _, w, h in rows:
                sizes.append(max(w * 1920, h * 1080))
        sizes.sort()
        return round(sizes[len(sizes) // 2]) if sizes else 40.0

    def export_zip(self, annotator: str = "") -> bytes:
        """The finished labels as a zip (to send back to whoever builds the dataset)."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.out_dir.glob("*")):
                archive.write(path, f"{HUMAN_DIR}/{path.name}")
            archive.writestr("batch.txt", f"{self.root.name}\n{annotator}\n")
        return buffer.getvalue()


_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def image_size(path: Path) -> tuple[int, int]:
    key = str(path)
    if key not in _SIZE_CACHE:
        with Image.open(path) as im:
            _SIZE_CACHE[key] = im.size
    return _SIZE_CACHE[key]


def load_batch(root: str | Path) -> Batch:
    root = Path(root)
    images = sorted(p for p in (root / "images").glob("*") if p.suffix.lower() in IMAGE_EXT)
    if not images:
        raise FileNotFoundError(f"No pictures in {root / 'images'}")
    review_list = root / "review" / "review_list.txt"
    review_mode = review_list.is_file()
    if review_mode:
        wanted = {Path(n.strip()).stem for n in review_list.read_text().splitlines() if n.strip()}
        images = [p for p in images if p.stem in wanted]

    suggestions: dict[str, list] = {}
    classes = None
    for zip_path in (root / "cvat_prelabels.zip", root / "review" / "cvat_review.zip"):
        labels, names = _zip_labels(zip_path)
        suggestions.update(labels)
        classes = names or classes
    for auto in (root / "labels_suggested", root / "labels_auto"):   # boxes to start from (not training labels)
        if auto.is_dir():
            for path in auto.glob("*.txt"):
                suggestions.setdefault(path.stem, _read_yolo_text(path.read_text()))
            if (auto / "obj.names").is_file():
                classes = classes or [n.strip() for n in (auto / "obj.names").read_text().splitlines() if n.strip()]
    human_names = root / HUMAN_DIR / "obj.names"
    if human_names.is_file():
        classes = [n.strip() for n in human_names.read_text().splitlines() if n.strip()]
    return Batch(root, images, classes or DEFAULT_CLASSES, review_mode, suggestions)


def find_batches(base: str | Path) -> list[Path]:
    base = Path(base)
    if (base / "images").is_dir():
        return [base]
    return sorted(p.parent for p in base.glob("*/images") if p.is_dir())
