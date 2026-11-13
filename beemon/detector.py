"""Detector wrapper around Ultralytics YOLO with optional sliced (tiled) inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detections:
    """Boxes for one frame. ``xyxy`` in pixels of the full frame."""

    xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    conf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    motion_in: np.ndarray | None = None
    motion_ratio: np.ndarray | None = None

    def __post_init__(self):
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(self.conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(self.cls, dtype=np.int32).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    @property
    def centers(self) -> np.ndarray:
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1)

    @property
    def wh(self) -> np.ndarray:
        return np.stack([self.xyxy[:, 2] - self.xyxy[:, 0], self.xyxy[:, 3] - self.xyxy[:, 1]], axis=1)

    def subset(self, index) -> "Detections":
        index = np.asarray(index)
        return Detections(
            self.xyxy[index],
            self.conf[index],
            self.cls[index],
            None if self.motion_in is None else self.motion_in[index],
            None if self.motion_ratio is None else self.motion_ratio[index],
        )


def nms(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-agnostic non-maximum suppression; returns kept indices (highest score first)."""
    if len(scores) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = xyxy.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def tile_windows(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping square windows covering the image (last row/column aligned to the border)."""
    size = int(size)
    step = max(1, int(round(size * (1.0 - overlap))))

    def starts(total: int) -> list[int]:
        if total <= size:
            return [0]
        values = list(range(0, total - size, step))
        values.append(total - size)
        return sorted(set(values))

    return [(x, y, min(width, x + size), min(height, y + size)) for y in starts(height) for x in starts(width)]


def friendly_names(names: dict | list) -> dict[int, str]:
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
    if names == {0: "item"}:  # Ultralytics names single-class models "item"
        return {0: "insect"}
    return {int(k): str(v) for k, v in names.items()}


class YoloDetector:
    """Runs a YOLO detection model on one image, optionally in overlapping tiles."""

    def __init__(
        self,
        weights: str | Path,
        imgsz: int = 1280,
        conf: float = 0.1,
        iou: float = 0.5,
        device: str | None = None,
        classes: list[int] | None = None,
        max_det: int = 300,
        tiles: dict | None = None,
        augment: bool = False,
    ):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Detector weights not found: {weights}\n"
                "Copy your trained best.pt into models/ and set detector.weights in configs/pipeline.yaml "
                "(or pass --weights)."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if getattr(self.model, "task", "detect") != "detect":
            raise ValueError(f"{weights} is a '{self.model.task}' model, expected a detection model.")
        self.names = friendly_names(self.model.names)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.classes = classes
        self.max_det = int(max_det)
        self.tiles = tiles or {}
        self.augment = bool(augment)

    def _predict(self, images: list[np.ndarray], imgsz: int) -> list:
        kwargs = dict(imgsz=imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det, verbose=False,
                      augment=self.augment)
        if self.device not in (None, ""):
            kwargs["device"] = self.device
        if self.classes:
            kwargs["classes"] = self.classes
        return self.model.predict(images, **kwargs)

    @staticmethod
    def _to_arrays(result, dx: float = 0.0, dy: float = 0.0):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        xyxy[:, [0, 2]] += dx
        xyxy[:, [1, 3]] += dy
        return xyxy, boxes.conf.cpu().numpy().astype(np.float32), boxes.cls.cpu().numpy().astype(np.int32)

    def __call__(self, image: np.ndarray) -> Detections:
        if not self.tiles.get("enabled"):
            return Detections(*self._to_arrays(self._predict([image], self.imgsz)[0]))

        height, width = image.shape[:2]
        size = int(self.tiles.get("size", 960))
        windows = tile_windows(width, height, size, float(self.tiles.get("overlap", 0.2)))
        crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in windows]
        parts = [self._to_arrays(r, x0, y0) for r, (x0, y0, _, _) in zip(self._predict(crops, size), windows)]
        if self.tiles.get("include_full_frame", True):
            parts.append(self._to_arrays(self._predict([image], self.imgsz)[0]))
        xyxy = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 4), np.float32)
        conf = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.float32)
        cls = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int32)
        keep = nms(xyxy, conf, self.iou)
        return Detections(xyxy[keep], conf[keep], cls[keep])


class CropClassifier:
    """Optional second stage: YOLO classification model applied to crops of a track."""

    def __init__(self, weights: str | Path, imgsz: int = 128, device: str | None = None):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Classifier weights not found: {weights}")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if getattr(self.model, "task", None) != "classify":
            raise ValueError(f"{weights} is not a classification model.")
        self.names = friendly_names(self.model.names)
        self.imgsz = int(imgsz)
        self.device = device

    def probabilities(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return an (N, n_classes) array of class probabilities."""
        if not crops:
            return np.zeros((0, len(self.names)), dtype=np.float32)
        kwargs = dict(imgsz=self.imgsz, verbose=False)
        if self.device not in (None, ""):
            kwargs["device"] = self.device
        results = self.model.predict(crops, **kwargs)
        return np.stack([r.probs.data.cpu().numpy() for r in results]).astype(np.float32)


def crop_box(frame: np.ndarray, box: np.ndarray, pad: float = 0.25, square: bool = True) -> np.ndarray:
    """Crop a (padded, optionally square) region around a box, clamped to the frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    if square:
        w = h = max(w, h, 8.0)
    x0, y0 = int(max(0, round(cx - w / 2))), int(max(0, round(cy - h / 2)))
    x1i, y1i = int(min(width, round(cx + w / 2))), int(min(height, round(cy + h / 2)))
    if x1i <= x0 or y1i <= y0:
        return np.zeros((8, 8, 3), dtype=frame.dtype)
    return frame[y0:y1i, x0:x1i].copy()
