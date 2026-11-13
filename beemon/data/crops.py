"""Crop dataset for the optional species classifier (two-stage: detect "insect", then classify the crop).

Output is an image-folder dataset (``train/<class>/*.jpg``, ``val/...``, ``test/...``) that
`beemon train --preset classify` understands. With ``--fp-weights`` the detector is run on the
training images and its false positives (no matching label) become the ``not_insect`` class, so
the classifier also acts as a second false-positive filter in the pipeline.
Build the crops from an RGB dataset: the pipeline classifies RGB crops.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from beemon.data.common import data_yaml_split_dirs, label_path_for, list_images, read_labels, yolo_to_xyxy
from beemon.detector import crop_box


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(1e-9, area_a[:, None] + area_b[None, :] - inter)


def make_crops(data_yaml, out, size: int = 128, pad: float = 0.25, fp_weights=None, fp_conf: float = 0.3,
               fp_imgsz: int = 1280, fp_max_per_split: int = 3000, reject_class: str = "not_insect",
               log: Callable[[str], None] = print) -> Path:
    data_yaml = Path(data_yaml)
    info_path = data_yaml.parent / "dataset_info.yaml"
    if info_path.is_file() and (yaml.safe_load(info_path.read_text(encoding="utf-8")) or {}).get("mode") == "mie":
        raise ValueError("This is an MIE dataset; build crops from the RGB version of the same labels.")
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    out = Path(out)
    model = None
    if fp_weights:
        from ultralytics import YOLO

        model = YOLO(str(fp_weights))
    counts = {}
    for split in ("train", "val", "test"):
        dirs = data_yaml_split_dirs(data_yaml, split)
        counter = Counter()
        for folder in dirs:
            for image_path in list_images(folder):
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                labels = read_labels(label_path_for(image_path))
                boxes = yolo_to_xyxy(labels, width, height)
                for k, (box, cls) in enumerate(zip(boxes, labels[:, 0].astype(int))):
                    target = out / split / names[cls]
                    target.mkdir(parents=True, exist_ok=True)
                    crop = cv2.resize(crop_box(image, box, pad), (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(target / f"{image_path.stem}_{k}.jpg"), crop)
                    counter[names[cls]] += 1
                if model is not None and counter[reject_class] < fp_max_per_split:
                    result = model.predict(image, imgsz=fp_imgsz, conf=fp_conf, verbose=False)[0]
                    if result.boxes is not None and len(result.boxes):
                        pred = result.boxes.xyxy.cpu().numpy()
                        overlap = _iou(pred, boxes).max(axis=1) if len(boxes) else np.zeros(len(pred))
                        for k, box in enumerate(pred[overlap < 0.1]):
                            target = out / split / reject_class
                            target.mkdir(parents=True, exist_ok=True)
                            crop = cv2.resize(crop_box(image, box, pad), (size, size), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(str(target / f"{image_path.stem}_fp{k}.jpg"), crop)
                            counter[reject_class] += 1
        counts[split] = dict(counter)
        log(f"{split}: {dict(counter)}")
    all_classes = set().union(*(set(c) for c in counts.values()))
    for split, counter in counts.items():
        missing = sorted(all_classes - set(counter))
        if missing and split in ("train", "val"):
            log(f"WARNING: {split} has no crops of {missing}. Ultralytics needs every class in train AND val: "
                "label more of them or merge rare classes.")
    (out / "crops_info.yaml").write_text(yaml.safe_dump({"from": str(data_yaml.resolve()), "size": size,
                                                         "pad": pad, "counts": counts}, sort_keys=False),
                                         encoding="utf-8")
    log(f"crop dataset in {out}  -> beemon train --preset classify --data {out}")
    return out
