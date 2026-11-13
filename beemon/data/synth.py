"""Synthetic training images by copy-paste: real insect crops pasted onto empty frames of your own videos.

Why: rare classes (bumblebee, moth, hoverfly) get more examples, and the detector sees insects
on exactly the background it has to work on. The crops get a soft elliptical edge so the
rectangular crop border does not become a shortcut feature, and (in RGB mode) their brightness
and contrast are matched to the place they are pasted.

Works for MIE datasets too: crops and backgrounds then both come from MIE images, so a pasted
insect keeps its motion signal in the red channel (no colour matching is applied).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from beemon.data.common import (
    data_yaml_split_dirs, label_path_for, list_images, read_labels, write_labels, yolo_to_xyxy, xyxy_to_yolo,
)


def _collect_crops(data_yaml: Path, pad: float, min_size: int, max_per_class: int, rng: random.Random):
    pool = defaultdict(list)
    for folder in data_yaml_split_dirs(data_yaml, "train"):
        for image_path in list_images(folder):
            labels = read_labels(label_path_for(image_path))
            if not len(labels):
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for (x1, y1, x2, y2), cls in zip(yolo_to_xyxy(labels, width, height), labels[:, 0].astype(int)):
                w, h = x2 - x1, y2 - y1
                if min(w, h) < min_size:
                    continue
                px, py = w * pad, h * pad
                X1, Y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
                X2, Y2 = int(min(width, x2 + px)), int(min(height, y2 + py))
                crop = image[Y1:Y2, X1:X2].copy()
                inner = (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)  # object box inside the crop
                pool[int(cls)].append((crop, inner))
    for cls in pool:  # cap memory use for very frequent classes
        if len(pool[cls]) > max_per_class:
            pool[cls] = rng.sample(pool[cls], max_per_class)
    return pool


def _backgrounds(data_yaml: Path, extra: Path | None) -> list[Path]:
    paths = []
    for folder in data_yaml_split_dirs(data_yaml, "train"):
        paths += [p for p in list_images(folder) if len(read_labels(label_path_for(p))) == 0]
    if extra:
        paths += list_images(extra)
    return paths


def _soft_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, int(w * 0.46)), max(1, int(h * 0.46))), 0, 0, 360, 1.0, -1)
    sigma = max(1.0, 0.08 * min(h, w))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)


def _match_colors(crop: np.ndarray, patch: np.ndarray, strength: float = 0.5) -> np.ndarray:
    c = crop.astype(np.float32)
    p = patch.astype(np.float32)
    c_mean, c_std = c.mean(axis=(0, 1)), c.std(axis=(0, 1)) + 1e-3
    p_mean, p_std = p.mean(axis=(0, 1)), p.std(axis=(0, 1)) + 1e-3
    target_mean = c_mean * (1 - strength) + p_mean * strength
    target_std = c_std * (1 - strength) + p_std * strength
    return np.clip((c - c_mean) / c_std * target_std + target_mean, 0, 255).astype(np.uint8)


def make_synthetic(data_yaml, out, n_images: int = 500, per_image=(1, 4), extra_backgrounds=None,
                   scale=(0.8, 1.25), balance_classes: bool = True, pad: float = 0.3, min_size: int = 8,
                   mie: bool | None = None, seed: int = 0, log: Callable[[str], None] = print) -> Path:
    data_yaml = Path(data_yaml)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists")
    rng = random.Random(seed)
    info_path = data_yaml.parent / "dataset_info.yaml"
    if mie is None and info_path.is_file():
        mie = (yaml.safe_load(info_path.read_text(encoding="utf-8")) or {}).get("mode") == "mie"
    pool = _collect_crops(data_yaml, pad, min_size, 3000, rng)
    if not pool:
        raise ValueError("No insect boxes found in the train split to copy.")
    backgrounds = _backgrounds(data_yaml, Path(extra_backgrounds) if extra_backgrounds else None)
    if not backgrounds:
        raise ValueError("No empty background frames: label some frames without insects (empty label files) "
                         "or pass --backgrounds with a folder of insect-free frames.")
    classes = sorted(pool)
    log(f"crops per class: { {c: len(pool[c]) for c in classes} }, backgrounds: {len(backgrounds)}")

    img_dir, lbl_dir = out / "images" / "train", out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    pasted = defaultdict(int)
    for index in range(n_images):
        background = cv2.imread(str(rng.choice(backgrounds)))
        if background is None:
            continue
        height, width = background.shape[:2]
        boxes, classes_out = [], []
        for _ in range(rng.randint(*per_image)):
            cls = rng.choice(classes) if balance_classes else rng.choices(classes, [len(pool[c]) for c in classes])[0]
            crop, inner = rng.choice(pool[cls])
            s = rng.uniform(*scale)
            crop = cv2.resize(crop, (max(4, int(crop.shape[1] * s)), max(4, int(crop.shape[0] * s))))
            ix1, iy1, ix2, iy2 = [v * s for v in inner]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
                ix1, ix2 = crop.shape[1] - ix2, crop.shape[1] - ix1
            if rng.random() < 0.5:
                crop = crop[::-1, :]
                iy1, iy2 = crop.shape[0] - iy2, crop.shape[0] - iy1
            ch, cw = crop.shape[:2]
            if cw >= width or ch >= height:
                continue
            for _attempt in range(20):
                x0, y0 = rng.randint(0, width - cw), rng.randint(0, height - ch)
                new = np.array([x0 + ix1, y0 + iy1, x0 + ix2, y0 + iy2])
                if all(new[2] < b[0] or new[0] > b[2] or new[3] < b[1] or new[1] > b[3] for b in boxes):
                    break
            else:
                continue
            patch = background[y0:y0 + ch, x0:x0 + cw]
            source = np.ascontiguousarray(crop) if mie else _match_colors(np.ascontiguousarray(crop), patch)
            alpha = _soft_mask(ch, cw)[..., None]
            background[y0:y0 + ch, x0:x0 + cw] = (source * alpha + patch * (1 - alpha)).astype(np.uint8)
            boxes.append(new)
            classes_out.append(cls)
            pasted[cls] += 1
        stem = f"synth_{index:06d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), background, [cv2.IMWRITE_JPEG_QUALITY, 95])
        labels = xyxy_to_yolo(np.array(boxes).reshape(-1, 4), np.array(classes_out), width, height)
        write_labels(lbl_dir / f"{stem}.txt", labels)

    source = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    train_dirs = [str(p) for p in data_yaml_split_dirs(data_yaml, "train")] + [str(img_dir.resolve())]
    merged = {**{k: v for k, v in source.items() if k != "path"},
              "train": train_dirs,
              "val": [str(p) for p in data_yaml_split_dirs(data_yaml, "val")],
              "test": [str(p) for p in data_yaml_split_dirs(data_yaml, "test")]}
    (out / "data.yaml").write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    (out / "dataset_info.yaml").write_text(yaml.safe_dump(
        {"synthetic_from": str(data_yaml.resolve()), "images": n_images, "pasted_per_class": dict(pasted),
         "mode": "mie" if mie else "rgb"}, sort_keys=False), encoding="utf-8")
    log(f"{n_images} synthetic images in {out}; pasted per class {dict(pasted)}")
    log(f"train on original + synthetic with:  --data {out / 'data.yaml'}")
    return out
