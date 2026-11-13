"""Training presets around Ultralytics (run on a GPU machine).

Stages (see README):
  pretrain  — COCO weights -> public dataset (BuzzSet). Learns "what an insect looks like".
  finetune  — pretrain best.pt -> YOUR labelled frames (RGB or MIE). Learns your camera and field.
  classify  — optional crop classifier for the two-stage setup.
Any preset value can be overridden (``--epochs 50``) and any other Ultralytics argument passed
with ``--set key=value``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PRESETS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "weights": "yolo11s.pt", "imgsz": 1280, "epochs": 150, "patience": 30, "batch": 16, "lr0": 0.01,
        "warmup_epochs": 3.0, "close_mosaic": 15, "cos_lr": True,
    },
    "finetune": {
        "weights": None, "imgsz": 1280, "epochs": 100, "patience": 25, "batch": 16, "lr0": 0.002,
        "warmup_epochs": 1.0, "close_mosaic": 10, "cos_lr": True,
        # field video: allow vertical flips and rotations (top-down camera has no "up")
        "flipud": 0.5, "degrees": 10.0,
    },
    "classify": {
        "weights": "yolo11s-cls.pt", "imgsz": 128, "epochs": 60, "patience": 15, "batch": 128, "lr0": 0.001,
        "warmup_epochs": 2.0, "cos_lr": True,
    },
}


def train(preset: str, data: str, weights: str | None = None, name: str | None = None, project: str = "runs",
          single_class: bool = False, device: str | None = None, workers: int = 8, cache: str = "disk",
          resume: bool = False, test_after: bool = True, extra: dict | None = None, **overrides) -> Path:
    from ultralytics import YOLO

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}: choose from {list(PRESETS)}")
    args = {**PRESETS[preset], **{k: v for k, v in overrides.items() if v is not None}}
    if isinstance(args.get("batch"), float) and args["batch"] >= 1:
        args["batch"] = int(args["batch"])  # <1 means "fraction of GPU memory" in Ultralytics
    weights = weights or args.pop("weights", None)
    args.pop("weights", None)
    if not weights:
        raise ValueError(f"--weights is required for the {preset} preset (use the best.pt of the previous stage)")
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data}")

    model = YOLO(str(weights))
    if resume:
        results = model.train(resume=True)
    else:
        args.update({
            "data": str(data_path.resolve()),
            "project": str(Path(project).resolve()),
            "name": name or f"{Path(weights).stem}_{preset}",
            "exist_ok": False,
            "workers": workers,
            "cache": False if cache == "none" else cache,
            "plots": True,
        })
        if device is not None:
            args["device"] = device
        if single_class and preset != "classify":
            args["single_cls"] = True
        args.update(extra or {})
        print("Training with:\n" + yaml.safe_dump(args, sort_keys=False))
        results = model.train(**args)
    save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    if test_after and preset != "classify" and best.is_file():
        evaluate(best, data_path, split="test", imgsz=args.get("imgsz", 1280), device=device,
                 project=str(save_dir), name="test_eval")
    return best


def evaluate(weights, data, split: str = "test", imgsz: int = 1280, device=None, project=None, name="eval") -> dict:
    """Detection metrics of a model on one split (precision, recall, mAP50, mAP50-95)."""
    from ultralytics import YOLO

    from beemon.data.common import data_yaml_split_dirs, list_images

    data = Path(data)
    content = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not content.get(split) or not any(list_images(d) for d in data_yaml_split_dirs(data, split) if d.exists()):
        print(f"No images in the '{split}' split of {data}; skipping evaluation "
              "(label frames of your test videos, docs/TRAINING.md).")
        return {}
    from beemon.config import PROJECT_DIR

    # always pass a project: otherwise Ultralytics writes to its global runs_dir (another project!)
    kwargs = dict(data=str(data.resolve()), split=split, imgsz=imgsz, plots=True, name=name,
                  project=str(project or PROJECT_DIR / "runs" / "eval"))
    if device is not None:
        kwargs["device"] = device
    metrics = YOLO(str(weights)).val(**kwargs)
    box = metrics.box
    result = {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
              "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4)}
    print(f"{split}: P={result['precision']:.3f} R={result['recall']:.3f} "
          f"mAP50={result['mAP50']:.3f} mAP50-95={result['mAP50-95']:.3f}")
    return result


def clean_weights(src, dst=None) -> Path:
    """Make a trained .pt safe to share: drop the optimizer state and remove the folder paths of the
    training computer (Ultralytics stores them in ``train_args`` and inside the model)."""
    import torch

    src = Path(src)
    dst = Path(dst) if dst else src
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    def clean(value):
        if isinstance(value, str) and ("/" in value or "\\" in value) and Path(value).is_absolute():
            return Path(value).name
        if isinstance(value, Path) and value.is_absolute():
            return value.name
        return value

    def clean_args(args):
        if isinstance(args, dict):
            return {k: clean(v) for k, v in args.items()}
        if args is not None and hasattr(args, "__dict__"):
            for k, v in vars(args).items():
                setattr(args, k, clean(v))
        return args

    ckpt["train_args"] = clean_args(ckpt.get("train_args"))
    for key in ("model", "ema"):
        model = ckpt.get(key)
        if model is not None and hasattr(model, "args"):
            model.args = clean_args(model.args)
    ckpt["optimizer"] = None
    ckpt["git"] = None
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    return dst
