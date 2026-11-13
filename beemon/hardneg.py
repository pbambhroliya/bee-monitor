"""Hard negatives: let the model look at pictures of a video, a person deletes the wrong boxes.

Every box the person deletes is a *hard negative* (something the model wrongly calls an insect:
flower centre, leaf, the orange stake ...). Training on these pictures teaches the model not to do it.

The files use common formats, so work from other review tools can be combined with it:

* work folder  ``contributions/work/<video>/review.json`` + ``frames/<video>/frame_*.jpg``
  (``beemon import-hardneg`` reads such session folders);
* the download ``hardneg_<name>_<date>.zip`` — a plain YOLO dataset (``images/{train,val,test}``,
  ``labels/...``, ``data.yaml``) plus ``review_report.csv`` (one row per box and decision) and
  ``contribution.json``. Other tools' YOLO dataset zips can be imported the same way.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Callable

import cv2

from beemon.config import PROJECT_DIR

CONTRIB_DIR = PROJECT_DIR / "contributions"
WORK_DIR = CONTRIB_DIR / "work"
RECEIVED_DIR = CONTRIB_DIR / "received"
DEFAULT_CLASSES = ["bee", "bumblebee", "hoverfly", "other_insect", "moth"]
CSV_FIELDS = ("video_filename", "frame_number", "timestamp_seconds", "frame_filename", "detection_id",
              "predicted_class", "predicted_class_id", "final_class_id", "final_class", "confidence",
              "x1", "y1", "x2", "y2", "review_status", "final_annotation_status", "frame_annotation_complete")


# ---------------------------------------------------------------------------------- state
# The picture finder (a background process) and the page never write the same file:
#   finder -> found.jsonl (one found picture per line) + finder.json (progress) + crops/
#   page   -> review.json (the decisions; it also holds the found pictures it has seen)
def session_dir(video: str) -> Path:
    return WORK_DIR / video


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def finder_status(video: str) -> dict:
    return _read_json(session_dir(video) / "finder.json", {})


def _write_finder(video: str, status: dict) -> None:
    folder = session_dir(video)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "finder.json.tmp"
    tmp.write_text(json.dumps(status), encoding="utf-8")
    tmp.replace(folder / "finder.json")


def load_state(video: str) -> dict:
    folder = session_dir(video)
    state = _read_json(folder / "review.json", None) or {"class_names": list(DEFAULT_CLASSES), "videos": {},
                                                          "frames": []}
    finder = finder_status(video)
    if finder.get("class_names"):
        state["class_names"] = finder["class_names"]
    if finder.get("video_info"):
        state["videos"].setdefault(video, {}).update(finder["video_info"])
    found = folder / "found.jsonl"
    if found.is_file():
        known = {f["frame_id"] for f in state["frames"]}
        added = False
        for line in found.read_text(encoding="utf-8").splitlines():
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written right now
            if frame["frame_id"] not in known:
                state["frames"].append(frame)
                known.add(frame["frame_id"])
                added = True
        if added:
            state["frames"].sort(key=lambda f: f["frame_number"])
    return state


def save_state(video: str, state: dict) -> None:
    folder = session_dir(video)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "review.json.tmp"
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(folder / "review.json")


def sessions() -> list[str]:
    if not WORK_DIR.is_dir():
        return []
    return sorted(p.name for p in WORK_DIR.iterdir() if (p / "review.json").is_file() or (p / "found.jsonl").is_file())


def frame_done(frame: dict) -> bool:
    """Reviewed completely: every box decided and the person confirmed the picture."""
    return bool(frame["annotation_complete"]) and bool(frame["detections"]) and all(
        d["status"] in ("true_positive", "false_positive") for d in frame["detections"])


def counts(state: dict) -> dict:
    frames = state["frames"]
    done = [f for f in frames if frame_done(f)]
    deleted = sum(d["status"] == "false_positive" for f in done for d in f["detections"])
    kept = sum(d["status"] == "true_positive" for f in done for d in f["detections"])
    return {"pictures": len(frames), "reviewed": len(done), "wrong_boxes": deleted, "insects": kept}


def letter(index: int) -> str:
    """Name of the k-th model box on a picture: A, B, ... Z, AA, AB ..."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(65 + rest) + name
    return name


def crop_path(video: str, detection_id: str) -> Path:
    return session_dir(video) / "crops" / f"{detection_id}.jpg"


# ---------------------------------------------------------------------------------- finding pictures
def _spread_order(candidates: list[int]) -> list[list[int]]:
    """Candidates in passes that each cover the whole video (coarse first), each pass sorted."""
    passes, seen, step = [], set(), 1
    while step * 2 < len(candidates):
        step *= 2
    while step >= 1:
        level = [c for i, c in enumerate(candidates) if i % step == 0 and c not in seen]
        seen.update(level)
        if level:
            passes.append(level)
        step //= 2
    return passes


def _motion_strip(frames: list, box: list[float], size: int = 120):
    """The box region 1 s before, now and 1 s after, side by side: an insect moves, a flower does not."""
    import numpy as np

    x1, y1, x2, y2 = box
    height, width = frames[1].shape[:2]
    side = max(72.0, 2.6 * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(min(max(0, cx - side / 2), max(0, width - side)))
    top = int(min(max(0, cy - side / 2), max(0, height - side)))
    side = int(min(side, width, height))
    scale = size / side
    tiles = []
    for frame in frames:
        tile = cv2.resize(frame[top:top + side, left:left + side], (size, size), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (int((x1 - left) * scale) - 2, int((y1 - top) * scale) - 2),
                      (int((x2 - left) * scale) + 2, int((y2 - top) * scale) + 2), (0, 220, 255), 1)
        tiles.append(tile)
    gap = np.full((size, 4, 3), 255, np.uint8)
    return np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]])


def find_pictures(video: str, video_path: str | Path, model, model_path: str, every_s: float = 10.0,
                  want: int = 30, imgsz: int = 1280, conf: float = 0.25,
                  progress: Callable[[float, str], None] | None = None) -> dict:
    """Run the model on pictures spread over the video until ``want`` pictures with boxes exist.

    Resumable: pictures already checked are skipped; every found picture is saved at once, so the
    person can start checking while the search goes on.
    """
    video_path = Path(video_path)
    folder = session_dir(video)
    state = load_state(video)
    names = [str(model.names[i]) for i in sorted(model.names)]
    if state["frames"] and state["class_names"] != names:
        raise ValueError(f"This model has other insect types ({names}) than the pictures already found "
                         f"({state['class_names']}). Use the same model as before.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open the video {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    old = finder_status(video)
    checked = list(old.get("checked_frames", []))
    status = {"status": "running", "pid": os.getpid(), "want": want, "found": len(state["frames"]),
              "looked": 0, "class_names": names, "checked_frames": checked, "started": time.time(),
              "video_info": {"filename": video_path.name, "path": str(video_path), "fps": fps,
                             "total_frames": total, "model": Path(model_path).name, "imgsz": imgsz,
                             "confidence": conf, "every_s": every_s}}
    _write_finder(video, status)
    step = max(1, int(round(every_s * fps)))
    seen = set(checked)
    candidates = [n for n in range(step // 2, total, step) if n not in seen]
    todo = [n for level in _spread_order(candidates) for n in level]
    images = folder / "frames" / video
    images.mkdir(parents=True, exist_ok=True)
    (folder / "crops").mkdir(exist_ok=True)
    offset = max(1, int(round(fps)))
    started = time.time()

    def read(number):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, number)))
        ok, image = capture.read()
        return image if ok else None

    try:
        with (folder / "found.jsonl").open("a", encoding="utf-8") as found:
            for i, number in enumerate(todo):
                if status["found"] >= want:
                    break
                frame = read(number)
                checked.append(number)
                status["looked"] = i + 1
                if frame is not None:
                    result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
                    if result.boxes is not None and len(result.boxes):
                        _save_found(video, video_path, frame, number, fps, result, names, images, found,
                                    [read(number - offset), frame, read(number + offset)])
                        status["found"] += 1
                spent = time.time() - started
                status["s_per_picture"] = round(spent / (i + 1), 2)
                _write_finder(video, status)
                if progress:
                    progress(min(1.0, status["found"] / max(1, want)),
                             f"{status['found']} of {want} pictures with boxes · looked at {i + 1} pictures · "
                             f"{spent / (i + 1):.1f} s per picture")
        status["status"] = "done" if status["found"] >= want else "video_end"
    except Exception as error:  # the page shows it
        status.update(status="error", error=str(error))
        raise
    finally:
        capture.release()
        _write_finder(video, status)
    return load_state(video)


def _save_found(video, video_path, frame, number, fps, result, names, images, found, around) -> None:
    frame_id = f"{video}_{number:09d}"
    path = images / f"frame_{number:09d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    height, width = frame.shape[:2]
    around = [a if a is not None else frame for a in around]
    detections = []
    for k, (box, cls, score) in enumerate(zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                              result.boxes.conf.tolist()), 1):
        cls = int(cls)
        x1, y1, x2, y2 = (float(v) for v in box)
        detection_id = f"{frame_id}-d{k:03d}"
        box = [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]
        cv2.imwrite(str(crop_path(video, detection_id)), _motion_strip(around, box), [cv2.IMWRITE_JPEG_QUALITY, 90])
        detections.append({
            "detection_id": detection_id, "class_id": cls, "class_name": names[cls],
            "predicted_class_id": cls, "predicted_class": names[cls], "confidence": float(score),
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "status": "unreviewed", "manual": False,
            "history": []})
    record = {"frame_id": frame_id, "video_id": video, "video_filename": Path(video_path).name,
              "frame_number": number, "timestamp_seconds": number / fps, "image_path": str(path.resolve()),
              "width": width, "height": height, "annotation_complete": False, "detections": detections}
    found.write(json.dumps(record) + "\n")
    found.flush()


def start_finder(video: str, video_path, weights, want: int, every_s: float, imgsz: int) -> int:
    """Run the finder as a background process (the page stays usable). Returns its process id."""
    import subprocess
    import sys

    folder = session_dir(video)
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "beemon", "hardneg-find", video, "--video", str(video_path),
               "--weights", str(weights), "--want", str(want), "--every-s", str(every_s), "--imgsz", str(imgsz)]
    extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform.startswith("win") \
        else {"start_new_session": True}
    with (folder / "finder.log").open("w") as log:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT, **extra)
    status = finder_status(video)
    status.update(status="starting", pid=process.pid, want=want, error=None)
    _write_finder(video, status)
    return process.pid


def image_path(video: str, frame: dict) -> Path:
    path = Path(frame["image_path"])
    if path.is_absolute() and path.is_file():
        return path
    return session_dir(video) / "frames" / video / path.name   # folder was moved or copied


# ---------------------------------------------------------------------------------- reviewing
def boxes_for_canvas(frame: dict) -> list[dict]:
    """Boxes to show: everything not deleted. Model boxes keep their id and letter (A, B, ...)."""
    boxes = []
    model_boxes = [d for d in frame["detections"] if not d.get("manual")]
    for d in frame["detections"]:
        if d["status"] == "false_positive":
            continue
        box = {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"], "cls": int(d["class_id"])}
        if not d.get("manual"):
            box.update(id=d["detection_id"], label=letter(model_boxes.index(d)))
        boxes.append(box)
    return boxes


def save_review(video: str, state: dict, frame_id: str, boxes: list[dict]) -> None:
    """The boxes the person kept/moved/added on the canvas -> review statuses:
    kept model box = true_positive, deleted model box = false_positive, new box = manual true_positive."""
    frame = next(f for f in state["frames"] if f["frame_id"] == frame_id)
    names = state["class_names"]
    width, height = float(frame["width"]), float(frame["height"])
    by_id = {b.get("id"): b for b in boxes if b.get("id")}
    detections = []
    for d in frame["detections"]:
        if d.get("manual"):
            continue  # manual boxes are rebuilt from the canvas below
        box = by_id.get(d["detection_id"])
        d["history"].append(d["status"])
        if box is None:
            d["status"] = "false_positive"
        else:
            d["status"] = "true_positive"
            d.update(_clip(box, width, height))
            d["class_id"] = int(box.get("cls", d["class_id"]))
            d["class_name"] = names[d["class_id"]]
        detections.append(d)
    manual = [b for b in boxes if not b.get("id")]
    for k, box in enumerate(manual, 1):
        cls = int(box.get("cls", 0))
        detections.append({"detection_id": f"{frame_id}-manual{k:03d}", "class_id": cls, "class_name": names[cls],
                           "predicted_class_id": "", "predicted_class": "manual annotation", "confidence": 0.0,
                           **_clip(box, width, height), "status": "true_positive", "manual": True, "history": []})
    frame["detections"] = [d for d in detections if d["x2"] - d["x1"] >= 1 and d["y2"] - d["y1"] >= 1]
    frame["annotation_complete"] = True
    save_state(video, state)


def reset_review(video: str, state: dict, frame_id: str) -> None:
    frame = next(f for f in state["frames"] if f["frame_id"] == frame_id)
    frame["detections"] = [d for d in frame["detections"] if not d.get("manual")]
    for d in frame["detections"]:
        d["status"] = "unreviewed"
        d["class_id"], d["class_name"] = d["predicted_class_id"], d["predicted_class"]
    frame["annotation_complete"] = False
    save_state(video, state)


def _clip(box: dict, width: float, height: float) -> dict:
    x1, x2 = sorted((max(0.0, min(width, float(box["x1"]))), max(0.0, min(width, float(box["x2"])))))
    y1, y2 = sorted((max(0.0, min(height, float(box["y1"]))), max(0.0, min(height, float(box["y2"])))))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


# ---------------------------------------------------------------------------------- export
def report_csv(frames: list[dict]) -> str:
    """One row per box with its review decision (review_report.csv)."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for frame in frames:
        for d in frame["detections"]:
            status = d["status"]
            final = ("excluded_uncertain" if status == "uncertain" else "excluded_unreviewed" if status == "unreviewed"
                     else "excluded_incomplete_frame" if not frame["annotation_complete"]
                     else "excluded_rejected" if status == "false_positive" else "included")
            writer.writerow({
                "video_filename": frame["video_filename"], "frame_number": frame["frame_number"],
                "timestamp_seconds": f"{frame['timestamp_seconds']:.6f}",
                "frame_filename": f"{frame['frame_id']}.jpg", "detection_id": d["detection_id"],
                "predicted_class": d.get("predicted_class", d["class_name"]),
                "predicted_class_id": d.get("predicted_class_id", d["class_id"]),
                "final_class_id": d["class_id"], "final_class": d["class_name"],
                "confidence": "" if d.get("manual") else f"{d['confidence']:.6f}",
                **{k: f"{d[k]:.2f}" for k in ("x1", "y1", "x2", "y2")},
                "review_status": status, "final_annotation_status": final,
                "frame_annotation_complete": frame["annotation_complete"]})
    return out.getvalue()


def _yolo_lines(frame: dict) -> list[str]:
    w, h = float(frame["width"]), float(frame["height"])
    return [f"{int(d['class_id'])} {(d['x1'] + d['x2']) / 2 / w:.6f} {(d['y1'] + d['y2']) / 2 / h:.6f} "
            f"{(d['x2'] - d['x1']) / w:.6f} {(d['y2'] - d['y1']) / h:.6f}"
            for d in frame["detections"] if d["status"] == "true_positive"]


def export_zip(videos: list[str], contributor: str = "") -> tuple[bytes, dict]:
    """All reviewed pictures of these videos as one zip in the YOLO dataset layout.

    Split by video (seed 42, 10 % val, 10 % test) when there are 3 or more
    videos; otherwise everything is in ``train`` (the person who trains re-splits anyway).
    """
    states = {v: load_state(v) for v in videos}
    eligible = [(v, f) for v, s in states.items() for f in s["frames"] if frame_done(f)]
    if not eligible:
        raise ValueError("No reviewed pictures yet.")
    names = next(iter(states.values()))["class_names"]
    used = sorted({v for v, _ in eligible})
    split = {v: "train" for v in used}
    if len(used) >= 3:
        order = list(used)
        random.Random(42).shuffle(order)
        n_val = min(max(1, round(len(order) * 0.1)), len(order) - 2)
        n_test = min(max(1, round(len(order) * 0.1)), len(order) - n_val - 1)
        n_train = len(order) - n_val - n_test
        split.update({v: "val" for v in order[n_train:n_train + n_val]})
        split.update({v: "test" for v in order[n_train + n_val:]})
    stats = {"train": 0, "val": 0, "test": 0}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for video, frame in eligible:
            part = split[video]
            archive.write(image_path(video, frame), f"images/{part}/{frame['frame_id']}.jpg")
            lines = _yolo_lines(frame)
            archive.writestr(f"labels/{part}/{frame['frame_id']}.txt", "\n".join(lines) + ("\n" if lines else ""))
            stats[part] += 1
        has = {p for p in stats if stats[p]}
        data = {"train": "images/train", "val": "images/val" if "val" in has else "images/train",
                "test": "images/test" if "test" in has else "images/train", "nc": len(names), "names": names}
        import yaml

        archive.writestr("data.yaml", yaml.safe_dump(data, sort_keys=False))
        archive.writestr("review_report.csv", report_csv([f for s in states.values() for f in s["frames"]]))
        summary = {"contributor": contributor, "videos": used, "pictures": len(eligible),
                   "wrong_boxes_deleted": sum(d["status"] == "false_positive" for _, f in eligible
                                              for d in f["detections"]),
                   "model": {v: s["videos"].get(v, {}).get("model") for v, s in states.items()},
                   "created": time.strftime("%Y-%m-%d %H:%M"), "split": stats}
        archive.writestr("contribution.json", json.dumps(summary, indent=1))
    return buffer.getvalue(), summary


# ---------------------------------------------------------------------------------- import (for the trainer)
def import_contribution(zip_path: str | Path, review_csv: str | Path | None = None,
                        log: Callable[[str], None] = print) -> tuple[Path, list[str]]:
    """A received zip (from this page or another review tool) -> a labelling batch in data/labeling/ that
    `beemon build-dataset` / the workflow use for training. Returns (batch folder, video names)."""
    from beemon.data.bridge import import_dataset_zip

    zip_path = Path(zip_path)
    out = PROJECT_DIR / "data" / "labeling" / f"hardneg_{zip_path.stem}"
    import_dataset_zip(zip_path, out, review_csv=review_csv, log=log)
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        videos = sorted({row["video"] for row in csv.DictReader(handle)})
    return out, videos
