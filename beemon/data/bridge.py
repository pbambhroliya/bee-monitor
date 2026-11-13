"""Import hard-negative / review work that was done in another tool, and export datasets for it.

Manual work is done once and used by both projects:

* ``import_hardneg``  — a review session folder (``<session>/`` with
  ``review.json``) or its exported dataset ZIP becomes a labelling batch here:
    - fully reviewed frames  -> ``labels_human/``  (kept boxes; rejected boxes = background) — used for training
    - unfinished frames      -> ``labels_suggested/`` + ``review/review_list.txt`` — finish them in the bee labeller
* ``zip_dataset``     — any dataset built here, zipped in the plain YOLO layout
  (images/{train,val,test}, labels/..., data.yaml), so the old training scripts can use it.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import yaml

from beemon.data.common import frame_name, write_labels, xyxy_to_yolo
from beemon.data.extract import MANIFEST_FIELDS

SUGGESTED_DIR = "labels_suggested"


def _write_batch(out: Path, class_names: list[str], frames: list[dict], log) -> dict:
    """frames: dicts with image (path), video, frame, time_s, width, height, boxes [(x1,y1,x2,y2,cls)],
    complete (bool), suggestions [(x1,y1,x2,y2,cls)], video_path."""
    for sub in ("images", "labels_human", SUGGESTED_DIR, "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("labels_human", SUGGESTED_DIR):
        (out / sub / "obj.names").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    manifest, review, counts = [], [], defaultdict(int)
    for f in frames:
        name = frame_name(f["video"], f["frame"]) + ".jpg"
        target = out / "images" / name
        if Path(f["image"]).suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(f["image"], target)
        else:
            cv2.imwrite(str(target), cv2.imread(str(f["image"])))
        manifest.append({"image": name, "video": f["video"], "video_path": f.get("video_path", ""),
                         "frame": f["frame"], "time_s": f.get("time_s", ""), "fps": f.get("fps", ""),
                         "width": f["width"], "height": f["height"]})

        def yolo(boxes):
            if not boxes:
                return xyxy_to_yolo([], [], f["width"], f["height"])
            xyxy = [b[:4] for b in boxes]
            return xyxy_to_yolo(xyxy, [b[4] for b in boxes], f["width"], f["height"])

        if f["complete"]:
            write_labels(out / "labels_human" / (Path(name).stem + ".txt"), yolo(f["boxes"]))
            counts["reviewed (training labels)"] += 1
            counts["  of which background only"] += int(not f["boxes"])
        else:
            write_labels(out / SUGGESTED_DIR / (Path(name).stem + ".txt"), yolo(f["suggestions"]))
            review.append(name)
            counts["unfinished (to finish in the labeller)"] += 1
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    review_file = out / "review" / "review_list.txt"
    if review:
        review_file.write_text("\n".join(review) + "\n", encoding="utf-8")
    elif review_file.exists():
        review_file.unlink()
    log(f"imported into {out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return dict(counts)


def import_session(session: str | Path, out: str | Path, log: Callable[[str], None] = print) -> dict:
    """Import a review session folder (contains review.json and frames/)."""
    session = Path(session)
    state = json.loads((session / "review.json").read_text(encoding="utf-8"))
    class_names = state.get("class_names") or []
    if not state.get("frames"):
        raise ValueError(f"{session} contains no reviewed frames")
    uploads = list((session / "uploads").glob("*")) if (session / "uploads").is_dir() else []
    frames = []
    for f in state["frames"]:
        image = Path(f["image_path"])
        if not image.is_file():  # session copied from another computer: look inside this folder
            image = session / "frames" / image.parent.name / image.name
        if not image.is_file():
            log(f"missing picture {f['image_path']} — skipped")
            continue
        video = Path(f["video_filename"]).stem
        video_path = next((str(p) for p in uploads if p.name.endswith("_" + f["video_filename"])), "")
        statuses = {d["status"] for d in f["detections"]}
        complete = bool(f.get("annotation_complete")) and statuses <= {"true_positive", "false_positive"}
        kept = [(d["x1"], d["y1"], d["x2"], d["y2"], d["class_id"]) for d in f["detections"]
                if d["status"] == "true_positive"]
        suggestions = [(d["x1"], d["y1"], d["x2"], d["y2"], d["class_id"]) for d in f["detections"]
                       if d["status"] != "false_positive"]
        frames.append({"image": image, "video": video, "frame": int(f["frame_number"]),
                       "time_s": f.get("timestamp_seconds", ""), "width": int(f["width"]), "height": int(f["height"]),
                       "complete": complete, "boxes": kept, "suggestions": suggestions, "video_path": video_path})
    return _write_batch(Path(out), class_names, frames, log)


def import_dataset_zip(zip_path: str | Path, out: str | Path, review_csv: str | Path | None = None,
                       log: Callable[[str], None] = print) -> dict:
    """Import a reviewed YOLO dataset ZIP (all frames in it are fully reviewed).

    Its file names are ``<video code>_<frame>.jpg``. With the review CSV of the same session the
    video codes are translated back to video names (needed for the split by video and for MIE).
    """
    out = Path(out)
    tmp = out / "_zip"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp)
    data_yaml = next(tmp.rglob("data.yaml"))
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    pattern = re.compile(r"^(?P<code>.+)_(?P<frame>\d+)$")
    if review_csv is None and (tmp / "review_report.csv").is_file():
        review_csv = tmp / "review_report.csv"   # zips from this project carry it inside
    by_code = defaultdict(set)
    images = sorted(p for p in tmp.rglob("images/*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for image in images:
        match = pattern.match(image.stem)
        if match:
            by_code[match["code"]].add(int(match["frame"]))
    code_to_video = {code: code for code in by_code if not re.fullmatch(r"[0-9a-f]{8,}", code)}  # our names
    if review_csv and len(code_to_video) < len(by_code):  # match each code to the video whose reviewed frame numbers contain its frames
        frames_of_video = defaultdict(set)
        with Path(review_csv).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frames_of_video[Path(row["video_filename"]).stem].add(int(row["frame_number"]))
        for code, numbers in by_code.items():
            if code in code_to_video:
                continue
            best = max(frames_of_video, key=lambda v: len(numbers & frames_of_video[v]), default=None)
            if best and numbers <= frames_of_video[best]:
                # the review CSV only knows the file name; RIMG0001 exists in several plots -> keep them apart
                code_to_video[code] = best if best not in code_to_video.values() else f"{best}_{code[:6]}"
    frames = []
    for image in images:
        match = pattern.match(image.stem)
        if not match:
            continue
        label = Path(*[("labels" if part == "images" else part) for part in image.parts]).with_suffix(".txt")
        height, width = cv2.imread(str(image)).shape[:2]
        boxes = []
        if label.is_file():
            for line in label.read_text().splitlines():
                c, cx, cy, w, h = (float(v) for v in line.split()[:5])
                boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width,
                              (cy + h / 2) * height, int(c)))
        frames.append({"image": image, "video": code_to_video.get(match["code"], "video_" + match["code"]),
                       "frame": int(match["frame"]), "width": width, "height": height, "complete": True,
                       "boxes": boxes, "suggestions": []})
    if len(code_to_video) < len(by_code):
        log("NOTE: video names unknown (pass --review-csv to recover them). Frames are grouped by the app's "
            "video code, which is fine for training but not for MIE or the test-video split.")
    result = _write_batch(out, class_names, frames, log)
    shutil.rmtree(tmp)
    return result


def zip_dataset(dataset_dir: str | Path, zip_path: str | Path, log: Callable[[str], None] = print) -> Path:
    """Zip a dataset folder in the plain YOLO layout, with a portable data.yaml (no absolute path)."""
    dataset_dir, zip_path = Path(dataset_dir), Path(zip_path)
    data = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    data.pop("path", None)  # Ultralytics then uses the folder of data.yaml
    names = data["names"]
    data["names"] = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
    data["nc"] = len(data["names"])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file() and path.name != "data.yaml" and path.suffix != ".cache":
                archive.write(path, f"{dataset_dir.name}/{path.relative_to(dataset_dir)}")
        archive.writestr(f"{dataset_dir.name}/data.yaml", yaml.safe_dump(data, sort_keys=False))
    log(f"portable YOLO dataset: {zip_path}  (unzip anywhere and train with  data=<folder>/data.yaml)")
    return zip_path
