"""Hard example mining from processed videos (the "data flywheel").

* Hard POSITIVES: frames inside a valid track where the detector MISSED the bee (gap rows,
  the tracker still knows where it is). These are exactly the bees your model cannot see yet.
* Hard NEGATIVES: frames with tracks the filters rejected (static object, background hotspot,
  short flicker with high confidence). These are things your model wrongly calls a bee.

The selected frames are written like a labelling batch (images/, manifest.csv) plus CVAT
pre-labels containing every tracked box of that frame. In CVAT: fix the boxes (add the missed
bee, delete the false ones), export YOLO 1.1, then add the batch to `beemon build-dataset`.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from beemon.data.common import frame_name, load_split, write_cvat_yolo_package, xyxy_to_yolo
from beemon.data.extract import MANIFEST_FIELDS
from beemon.video import iter_frames, probe


def mine_hard_examples(result_dir, out_dir, video=None, split_yaml=None, max_positive: int = 60,
                       max_negative: int = 60, per_track: int = 2, negative_min_conf: float = 0.3,
                       include_test: bool = False, seed: int = 0, log: Callable[[str], None] = print) -> Path:
    result_dir, out_dir = Path(result_dir), Path(out_dir)
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    video_info, run_info = run["video"], run["run"]
    stem = video_info["video"]
    if stem in load_split(split_yaml)["test_videos"] and not include_test:
        raise ValueError(f"{stem} is a TEST video: mining it would leak test data into training.")
    video_path = Path(video or video_info["path"])
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path} (pass --video)")
    tracks = pd.read_csv(result_dir / "tracks.csv")
    tracks["detected"] = tracks["detected"].astype(bool)
    summary = pd.read_csv(result_dir / "tracks_summary.csv", keep_default_na=False)
    rng = random.Random(seed)

    chosen: dict[int, str] = {}
    # hard positives: missed detections inside valid tracks
    valid_ids = set(summary.loc[summary["valid"].astype(str).str.lower() == "true", "track_id"])
    gaps = tracks[(~tracks["detected"]) & tracks["track_id"].isin(valid_ids)]
    positives = []
    for _, group in gaps.groupby("track_id"):
        frames = sorted(group["frame"].unique())
        positives += rng.sample(frames, min(per_track, len(frames)))
    rng.shuffle(positives)
    for frame in positives[:max_positive]:
        chosen[int(frame)] = "hard_positive"
    # hard negatives: rejected tracks that the detector was fairly sure about
    rejected = summary[(summary["valid"].astype(str).str.lower() != "true")
                       & (summary["max_conf"].astype(float) >= negative_min_conf)
                       & summary["reason"].isin(["static object", "background hotspot", "too short"])]
    negatives = []
    for record in rejected.itertuples(index=False):
        rows = tracks[(tracks["track_id"] == record.track_id) & tracks["detected"]]
        frames = sorted(rows["frame"].unique())
        negatives += rng.sample(frames, min(per_track, len(frames)))
    rng.shuffle(negatives)
    for frame in negatives[:max_negative]:
        chosen.setdefault(int(frame), "hard_negative")
    if not chosen:
        log("nothing to mine (no gaps in valid tracks and no rejected confident tracks)")
        return out_dir

    names = [run_info["class_names"][k] for k in sorted(run_info["class_names"], key=int)]
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    manifest, prelabels, kinds = [], [], defaultdict(int)
    by_frame = {f: g for f, g in tracks.groupby("frame")}
    for pack in iter_frames(video_path, sorted(chosen), 0):
        rows = by_frame.get(pack.index)
        name = frame_name(stem, pack.index) + ".jpg"
        cv2.imwrite(str(image_dir / name), pack.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if rows is not None and len(rows):
            xyxy = np.stack([rows["cx"] - rows["w"] / 2, rows["cy"] - rows["h"] / 2,
                             rows["cx"] + rows["w"] / 2, rows["cy"] + rows["h"] / 2], axis=1)
            cls = rows["cls_id"].where(rows["cls_id"] >= 0, 0).to_numpy()
            labels = xyxy_to_yolo(xyxy, cls, info.width, info.height)
        else:
            labels = np.zeros((0, 5))
        prelabels.append((name, labels))
        manifest.append({"image": name, "video": stem, "video_path": str(video_path.resolve()), "frame": pack.index,
                         "time_s": round(pack.time_s, 3), "fps": info.fps, "width": info.width, "height": info.height})
        kinds[chosen[pack.index]] += 1
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    with (out_dir / "mined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "kind"])
        writer.writerows([(frame_name(stem, f) + ".jpg", k) for f, k in sorted(chosen.items())])
    write_cvat_yolo_package(out_dir / "cvat_prelabels.zip", names, prelabels)
    log(f"mined {dict(kinds)} frames into {out_dir}; review them in CVAT with cvat_prelabels.zip")
    return out_dir
