"""Automatic labelling of whole videos with the current model ("teacher") + tracking.

Raw detections of a model are not good enough as training labels — that would teach the next
model its own mistakes. Labels are therefore taken only from what the tracker and the filters
confirm, and every frame gets a category:

  positive       contains confident, valid tracks only -> their boxes become labels
  hard_positive  a positive frame where the model MISSED a tracked bee; the box comes from the
                 tracker's prediction -> exactly the cases the next model has to learn
  hard_negative  no valid track, but rejected tracks (static flower centres, hotspots)
                 -> empty label: "this is background", teaches the model to stop
  background     no tracks at all -> empty label
  uncertain      contains weak valid tracks or confident short glimpses -> NOT labelled;
                 these frames are sent to a human (CVAT) together with a random sample

Output is a labelling batch like the manual ones (images/, manifest.csv) plus
``labels_auto/`` and ``review/``. Human corrections exported from CVAT into the batch always
override the automatic labels (see ``beemon build-dataset``).
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from beemon.data.common import (
    AUTO_LABEL_DIR, frame_name, load_split, read_labels, write_cvat_yolo_package, write_labels, xyxy_to_yolo,
)
from beemon.data.extract import MANIFEST_FIELDS
from beemon.video import iter_frames

DEFAULTS = {
    "min_track_conf": 0.35,
    "min_det_ratio": 0.5,
    "frames_per_video": 300,
    "min_gap_s": 0.5,
    "background_fraction": 0.3,
    "hard_positive_share": 0.3,
    "hard_negative_share": 0.7,
    "glimpse_conf": 0.5,
    "min_box_conf": 0.25,      # a labelled box must be at least this confident ...
    "min_box_motion": 1.5,     # ... and move more than its surroundings (bees ~2-9, flower centres ~1)
    "static_motion_max": 1.3,
    "review_random": 60,
    "review_uncertain": 150,
}


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _spread(frames: list[tuple[int, float]], n: int, min_gap_s: float, rng: random.Random) -> list[int]:
    """Choose up to n frames spread over time, at least ``min_gap_s`` apart."""
    if n <= 0 or not frames:
        return []
    frames = sorted(frames, key=lambda f: f[1])
    order = list(range(len(frames)))
    if len(frames) > n:  # evenly spaced first, then fill randomly
        even = np.linspace(0, len(frames) - 1, n).round().astype(int).tolist()
        even_set = set(even)
        rest = [i for i in order if i not in even_set]
        rng.shuffle(rest)
        order = even + rest
    picked, times = [], []
    for i in order:
        frame, t = frames[i]
        if all(abs(t - other) >= min_gap_s for other in times):
            picked.append(frame)
            times.append(t)
            if len(picked) >= n:
                break
    return picked


def classify_frames(result_dir: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Per processed frame: category + the label rows. Returns (frames, rows, good_track_ids)."""
    tracks = pd.read_csv(result_dir / "tracks.csv", keep_default_na=False)
    summary = pd.read_csv(result_dir / "tracks_summary.csv", keep_default_na=False)
    log = pd.read_csv(result_dir / "frame_log.csv")
    if len(tracks):
        tracks["detected"] = _bool(tracks["detected"])
    if len(summary):
        summary["valid"] = _bool(summary["valid"])
        long_stay = _bool(summary["long_stay"]) if "long_stay" in summary else False
        # a track that never moves and has no motion of its own may be a flower centre or a resting bee:
        # never trust it automatically (it goes to review), whatever the counting filters decided
        motion = summary["motion_ratio_mean"].astype(float)
        stationary_calm = (summary["max_radius_px"].astype(float) <= np.maximum(8.0, summary["size_px"].astype(float))
                           ) & (motion > 0) & (motion < cfg["static_motion_max"])
        good = summary["valid"] & (summary["mean_conf"].astype(float) >= cfg["min_track_conf"]) & (
            summary["det_ratio"].astype(float) >= cfg["min_det_ratio"]) & ~long_stay & ~stationary_calm
        weak = summary["valid"] & ~good
        glimpse = (summary["reason"] == "too short") & (summary["max_conf"].astype(float) >= cfg["glimpse_conf"])
        rejected = ~summary["valid"] & ~glimpse
        good_ids, weak_ids = set(summary.loc[good, "track_id"]), set(summary.loc[weak | glimpse, "track_id"])
        rejected_ids = set(summary.loc[rejected, "track_id"])
    else:
        good_ids = weak_ids = rejected_ids = set()

    if len(tracks):
        tracks["trusted"] = _trusted_rows(tracks, cfg)
    by_frame = tracks.groupby("frame") if len(tracks) else {}
    records = []
    for row in log[log["detector_run"] == 1].itertuples(index=False):
        frame_rows = by_frame.get_group(row.frame) if len(tracks) and row.frame in by_frame.groups else None
        ids = set(frame_rows["track_id"]) if frame_rows is not None else set()
        drifted = frame_rows is not None and (~frame_rows.loc[frame_rows["track_id"].isin(good_ids), "trusted"]).any()
        if ids & weak_ids or drifted:
            kind = "uncertain"
        elif ids & good_ids:
            gap = frame_rows[frame_rows["track_id"].isin(good_ids) & ~frame_rows["detected"]]
            kind = "hard_positive" if len(gap) else "positive"
        elif ids & rejected_ids:
            kind = "hard_negative"
        else:
            kind = "background"
        records.append({"frame": int(row.frame), "time_s": float(row.time_s), "kind": kind})
    return pd.DataFrame.from_records(records, columns=["frame", "time_s", "kind"]), tracks, good_ids


def _trusted_rows(tracks: pd.DataFrame, cfg: dict) -> pd.Series:
    """Is every single box of a track trustworthy as a label?

    Detected boxes: confident enough AND moving more than their surroundings. This catches a
    track that started on a bee and then drifted onto a flower centre after the bee left.
    Predicted (gap) boxes: trusted when the detections before and after them are trusted.
    Without motion information (motion disabled) only the confidence is checked.
    """
    has_motion = float(tracks["motion_ratio"].max()) > 0
    det = tracks["detected"]
    ok = det & (tracks["conf"] >= cfg["min_box_conf"])
    if has_motion:
        ok &= tracks["motion_ratio"] >= cfg["min_box_motion"]
    trusted = ok.copy()
    for _, group in tracks.sort_values("frame").groupby("track_id"):
        values = ok[group.index].astype(float).where(det[group.index])  # NaN on gap rows
        before = values.ffill().fillna(0.0) > 0.5
        after = values.bfill().fillna(0.0) > 0.5
        gaps = ~det[group.index]
        trusted[group.index[gaps]] = (before & after)[gaps]
    return trusted


def autolabel_results(result_dirs: list[str | Path], out_dir: str | Path, class_names: list[str], split_yaml=None,
                      cfg: dict | None = None, seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """Turn processed videos into an automatically labelled batch (+ review package)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    out_dir = Path(out_dir)
    images_dir, labels_dir, review_dir = out_dir / "images", out_dir / AUTO_LABEL_DIR, out_dir / "review"
    for folder in (images_dir, labels_dir, review_dir):
        folder.mkdir(parents=True, exist_ok=True)
    (labels_dir / "obj.names").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    test_videos = set(load_split(split_yaml)["test_videos"])
    rng = random.Random(seed)

    manifest, catalogue = [], []
    uncertain_pool: list[tuple[str, np.ndarray]] = []
    auto_items: list[str] = []
    prelabels: dict[str, np.ndarray] = {}
    totals = Counter()
    result_dirs = [Path(r) for r in result_dirs]
    for result_dir in result_dirs:
        run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
        video, model_names = run["video"], run["run"]["class_names"]
        stem = video["video"]
        if stem in test_videos:
            log(f"skip {stem}: TEST video (never auto-labelled)")
            continue
        video_path = Path(video["path"])
        if not video_path.is_file():
            log(f"skip {stem}: video not found at {video_path}")
            continue
        frames, rows, good_ids = classify_frames(result_dir, cfg)
        if frames.empty:
            continue
        n = int(cfg["frames_per_video"])
        n_bg = int(round(n * float(cfg["background_fraction"])))
        n_pos = n - n_bg

        def pool(*kinds):
            sel = frames[frames["kind"].isin(kinds)]
            return list(zip(sel["frame"], sel["time_s"]))

        hard_pos = _spread(pool("hard_positive"), int(n_pos * cfg["hard_positive_share"]), cfg["min_gap_s"], rng)
        pos = _spread([f for f in pool("positive", "hard_positive") if f[0] not in set(hard_pos)],
                      n_pos - len(hard_pos), cfg["min_gap_s"], rng)
        hard_neg = _spread(pool("hard_negative"), int(n_bg * cfg["hard_negative_share"]), cfg["min_gap_s"], rng)
        bg = _spread([f for f in pool("hard_negative", "background") if f[0] not in set(hard_neg)],
                     n_bg - len(hard_neg), cfg["min_gap_s"], rng)
        per_video = max(5, int(np.ceil(2 * int(cfg["review_uncertain"]) / max(1, len(result_dirs)))))
        uncertain = _spread(pool("uncertain"), per_video, cfg["min_gap_s"], rng)
        kinds = {f: k for f, k in zip(frames["frame"], frames["kind"])}
        chosen = {f: kinds[f] for f in hard_pos + pos + hard_neg + bg}
        for f in uncertain:
            chosen.setdefault(f, "uncertain")

        rows_by_frame = {f: g for f, g in rows.groupby("frame")} if len(rows) else {}
        class_of = _track_classes(result_dir, model_names, class_names)
        for pack in iter_frames(video_path, sorted(chosen), 0):
            name = frame_name(stem, pack.index) + ".jpg"
            kind = chosen[pack.index]
            frame_rows = rows_by_frame.get(pack.index)
            labels = _labels(frame_rows, good_ids, class_of, video["width"], video["height"])
            cv2.imwrite(str(images_dir / name), pack.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({"image": name, "video": stem, "video_path": str(video_path.resolve()),
                             "frame": pack.index, "time_s": round(pack.time_s, 3), "fps": video["fps"],
                             "width": video["width"], "height": video["height"]})
            if kind == "uncertain":  # no automatic label: a human decides
                everything = _labels(frame_rows, None, class_of, video["width"], video["height"])
                uncertain_pool.append((name, everything))
            else:
                write_labels(labels_dir / (Path(name).stem + ".txt"), labels)
                auto_items.append(name)
                prelabels[name] = labels
            catalogue.append({"image": name, "video": stem, "frame": pack.index, "kind": kind, "boxes": len(labels)})
            totals[kind] += 1
        log(f"{stem}: " + ", ".join(f"{k} {v}" for k, v in Counter(chosen.values()).items()))

    _write_csv(out_dir / "manifest.csv", MANIFEST_FIELDS, manifest, append=True)
    _write_csv(out_dir / "autolabel.csv", ["image", "video", "frame", "kind", "boxes"], catalogue, append=True)

    # review package: all uncertain frames (up to the limit) + a random sample of automatic labels
    rng.shuffle(uncertain_pool)
    uncertain_sel = uncertain_pool[: int(cfg["review_uncertain"])]
    random_sel = rng.sample(auto_items, min(int(cfg["review_random"]), len(auto_items)))
    review = [(n, labels) for n, labels in uncertain_sel] + [(n, prelabels[n]) for n in random_sel]
    if review:
        write_cvat_yolo_package(review_dir / "cvat_review.zip", class_names, review)
        (review_dir / "review_list.txt").write_text("\n".join(n for n, _ in review) + "\n", encoding="utf-8")
        (review_dir / "random_sample.txt").write_text("\n".join(random_sel) + "\n", encoding="utf-8")
        (review_dir / "HOW_TO_REVIEW.md").write_text(_HOW_TO.format(n=len(review), u=len(uncertain_sel),
                                                                    r=len(random_sel)), encoding="utf-8")
    info = {"frames": dict(totals), "review_frames": len(review), "auto_labelled": len(auto_items)}
    (out_dir / "autolabel_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"auto-labelled batch {out_dir}: {dict(totals)}; review package: {len(review)} frames")
    return info


def _track_classes(result_dir: Path, model_names: dict, class_names: list[str]) -> dict[int, int]:
    """Track id -> class id in ``class_names`` (track-level vote of the teacher)."""
    summary = pd.read_csv(result_dir / "tracks_summary.csv", keep_default_na=False)
    mapping = {}
    for track_id, name in zip(summary.get("track_id", []), summary.get("cls_name", [])):
        mapping[int(track_id)] = class_names.index(name) if name in class_names else 0
    return mapping


def _labels(rows: pd.DataFrame | None, track_ids: set[int] | None, class_of: dict[int, int], width: int,
            height: int) -> np.ndarray:
    if rows is None or not len(rows):
        return np.zeros((0, 5))
    if track_ids is not None:
        rows = rows[rows["track_id"].isin(track_ids)]
    if not len(rows):
        return np.zeros((0, 5))
    xyxy = np.stack([rows["cx"] - rows["w"] / 2, rows["cy"] - rows["h"] / 2, rows["cx"] + rows["w"] / 2,
                     rows["cy"] + rows["h"] / 2], axis=1).astype(float)
    cls = np.array([class_of.get(int(t), 0) for t in rows["track_id"]])
    return xyxy_to_yolo(xyxy, cls, int(width), int(height))


def _write_csv(path: Path, fields, records, append: bool) -> None:
    exists = path.is_file() and append
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(records)


def label_quality(batch: str | Path, max_dist_factor: float = 1.0) -> dict:
    """Compare automatic labels with the human-corrected ones for the random review sample.

    Precision = share of automatic boxes that the reviewer kept; recall = share of the boxes in
    the corrected labels that the automatic labeller had found. Also reports frame-level errors.
    """
    from beemon.data.common import find_label_files, yolo_to_xyxy
    from scipy.optimize import linear_sum_assignment

    batch = Path(batch)
    sample_file = batch / "review" / "random_sample.txt"
    if not sample_file.is_file():
        raise FileNotFoundError(f"No review sample in {batch}")
    human = {stem: path for stem, (path, _, auto) in find_label_files(batch).items() if not auto}
    if not human:
        raise FileNotFoundError(f"No CVAT export found in {batch} (unzip it into {batch}/cvat_export/)")
    from beemon.data.common import cvat_listed

    listed = cvat_listed(batch)
    tp = fp = fn = frames = wrong_frames = 0
    for name in sample_file.read_text(encoding="utf-8").split():
        stem = Path(name).stem
        if stem not in human and stem not in listed:
            continue  # not reviewed yet
        auto = read_labels(batch / AUTO_LABEL_DIR / f"{stem}.txt")
        corrected = read_labels(human[stem]) if stem in human else np.zeros((0, 5))
        a, h = yolo_to_xyxy(auto, 1920, 1080), yolo_to_xyxy(corrected, 1920, 1080)
        frames += 1
        matched = 0
        if len(a) and len(h):
            ca, ch = (a[:, :2] + a[:, 2:]) / 2, (h[:, :2] + h[:, 2:]) / 2
            size = np.sqrt((h[:, 2] - h[:, 0]) * (h[:, 3] - h[:, 1]))
            dist = np.hypot(ca[:, None, 0] - ch[None, :, 0], ca[:, None, 1] - ch[None, :, 1])
            ok = dist <= np.maximum(10, max_dist_factor * size)[None, :]
            r, c = linear_sum_assignment(np.where(ok, dist, 1e6))
            matched = int(sum(ok[i, j] for i, j in zip(r, c)))
        tp += matched
        fp += len(a) - matched
        fn += len(h) - matched
        wrong_frames += int(matched != len(a) or matched != len(h))
    return {"frames_checked": frames, "frames_with_errors": wrong_frames,
            "box_precision": round(tp / max(1, tp + fp), 3), "box_recall": round(tp / max(1, tp + fn), 3),
            "boxes_auto": tp + fp, "boxes_human": tp + fn}


_HOW_TO = """# Review this batch ({n} pictures: {u} uncertain + {r} random checks)

Easiest — the bee labeller (no CVAT needed):
  * yourself:   streamlit run beemon/labeler/app.py  -> choose this picture set
  * teammates:  python -m beemon labeling-package data/labeling/<this batch>   -> send share/<batch>.zip
                they follow START_HERE.md and send back their labels zip;
                import it with
                python -m beemon labeling-import <their zip> --batch data/labeling/<this batch>

Correct EVERY picture: delete wrong boxes, add missed insects, fix the type; a picture without insects
is saved empty. Then check how good the automatic labels were:
  python -m beemon workflow quality --round <round>
Your corrections automatically replace the automatic labels when the dataset is built.

Alternative — CVAT: upload the pictures listed in review_list.txt, upload cvat_review.zip (YOLO 1.1) as
annotations, correct, export YOLO 1.1 and unzip into <this batch>/cvat_export/.
"""
