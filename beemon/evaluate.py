"""Evaluation: tracking metrics (MOTA, IDF1) against CVAT ground truth, and visit counts vs. manual counts.

Matching uses the distance between box CENTRES (``max_dist`` px) instead of box overlap,
because for 20-40 px insects a few pixels of annotation offset already destroy the IoU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MOT_COLUMNS = ["frame", "id", "x", "y", "w", "h"]


def read_mot(path: str | Path) -> pd.DataFrame:
    """Read a MOT 1.1 file (CVAT export ``gt/gt.txt`` or our ``tracks_mot.txt``)."""
    path = Path(path)
    if path.is_dir():
        candidates = list(path.rglob("gt.txt"))
        if not candidates:
            raise FileNotFoundError(f"No gt.txt inside {path}")
        path = candidates[0]
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=MOT_COLUMNS)
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, :8] if data.shape[1] >= 8 else data
    frame = data.iloc[:, :6].copy()
    frame.columns = MOT_COLUMNS
    if data.shape[1] >= 7:  # CVAT: column 7 is "not ignored" flag (1 = use), 0 rows are ignored
        flag = data.iloc[:, 6]
        if set(pd.unique(flag)) <= {0, 1}:
            frame = frame[flag.to_numpy() == 1]
    frame["cx"] = frame["x"] + frame["w"] / 2
    frame["cy"] = frame["y"] + frame["h"] / 2
    return frame.astype({"frame": int, "id": int})


def clip_mapping(clip_json: str | Path | None) -> tuple[int, int]:
    """Return (first_video_frame, stride) of a clip made by `beemon clip`."""
    if not clip_json:
        return 0, 1
    data = json.loads(Path(clip_json).read_text(encoding="utf-8"))
    return int(data["start_frame"]), int(data.get("stride", 1))


def mot_metrics(gt: pd.DataFrame, pred: pd.DataFrame, max_dist: float = 30.0, frames=None) -> dict:
    """CLEAR-MOT (MOTA, ID switches) and identity metrics (IDF1) with centre-distance matching.

    ``gt`` and ``pred`` use the same (absolute, 1-based) frame numbers. ``frames`` restricts the
    evaluation to frames the tracker actually processed.
    """
    if frames is not None:
        frames = set(int(f) for f in frames)
        gt = gt[gt["frame"].isin(frames)]
        pred = pred[pred["frame"].isin(frames)]
    eval_frames = sorted(set(gt["frame"]) | set(pred["frame"]))
    gt_by = {f: g for f, g in gt.groupby("frame")}
    pr_by = {f: p for f, p in pred.groupby("frame")}
    tp = fp = fn = idsw = 0
    last_match: dict[int, int] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    distances = []
    for f in eval_frames:
        g = gt_by.get(f, gt.iloc[0:0])
        p = pr_by.get(f, pred.iloc[0:0])
        if len(g) and len(p):
            d = np.hypot(g["cx"].to_numpy()[:, None] - p["cx"].to_numpy()[None], g["cy"].to_numpy()[:, None]
                         - p["cy"].to_numpy()[None])
            gids, pids = g["id"].to_numpy(), p["id"].to_numpy()
            for i, j in zip(*np.nonzero(d <= max_dist)):  # identity co-occurrence for IDF1
                pair_counts[(gids[i], pids[j])] = pair_counts.get((gids[i], pids[j]), 0) + 1
            cost = np.where(d <= max_dist, d, 1e6)
            # prefer keeping last frame's assignments (standard CLEAR-MOT behaviour)
            for i, gid in enumerate(gids):
                if gid in last_match:
                    hit = np.flatnonzero(pids == last_match[gid])
                    if hit.size and d[i, hit[0]] <= max_dist:
                        cost[i, hit[0]] = -1.0
            rows, cols = linear_sum_assignment(cost)
            matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e5]
            for r, c in matched:
                gid, pid = gids[r], pids[c]
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                last_match[gid] = pid
                distances.append(d[r, c])
            tp += len(matched)
            fp += len(p) - len(matched)
            fn += len(g) - len(matched)
        else:
            fp += len(p)
            fn += len(g)

    n_gt, n_pred = len(gt), len(pred)
    gt_ids = sorted(set(gt["id"]))
    pr_ids = sorted(set(pred["id"]))
    idtp = 0
    if gt_ids and pr_ids and pair_counts:
        gi = {g: i for i, g in enumerate(gt_ids)}
        pi = {p: i for i, p in enumerate(pr_ids)}
        overlap = np.zeros((len(gt_ids), len(pr_ids)))
        for (g, p), count in pair_counts.items():
            overlap[gi[g], pi[p]] = count
        rows, cols = linear_sum_assignment(-overlap)
        idtp = int(overlap[rows, cols].sum())
    idfp, idfn = n_pred - idtp, n_gt - idtp
    return {
        "frames": len(eval_frames),
        "gt_boxes": n_gt,
        "pred_boxes": n_pred,
        "gt_tracks": len(gt_ids),
        "pred_tracks": len(pr_ids),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "MOTA": round(1 - (fn + fp + idsw) / max(1, n_gt), 4),
        "IDF1": round(2 * idtp / max(1, 2 * idtp + idfp + idfn), 4),
        "ID_switches": idsw,
        "FP": fp,
        "FN": fn,
        "mean_center_error_px": round(float(np.mean(distances)), 2) if distances else None,
    }


def evaluate_tracking(gt_path, result_dir, clip_json=None, max_dist: float = 30.0, all_tracks: bool = False) -> dict:
    result_dir = Path(result_dir)
    gt = read_mot(gt_path)
    first, stride = clip_mapping(clip_json)
    gt["frame"] = first + (gt["frame"] - 1) * stride + 1  # -> absolute 1-based video frame numbers
    pred = read_mot(result_dir / ("tracks_mot_all.txt" if all_tracks else "tracks_mot.txt"))
    log = pd.read_csv(result_dir / "frame_log.csv")
    processed = set((log.loc[log["detector_run"] == 1, "frame"] + 1).astype(int))
    gt_frames = set(gt["frame"])
    overlap = processed & gt_frames
    if not overlap:
        raise ValueError(
            "No frame of the ground truth was processed by the pipeline. Process the same video and time window "
            "as the clip (and check --clip)."
        )
    metrics = mot_metrics(gt, pred, max_dist=max_dist, frames=overlap)
    metrics["gt_frames_not_processed"] = len(gt_frames - processed)
    return metrics


def evaluate_counts(results_root: str | Path, manual_csv: str | Path) -> pd.DataFrame:
    """Compare pipeline visit counts with manual counts (CSV with columns video, visits)."""
    manual = pd.read_csv(manual_csv)
    if not {"video", "visits"} <= set(manual.columns):
        raise ValueError("Manual counts CSV needs the columns: video, visits")
    rows = []
    for record in manual.itertuples(index=False):
        summary_path = Path(results_root) / str(record.video) / "summary.json"
        predicted = None
        if summary_path.is_file():
            predicted = json.loads(summary_path.read_text(encoding="utf-8")).get("visits")
        error = None if predicted is None else predicted - record.visits
        rows.append({
            "video": record.video,
            "manual_visits": record.visits,
            "pipeline_visits": predicted,
            "error": error,
            "abs_error_pct": None if predicted is None or record.visits == 0 else round(
                abs(error) / record.visits * 100, 1),
        })
    return pd.DataFrame(rows)
