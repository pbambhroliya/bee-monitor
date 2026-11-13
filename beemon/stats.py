"""Aggregated monitoring statistics: visits, dwell time, activity over time, spatial heatmap."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _valid(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["valid"].astype(bool)] if "valid" in summary and len(summary) else summary.iloc[0:0]


def activity_series(summary: pd.DataFrame, start_s: float, end_s: float, bin_s: float = 5.0) -> pd.DataFrame:
    """Per time bin: number of valid bees present and number of visits that started."""
    bin_s = max(0.5, float(bin_s))
    edges = np.arange(start_s, end_s + bin_s, bin_s)
    if len(edges) < 2:
        edges = np.array([start_s, end_s])
    valid = _valid(summary)
    records = []
    for a, b in zip(edges[:-1], edges[1:]):
        present = valid[(valid["start_s"] < b) & (valid["end_s"] >= a)]
        overlap = (np.minimum(present["end_s"], b) - np.maximum(present["start_s"], a)).clip(lower=0).sum()
        records.append({
            "t_start_s": round(float(a), 3),
            "t_end_s": round(float(b), 3),
            "bees_present": int(len(present)),
            "new_visits": int(((valid["start_s"] >= a) & (valid["start_s"] < b)).sum()),
            "mean_bees": float(overlap / (b - a)),
        })
    return pd.DataFrame.from_records(records)


def heatmap_seconds(rows: pd.DataFrame, summary: pd.DataFrame, width: int, height: int, grid=(64, 36),
                    seconds_per_row: float = 1 / 15) -> np.ndarray:
    """Seconds that valid bees spent in each grid cell."""
    gw, gh = int(grid[0]), int(grid[1])
    counts = np.zeros((gh, gw), dtype=np.float32)
    valid_ids = set(_valid(summary)["track_id"])
    if rows is None or rows.empty or not valid_ids:
        return counts
    sel = rows[rows["detected"] & rows["track_id"].isin(valid_ids)]
    col = np.clip((sel["cx"].to_numpy() / width * gw).astype(int), 0, gw - 1)
    row = np.clip((sel["cy"].to_numpy() / height * gh).astype(int), 0, gh - 1)
    np.add.at(counts, (row, col), seconds_per_row)
    return counts


def heatmap_overlay(counts: np.ndarray, background: np.ndarray | None, size=(960, 540), alpha: float = 0.6) -> np.ndarray:
    width, height = size
    base = (cv2.resize(background, (width, height)) if background is not None
            else np.full((height, width, 3), 40, np.uint8))
    if counts.max() <= 0:
        return base
    norm = np.sqrt(counts / counts.max())  # sqrt: keeps rarely visited cells visible
    heat = cv2.resize((norm * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(1, width // 120))
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    weight = (heat.astype(np.float32) / 255.0 * alpha)[..., None]
    return (base * (1 - weight) + colored * weight).astype(np.uint8)


def video_summary(summary: pd.DataFrame, video: dict, run: dict) -> dict:
    valid = _valid(summary)
    processed_s = max(1e-6, float(video.get("processed_s", video.get("duration_s", 0))))
    dwell = valid["duration_s"] if len(valid) else pd.Series(dtype=float)
    rejected = summary.loc[~summary["valid"].astype(bool), "reason"].value_counts().to_dict() if len(summary) else {}
    return {
        **video,
        **run,
        "tracks_total": int(len(summary)),
        "tracks_confirmed": int(summary["confirmed"].sum()) if len(summary) else 0,
        "visits": int(len(valid)),
        "visits_per_minute": round(len(valid) / processed_s * 60, 3),
        "visits_per_class": valid["cls_name"].value_counts().to_dict() if len(valid) else {},
        "rejected": {str(k): int(v) for k, v in rejected.items()},
        "long_stay": int(summary.get("long_stay", pd.Series(dtype=bool)).astype(bool).sum()) if len(summary) else 0,
        "dwell_s": {
            "mean": round(float(dwell.mean()), 3) if len(dwell) else 0.0,
            "median": round(float(dwell.median()), 3) if len(dwell) else 0.0,
            "p90": round(float(dwell.quantile(0.9)), 3) if len(dwell) else 0.0,
            "max": round(float(dwell.max()), 3) if len(dwell) else 0.0,
        },
        "bee_seconds": round(float(dwell.sum()), 3),
        "mean_bees_present": round(float(dwell.sum()) / processed_s, 4),
    }


def collect_sessions(results_root: str | Path, videos_csv: str | Path | None = None) -> pd.DataFrame:
    """One row per processed video (from results/*/summary.json), joined with videos.csv."""
    records = []
    for path in sorted(Path(results_root).glob("*/summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            "video": path.parent.name,   # unique name (the file name RIMG0001 exists in several plots)
            "file": data.get("video", ""),
            "result_dir": str(path.parent),
            "processed_s": data.get("processed_s"),
            "visits": data.get("visits"),
            "visits_per_minute": data.get("visits_per_minute"),
            "mean_dwell_s": data.get("dwell_s", {}).get("mean"),
            "mean_bees_present": data.get("mean_bees_present"),
            "tracks_total": data.get("tracks_total"),
            "tracker": data.get("tracker"),
            "model": Path(str(data.get("model", ""))).name,
            "input": data.get("input"),
            **{f"class_{k}": v for k, v in data.get("visits_per_class", {}).items()},
        })
    sessions = pd.DataFrame.from_records(records)
    if videos_csv and Path(videos_csv).is_file() and not sessions.empty:
        meta = pd.read_csv(videos_csv, dtype=str)
        if "video" in meta:
            sessions = sessions.merge(meta, on="video", how="left")
    return sessions
