"""From raw tracks to bee visits: per-track features and the filter rules.

Rules (all thresholds in ``filters:`` of the config):

* too short   — not confirmed, shorter than ``min_duration_s`` or fewer than ``min_detections``.
* static object — the "same place for too long" rule: the track stays within
  ``static_radius_factor`` x its box size of its mean position for at least ``static_min_s``
  AND barely moves itself (mean motion ratio < ``static_motion_max``). A real bee that sits
  still on a flower while feeding still moves its body/wings, so it is only marked
  ``long_stay`` (kept, but worth a look in the gallery).
* background hotspot — the track lies mostly in grid cells where the detector fires in more
  than ``hotspot_fraction`` of all processed frames (a flower bud, a stake, a leaf that the
  detector keeps calling "bee"). A real bee rarely stays in one cell for 40% of a video.
* low confidence — mean confidence below ``min_mean_conf`` (off by default).
* classifier — the optional crop classifier voted for ``not_insect``.

The functions are pure (DataFrames in, DataFrames out), so the dashboard can re-run them
with different thresholds without re-processing the video.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beemon.tracker import ROW_FIELDS

FLAG_COLUMNS = ["flag_short", "flag_static", "flag_hotspot", "flag_lowconf", "flag_classifier"]
REASONS = {
    "flag_short": "too short",
    "flag_static": "static object",
    "flag_hotspot": "background hotspot",
    "flag_lowconf": "low confidence",
    "flag_classifier": "classifier: not an insect",
}

SUMMARY_COLUMNS = [
    "track_id", "confirmed", "start_frame", "end_frame", "start_s", "end_s", "duration_s", "n_det", "n_gap",
    "det_ratio", "mean_conf", "max_conf", "best_frame", "cls_id", "cls_name", "cls_share", "cx_mean", "cy_mean",
    "path_px", "net_disp_px", "max_radius_px", "size_px", "motion_in_mean", "motion_ratio_mean", "hotspot_share",
]


def rows_frame(tracks) -> pd.DataFrame:
    rows = [row for track in tracks for row in track.rows]
    frame = pd.DataFrame(rows, columns=list(ROW_FIELDS))
    return frame.astype({"frame": int, "track_id": int, "cls_id": int, "detected": bool}) if len(frame) else frame


def summarize_tracks(rows: pd.DataFrame, names: dict[int, str], confirmed_ids: set[int] | None = None) -> pd.DataFrame:
    """One row per track with the features the filters and statistics need."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    records = []
    for track_id, group in rows.groupby("track_id", sort=True):
        det = group[group["detected"]]
        if det.empty:
            continue
        xy = det[["cx", "cy"]].to_numpy(float)
        center = xy.mean(axis=0)
        steps = np.hypot(*np.diff(xy, axis=0).T) if len(xy) > 1 else np.zeros(0)
        votes = det.groupby("cls_id")["conf"].sum()
        cls_id = int(votes.idxmax())
        best = det.loc[det["conf"].idxmax()]
        records.append({
            "track_id": int(track_id),
            "confirmed": bool(confirmed_ids is None or track_id in confirmed_ids),
            "start_frame": int(det["frame"].min()),
            "end_frame": int(det["frame"].max()),
            "start_s": float(det["time_s"].min()),
            "end_s": float(det["time_s"].max()),
            "duration_s": float(det["time_s"].max() - det["time_s"].min()),
            "n_det": int(len(det)),
            "n_gap": int((~group["detected"]).sum()),
            "det_ratio": float(len(det) / max(1, len(group))),
            "mean_conf": float(det["conf"].mean()),
            "max_conf": float(det["conf"].max()),
            "best_frame": int(best["frame"]),
            "cls_id": cls_id,
            "cls_name": names.get(cls_id, str(cls_id)),
            "cls_share": float(votes.max() / max(1e-9, votes.sum())),
            "cx_mean": float(center[0]),
            "cy_mean": float(center[1]),
            "path_px": float(steps.sum()),
            "net_disp_px": float(np.hypot(*(xy[-1] - xy[0]))),
            "max_radius_px": float(np.hypot(*(xy - center).T).max()),
            "size_px": float(np.sqrt(np.maximum(1.0, det["w"] * det["h"])).mean()),
            "motion_in_mean": float(det["motion_in"].mean()),
            "motion_ratio_mean": float(det["motion_ratio"].mean()),
            "hotspot_share": 0.0,
        })
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)


def hotspot_map(detections: pd.DataFrame, n_frames: int, width: int, height: int, grid=(32, 18)) -> np.ndarray:
    """Fraction of detector frames in which each grid cell contains at least one detection."""
    gw, gh = int(grid[0]), int(grid[1])
    occupancy = np.zeros((gh, gw), dtype=np.float32)
    if detections is None or detections.empty or n_frames <= 0:
        return occupancy
    cx = ((detections["x1"] + detections["x2"]) / 2).to_numpy()
    cy = ((detections["y1"] + detections["y2"]) / 2).to_numpy()
    col = np.clip((cx / width * gw).astype(int), 0, gw - 1)
    row = np.clip((cy / height * gh).astype(int), 0, gh - 1)
    cells = pd.DataFrame({"frame": detections["frame"].to_numpy(), "row": row, "col": col}).drop_duplicates()
    counts = cells.groupby(["row", "col"]).size()
    for (r, c), count in counts.items():
        occupancy[r, c] = count / n_frames
    return occupancy


def _hotspot_share(rows: pd.DataFrame, hot: np.ndarray, width: int, height: int) -> pd.Series:
    gh, gw = hot.shape
    det = rows[rows["detected"]]
    if det.empty:
        return pd.Series(dtype=float)
    col = np.clip((det["cx"].to_numpy() / width * gw).astype(int), 0, gw - 1)
    row = np.clip((det["cy"].to_numpy() / height * gh).astype(int), 0, gh - 1)
    inside = pd.Series(hot[row, col], index=det.index)
    return inside.groupby(det["track_id"]).mean()


def apply_filters(summary: pd.DataFrame, rows: pd.DataFrame, hotspots: np.ndarray | None, cfg: dict,
                  video: dict) -> pd.DataFrame:
    """Add flag columns, ``long_stay``, ``valid`` and ``reason`` to the track summary."""
    out = summary.copy()
    if out.empty:
        for column in FLAG_COLUMNS + ["long_stay", "valid"]:
            out[column] = pd.Series(dtype=bool)
        out["reason"] = pd.Series(dtype=str)
        return out

    out["flag_short"] = (
        ~out["confirmed"].astype(bool)
        | (out["duration_s"] < float(cfg.get("min_duration_s", 0.4)))
        | (out["n_det"] < int(cfg.get("min_detections", 4)))
    )

    stationary = out["max_radius_px"] <= np.maximum(8.0, float(cfg.get("static_radius_factor", 1.0)) * out["size_px"])
    long_enough = out["duration_s"] >= float(cfg.get("static_min_s", 3.0))
    has_motion = out["motion_ratio_mean"] > 0
    calm = ~has_motion | (out["motion_ratio_mean"] < float(cfg.get("static_motion_max", 1.3)))
    out["flag_static"] = stationary & long_enough & calm
    out["long_stay"] = stationary & long_enough & ~calm

    out["hotspot_share"] = 0.0
    use_hotspots = (
        hotspots is not None
        and hotspots.size > 0
        and float(video.get("processed_s", video.get("duration_s", 0))) >= float(cfg.get("hotspot_min_video_s", 20))
    )
    if use_hotspots:
        hot = hotspots >= float(cfg.get("hotspot_fraction", 0.4))
        share = _hotspot_share(rows, hot.astype(float), int(video["width"]), int(video["height"]))
        out["hotspot_share"] = out["track_id"].map(share).fillna(0.0)
    out["flag_hotspot"] = out["hotspot_share"] >= float(cfg.get("hotspot_track_fraction", 0.5))

    out["flag_lowconf"] = out["mean_conf"] < float(cfg.get("min_mean_conf", 0.0))
    reject = cfg.get("reject_class", "not_insect")
    out["flag_classifier"] = (out["classifier_class"] == reject) if "classifier_class" in out else False

    flags = out[FLAG_COLUMNS].astype(bool)
    out["valid"] = ~flags.any(axis=1)
    out["reason"] = [
        next((REASONS[c] for c in FLAG_COLUMNS if row[c]), "") for _, row in flags.iterrows()
    ]
    return out
