"""End-to-end processing of one video: detect -> track -> filter -> statistics -> outputs.

Outputs in ``results/<video name>/``:

    summary.json         totals: visits, dwell time, rejected tracks, settings used
    tracks_summary.csv   one row per track (features, flags, valid, reason)
    tracks_features.csv  the same before filtering (used by `beemon refilter`)
    tracks.csv           one row per track per processed frame (detected or predicted gap)
    detections.csv       every raw detection
    frame_log.csv        per processed frame: detector run?, number of detections
    activity.csv         bees present / new visits per time bin
    tracks_mot.txt       valid tracks in MOT format (for tracking evaluation)
    hotspots.npy         background hotspot map
    background.jpg       median background frame, heatmap.png  time-spent heatmap
    thumbs/              best crop of every track
    annotated.mp4        video with boxes, ids and trails (optional)
    config_used.yaml     exact configuration of this run
"""

from __future__ import annotations

import json
import shutil
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from beemon.config import resolve_path, write_yaml
from beemon.detector import CropClassifier, Detections, YoloDetector, crop_box
from beemon.filters import apply_filters, hotspot_map, rows_frame, summarize_tracks
from beemon.motion import MotionComputer, box_motion, mie_image
from beemon.stats import activity_series, heatmap_overlay, heatmap_seconds, video_summary
from beemon.tracker import TrackerConfig, make_tracker
from beemon.video import iter_frames, frame_range, probe

DETECTION_COLUMNS = ["frame", "time_s", "x1", "y1", "x2", "y2", "conf", "cls_id", "motion_in", "motion_ratio"]


def build_detector(cfg: dict) -> YoloDetector:
    d = cfg["detector"]
    weights = resolve_path(d["weights"])
    return YoloDetector(weights, d.get("imgsz", 1280), d.get("conf", 0.1), d.get("iou", 0.5), d.get("device"),
                        d.get("classes"), d.get("max_det", 300), d.get("tiles"), bool(d.get("augment", False)))


def build_classifier(cfg: dict) -> CropClassifier | None:
    c = cfg.get("classifier") or {}
    if not c.get("weights"):
        return None
    return CropClassifier(resolve_path(c["weights"]), c.get("imgsz", 128), cfg["detector"].get("device"))


class _Progress:
    def __init__(self, path: Path, video: str, total: int, log: Callable[[str], None]):
        self.path, self.video, self.total, self.log = path, video, max(1, total), log
        self.start = time.time()
        self.last = 0.0

    def update(self, done: int, status: str = "running", force: bool = False, extra: dict | None = None):
        now = time.time()
        if not force and now - self.last < 2.0:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else None
        data = {"video": self.video, "status": status, "done": done, "total": self.total,
                "elapsed_s": round(elapsed, 1), "frames_per_s": round(rate, 2),
                "eta_s": None if eta is None else round(eta, 1), **(extra or {})}
        self.path.write_text(json.dumps(data), encoding="utf-8")
        if status == "running":
            eta_text = "?" if eta is None else f"{eta / 60:.1f} min"
            self.log(f"  {self.video}: {done}/{self.total} frames  {rate:.2f} fps  ETA {eta_text}")


def adapt_to_stride(cfg: dict, fps: float) -> dict:
    """Big frame skips (e.g. 1 analysed frame per second) make the count-based rules too strict:
    a 3-second visit then has only 3 detections. Express them in time again (``video.adapt_to_stride``)."""
    vcfg = cfg["video"]
    stride = max(1, int(vcfg.get("frame_stride", 1)))
    if stride <= 3 or not vcfg.get("adapt_to_stride", True):
        return cfg
    import copy

    cfg = copy.deepcopy(cfg)
    dt = stride / max(fps, 1e-6)                      # seconds between analysed frames
    tcfg, fcfg, vcfg = cfg["tracker"], cfg["filters"], cfg["video"]
    tcfg["min_hits"] = min(int(tcfg.get("min_hits", 3)), 2)
    tcfg["max_lost_s"] = max(float(tcfg.get("max_lost_s", 1.0)), 2.5 * dt)
    tcfg["tentative_max_lost_s"] = max(float(tcfg.get("tentative_max_lost_s", 0.3)), 1.5 * dt)
    tcfg["max_gate_px"] = max(float(tcfg.get("max_gate_px", 250)),
                              min(600.0, 0.4 * float(tcfg.get("max_speed_px_s", 1200)) * dt))
    fcfg["min_detections"] = min(int(fcfg.get("min_detections", 4)), 2)
    vcfg["idle_stride"] = 1 if stride >= 10 else min(int(vcfg.get("idle_stride", 1)), 2)
    return cfg


def process_video(video_path: str | Path, cfg: dict, out_root: str | Path = "results", detector=None,
                  classifier=None, log: Callable[[str], None] = print, name: str | None = None) -> Path:
    """Run the full pipeline on one video and write all outputs. Returns the output folder."""
    info = probe(video_path)
    cfg = adapt_to_stride(cfg, info.fps)
    out_dir = Path(out_root) / (name or info.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, out_dir / "config_used.yaml")

    dcfg, mcfg, vcfg, ocfg = cfg["detector"], cfg["motion"], cfg["video"], cfg["output"]
    if detector is None:
        detector = build_detector(cfg)
    if classifier is None:
        classifier = build_classifier(cfg)
    input_mode = str(dcfg.get("input", "rgb")).lower()
    if input_mode not in ("rgb", "mie"):
        raise ValueError(f"detector.input must be rgb or mie, got {input_mode!r}")
    motion_on = bool(mcfg.get("enabled", True)) or input_mode == "mie"
    offset = int(mcfg.get("offset_frames", 4)) if motion_on else 0

    start, end = frame_range(info, vcfg.get("start_s"), vcfg.get("end_s"))
    stride = max(1, int(vcfg.get("frame_stride", 1)))
    idle_stride = max(1, int(vcfg.get("idle_stride", 1)))
    targets = list(range(start, end, stride))
    effective_fps = info.fps / stride
    tracker_kind = cfg["tracker"].get("type", "beetrack")
    tracker = make_tracker(tracker_kind, TrackerConfig.from_dict(cfg["tracker"]), (info.width, info.height),
                           effective_fps)
    motion = MotionComputer(mcfg.get("blur", 5), offset, local_kernel=int(mcfg.get("local_kernel", 0) or 0))

    ccfg = cfg.get("classifier") or {}
    max_crops = int(ccfg.get("max_crops_per_track", 12))
    crop_pad = float(ccfg.get("pad", 0.25))
    crop_size = int(ccfg.get("imgsz", 128))
    crops: dict[int, list[np.ndarray]] = defaultdict(list)
    crop_seen: dict[int, int] = defaultdict(int)
    thumbs: dict[int, tuple[float, np.ndarray]] = {}
    rng = random.Random(0)

    detections: list[tuple] = []
    frame_log: list[tuple] = []
    background_every = max(1, len(targets) // 15)
    background_frames: list[np.ndarray] = []
    progress = _Progress(out_dir / "progress.json", info.stem, len(targets), log)
    log(f"Processing {info.path} ({info.width}x{info.height}, {info.fps:.2f} fps, {info.duration_s:.1f} s) -> {out_dir}")
    log(f"  {len(targets)} frames (stride {stride}), input={input_mode}, tracker={tracker_kind}")
    progress.update(0, force=True)
    live = None
    if ocfg.get("live_preview", True):
        from beemon.live import LivePreview

        live = LivePreview(out_dir, float(ocfg.get("live_every_s", 1.0)),
                           min_hits=int(cfg["tracker"].get("min_hits", 3)))

    idle_counter = 0
    detector_frames = 0
    cache_dir = None
    if ocfg.get("save_video", True) and stride >= 3:
        scale = float(ocfg.get("video_scale", 0.5))
        cache_size = (int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2)
        cache_dir = out_dir / "_frames"
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True)
    try:
        for done, pack in enumerate(iter_frames(info.path, targets, offset), start=1):
            run = True
            if idle_stride > 1 and not tracker.has_active():
                run = idle_counter % idle_stride == 0
                idle_counter += 1
            else:
                idle_counter = 0
            n_det = 0
            if run:
                detector_frames += 1
                fd = motion.difference(pack) if motion_on else None
                image = mie_image(pack.frame, fd) if input_mode == "mie" else pack.frame
                dets: Detections = detector(image)
                n_det = len(dets)
                if fd is not None and n_det:
                    dets.motion_in, dets.motion_ratio = box_motion(fd, dets.xyxy)
                matched = tracker.update(pack.time_s, pack.index, dets)
                if live is not None:
                    live.observe(pack.frame, pack.time_s, dets, matched)
                for j in range(n_det):
                    x1, y1, x2, y2 = dets.xyxy[j]
                    detections.append((pack.index, pack.time_s, x1, y1, x2, y2, dets.conf[j], dets.cls[j],
                                       0.0 if dets.motion_in is None else dets.motion_in[j],
                                       0.0 if dets.motion_ratio is None else dets.motion_ratio[j]))
                for track_id, j in matched:
                    conf = float(dets.conf[j])
                    if ocfg.get("save_thumbnails", True) and conf > thumbs.get(track_id, (-1.0, None))[0]:
                        thumb = crop_box(pack.frame, dets.xyxy[j], pad=0.6)
                        thumbs[track_id] = (conf, cv2.resize(thumb, (96, 96), interpolation=cv2.INTER_AREA))
                    if classifier is not None:
                        crop = cv2.resize(crop_box(pack.frame, dets.xyxy[j], pad=crop_pad), (crop_size, crop_size))
                        crop_seen[track_id] += 1
                        if len(crops[track_id]) < max_crops:
                            crops[track_id].append(crop)
                        else:  # reservoir sampling: keep crops spread over the whole track
                            k = rng.randrange(crop_seen[track_id])
                            if k < max_crops:
                                crops[track_id][k] = crop
            frame_log.append((pack.index, pack.time_s, int(run), n_det))
            if (done - 1) % background_every == 0 and len(background_frames) < 15:
                background_frames.append(cv2.resize(pack.frame, (960, 540), interpolation=cv2.INTER_AREA))
            if cache_dir is not None:  # small copy for the annotated video: no 2nd pass through the video
                small = cv2.resize(pack.frame, cache_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(cache_dir / f"{pack.index:08d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            progress.update(done, extra=live.stats() if live is not None else None)
    except Exception as error:
        progress.update(len(frame_log), status="error", force=True, extra={"error": str(error)})
        raise

    tracks = tracker.finalize()
    rows = rows_frame(tracks)
    confirmed = {tr.id for tr in tracks if tr.ever_confirmed}
    names = getattr(detector, "names", {0: "insect"})
    features = summarize_tracks(rows, names, confirmed)

    if classifier is not None and len(features):
        features = _classify_tracks(features, crops, classifier)

    det_df = pd.DataFrame(detections, columns=DETECTION_COLUMNS)
    log_df = pd.DataFrame(frame_log, columns=["frame", "time_s", "detector_run", "n_det"])
    processed_s = (end - start) / info.fps
    video = {**info.as_dict(), "video": info.stem, "start_s": start / info.fps, "end_s": end / info.fps,
             "processed_s": processed_s, "frame_stride": stride}
    hot_grid = cfg["filters"].get("hotspot_grid", [32, 18])
    hotspots = hotspot_map(det_df, detector_frames, info.width, info.height, hot_grid)

    out_dir.joinpath("thumbs").mkdir(exist_ok=True)
    for track_id, (_, image) in thumbs.items():
        cv2.imwrite(str(out_dir / "thumbs" / f"track_{track_id:05d}.jpg"), image)
    background = np.median(np.stack(background_frames), axis=0).astype(np.uint8) if background_frames else None
    if background is not None:
        cv2.imwrite(str(out_dir / "background.jpg"), background)

    rows.to_csv(out_dir / "tracks_raw.csv", index=False)
    features.to_csv(out_dir / "tracks_features.csv", index=False)
    det_df.to_csv(out_dir / "detections.csv", index=False)
    log_df.to_csv(out_dir / "frame_log.csv", index=False)
    np.save(out_dir / "hotspots.npy", hotspots)
    run = {
        "model": str(dcfg.get("weights")),
        "input": input_mode,
        "tracker": tracker_kind,
        "frames_processed": len(frame_log),
        "detector_frames": detector_frames,
        "detections": int(len(det_df)),
        "processing_time_s": round(time.time() - progress.start, 1),
        "class_names": {int(k): v for k, v in names.items()},
    }
    (out_dir / "run.json").write_text(json.dumps({"video": video, "run": run}, indent=2), encoding="utf-8")

    summary = postprocess(out_dir, cfg, log=log, render=bool(ocfg.get("save_video", True)))
    progress.update(len(frame_log), status="done", force=True, extra=live.stats() if live is not None else None)
    log(f"Done: {summary['visits']} bee visits ({summary['tracks_total']} tracks), "
        f"{run['processing_time_s'] / 60:.1f} min. Results in {out_dir}")
    return out_dir


def _classify_tracks(features: pd.DataFrame, crops: dict[int, list[np.ndarray]], classifier: CropClassifier):
    """Track voting: average the classifier probabilities over the stored crops of each track."""
    labels, shares = [], []
    for track_id in features["track_id"]:
        images = crops.get(int(track_id), [])
        if not images:
            labels.append("")
            shares.append(0.0)
            continue
        mean = classifier.probabilities(images).mean(axis=0)
        best = int(mean.argmax())
        labels.append(classifier.names.get(best, str(best)))
        shares.append(float(mean[best]))
    features = features.copy()
    features["classifier_class"] = labels
    features["classifier_share"] = shares
    has_label = features["classifier_class"] != ""
    features.loc[has_label, "cls_name"] = features.loc[has_label, "classifier_class"]
    return features


def postprocess(out_dir: str | Path, cfg: dict, log: Callable[[str], None] = print, render: bool = False) -> dict:
    """Apply the filters and statistics to saved raw tracks. Fast: no detection is re-run.

    Used at the end of ``process_video`` and by ``beemon refilter`` to try other thresholds.
    """
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    video, run = meta["video"], meta["run"]
    rows = pd.read_csv(out_dir / "tracks_raw.csv")
    if len(rows):
        rows["detected"] = rows["detected"].astype(bool)
    features = pd.read_csv(out_dir / "tracks_features.csv", keep_default_na=False)
    if len(features):
        features["confirmed"] = features["confirmed"].astype(str).str.lower().isin(["true", "1"])
    hotspots = np.load(out_dir / "hotspots.npy")

    fcfg = {**cfg["filters"], "reject_class": (cfg.get("classifier") or {}).get("reject_class", "not_insect")}
    summary = apply_filters(features, rows, hotspots, fcfg, video)
    summary.to_csv(out_dir / "tracks_summary.csv", index=False)

    status = summary.set_index("track_id")[["valid", "reason", "cls_name"]] if len(summary) else None
    tracks = rows.copy()
    if status is not None and len(tracks):
        tracks = tracks.join(status, on="track_id")
    tracks.to_csv(out_dir / "tracks.csv", index=False)
    _write_mot(tracks, out_dir / "tracks_mot.txt", only_valid=True)
    _write_mot(tracks, out_dir / "tracks_mot_all.txt", only_valid=False)

    ocfg = cfg["output"]
    activity = activity_series(summary, video["start_s"], video["end_s"], ocfg.get("activity_bin_s", 5.0))
    activity.to_csv(out_dir / "activity.csv", index=False)
    seconds_per_row = video["frame_stride"] / video["fps"]
    heat = heatmap_seconds(rows, summary, video["width"], video["height"], ocfg.get("heatmap_grid", [64, 36]),
                           seconds_per_row)
    np.save(out_dir / "heatmap.npy", heat)
    background = cv2.imread(str(out_dir / "background.jpg")) if (out_dir / "background.jpg").is_file() else None
    cv2.imwrite(str(out_dir / "heatmap.png"), heatmap_overlay(heat, background))

    result = video_summary(summary, {**video}, {**run, "filters": cfg["filters"]})
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    log(f"  filters: {result['visits']} valid visits, rejected {result['rejected']}")

    if render:
        from beemon.render import render_video

        render_video(video["path"], tracks, summary, out_dir / "annotated.mp4", cfg, video, log=log,
                     frame_dir=out_dir / "_frames")
    shutil.rmtree(out_dir / "_frames", ignore_errors=True)
    return result


def _write_mot(tracks: pd.DataFrame, path: Path, only_valid: bool) -> None:
    """MOT challenge format: frame(1-based), id, left, top, width, height, conf, -1, -1, -1."""
    lines = []
    if len(tracks):
        sel = tracks[tracks["detected"]]
        if only_valid and "valid" in sel:
            sel = sel[sel["valid"].astype(bool)]
        for row in sel.itertuples(index=False):
            lines.append(f"{row.frame + 1},{row.track_id},{row.cx - row.w / 2:.1f},{row.cy - row.h / 2:.1f},"
                         f"{row.w:.1f},{row.h:.1f},{row.conf:.3f},-1,-1,-1")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
