"""Track a video: choose a model and a video, start, watch the detection and tracking live."""

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from app_pages._shared import model_picker, running, stop, video_picker
from beemon.catalog import load_memory, remember
from beemon.config import PROJECT_DIR, load_config
from beemon.video import probe

RESULTS = PROJECT_DIR / "results"
JOBS = RESULTS / "_app" / "jobs"
STRIDES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90]
BUDGETS = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360]
DECODE_FPS = 90.0     # reading (decoding) a 1080p video on a laptop: frames per second
SEEK_S = 0.45         # jumping to a far frame instead (beemon.video.SEEK_GAP)
# seconds per frame on a normal laptop CPU (measured on an i5-6300U): model size -> {image size: s}
CPU_SPEED = {"n": {640: 0.12, 960: 0.23, 1280: 0.38}, "s": {640: 0.25, 960: 0.5, 1280: 0.89},
             "m": {640: 0.65, 960: 1.36, 1280: 2.33}, "l": {640: 0.85, 960: 1.74, 1280: 3.06},
             "x": {640: 1.6, 960: 3.2, 1280: 5.5}}


def seconds_per_frame(weights, info: dict, imgsz: int) -> float:
    """Model time per analysed frame: measured on this computer by earlier runs, else the table."""
    learned = load_memory().get("speed", {}).get(f"{Path(str(weights)).name}@{imgsz}")
    if learned:
        return float(learned)
    return 1.3 * CPU_SPEED.get(info.get("scale") or "m", CPU_SPEED["m"]).get(int(imgsz), 1.5)


def estimate_s(stride: int, length_s: float, fps: float, per_frame: float, save_video: bool) -> float:
    """Reading the video + the model on every analysed frame + the annotated video."""
    targets = length_s * fps / stride
    reading = targets * SEEK_S if stride > 48 else length_s * fps / DECODE_FPS
    video = (targets / 150 + (0 if stride >= 3 else length_s * fps / DECODE_FPS)) if save_video else 0
    return reading + targets * per_frame + video


def learn_speed(job: dict, progress: dict) -> None:
    """After a finished run: remember how long the model took per frame on this computer."""
    if "imgsz" not in job or not progress.get("done") or progress.get("status") != "done":
        return
    key = f"{Path(job['model']).name}@{job['imgsz']}"
    if load_memory().get("speed_from", {}).get(key) == job["started"]:
        return
    targets = float(progress["done"])
    other = estimate_s(job["stride"], job["length_s"], job["fps"], 0.0, job["save_video"])  # reading + video
    per_frame = max(0.05, (float(progress.get("elapsed_s", 0)) - other) / targets)
    remember("speed", key, round(per_frame, 3))
    remember("speed_from", key, job["started"])


def log_tail(job: dict) -> None:
    path = Path(job["log"])
    if path.is_file():
        lines = path.read_text(errors="replace").splitlines()
        st.code("\n".join(lines[-15:]) or "(no output yet)", language="text")


@st.fragment(run_every=1)
def live_view():
    jobs = sorted(JOBS.glob("*.json"), reverse=True) if JOBS.is_dir() else []
    if not jobs:
        return
    st.subheader("Live view")
    for i, job_file in enumerate(jobs[:5]):
        job = json.loads(job_file.read_text(encoding="utf-8"))
        alive = running(job["pid"])
        result = Path(job["result"])
        progress = {}
        try:
            progress = json.loads((result / "progress.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        if i > 0 and not alive:
            with st.expander(f"{job['started']} · {job['name']} · ended"):
                log_tail(job)
            continue
        state = ("running" if alive else "error" if progress.get("status") == "error"
                 else "finished" if progress.get("status") == "done" else "stopped")
        with st.container(border=True):
            top = st.container(horizontal=True, vertical_alignment="center")
            badge = {"running": ":green-badge[running]", "finished": ":blue-badge[finished]",
                     "error": ":red-badge[error]", "stopped": ":gray-badge[stopped]"}[state]
            top.markdown(f"**{job['name']}** {badge} · model `{Path(job['model']).name}`")
            if alive and top.button("Stop", key=f"stop{job['pid']}", icon=":material/stop:"):
                stop(job["pid"])
            done, total = int(progress.get("done", 0)), int(progress.get("total", 0) or 0)
            if alive and total:
                eta = progress.get("eta_s")
                st.progress(done / max(1, total),
                            text=f"{done} / {total} frames · {progress.get('frames_per_s', 0)} frames/s · "
                                 + (f"about {eta / 60:.0f} min left" if eta else "starting…"))
            left, right = st.columns([3, 1])
            live = result / "live.jpg"
            if live.is_file():
                left.image(live.read_bytes(), width="stretch",
                           caption="coloured box = tracked insect (number, confidence) · yellow ? = new, not yet "
                                   "confirmed · grey = detection without a track")
            elif alive:
                left.info("Loading the model and the video…", icon=":material/hourglass_top:")
            right.metric("Insects now", progress.get("bees_now", 0))
            right.metric("Tracks so far", progress.get("tracks_so_far", 0),
                         help="Before the final filters (flower centres etc. are removed at the end).")
            if state == "finished":
                learn_speed(job, progress)
            if state == "finished" and (result / "summary.json").is_file():
                summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
                right.metric("Visits (final)", summary.get("visits", 0))
                st.success("Finished. Open **Results** for the activity chart, heatmap and annotated video.",
                           icon=":material/check_circle:")
            elif state == "error":
                st.error(progress.get("error", "Processing failed — see the log."))
            elif state == "stopped":
                st.warning("Stopped before the end — no results were written.")
            with st.expander("Log"):
                log_tail(job)


st.caption("Choose a model and a video and press Start. You can switch pages or close the browser: "
           "the processing continues in the background.")

with st.container(border=True):
    st.markdown("**1 · Model**")
    weights, info = model_picker("track_model")

with st.container(border=True):
    st.markdown("**2 · Video**")
    video, name, _ = video_picker("track_video")
    vinfo = None
    if video is not None:
        try:
            vinfo = probe(video)
            st.caption(f"{vinfo.width}×{vinfo.height} · {vinfo.fps:.0f} fps · {vinfo.duration_s / 60:.1f} min · "
                       f"saved as `results/{name}`")
        except ValueError as error:
            st.error(str(error))
            video = None

cfg = load_config()
with st.expander("More settings", icon=":material/tune:"):
    a, b = st.columns(2)
    default_size = info.get("imgsz") if info.get("imgsz") in (640, 960, 1280) else 1280
    imgsz = a.segmented_control("Image size", [640, 960, 1280], default=default_size, key="track_imgsz",
                                help="Use the size the model was trained with. Smaller = faster (more frames in the "
                                     "same time), but small insects are missed.") or default_size
    tracker = b.segmented_control("Tracker", ["beetrack", "bytetrack"], default="beetrack", key="track_tracker",
                                  help="beetrack: made for insects (default). bytetrack: standard method, to compare.")
    window = (0.0, vinfo.duration_s if vinfo else 0.0)
    if vinfo is not None:
        window = st.slider("Part of the video (seconds)", 0.0, float(round(vinfo.duration_s, 1)),
                           (0.0, float(round(vinfo.duration_s, 1))), 1.0, key="track_window")
    save_video = st.toggle("Save an annotated video", value=True, key="track_save")

stride = 2
if vinfo is not None:
    with st.container(border=True):
        st.markdown("**3 · Speed**")
        per_frame = seconds_per_frame(weights, info, int(imgsz))
        length_s = window[1] - window[0]
        mode = st.segmented_control("Choose the speed by", ["Time limit", "Frames"], default="Time limit",
                                    key="track_speed_mode")
        if mode == "Frames":
            stride = st.select_slider("Analyse every …th frame", STRIDES, value=2, key="track_stride",
                                      format_func=lambda n: f"every {n}." if n > 1 else "every frame")
        else:
            budget = st.select_slider("Finish within", BUDGETS, value=30, key="track_budget",
                                      format_func=lambda m: f"{m} min" if m < 60 else f"{m / 60:g} h")
            fitting = [n for n in STRIDES if estimate_s(n, length_s, vinfo.fps, per_frame, save_video) <= budget * 60]
            stride = fitting[0] if fitting else STRIDES[-1]
            if not fitting:
                st.warning("Even the fastest setting needs longer. Choose a smaller image size (More settings), "
                           "a shorter part of the video, or a longer time.", icon=":material/timer_off:")
        dt = stride / vinfo.fps
        total = estimate_s(stride, length_s, vinfo.fps, per_frame, save_video)
        targets = int(length_s * vinfo.fps / stride)
        st.markdown(f"Every **{stride}.** frame · {1 / dt:.1f} frames per second of video · {targets} frames · "
                    f"about **{total / 60:.0f} min** on this computer")
        if dt <= 0.1:
            st.caption(":green[:material/check_circle: Best tracking: every visit is followed.]")
        elif dt <= 0.35:
            st.caption(":green[:material/check_circle: Good: also fast flights are followed.]")
        elif dt <= 1.05:
            st.caption(f":orange[:material/info: Rough: visits shorter than about {max(1, round(2 * dt))} s can be "
                       "missed, so the visit count is lower than the truth. Good for comparing videos.]")
        else:
            st.caption(f":red[:material/warning: Very rough: only visits longer than about {round(2 * dt)} s are "
                       "counted. Good for a quick overview; for real counts use a computer with an NVIDIA card "
                       "(every frame, about as fast as the video plays).]")
        if mode != "Frames" and dt > 0.35 and int(imgsz) > 640:
            small = seconds_per_frame(weights, info, 640)
            fits = [n for n in STRIDES if estimate_s(n, length_s, vinfo.fps, small, save_video) <= budget * 60]
            if fits and fits[0] < stride:
                st.caption(f":material/lightbulb: With image size 640 (More settings) the same time allows every "
                           f"{fits[0]}. frame ({vinfo.fps / fits[0]:.1f} per second), but tiny insects are missed "
                           "more often.")

if st.button("Start", type="primary", icon=":material/play_arrow:", disabled=weights is None or video is None):
    JOBS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = JOBS / f"{stamp}.log"
    command = [sys.executable, "-m", "beemon", "process", str(video), "--weights", str(weights),
               "--out", str(RESULTS), "--name", name, "--tracker", tracker or "beetrack",
               "--set", f"detector.imgsz={int(imgsz)}", "--set", f"video.frame_stride={int(stride)}",
               "--set", "output.live_preview=true", "--start-s", str(window[0])]
    if window[1] < vinfo.duration_s - 0.5:
        command += ["--end-s", str(window[1])]
    if not save_video:
        command.append("--no-video")
    extra = {"start_new_session": True} if not sys.platform.startswith("win") else \
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT, **extra)
    (JOBS / f"{stamp}.json").write_text(json.dumps(
        {"pid": process.pid, "command": command, "log": str(log_path), "name": name, "video": str(video),
         "result": str(RESULTS / name), "model": str(weights), "started": stamp, "imgsz": int(imgsz),
         "stride": int(stride), "save_video": bool(save_video), "fps": vinfo.fps,
         "length_s": window[1] - window[0]}), encoding="utf-8")
    st.toast("Started", icon=":material/play_arrow:")

live_view()
