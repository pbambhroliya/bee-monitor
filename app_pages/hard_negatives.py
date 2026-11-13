"""Hard negatives: the model looks at pictures of a video, you delete its wrong boxes, you send the file."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app_pages._shared import file_field, model_picker, running, stop
from beemon import hardneg
from beemon.catalog import STATUS_TEXT, find_local, load_catalog, load_memory, remember, set_status
from beemon.labeler.bee_canvas import bee_canvas

# ------------------------------------------------------------------ data for this run
memory = load_memory()
rows = load_catalog()
by_name = {row["video"]: row for row in rows}
mine = {v: hardneg.counts(hardneg.load_state(v)) for v in hardneg.sessions()}
totals = {k: sum(c[k] for c in mine.values()) for k in ("reviewed", "wrong_boxes", "insects")}

# ------------------------------------------------------------------ header
top = st.container(horizontal=True, vertical_alignment="bottom")
with top:
    st.markdown("#### Teach the model what is *not* an insect")
    if "hn_name" not in st.session_state:
        st.session_state["hn_name"] = memory.get("me", {}).get("name", "")
    name = st.text_input("Your name", key="hn_name", placeholder="first name", width=220)
if name.strip() and name.strip() != memory.get("me", {}).get("name"):
    remember("me", "name", name.strip())

with st.expander("How it works", icon=":material/help:", expanded=not mine):
    st.markdown(
        "The model sometimes boxes things that are not insects: flower centres, buds, leaves, the orange stake. "
        "Each wrong box you delete is a **hard negative**, and the next model learns not to repeat that mistake.\n\n"
        "1. **Choose a video**: a free one, ideally already downloaded.\n"
        "2. **Find pictures**: the model looks at the video in the background. You can start checking the first "
        "pictures straight away.\n"
        "3. **Check each picture**: delete the wrong boxes, keep the insects. Unsure? The *Does it move?* panel "
        "shows each box 1 second before and after: insects move, flowers do not.\n"
        "4. **Send your work**: download one zip and send it to the person who trains the model.")

row = st.container(horizontal=True)
row.metric("Videos started", len(mine), border=True)
row.metric("Pictures checked", totals["reviewed"], border=True)
row.metric("Wrong boxes deleted", totals["wrong_boxes"], border=True,
           help="Hard negatives: each one teaches the model something.")
row.metric("Insects confirmed", totals["insects"], border=True)

# ------------------------------------------------------------------ 1 · choose a video
with st.container(border=True):
    st.markdown("**1 · Choose a video**")
    local = {r["video"]: find_local(r) for r in rows}
    table = pd.DataFrame([{
        "Video": r["video"], "Date": r["date"],
        "Plot": r["sciebo_folder"].rstrip("/").split("/")[-1].removeprefix(r["video"][:9]),
        "File": r["file"], "Team status": STATUS_TEXT.get(r["status"], r["status"] or "free"),
        "On this computer": local[r["video"]] is not None,
        "Checked by me": (f"{mine[r['video']]['reviewed']} / {mine[r['video']]['pictures']}"
                          if r["video"] in mine else ""),
    } for r in rows], columns=["Video", "Date", "Plot", "File", "Team status", "On this computer",
                               "Checked by me"])
    show = st.segmented_control("Show", ["All", "Free", "On this computer", "Started by me"], default="All",
                                key="hn_filter", label_visibility="collapsed")
    if show == "Free":
        table = table[table["Team status"] == STATUS_TEXT["todo"]]
    elif show == "On this computer":
        table = table[table["On this computer"]]
    elif show == "Started by me":
        table = table[table["Checked by me"] != ""]
    event = st.dataframe(table, hide_index=True, width="stretch", height=min(390, 38 + 35 * max(1, len(table))),
                         on_select="rerun", selection_mode="single-row", key="hn_table",
                         column_config={"On this computer": st.column_config.CheckboxColumn(width="small")})
    if event.selection.rows:
        picked = table.iloc[event.selection.rows[0]]["Video"]
        if picked != st.session_state.get("hn_video"):
            st.session_state["hn_video"] = picked
            remember("me", "last_video", picked)
    if st.session_state.get("hn_video") not in by_name:
        st.session_state["hn_video"] = memory.get("me", {}).get("last_video")
    video = st.session_state.get("hn_video")
    if video not in by_name:
        st.info("Click a row to choose a video.", icon=":material/touch_app:")
        st.stop()
    info = by_name[video]
    video_path = local.get(video)
    st.markdown(f"Selected: **{video}**")
    if info["status"] in ("done", "test", "skip"):
        st.warning(f"This video is marked **{info['status']}** in videos.csv"
                   + (f" ({info['notes']})" if info["notes"] else "") + ". Better choose a free one.",
                   icon=":material/block:")
    elif info["status"] == "taken":
        st.warning(f"{info['taken_by'] or 'Someone'} is already working on this video.", icon=":material/group:")
    if video_path is None:
        folder = info["sciebo_folder"].rstrip("/").split("/")[-1]
        st.warning(f"Not on this computer yet. Download **{info['file']}** from Sciebo "
                   f"`video_records_2024/{info['sciebo_folder']}` into `videos/{folder}/`, or choose the file:",
                   icon=":material/download:")
        video_path = file_field("Video file", f"hn_path_{video}", "video", f"…/{info['file']}")
        if video_path is not None:
            remember("videos", video, str(video_path))
    else:
        st.caption(f"`{video_path}`")

state = hardneg.load_state(video)
frames = state["frames"]

# ------------------------------------------------------------------ 2 · find pictures
with st.container(border=True):
    st.markdown("**2 · Let the model find pictures**")
    with st.expander("Model", icon=":material/neurology:"):
        weights, model_info = model_picker("hn_model")
    a, b = st.columns(2)
    want = a.segmented_control("Pictures", [20, 30, 50], default=30, key="hn_want",
                               help="Only pictures in which the model found something are kept.") or 30
    every = b.segmented_control("One picture every", [5, 10, 20], default=10, key="hn_every",
                                format_func=lambda v: f"{v} s", help="Spread over the whole video.") or 10

    @st.fragment(run_every=2)
    def finder_panel():
        status = hardneg.finder_status(video)
        alive = bool(status.get("pid")) and status.get("status") in ("starting", "running") and \
            running(int(status["pid"]))
        found = len(hardneg.load_state(video)["frames"])
        if alive:
            st.session_state["hn_was_running"] = True
            left, right = st.columns([4, 1], vertical_alignment="center")
            per = status.get("s_per_picture")
            left.progress(min(1.0, found / max(1, status.get("want", want))),
                          text=f"Looking at the video… {found} of {status.get('want', want)} pictures found · "
                               f"{status.get('looked', 0)} looked at"
                               + (f" · {per:.1f} s per picture" if per else " · loading the model"))
            if right.button("Stop", icon=":material/stop:", key="hn_stop"):
                stop(int(status["pid"]))
            if (found and not frames) or found - len(frames) >= 5:
                st.rerun(scope="app")  # new pictures are there: add them to the list below
            return
        if st.session_state.get("hn_was_running"):
            st.session_state["hn_was_running"] = False
            st.rerun(scope="app")  # the search ended: refresh the picture list
        if status.get("status") == "error":
            st.error(f"The search stopped: {status.get('error')}", icon=":material/error:")
        elif status.get("status") == "video_end":
            st.caption(f"The whole video was searched: {found} pictures with boxes. Choose *every 5 s* to look "
                       "at more pictures.")
        label = "Find pictures" if not found else f"Find {int(want)} more pictures"
        if st.button(label, icon=":material/search:", type="primary" if not found else "secondary",
                     disabled=weights is None or video_path is None, key="hn_find"):
            hardneg.start_finder(video, video_path, weights, found + int(want), float(every),
                                 int(model_info.get("imgsz") or 1280))
            st.session_state["hn_was_running"] = True
            st.rerun(scope="fragment")

    finder_panel()


# ------------------------------------------------------------------ 3 · check the pictures
def go_to(video_name: str, pills_key: str, ids: list[str]) -> None:
    number = st.session_state.get(pills_key)
    if number:
        st.session_state[f"hn_current_{video_name}"] = ids[number - 1]
        st.session_state[pills_key] = None


if frames:
    with st.container(border=True):
        done = [f for f in frames if hardneg.frame_done(f)]
        head = st.container(horizontal=True, vertical_alignment="center")
        head.markdown("**3 · Check each picture**")
        head.progress(len(done) / len(frames), text=f"{len(done)} of {len(frames)} checked")
        ids = [f["frame_id"] for f in frames]
        key = f"hn_current_{video}"
        if st.session_state.get(key) not in ids:
            st.session_state[key] = next((f["frame_id"] for f in frames if not hardneg.frame_done(f)), ids[0])
        frame = frames[ids.index(st.session_state[key])]
        position = ids.index(frame["frame_id"])
        is_done = hardneg.frame_done(frame)

        left, right = st.columns([3, 1])
        with left:
            at = int(frame["timestamp_seconds"])
            st.markdown(f"Picture **{position + 1}** of {len(frames)} · at {at // 60}:{at % 60:02d} in the video · "
                        + (":green-badge[:material/check: checked]" if is_done else ":orange-badge[to check]"))
            token = frame["frame_id"] + ("+" if is_done else "")
            action = bee_canvas(hardneg.image_path(video, frame), hardneg.boxes_for_canvas(frame),
                                state["class_names"], token=token, box_size=45, key="hn_canvas")
            if action and action.get("token") == token:
                kind = action.get("type")
                if kind == "save":
                    hardneg.save_review(video, state, frame["frame_id"], action.get("boxes") or [])
                if kind in ("save", "skip"):
                    later = [f for f in frames[position + 1:] if not hardneg.frame_done(f)]
                    st.session_state[key] = later[0]["frame_id"] if later else ids[min(position + 1, len(ids) - 1)]
                elif kind == "back":
                    st.session_state[key] = ids[max(0, position - 1)]
                st.rerun()
        with right:
            st.markdown("**Does it move?**")
            st.caption("Each box 1 s before · now · 1 s after. An insect moves or changes. A flower, leaf or "
                       "the stake looks the same → delete that box.")
            model_boxes = [d for d in frame["detections"] if not d.get("manual")]
            with st.container(height=520, border=False):
                for k, d in enumerate(model_boxes):
                    strip = hardneg.crop_path(video, d["detection_id"])
                    verdict = {"false_positive": ":red-badge[deleted]", "true_positive": ":green-badge[kept]"}.get(
                        d["status"], "")
                    st.markdown(f"**{hardneg.letter(k)}** · {d['predicted_class']} {d['confidence']:.2f} {verdict}")
                    if strip.is_file():
                        st.image(str(strip), width="stretch")
                    else:
                        st.caption("(no motion pictures for this box)")

        nav = st.container(horizontal=True, vertical_alignment="center")
        pills_key = f"hn_pills_{video}"
        nav.pills("Go to picture", list(range(1, len(frames) + 1)), key=pills_key, on_change=go_to,
                  args=(video, pills_key, ids), label_visibility="collapsed",
                  format_func=lambda n: f"{n} ✓" if hardneg.frame_done(frames[n - 1]) else str(n))
        if is_done and nav.button("Undo this picture", icon=":material/undo:"):
            hardneg.reset_review(video, state, frame["frame_id"])
            st.rerun()
        st.caption("Keys: `Del` delete box · `Enter` save & next · `1`–`5` insect type · mouse wheel zoom · "
                   "`0` whole picture. *No insect here* removes all boxes. Not sure → *Skip*.")

# ------------------------------------------------------------------ 4 · send
reviewed = [v for v, c in mine.items() if c["reviewed"]]
if reviewed:
    with st.container(border=True):
        st.markdown("**4 · Send your work**")
        st.caption(f"{totals['reviewed']} checked pictures from {len(reviewed)} video(s) on this computer. Download "
                   "one file with everything and send it to the person who trains the model (e.g. upload it to the "
                   "shared Sciebo folder). You can keep working and send a new file later.")
        who = (name.strip() or "someone").replace(" ", "_")
        st.download_button("Download my work (.zip)", data=lambda: hardneg.export_zip(reviewed, name.strip())[0],
                           file_name=f"hardneg_{who}_{time.strftime('%Y%m%d')}.zip", mime="application/zip",
                           type="primary", icon=":material/download:", on_click="ignore")
        gallery = st.expander(f"What you taught the model: {totals['wrong_boxes']} wrong boxes",
                              icon=":material/school:", on_change="rerun", key="hn_gallery")
        if gallery.open:
            strips = [hardneg.crop_path(v, d["detection_id"]) for v in reviewed
                      for f in hardneg.load_state(v)["frames"] if hardneg.frame_done(f)
                      for d in f["detections"] if d["status"] == "false_positive"]
            strips = [s for s in strips if s.is_file()][-48:]
            with gallery:
                if not strips:
                    st.caption("No pictures of them yet.")
                grid = st.columns(4)
                for i, strip in enumerate(strips):
                    grid[i % 4].image(str(strip), width="stretch")

# ------------------------------------------------------------------ for the person who trains
with st.expander("For the person who trains the model: add files you received", icon=":material/inbox:"):
    st.caption("Add zips from this page, or any YOLO dataset zip (`images/`, `labels/`, `data.yaml`) from another "
               "review tool. If that tool wrote a review_report.csv, add it too: it tells which video each picture "
               "came from. Imported pictures go to `data/labeling/hardneg_*` and are used by `beemon build-dataset` "
               "for the next training. Their videos are marked *done* in videos.csv.")
    uploads = st.file_uploader("Received zips", type="zip", accept_multiple_files=True, key="hn_received")
    report = st.file_uploader("review_report.csv (optional)", type="csv", key="hn_report")
    if uploads and st.button("Import", icon=":material/move_to_inbox:"):
        hardneg.RECEIVED_DIR.mkdir(parents=True, exist_ok=True)
        report_path = None
        if report is not None:
            report_path = hardneg.RECEIVED_DIR / report.name
            report_path.write_bytes(report.getvalue())
        for upload in uploads:
            target = hardneg.RECEIVED_DIR / Path(upload.name).name
            target.write_bytes(upload.getvalue())
            messages = []
            try:
                batch, videos = hardneg.import_contribution(target, report_path, log=messages.append)
                set_status(videos, "done")
                st.success(f"**{upload.name}** → `{batch.relative_to(hardneg.CONTRIB_DIR.parent)}`  \n"
                           + "  \n".join(messages), icon=":material/check:")
            except Exception as error:  # show the problem, continue with the other files
                st.error(f"{upload.name}: {error}")
