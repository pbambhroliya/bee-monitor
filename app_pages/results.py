"""Results: what the processed videos show."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from beemon.catalog import CATALOG
from beemon.config import PROJECT_DIR
from beemon.stats import activity_series, collect_sessions

RESULTS = PROJECT_DIR / "results"


@st.cache_data(show_spinner=False, ttl=60)
def load_sessions(stamp: float) -> pd.DataFrame:
    return collect_sessions(RESULTS, CATALOG)


@st.cache_data(show_spinner=False, max_entries=16)
def load_tracks(folder: str, stamp: float) -> pd.DataFrame:
    tracks = pd.read_csv(Path(folder) / "tracks_summary.csv", keep_default_na=False)
    if len(tracks):
        tracks["valid"] = tracks["valid"].astype(str).str.lower() == "true"
    return tracks


summaries = list(RESULTS.glob("*/summary.json"))
if not summaries:
    st.info("No processed videos yet. Go to **Track a video**, choose a model and a video and press Start.",
            icon=":material/videocam:")
    st.stop()

sessions = load_sessions(max(p.stat().st_mtime for p in summaries))
minutes = sessions["processed_s"].sum() / 60
row = st.container(horizontal=True)
row.metric("Videos", len(sessions), border=True)
row.metric("Insect visits", int(sessions["visits"].sum()), border=True)
row.metric("Video analysed", f"{minutes:.0f} min" if minutes >= 2 else f"{minutes * 60:.0f} s", border=True)
row.metric("Visits per minute", f"{sessions['visits'].sum() / max(minutes, 1e-6):.2f}", border=True)

columns = [c for c in ["video", "date", "visits", "visits_per_minute", "mean_dwell_s", "processed_s", "model"]
           if c in sessions]
st.dataframe(sessions[columns], hide_index=True, width="stretch",
             column_config={"video": "Video", "date": "Date", "visits": "Visits",
                            "visits_per_minute": st.column_config.NumberColumn("Visits / min", format="%.2f"),
                            "mean_dwell_s": st.column_config.NumberColumn("Mean visit (s)", format="%.1f"),
                            "processed_s": st.column_config.NumberColumn("Analysed (s)", format="%.0f"),
                            "model": "Model"})

st.subheader("One video")
folder = RESULTS / st.selectbox("Video", sorted(sessions["video"]), key="results_video",
                                label_visibility="collapsed")
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
tracks = load_tracks(str(folder), (folder / "tracks_summary.csv").stat().st_mtime)
valid = tracks[tracks["valid"]] if len(tracks) else tracks
processed = max(1e-6, float(summary["processed_s"]))
row = st.container(horizontal=True)
row.metric("Visits", len(valid), border=True)
row.metric("Visits per minute", f"{len(valid) / processed * 60:.2f}", border=True)
row.metric("Mean visit", f"{valid['duration_s'].mean():.1f} s" if len(valid) else "–", border=True)
row.metric("Rejected tracks", int((~tracks["valid"]).sum()) if len(tracks) else 0, border=True,
           help="Tracks that were not counted: too short, standing still (flower centre, stake), or in a spot "
                "where something is detected all the time.")
st.caption((f"Analysed {processed / 60:.1f} min" if processed >= 120 else f"Analysed {processed:.0f} s")
           + f" · model `{Path(str(summary.get('model', ''))).name}` · "
           f"processing took {summary.get('processing_time_s', 0) / 60:.0f} min")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Activity over time**")
    activity = activity_series(tracks, summary["start_s"], summary["end_s"], 10)
    if len(activity):
        base = alt.Chart(activity).encode(x=alt.X("t_start_s:Q", title="time in video (s)"))
        chart = base.mark_area(opacity=0.35, color="#d98e04").encode(
            y=alt.Y("mean_bees:Q", title="insects in view (mean)")) + base.mark_bar(color="#7a4a00", size=4).encode(
            y="new_visits:Q", tooltip=["t_start_s", "new_visits"])
        st.altair_chart(chart, width="stretch")
        st.caption("Area: insects in view on average. Bars: visits that start in these 10 seconds.")
with right:
    st.markdown("**Where insects spend time**")
    heatmap = folder / "heatmap.png"
    if heatmap.is_file():
        st.image(str(heatmap), width="stretch")

if len(valid):
    st.markdown("**Visits per insect type**")
    per_type = valid["cls_name"].value_counts().rename_axis("type").reset_index(name="visits")
    st.altair_chart(alt.Chart(per_type).mark_bar(color="#d98e04").encode(
        x="visits:Q", y=alt.Y("type:N", sort="-x", title=None)), width="stretch")

video_file = folder / "annotated.mp4"
if video_file.is_file():
    st.markdown("**Annotated video**")
    st.video(str(video_file))
    st.caption("Coloured: counted visits. Grey: rejected tracks with the reason.")
