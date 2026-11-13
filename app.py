"""Bee monitoring app.   Start it with:   streamlit run app.py      (or run.sh / run.bat)

Pages (files in app_pages/):
  Results          what the processed videos show: bee visits, activity over time, heatmap, video
  Track a video    choose a model and a video, watch detection and tracking live
  Hard negatives   help the model: delete its wrong boxes, send the result (see CONTRIBUTING.md)
"""

import streamlit as st

st.set_page_config(page_title="Bee monitoring", page_icon="🐝", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/results.py", title="Results", icon=":material/insights:", default=True),
        st.Page("app_pages/track.py", title="Track a video", icon=":material/videocam:"),
        st.Page("app_pages/hard_negatives.py", title="Hard negatives", icon=":material/rule:"),
    ],
    position="top",
)
page.run()
