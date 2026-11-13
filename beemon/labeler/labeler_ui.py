"""The labeller screen (used by the standalone app.py)."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from bee_canvas import bee_canvas
from labeling import load_batch

NAME = re.compile(r"^(?P<video>.+)__f(?P<frame>\d+)$")


@st.cache_data(show_spinner=False, max_entries=8)
def _load(root: str, stamp: float):
    return load_batch(root)


def _describe(stem: str) -> str:
    match = NAME.match(stem)
    return f"video {match['video']}, frame {int(match['frame'])}" if match else stem


def labeller_page(batches: list[Path]) -> None:
    st.title("Bee labeller")
    if not batches:
        st.warning("No pictures found. Put a picture set (a folder with an `images` folder inside) into the data "
                   "folder, then reload.", title="Nothing to label yet")
        return

    with st.sidebar:
        st.header("Your settings")
        root = st.selectbox("Picture set", batches, format_func=lambda p: p.name, key="bl_batch")
        name = st.text_input("Your name", key="bl_name", placeholder="e.g. Aniket")
        st.caption("Sharing the work: everybody uses the same number of people and a different number.")
        people = st.number_input("People labelling this set", 1, 20, 1, key="bl_people")
        me = st.number_input("I am person number", 1, int(people), 1, key="bl_me")
        show = st.segmented_control("Show", ["To do", "All"], default="To do", key="bl_show")

    try:
        batch = _load(str(root), (root / "images").stat().st_mtime)
    except FileNotFoundError:
        st.info(f"The picture set `{root.name}` has no pictures yet. If frames are still being extracted, wait "
                "until that is finished and press R to reload.", title="No pictures yet")
        return
    part = batch.images[int(me) - 1::int(people)]
    done = batch.done()
    my_done = sum(p.stem in done for p in part)
    with st.sidebar:
        size = st.slider("Size of a new box (pixels)", 15, 120, int(batch.typical_box_size()), key="bl_size",
                         help="Size of the box that appears when you click on a bee. Change one box with +/-.")
        st.progress(my_done / max(1, len(part)), text=f"{my_done} of {len(part)} pictures done")
        st.download_button("Download my labels", batch.export_zip(name or "anonymous"),
                           file_name=f"labels_{root.name}_{(name or 'person' + str(me)).replace(' ', '_')}.zip",
                           mime="application/zip", icon=":material/download:", width="stretch",
                           disabled=my_done == 0, help="Send this file to the person who builds the dataset.")

    if batch.review_mode:
        st.info("Check every picture: the boxes are the computer's guesses. Delete wrong boxes (flowers, stems, "
                "the orange stake), add missing insects, fix the type. A picture without insects is saved empty.",
                title="Review mode")

    stems = [p.stem for p in part]
    current = st.session_state.get("bl_cur")
    if current not in stems:
        current = next((s for s in stems if show == "All" or s not in done), None)
        st.session_state["bl_cur"] = current
    if current is None:
        st.success(f"All {len(part)} pictures of your part are done. Download your labels (sidebar) and send the file.",
                   title="Finished")
        return

    image = part[stems.index(current)]
    boxes, saved = batch.boxes_for(image)
    position = stems.index(current) + 1
    st.markdown(f"**Picture {position} of {len(part)}** · {_describe(current)}"
                + ("  ·  :green-badge[saved]" if saved else "  ·  :orange-badge[not saved yet]"))

    token = f"{root.name}/{current}"
    action = bee_canvas(image, boxes, batch.classes, token=token, box_size=size, key="bl_canvas")
    if action and action.get("token") == token:
        kind = action.get("type")
        index = stems.index(current)
        if kind == "save":
            batch.save(image, action.get("boxes") or [], name or f"person {me}")
            done.add(current)
        if kind in ("save", "skip"):
            following = stems[index + 1:]
            st.session_state["bl_cur"] = next((s for s in following if show == "All" or s not in done), None)
        elif kind == "back":
            st.session_state["bl_cur"] = stems[max(0, index - 1)]
        st.rerun()

    with st.expander("How to label", icon=":material/help:"):
        st.markdown(
            "- **Click on an insect** → a box appears. Press `+` / `-` (or the size buttons) to fit it.\n"
            "- **Click on a box** to select it. Drag it to move it, press `Del` to delete it.\n"
            "- **Type** of the insect: buttons at the top or keys `1`-`9` (applies to the selected box and to "
            "new boxes).\n"
            "- **Zoom** with the mouse wheel, **move around** by dragging the picture, `0` = whole picture.\n"
            "- **No insect in the picture?** Click *No insect here* (removes all boxes), then save.\n"
            "- **Save & next** (or `Enter`) saves the picture. Your work is saved immediately on this computer.\n"
            "- The box should hug the whole insect including the wings. Half hidden but more than half visible: "
            "box it. Not sure what it is: use *other_insect*.")
