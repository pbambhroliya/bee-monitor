"""Bee labeller — a small app to put boxes on bees.   Start it with:

    streamlit run beemon/labeler/app.py           (inside the bee-monitor project)
    streamlit run app.py                          (inside an unzipped labelling package)
    streamlit run app.py -- --data /some/folder   (pictures somewhere else)

Only needs Streamlit (pip install streamlit).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

from labeler_ui import labeller_page  # noqa: E402
from labeling import find_batches  # noqa: E402

st.set_page_config(page_title="Bee labeller", page_icon="🐝", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="folder with picture sets (or one picture set)")
args, _ = parser.parse_known_args(sys.argv[1:])

places = ([Path(args.data).expanduser()] if args.data else
          [HERE.parent / "data", HERE.parents[1] / "data" / "labeling"])  # unzipped package / inside the project
batches = []
for place in places:
    batches += [b for b in find_batches(place) if b not in batches]

labeller_page(batches)
