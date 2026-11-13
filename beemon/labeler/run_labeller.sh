#!/bin/bash
# Starts the bee labeller. First start: installs Streamlit into a local .venv folder.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  echo "Installing (only the first time)..."
  if python3 -m venv .venv 2>/dev/null && .venv/bin/pip install -q -r requirements.txt; then
    echo "Installed."
  else
    echo "Could not create a virtual environment; installing Streamlit for your user instead."
    rm -rf .venv
    python3 -m pip install --user -q -r requirements.txt || { echo "Please install Python 3.10+ and try again."; exit 1; }
    exec python3 -m streamlit run labeler/app.py
  fi
fi
exec .venv/bin/streamlit run labeler/app.py
