#!/usr/bin/env bash
# Start the bee monitoring app (Linux / Mac):   bash run.sh
# The first start installs everything into the folder .venv (5-15 minutes, needs internet).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First start: installing (5-15 minutes)..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    # no NVIDIA graphics card: the small CPU version of PyTorch is enough
    .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install -e . --no-deps
fi
exec .venv/bin/python -m streamlit run app.py
