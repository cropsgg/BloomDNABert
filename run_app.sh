#!/usr/bin/env bash
# Run the Bloom-DNABERT Gradio dashboard (use from project root).
set -e
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  PIP_DISABLE_PIP_VERSION_CHECK=1 .venv/bin/pip install -q -r requirements.txt
fi
echo "Starting dashboard at http://127.0.0.1:7860"
.venv/bin/python app.py
